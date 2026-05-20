import argparse
import gc
import json
import os
import sys
import time
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.metrics import compute_wer_cer
SNR_LEVELS = [0, 5, 10, 15, 20]
BASELINE_NAME = 'gigaam_v2_ctc'

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Evaluate GigaAM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--eval-root', type=str, required=True, help='Eval set root')
    p.add_argument('--farfield-manifest', type=str, default=None, help='Farfield manifest')
    p.add_argument('--results-dir', type=str, required=True, help='Output directory')
    p.add_argument('--extra-manifests', nargs='+', default=None, metavar='NAME=PATH', help='Extra manifests: name=path')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size')
    p.add_argument('--test-limit', type=int, default=None, help='Sample limit')
    p.add_argument('--force-rerun', action='store_true', help='Overwrite existing results')
    return p

def parse_extra_manifests(specs: list | None) -> dict:
    out = {}
    if not specs:
        return out
    for spec in specs:
        if '=' not in spec:
            raise SystemExit(f'[ERROR] invalid extra manifest: {spec!r}')
        name, path = spec.split('=', 1)
        name, path = (name.strip(), path.strip())
        if not os.path.exists(path):
            raise SystemExit(f'[ERROR] missing manifest: {path}')
        out[name] = path
    return out

def load_manifest(path: str, limit: int | None=None) -> list[dict]:
    manifest_dir = os.path.dirname(os.path.abspath(path))
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ('audio_filepath', 'audio', 'wav', 'path', 'file'):
                if key in rec and isinstance(rec[key], str):
                    if not os.path.isabs(rec[key]):
                        rec[key] = os.path.join(manifest_dir, rec[key])
                    if not os.path.exists(rec[key]):
                        raise FileNotFoundError(f'missing audio file: {rec[key]}')
                    break
            records.append(rec)
            if limit and len(records) >= limit:
                break
    return records

def collect_test_sets(args) -> dict[str, list[dict]]:
    sets = {}
    clean_path = os.path.join(args.eval_root, 'clean', 'manifest.jsonl')
    if not os.path.exists(clean_path):
        raise SystemExit(f'[ERROR] missing manifest: {clean_path}')
    sets['clean'] = load_manifest(clean_path, args.test_limit)
    for snr in SNR_LEVELS:
        path = os.path.join(args.eval_root, f'snr_{snr}db', 'manifest.jsonl')
        if os.path.exists(path):
            sets[f'snr_{snr}db'] = load_manifest(path, args.test_limit)
        else:
            print(f'[WARN] missing manifest: {path}')
    if args.farfield_manifest:
        if os.path.exists(args.farfield_manifest):
            sets['farfield'] = load_manifest(args.farfield_manifest, args.test_limit)
        else:
            print(f'[WARN] missing farfield manifest: {args.farfield_manifest}')
    extra = parse_extra_manifests(args.extra_manifests)
    for name, path in extra.items():
        sets[name] = load_manifest(path, args.test_limit)
    return sets

def get_audio_path(rec: dict) -> str:
    for key in ('audio_filepath', 'audio', 'wav', 'path', 'file'):
        if key in rec and isinstance(rec[key], str):
            return rec[key]
    raise KeyError(f'missing audio path: {rec}')

def get_reference(rec: dict) -> str:
    for key in ('sentence', 'text', 'transcription', 'transcript'):
        if key in rec and isinstance(rec[key], str):
            return rec[key]
    raise KeyError(f'missing reference text: {rec}')

def evaluate_gigaam_on_set(model, records: list[dict], desc: str, batch_size: int) -> dict:
    from gigaam.preprocess import load_audio, SAMPLE_RATE
    predictions = []
    references = []
    total_time = 0.0
    n = len(records)
    device = model._device
    dtype = model._dtype
    with tqdm(total=n, desc=desc) as pbar:
        for start in range(0, n, batch_size):
            batch = records[start:start + batch_size]
            wavs = []
            lengths = []
            refs = []
            for rec in batch:
                audio_path = get_audio_path(rec)
                wav = load_audio(audio_path, SAMPLE_RATE)
                wavs.append(wav)
                lengths.append(wav.shape[-1])
                refs.append(get_reference(rec))
            max_len = max(lengths)
            padded = torch.zeros(len(wavs), max_len, dtype=torch.float32)
            for i, wav in enumerate(wavs):
                padded[i, :wav.shape[-1]] = wav
            padded = padded.to(device).to(dtype)
            lengths_t = torch.tensor(lengths, device=device)
            t0 = time.time()
            with torch.inference_mode():
                encoded, encoded_len = model.forward(padded, lengths_t)
                decoded = model.decoding.decode(model.head, encoded, encoded_len)
            total_time += time.time() - t0
            for text, _, _ in decoded:
                predictions.append(text)
            references.extend(refs)
            pbar.update(len(batch))
    metrics = compute_wer_cer(predictions, references)
    metrics['total_time'] = round(total_time, 2)
    metrics['avg_time_per_sample'] = round(total_time / n, 4) if n > 0 else 0.0
    metrics['num_samples'] = n
    return metrics

def main():
    args = get_parser().parse_args()
    if not torch.cuda.is_available():
        print('[WARN] CUDA is not available')
    os.makedirs(args.results_dir, exist_ok=True)
    result_path = os.path.join(args.results_dir, f'{BASELINE_NAME}.json')
    existing: dict = {}
    if os.path.exists(result_path) and (not args.force_rerun):
        with open(result_path, encoding='utf-8') as f:
            existing = json.load(f)
        print(f'existing: {list(existing)}')
    test_sets = collect_test_sets(args)
    for name, recs in test_sets.items():
        print(f'  {name}: {len(recs)}')
    to_evaluate = {name: recs for name, recs in test_sets.items() if name not in existing}
    if not to_evaluate:
        print('up to date')
        _print_summary(existing)
        return
    print('\nloading gigaam')
    import gigaam
    model = gigaam.load_model('v2_ctc')
    model.eval()
    print(f'device: {model._device}, dtype: {model._dtype}')
    results = dict(existing)
    for name, records in to_evaluate.items():
        print(f'\ncondition: {name} ({len(records)})')
        m = evaluate_gigaam_on_set(model, records, desc=f'gigaam_v2_ctc | {name}', batch_size=args.batch_size)
        results[name] = m
        print(f"  WER={m['wer']:.2f}%  CER={m['cer']:.2f}%  time={m['total_time']:.1f}s")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nsaved: {result_path}')
    _print_summary(results)
    del model
    torch.cuda.empty_cache()
    gc.collect()

def _print_summary(results: dict):
    order = ['clean'] + [f'snr_{s}db' for s in SNR_LEVELS] + ['farfield', 'open_stt_calls', 'rudevices_10pct', 'cv25_test']
    print('\nGigaAM-v2-CTC WER/CER')
    print(f"{'Condition':<20} {'WER':>8} {'CER':>8}")
    print('-' * 40)
    for name in order:
        if name in results:
            m = results[name]
            print(f"  {name:<18} {m['wer']:>7.2f}% {m['cer']:>7.2f}%")
if __name__ == '__main__':
    main()
