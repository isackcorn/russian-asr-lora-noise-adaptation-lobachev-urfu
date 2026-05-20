import argparse
import gc
import json
import os
import sys
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset_loader import load_manifest_as_dataset
from src.evaluation.evaluator import evaluate_model
from src.models.lora_setup import load_processor
ALL_BASELINES = ['zero_shot', 'full_ft_clean', 'lora_clean', 'full_ft_noisy', 'lora_noisy', 'lora_noisy_r32']
SNR_LEVELS = [0, 5, 10, 15, 20]

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Evaluate checkpoints', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints-root', type=str, required=True, help='Checkpoint root')
    p.add_argument('--eval-root', type=str, required=True, help='Eval set root')
    p.add_argument('--farfield-manifest', type=str, default=None, help='Farfield manifest')
    p.add_argument('--results-dir', type=str, required=True, help='Output directory')
    p.add_argument('--model-id', type=str, default='openai/whisper-large-v3', help='Base model')
    p.add_argument('--baselines', nargs='+', default=ALL_BASELINES, choices=ALL_BASELINES, metavar='BASELINE', help='Baselines to evaluate')
    p.add_argument('--test-limit', type=int, default=None, help='Test sample limit')
    p.add_argument('--batch-size', type=int, default=16, help='Inference batch size')
    p.add_argument('--force-rerun', action='store_true', help='Overwrite existing results')
    p.add_argument('--extra-manifests', nargs='+', default=None, metavar='NAME=PATH', help='Extra manifests: name=path')
    return p

def parse_extra_manifests(specs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not specs:
        return out
    for spec in specs:
        if '=' not in spec:
            raise SystemExit(f'[ERROR] invalid extra manifest: {spec!r}')
        name, path = spec.split('=', 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise SystemExit(f'[ERROR] invalid extra manifest: {spec!r}')
        if name in {'clean', 'farfield'} or name.startswith('snr_'):
            raise SystemExit(f'[ERROR] reserved condition name: {name!r}')
        if not os.path.exists(path):
            raise SystemExit(f'[ERROR] missing manifest: {path}')
        out[name] = path
    return out

def load_test_sets(eval_root: str, test_limit: int | None, farfield_manifest: str | None, extra_manifests: dict[str, str] | None=None) -> dict:
    sets = {}
    clean_path = os.path.join(eval_root, 'clean', 'manifest.jsonl')
    if not os.path.exists(clean_path):
        raise SystemExit(f'[ERROR] missing manifest: {clean_path}')
    sets['clean'] = load_manifest_as_dataset(clean_path, limit=test_limit)
    for snr in SNR_LEVELS:
        path = os.path.join(eval_root, f'snr_{snr}db', 'manifest.jsonl')
        if not os.path.exists(path):
            raise FileNotFoundError(f'missing manifest: {path}')
        sets[f'snr_{snr}db'] = load_manifest_as_dataset(path, limit=test_limit)
    if farfield_manifest and os.path.exists(farfield_manifest):
        sets['farfield'] = load_manifest_as_dataset(farfield_manifest, limit=test_limit)
    elif farfield_manifest:
        raise FileNotFoundError(f'missing farfield manifest: {farfield_manifest}')
    for name, path in (extra_manifests or {}).items():
        sets[name] = load_manifest_as_dataset(path, limit=test_limit)
    return sets

def _eval_all(model, processor, test_sets: dict, label: str, device: str, dtype, batch_size: int) -> dict:
    results = {}
    for test_name, ds in test_sets.items():
        m = evaluate_model(model, processor, ds, device=device, dtype=dtype, desc=f'{label} | {test_name}', batch_size=batch_size)
        results[test_name] = m
        print(f"    {test_name}: WER={m['wer']:.2f}%  CER={m['cer']:.2f}%")
    return results

def load_model_for_baseline(baseline: str, args, dtype) -> tuple:
    if baseline == 'zero_shot':
        print(f'  loading model: {args.model_id}')
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype)
        return (model, 'zero_shot')
    ckpt_dir = os.path.join(args.checkpoints_root, baseline)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f'missing checkpoint: {ckpt_dir}')
    if baseline.startswith('full_ft'):
        print(f'  loading checkpoint: {ckpt_dir}')
        model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir, torch_dtype=dtype)
        model.config.use_cache = True
        return (model, baseline)
    if baseline.startswith('lora'):
        adapter_dir = os.path.join(ckpt_dir, 'lora-adapter')
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f'missing adapter: {adapter_dir}')
        print(f'  loading base: {args.model_id}')
        base = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype)
        print(f'  loading adapter: {adapter_dir}')
        model = PeftModel.from_pretrained(base, adapter_dir)
        return (model, baseline)
    raise ValueError(f'unknown baseline: {baseline}')

def print_summary(all_results: dict):
    standard = ['clean'] + [f'snr_{s}db' for s in SNR_LEVELS]
    seen = set(standard)
    extra_names: list[str] = []
    for res in all_results.values():
        for k in res.keys():
            if k not in seen:
                extra_names.append(k)
                seen.add(k)
    test_names = standard + extra_names
    col_w = 12
    print('\nWER summary')
    header = f"{'Baseline':<22}" + ''.join((f'{t:>{col_w}}' for t in test_names))
    print(header)
    print('-' * len(header))
    for name, res in all_results.items():
        row = f'{name:<22}'
        for t in test_names:
            row += f"{res[t]['wer']:>{col_w - 1}.2f}%" if t in res else f"{'-':>{col_w}}"
        print(row)

def main():
    args = get_parser().parse_args()
    if not torch.cuda.is_available():
        print('CUDA is not available', file=sys.stderr)
        sys.exit(1)
    device = 'cuda'
    dtype = torch.bfloat16
    os.makedirs(args.results_dir, exist_ok=True)
    print('evaluation')
    print(f'  model:       {args.model_id}')
    print(f'  baselines:   {args.baselines}')
    print(f'  checkpoints: {args.checkpoints_root}')
    print(f'  eval root:   {args.eval_root}')
    print(f"  farfield:    {args.farfield_manifest or '-'}")
    print(f'  output:      {args.results_dir}')
    extra = parse_extra_manifests(args.extra_manifests)
    if extra:
        print(f'  extra:       {list(extra.keys())}')
    print('\nloading test sets')
    test_sets = load_test_sets(args.eval_root, args.test_limit, args.farfield_manifest, extra_manifests=extra)
    for name, ds in test_sets.items():
        print(f'  {name}: {len(ds)}')
    processor = load_processor(model_id=args.model_id)
    all_results: dict = {}
    for baseline in args.baselines:
        result_path = os.path.join(args.results_dir, f'{baseline}.json')
        existing: dict = {}
        if os.path.exists(result_path) and (not args.force_rerun):
            with open(result_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        to_evaluate = {name: ds for name, ds in test_sets.items() if name not in existing}
        if not to_evaluate:
            print(f'\n[skip] {baseline}: up to date')
            all_results[baseline] = existing
            continue
        print(f'\nbaseline: {baseline}')
        if existing:
            print(f'  existing: {list(existing)}')
            print(f'  pending:  {list(to_evaluate)}')
        else:
            print(f'  pending:  {list(to_evaluate)}')
        model, label = load_model_for_baseline(baseline, args, dtype)
        new_results = _eval_all(model, processor, to_evaluate, label, device, dtype, args.batch_size)
        merged = {**existing, **new_results}
        all_results[baseline] = merged
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f'  saved: {result_path} ({len(merged)})')
        del model
        torch.cuda.empty_cache()
        gc.collect()
    summary_path = os.path.join(args.results_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f'\nsummary: {summary_path}')
    print_summary(all_results)
if __name__ == '__main__':
    main()
