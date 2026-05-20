import argparse
import json
import os
import random
import sys
import numpy as np
import soundfile as sf
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.augmentation import mix_at_exact_snr
DEFAULT_SOURCE = '/root/DisserRealization/data/golos_wav/test_opus/crowd/manifest.jsonl'
DEFAULT_NOISE_ROOT = '/root/DisserRealization/noise_data'
DEFAULT_OUT_DIR = '/root/DisserRealization/eval/thesis_test_sets'
DEFAULT_N_SAMPLES = 1500
SNR_LEVELS = [0, 5, 10, 15, 20]
SEED = 42

def _resolve_audio_path(manifest_path: str, record: dict) -> str:
    raw = record.get('wav') or record.get('audio_filepath')
    if not raw:
        raise KeyError("missing 'wav' or 'audio_filepath'")
    if os.path.isabs(raw):
        return raw
    base = os.path.dirname(os.path.abspath(manifest_path))
    for cand in (os.path.join(base, raw), os.path.join(os.path.dirname(base), raw)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError(f'missing audio file: {raw}')

def load_records(manifest_path: str) -> list[dict]:
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def collect_noise_files(noise_root: str) -> list[str]:
    splits_test = os.path.join(noise_root, 'splits', 'test')
    if not os.path.isdir(splits_test):
        raise SystemExit(f'[ERROR] missing noise split: {splits_test}')
    categories = ['musan_noise', 'musan_music', 'musan_speech', 'esc50', 'demand']
    files = []
    for cat in categories:
        cat_dir = os.path.join(splits_test, cat)
        if not os.path.isdir(cat_dir):
            raise FileNotFoundError(f'missing noise category: {cat_dir}')
        for root, _, fnames in os.walk(cat_dir):
            for fname in fnames:
                if fname.lower().endswith(('.wav', '.flac', '.opus', '.mp3')):
                    files.append(os.path.join(root, fname))
    if not files:
        raise SystemExit(f'[ERROR] no noise files: {splits_test}')
    return files

def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, n)

def _link_or_copy(src: str, dst: str):
    if os.path.exists(dst):
        return
    os.link(src, dst)

def build_clean(records: list[dict], source_manifest: str, out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    out_records = []
    for rec in records:
        src_wav = _resolve_audio_path(source_manifest, rec)
        rec_id = rec['id']
        dst_wav = os.path.join(out_dir, f'{rec_id}.wav')
        _link_or_copy(src_wav, dst_wav)
        out_records.append({'id': rec_id, 'text': rec['text'], 'wav': dst_wav, 'duration': rec['duration']})
    manifest = os.path.join(out_dir, 'manifest.jsonl')
    with open(manifest, 'w', encoding='utf-8') as f:
        for r in out_records:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')
    print(f'  clean: {len(out_records)} -> {out_dir}')
    return out_records

def build_snr(clean_records: list[dict], noise_files: list[str], snr_db: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_records = []
    for idx, rec in enumerate(tqdm(clean_records, desc=f'SNR={snr_db} dB')):
        clean_audio, sr = sf.read(rec['wav'], dtype='float32')
        if clean_audio.ndim > 1:
            clean_audio = clean_audio.mean(axis=1)
        noise_idx = (SEED + idx) % len(noise_files)
        noise_path = noise_files[noise_idx]
        noise_audio, noise_sr = sf.read(noise_path, dtype='float32')
        if noise_audio.ndim > 1:
            noise_audio = noise_audio.mean(axis=1)
        if noise_sr != sr:
            from scipy import signal
            gcd = np.gcd(int(sr), int(noise_sr))
            noise_audio = signal.resample_poly(noise_audio, up=sr // gcd, down=noise_sr // gcd).astype(np.float32)
        mixed = mix_at_exact_snr(clean_audio, noise_audio, snr_db, seed=SEED + idx)
        dst_wav = os.path.join(out_dir, f"{rec['id']}.wav")
        sf.write(dst_wav, mixed, sr)
        out_records.append({'id': rec['id'], 'text': rec['text'], 'wav': dst_wav, 'duration': rec['duration'], 'snr_db': snr_db})
    manifest = os.path.join(out_dir, 'manifest.jsonl')
    with open(manifest, 'w', encoding='utf-8') as f:
        for r in out_records:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')
    print(f'  snr_{snr_db}db: {len(out_records)} -> {out_dir}')

def main():
    p = argparse.ArgumentParser(description='Build eval sets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--source-manifest', default=DEFAULT_SOURCE, help='Source manifest')
    p.add_argument('--noise-root', default=DEFAULT_NOISE_ROOT, help='Noise root')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Output directory')
    p.add_argument('--n-samples', type=int, default=DEFAULT_N_SAMPLES, help='Sample count')
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    if not os.path.exists(args.source_manifest):
        raise SystemExit(f'[ERROR] missing source manifest: {args.source_manifest}')
    print('build eval sets')
    print(f'  source: {args.source_manifest}')
    print(f'  noise:  {args.noise_root}/splits/test/')
    print(f'  output: {args.out_dir}')
    print(f'  size:   {args.n_samples}')
    print(f'  seed:   {args.seed}')
    all_records = load_records(args.source_manifest)
    print(f'\nloaded: {len(all_records)}')
    sampled = sample_records(all_records, args.n_samples, args.seed)
    print(f'sampled: {len(sampled)}')
    noise_files = collect_noise_files(args.noise_root)
    print(f'noise files: {len(noise_files)}')
    clean_dir = os.path.join(args.out_dir, 'clean')
    clean_records = build_clean(sampled, args.source_manifest, clean_dir)
    for snr in SNR_LEVELS:
        snr_dir = os.path.join(args.out_dir, f'snr_{snr}db')
        build_snr(clean_records, noise_files, snr, snr_dir)
    print(f'\ndone: {args.out_dir}')
if __name__ == '__main__':
    main()
