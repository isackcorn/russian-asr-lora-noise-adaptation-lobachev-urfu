import argparse
import json
import os
import random
import sys
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

def read_manifest(manifest_csv: str) -> list[tuple[str, str, float]]:
    rows = []
    with open(manifest_csv, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip('\n').rstrip('\r')
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                raise ValueError(f'invalid CSV row: {lineno}')
            wav_path = parts[0].strip()
            text_path = parts[1].strip()
            try:
                duration = float(parts[2].strip())
            except ValueError:
                raise ValueError(f'invalid duration: {lineno}')
            rows.append((wav_path, text_path, duration))
    return rows

def resolve_path(audio_root: str, rel_path: str) -> str:
    rel_path = rel_path.lstrip('/')
    candidates = [os.path.join(audio_root, rel_path), os.path.join(audio_root, os.path.basename(os.path.dirname(rel_path)), os.path.basename(rel_path))]
    parts = Path(rel_path).parts
    for i in range(len(parts)):
        candidates.append(os.path.join(audio_root, *parts[i:]))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f'missing file: {rel_path}')

def read_text(text_path: str) -> str | None:
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def probe_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    if not info.samplerate:
        raise ValueError(f'invalid sample rate: {audio_path}')
    return info.frames / info.samplerate

def main():
    p = argparse.ArgumentParser(description='Convert Open STT manifest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--manifest-csv', required=True, help='Input CSV')
    p.add_argument('--audio-root', required=True, help='Audio root')
    p.add_argument('--out', required=True, help='Output manifest')
    p.add_argument('--sample-n', type=int, default=3000, help='Sample count')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-duration', type=float, default=30.0, help='Max duration')
    p.add_argument('--min-duration', type=float, default=0.5, help='Min duration')
    p.add_argument('--min-text-len', type=int, default=2, help='Min text length')
    p.add_argument('--verify-duration', action='store_true', help='Read duration from files')
    args = p.parse_args()
    if not os.path.isfile(args.manifest_csv):
        raise SystemExit(f'[ERROR] missing CSV: {args.manifest_csv}')
    if not os.path.isdir(args.audio_root):
        raise SystemExit(f'[ERROR] missing audio root: {args.audio_root}')
    print('convert open_stt')
    print(f'  csv:    {args.manifest_csv}')
    print(f'  root:   {args.audio_root}')
    print(f'  output: {args.out}')
    print(f'  sample: {args.sample_n}')
    print(f'  range:  [{args.min_duration}, {args.max_duration}]')
    rows = read_manifest(args.manifest_csv)
    print(f'\nrows: {len(rows)}')
    pre_filtered = [r for r in rows if args.min_duration <= r[2] <= args.max_duration]
    print(f'filtered: {len(pre_filtered)}')
    if not pre_filtered:
        raise SystemExit('[ERROR] no records after duration filter')
    rng = random.Random(args.seed)
    target = min(args.sample_n, len(pre_filtered))
    candidates = rng.sample(pre_filtered, target)
    out_records = []
    for wav_rel, txt_rel, dur in tqdm(candidates, desc='Resolve & validate'):
        if len(out_records) >= target:
            break
        audio_abs = resolve_path(args.audio_root, wav_rel)
        text_abs = resolve_path(args.audio_root, txt_rel)
        text = read_text(text_abs)
        if not text or len(text) < args.min_text_len:
            raise ValueError(f'invalid text: {text_abs}')
        if args.verify_duration:
            real_dur = probe_duration_seconds(audio_abs)
            if real_dur < args.min_duration or real_dur > args.max_duration:
                raise ValueError(f'invalid duration: {audio_abs}')
            dur = real_dur
        rec_id = os.path.splitext(os.path.basename(audio_abs))[0]
        out_records.append({'id': rec_id, 'text': text, 'wav': audio_abs, 'duration': round(dur, 3)})
    if not out_records:
        raise SystemExit('[ERROR] no valid records')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out_records:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')
    print(f'\ndone: {len(out_records)} -> {args.out}')
if __name__ == '__main__':
    main()
