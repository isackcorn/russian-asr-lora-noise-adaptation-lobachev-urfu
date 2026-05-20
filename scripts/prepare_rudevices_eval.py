import argparse
import json
import os
import random
import sys
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

def scan_pairs(audio_root: str) -> list[tuple[str, str, str]]:
    print(f'scan: {audio_root}')
    wav_index: dict[str, str] = {}
    txt_index: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(audio_root):
        for fname in filenames:
            stem, ext = os.path.splitext(fname)
            full = os.path.join(dirpath, fname)
            if ext == '.wav':
                wav_index[stem] = full
            elif ext == '.txt':
                txt_index[stem] = full
    common = sorted(set(wav_index) & set(txt_index))
    only_wav = len(wav_index) - len(common)
    only_txt = len(txt_index) - len(common)
    print(f'  wav: {len(wav_index)}  txt: {len(txt_index)}  pairs: {len(common)}')
    if only_wav or only_txt:
        print(f'  unmatched: wav={only_wav}, txt={only_txt}')
    return [(uid, wav_index[uid], txt_index[uid]) for uid in common]

def sample_pairs(pairs: list[tuple], fraction: float, sample_n: int | None, seed: int) -> list[tuple]:
    if sample_n is not None:
        n = min(sample_n, len(pairs))
    elif fraction >= 1.0:
        return list(pairs)
    else:
        n = max(1, int(len(pairs) * fraction))
    rng = random.Random(seed)
    return rng.sample(pairs, n)

def read_text(txt_path: str, min_len: int=2) -> str | None:
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if len(text) < min_len:
        raise ValueError(f'invalid text: {txt_path}')
    return text

def probe_duration(wav_path: str) -> float:
    info = sf.info(wav_path)
    if not info.samplerate:
        raise ValueError(f'invalid sample rate: {wav_path}')
    return info.frames / info.samplerate

def main():
    p = argparse.ArgumentParser(description='Convert RuDevices manifest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--audio-root', required=True, help='Audio root')
    p.add_argument('--out', required=True, help='Output manifest')
    p.add_argument('--fraction', type=float, default=0.1, help='Sample fraction')
    p.add_argument('--sample-n', type=int, default=None, help='Sample count')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-duration', type=float, default=30.0, help='Max duration')
    p.add_argument('--min-duration', type=float, default=0.5, help='Min duration')
    p.add_argument('--min-text-len', type=int, default=2, help='Min text length')
    args = p.parse_args()
    if not os.path.isdir(args.audio_root):
        raise SystemExit(f'[ERROR] missing audio root: {args.audio_root}')
    print('convert rudevices')
    print(f'  root:   {args.audio_root}')
    print(f'  output: {args.out}')
    if args.sample_n is not None:
        print(f'  sample: {args.sample_n}')
    else:
        print(f'  fraction: {args.fraction:.2f}')
    print(f'  seed:   {args.seed}')
    print(f'  range:  [{args.min_duration}, {args.max_duration}]')
    all_pairs = scan_pairs(args.audio_root)
    if not all_pairs:
        raise SystemExit('[ERROR] no wav/txt pairs')
    valid_records = []
    filtered_duration = 0
    for uid, wav_path, txt_path in tqdm(all_pairs, desc='Validate & probe'):
        text = read_text(txt_path, args.min_text_len)
        dur = probe_duration(wav_path)
        if dur < args.min_duration or dur > args.max_duration:
            filtered_duration += 1
            continue
        valid_records.append({'id': uid, 'text': text, 'wav': os.path.abspath(wav_path), 'duration': round(dur, 3)})
    if not valid_records:
        raise SystemExit('[ERROR] no valid records')
    out_records = sample_pairs(valid_records, args.fraction, args.sample_n, args.seed)
    print(f'\nsampled: {len(out_records)}')
    if not out_records:
        raise SystemExit('[ERROR] no sampled records')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out_records:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')
    durations = [r['duration'] for r in out_records]
    durations_sorted = sorted(durations)
    median = durations_sorted[len(durations) // 2]
    short_share = sum((1 for d in durations if d < 2.0)) / len(durations)
    print(f'\ndone: {len(out_records)} -> {args.out}')
    print(f'  filtered duration: {filtered_duration}')
    print(f'  median duration:  {median:.2f}')
    print(f'  short share:      {short_share * 100:.1f}%')
if __name__ == '__main__':
    main()
