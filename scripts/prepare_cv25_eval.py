import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
CV_PREFIX = 'cv-corpus-25.0-2026-03-09/ru'
ARCHIVE_CLIPS_PREFIX = f'{CV_PREFIX}/clips/'

def get_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--archive', required=True, help='Input archive')
    p.add_argument('--extract-dir', default='data/cv25', help='Extract directory')
    p.add_argument('--out-manifest', default='data/cv25/test_manifest.jsonl', help='Output manifest')
    p.add_argument('--split', default='test', choices=['test', 'dev', 'train'], help='Dataset split')
    p.add_argument('--workers', type=int, default=8, help='Worker count')
    p.add_argument('--skip-extract', action='store_true', help='Skip extract')
    p.add_argument('--skip-convert', action='store_true', help='Skip convert')
    return p

def extract_archive(archive: str, extract_dir: Path, split: str):
    tsv_member = f'{CV_PREFIX}/{split}.tsv'
    clips_prefix = ARCHIVE_CLIPS_PREFIX
    print(f'scan: {archive}')
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
    tsv_members = [m for m in members if m.name == tsv_member]
    clip_members = [m for m in members if m.name.startswith(clips_prefix) and m.name.endswith('.mp3')]
    print(f'  tsv: {len(tsv_members)}')
    print(f'  mp3: {len(clip_members)}')
    to_extract = tsv_members + clip_members
    print(f'extract: {len(to_extract)} -> {extract_dir}')
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tf:
        for m in tqdm(to_extract, desc='extract'):
            tf.extract(m, path=str(extract_dir), filter='data')
    print('extract done')

def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> tuple[bool, str]:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['ffmpeg', '-y', '-i', str(mp3_path), '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', str(wav_path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return (False, result.stderr.decode(errors='replace'))
    return (True, '')

def get_duration_seconds(wav_path: Path) -> float:
    result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(wav_path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f'ffprobe failed: {wav_path}')
    info = json.loads(result.stdout)
    streams = info['streams']
    for stream in streams:
        dur = stream.get('duration')
        if dur is not None:
            return float(dur)
    raise RuntimeError(f'missing duration: {wav_path}')

def main():
    args = get_parser().parse_args()
    archive = Path(args.archive).resolve()
    if not archive.exists():
        sys.exit(f'[ERROR] missing archive: {archive}')
    extract_dir = Path(args.extract_dir).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    ru_dir = extract_dir / CV_PREFIX
    if not args.skip_extract:
        tsv_path = ru_dir / f'{args.split}.tsv'
        if tsv_path.exists():
            print(f'[skip] tsv exists: {tsv_path}')
        else:
            extract_archive(str(archive), extract_dir, args.split)
    else:
        print('[skip] extract')
    tsv_path = ru_dir / f'{args.split}.tsv'
    clips_mp3_dir = ru_dir / 'clips'
    clips_wav_dir = extract_dir / 'clips_wav'
    if not tsv_path.exists():
        sys.exit(f'[ERROR] missing tsv: {tsv_path}')
    import csv
    rows = []
    with open(tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    print(f'tsv rows: {len(rows)} ({args.split})')
    if not args.skip_convert:
        clips_wav_dir.mkdir(parents=True, exist_ok=True)
        to_convert = []
        for row in rows:
            mp3_name = row['path']
            wav_name = mp3_name.replace('.mp3', '.wav')
            mp3_path = clips_mp3_dir / mp3_name
            wav_path = clips_wav_dir / wav_name
            if not wav_path.exists():
                to_convert.append((mp3_path, wav_path))
        if not to_convert:
            print(f'[skip] wav exists: {len(rows)}')
        else:
            print(f'convert: {len(to_convert)} (workers={args.workers})')
            errors = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(convert_mp3_to_wav, mp3, wav): (mp3, wav) for mp3, wav in to_convert}
                with tqdm(total=len(futures), desc='convert') as pbar:
                    for fut in as_completed(futures):
                        ok, msg = fut.result()
                        if not ok:
                            mp3, _ = futures[fut]
                            errors.append(f'{mp3}: {msg[:120]}')
                        pbar.update(1)
            if errors:
                raise RuntimeError(f'convert failed: {errors[:10]}')
            print('convert done')
    else:
        print('[skip] convert')
    print('write manifest')
    written = 0
    with open(out_manifest, 'w', encoding='utf-8') as fout:
        for row in tqdm(rows, desc='manifest'):
            mp3_name = row['path']
            wav_name = mp3_name.replace('.mp3', '.wav')
            wav_path = clips_wav_dir / wav_name
            if not wav_path.exists():
                raise FileNotFoundError(f'missing wav: {wav_path}')
            text = row['sentence'].strip()
            if not text:
                raise ValueError(f'empty text: {mp3_name}')
            duration = get_duration_seconds(wav_path)
            record = {'id': mp3_name.replace('.mp3', ''), 'text': text, 'wav': str(wav_path), 'duration': round(duration, 3)}
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1
    print(f'\nmanifest: {out_manifest}')
    print(f'  written: {written}')
    if written > 0:
        total_h = sum((json.loads(l)['duration'] for l in open(out_manifest, encoding='utf-8'))) / 3600
        print(f'  hours: {total_h:.2f}')
if __name__ == '__main__':
    main()
