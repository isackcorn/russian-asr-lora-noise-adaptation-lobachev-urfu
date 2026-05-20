import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List
SEED = 42
DEFAULT_NOISE_DIR = Path('/root/DisserRealization/noise_data')
DEMAND_TEST_SCENES = {'PCAFETER', 'OOFFICE', 'STRAFFIC', 'TBUS'}

def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    os.link(src, dst)

def _link_files(files: Iterable[Path], dst_dir: Path) -> int:
    count = 0
    for src in files:
        _link(src, dst_dir / src.name)
        count += 1
    return count

def split_esc50(noise_dir: Path, splits_dir: Path) -> None:
    src = noise_dir / 'esc50_16k'
    if not src.is_dir():
        raise FileNotFoundError(f'missing ESC-50: {src}')
    train_dir = splits_dir / 'train' / 'esc50'
    test_dir = splits_dir / 'test' / 'esc50'
    pat = re.compile('^(\\d+)-')
    train_files: List[Path] = []
    test_files: List[Path] = []
    for wav in sorted(src.glob('*.wav')):
        m = pat.match(wav.name)
        if not m:
            raise ValueError(f'invalid ESC-50 filename: {wav}')
        fold = int(m.group(1))
        (test_files if fold == 5 else train_files).append(wav)
    n_tr = _link_files(train_files, train_dir)
    n_te = _link_files(test_files, test_dir)
    print(f'ESC-50: train={n_tr} test={n_te}')

def split_musan_noise(noise_dir: Path, splits_dir: Path) -> None:
    src = noise_dir / 'musan' / 'noise'
    if not src.is_dir():
        raise FileNotFoundError(f'missing MUSAN noise: {src}')
    train_dir = splits_dir / 'train' / 'musan_noise'
    test_dir = splits_dir / 'test' / 'musan_noise'
    free_files = list((src / 'free-sound').rglob('*.wav')) if (src / 'free-sound').is_dir() else []
    bible_files = list((src / 'sound-bible').rglob('*.wav')) if (src / 'sound-bible').is_dir() else []
    n_tr = _link_files(free_files, train_dir)
    n_te = _link_files(bible_files, test_dir)
    print(f'MUSAN noise: train={n_tr} test={n_te}')

def split_musan_random(noise_dir: Path, splits_dir: Path, subset: str, ratio: float=0.2) -> None:
    src = noise_dir / 'musan' / subset
    if not src.is_dir():
        raise FileNotFoundError(f'missing MUSAN {subset}: {src}')
    train_dir = splits_dir / 'train' / f'musan_{subset}'
    test_dir = splits_dir / 'test' / f'musan_{subset}'
    files = sorted(src.rglob('*.wav'))
    rng = random.Random(SEED)
    rng.shuffle(files)
    n_test = int(len(files) * ratio)
    test_files = files[:n_test]
    train_files = files[n_test:]
    n_tr = _link_files(train_files, train_dir)
    n_te = _link_files(test_files, test_dir)
    print(f'MUSAN {subset}: train={n_tr} test={n_te}')

def split_demand(noise_dir: Path, splits_dir: Path) -> None:
    src = noise_dir / 'demand_mono_16k_first'
    if not src.is_dir():
        raise FileNotFoundError(f'missing DEMAND: {src}')
    train_dir = splits_dir / 'train' / 'demand'
    test_dir = splits_dir / 'test' / 'demand'
    train_files: List[Path] = []
    test_files: List[Path] = []
    for wav in sorted(src.glob('*.wav')):
        scene = wav.stem.split('_')[0]
        (test_files if scene in DEMAND_TEST_SCENES else train_files).append(wav)
    n_tr = _link_files(train_files, train_dir)
    n_te = _link_files(test_files, test_dir)
    print(f'DEMAND: train={n_tr} test={n_te}')

def main() -> int:
    parser = argparse.ArgumentParser(description='Split noise data')
    parser.add_argument('--noise-dir', type=Path, default=DEFAULT_NOISE_DIR, help='Noise root')
    parser.add_argument('--splits-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    noise_dir: Path = args.noise_dir
    splits_dir: Path = args.splits_dir or noise_dir / 'splits'
    if not noise_dir.is_dir():
        print(f'[ERROR] missing noise directory: {noise_dir}', file=sys.stderr)
        return 1
    print('split noise data')
    print(f'  source: {noise_dir}')
    print(f'  output: {splits_dir}')
    print(f'  seed:   {SEED}')
    print()
    split_esc50(noise_dir, splits_dir)
    split_musan_noise(noise_dir, splits_dir)
    split_musan_random(noise_dir, splits_dir, 'music')
    split_musan_random(noise_dir, splits_dir, 'speech')
    split_demand(noise_dir, splits_dir)
    print()
    print('done')
    print(f'  {splits_dir}/train/{{musan_noise,musan_music,musan_speech,esc50,demand}}/')
    print(f'  {splits_dir}/test/{{musan_noise,musan_music,musan_speech,esc50,demand}}/')
    return 0
if __name__ == '__main__':
    sys.exit(main())
