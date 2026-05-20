import argparse
import json
import os
import subprocess
import shutil
import sys
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
DATA_AUDIOSETS = Path('/root/DisserRealization/DataAudiosets')
DATA_DIR = Path('/root/DisserRealization/data')
NOISE_DIR = Path('/root/DisserRealization/noise_data')
Golos_ARCHIVE = DATA_AUDIOSETS / 'GOLOS' / 'golos_opus.tar.gz'
MUSAN_ARCHIVE = DATA_AUDIOSETS / 'MUSAN' / 'musan.tar.gz'
ESC50_ARCHIVE = DATA_AUDIOSETS / 'ESC-50' / 'master.zip'
DEMAND_ARCHIVE = DATA_AUDIOSETS / 'DEMAND' / 'archive.zip'
TARGET_SR = 16000
DEFAULT_GOLOS_MANIFESTS = [Path('train_opus') / '100hours.jsonl', Path('test_opus') / 'farfield' / 'manifest.jsonl', Path('test_opus') / 'crowd' / 'manifest.jsonl']
FULL_GOLOS_MANIFEST = Path('train_opus') / 'manifest.jsonl'

def extract_golos(output_dir: Path):
    print('\nextract Golos')
    if (output_dir / 'train_opus').exists():
        print(f'[skip] exists: {output_dir}')
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'archive: {Golos_ARCHIVE}')
    print(f'output: {output_dir}')
    with tarfile.open(Golos_ARCHIVE, 'r:gz') as tar:
        tar.extractall(path=output_dir)
    nested = output_dir / 'golos_opus'
    if nested.is_dir():
        for item in nested.iterdir():
            shutil.move(str(item), str(output_dir / item.name))
        nested.rmdir()
        print(f'normalized: {output_dir}')
    print('done')

def _convert_audio_ffmpeg(input_path: Path, output_path: Path, target_sr: int=TARGET_SR):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-y', '-i', str(input_path), '-ar', str(target_sr), '-ac', '1', '-hide_banner', '-loglevel', 'error', str(output_path)]
    subprocess.run(command, check=True)

def _resolve_golos_audio_path(manifest_base: Path, audio_filepath: str) -> Path:
    rel_path = Path(audio_filepath)
    candidates = [manifest_base / rel_path, manifest_base.parent / rel_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'missing audio file: {audio_filepath}')

def _convert_golos_record(task: tuple[str, str, dict, int]) -> dict:
    manifest_base_str, wav_manifest_base_str, record, target_sr = task
    manifest_base = Path(manifest_base_str)
    wav_manifest_base = Path(wav_manifest_base_str)
    source_path = _resolve_golos_audio_path(manifest_base, record['audio_filepath'])
    wav_rel_path = Path(record['audio_filepath']).with_suffix('.wav')
    wav_path = wav_manifest_base / wav_rel_path
    if not wav_path.exists():
        _convert_audio_ffmpeg(source_path, wav_path, target_sr=target_sr)
    return {'id': record['id'], 'text': record['text'], 'wav': wav_rel_path.as_posix(), 'duration': record['duration']}

def convert_golos_to_wav(raw_root: Path, output_dir: Path, include_full_train: bool, jobs: int):
    print('\nconvert Golos')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found')
    manifests = list(DEFAULT_GOLOS_MANIFESTS)
    if include_full_train:
        manifests.insert(0, FULL_GOLOS_MANIFEST)
    output_dir.mkdir(parents=True, exist_ok=True)
    for manifest_rel in manifests:
        input_manifest = raw_root / manifest_rel
        output_manifest = output_dir / manifest_rel
        if not input_manifest.exists():
            raise FileNotFoundError(f'missing manifest: {input_manifest}')
        print(f'\nmanifest: {manifest_rel}')
        print(f'input: {input_manifest}')
        print(f'output: {output_manifest}')
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_base = input_manifest.parent
        wav_manifest_base = output_manifest.parent
        with open(input_manifest, 'r', encoding='utf-8') as src:
            total = sum((1 for _ in src))
        converted = 0
        with input_manifest.open('r', encoding='utf-8') as src, open(output_manifest, 'w', encoding='utf-8') as dst, ProcessPoolExecutor(max_workers=jobs) as pool:
            task_iter = ((str(manifest_base), str(wav_manifest_base), json.loads(line), TARGET_SR) for line in src)
            for record in tqdm(pool.map(_convert_golos_record, task_iter, chunksize=16), total=total, desc=f'Golos -> WAV [{manifest_rel}]'):
                json.dump(record, dst, ensure_ascii=False)
                dst.write('\n')
                converted += 1
        print(f'done: {converted}')

def extract_musan(output_dir: Path):
    print('\nextract MUSAN')
    if (output_dir / 'noise').exists():
        print(f'[skip] exists: {output_dir}')
        return
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    print(f'archive: {MUSAN_ARCHIVE}')
    print(f'output: {output_dir}')
    with tarfile.open(MUSAN_ARCHIVE, 'r:gz') as tar:
        tar.extractall(path=parent)
    print('done')

def resample_audio(input_path: Path, output_path: Path, target_sr: int=16000):
    waveform, sr = torchaudio.load(str(input_path))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    torchaudio.save(str(output_path), waveform, target_sr)

def extract_esc50(output_dir: Path):
    print('\nextract ESC-50')
    existing = list(output_dir.glob('*.wav')) if output_dir.exists() else []
    if existing:
        print(f'[skip] exists: {output_dir} ({len(existing)})')
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f'archive: {ESC50_ARCHIVE}')
    with zipfile.ZipFile(ESC50_ARCHIVE, 'r') as z:
        z.extractall(path=temp_dir)
    audio_source = temp_dir / 'ESC-50-master' / 'audio'
    wav_files = sorted(audio_source.glob('*.wav'))
    print(f'files: {len(wav_files)}')
    for wav_path in tqdm(wav_files, desc='ESC-50'):
        out_path = output_dir / wav_path.name
        resample_audio(wav_path, out_path, target_sr=TARGET_SR)
    shutil.rmtree(temp_dir)
    print(f"done: {output_dir} ({len(list(output_dir.glob('*.wav')))})")

def extract_demand(output_dir: Path, mode: str='mean'):
    print(f'\nextract DEMAND ({mode})')
    existing = list(output_dir.glob('*.wav')) if output_dir.exists() else []
    if existing:
        print(f'[skip] exists: {output_dir} ({len(existing)})')
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f'archive: {DEMAND_ARCHIVE}')
    with zipfile.ZipFile(DEMAND_ARCHIVE, 'r') as z:
        z.extractall(path=temp_dir)
    demand_root = temp_dir / 'demand'
    if not demand_root.exists():
        raise FileNotFoundError(f'missing demand root: {demand_root}')
    scenes = [d for d in demand_root.iterdir() if d.is_dir()]
    print(f'scenes: {len(scenes)}')
    total_files = 0
    for scene_dir in tqdm(scenes, desc='DEMAND'):
        scene_name = scene_dir.name
        ch_files = sorted(scene_dir.glob('ch*.wav'))
        if not ch_files:
            raise FileNotFoundError(f'missing DEMAND channels: {scene_dir}')
        if mode == 'first':
            ch_file = ch_files[0]
            waveform, sr = torchaudio.load(str(ch_file))
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            out_path = output_dir / f'{scene_name}_ch01.wav'
            torchaudio.save(str(out_path), waveform, TARGET_SR)
            total_files += 1
        elif mode == 'mean':
            waveforms = []
            sr = None
            for ch_file in ch_files:
                w, s = torchaudio.load(str(ch_file))
                if sr is None:
                    sr = s
                waveforms.append(w)
            stacked = torch.stack(waveforms)
            mean_waveform = stacked.mean(dim=0)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                mean_waveform = resampler(mean_waveform)
            if mean_waveform.shape[0] > 1:
                mean_waveform = mean_waveform.mean(dim=0, keepdim=True)
            out_path = output_dir / f'{scene_name}_mean.wav'
            torchaudio.save(str(out_path), mean_waveform, TARGET_SR)
            total_files += 1
        elif mode == 'all':
            for ch_file in ch_files:
                waveform, sr = torchaudio.load(str(ch_file))
                if sr != TARGET_SR:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                out_path = output_dir / f'{scene_name}_{ch_file.stem}.wav'
                torchaudio.save(str(out_path), waveform, TARGET_SR)
                total_files += 1
    shutil.rmtree(temp_dir)
    print(f'done: {output_dir} ({total_files})')

def main():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--demand-mode', type=str, choices=['mean', 'first', 'all'], default='first', help='DEMAND mode')
    parser.add_argument('--skip-golos-wav', action='store_true', help='Skip Golos WAV conversion')
    parser.add_argument('--include-full-train', action='store_true', help='Include full train manifest')
    parser.add_argument('--golos-jobs', type=int, default=8, help='Worker count')
    parser.add_argument('--output-data', type=str, default=str(DATA_DIR), help='Speech data output')
    parser.add_argument('--output-noise', type=str, default=str(NOISE_DIR), help='Noise data output')
    parser.add_argument('--skip-golos', action='store_true', help='Skip Golos')
    parser.add_argument('--skip-musan', action='store_true', help='Skip MUSAN')
    parser.add_argument('--skip-esc50', action='store_true', help='Skip ESC-50')
    parser.add_argument('--skip-demand', action='store_true', help='Skip DEMAND')
    args = parser.parse_args()
    data_dir = Path(args.output_data)
    noise_dir = Path(args.output_noise)
    print('prepare data')
    print(f'DataAudiosets: {DATA_AUDIOSETS}')
    print(f'Speech data: {data_dir}')
    print(f'Noise data: {noise_dir}')
    print(f'DEMAND mode: {args.demand_mode}')
    print(f"Golos WAV: {('skip' if args.skip_golos_wav else 'enable')}")
    archives = {'Golos': (Golos_ARCHIVE, args.skip_golos), 'MUSAN': (MUSAN_ARCHIVE, args.skip_musan), 'ESC-50': (ESC50_ARCHIVE, args.skip_esc50), 'DEMAND': (DEMAND_ARCHIVE, args.skip_demand)}
    for name, (path, skip) in archives.items():
        if not skip and (not path.exists()):
            print(f'\n[ERROR] missing archive: {path}')
            sys.exit(1)
    if not args.skip_golos:
        extract_golos(data_dir / 'golos')
    if not args.skip_golos_wav:
        convert_golos_to_wav(raw_root=data_dir / 'golos', output_dir=data_dir / 'golos_wav', include_full_train=args.include_full_train, jobs=args.golos_jobs)
    if not args.skip_musan:
        extract_musan(noise_dir / 'musan')
    if not args.skip_esc50:
        extract_esc50(noise_dir / 'esc50_16k')
    if not args.skip_demand:
        demand_out = noise_dir / f'demand_mono_16k_{args.demand_mode}'
        extract_demand(demand_out, mode=args.demand_mode)
    print('\ndone')
if __name__ == '__main__':
    main()
