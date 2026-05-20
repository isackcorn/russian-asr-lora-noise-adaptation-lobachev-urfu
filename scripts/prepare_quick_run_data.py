import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.augmentation import mix_at_exact_snr
GOLOS_ROOT = '/root/DisserRealization/data/golos'
TRAIN_MANIFEST = os.path.join(GOLOS_ROOT, 'train_opus', '100hours.jsonl')
TEST_MANIFEST = os.path.join(GOLOS_ROOT, 'test_opus', 'farfield', 'manifest.jsonl')
MINI_TRAIN_DIR = '/root/DisserRealization/data/mini_train'
MINI_TEST_DIR = '/root/DisserRealization/data/mini_test'
EVAL_DIR = '/root/DisserRealization/eval/noisy_test_sets'
TRAIN_SIZE = 500
TEST_SIZE = 200
TARGET_SR = 16000
_NOISE_ROOT = '/root/DisserRealization/noise_data'
_SPLITS_TEST = os.path.join(_NOISE_ROOT, 'splits', 'test')
if os.path.isdir(_SPLITS_TEST):
    NOISE_POOL = [os.path.join(_SPLITS_TEST, 'musan_noise'), os.path.join(_SPLITS_TEST, 'musan_music'), os.path.join(_SPLITS_TEST, 'musan_speech'), os.path.join(_SPLITS_TEST, 'esc50'), os.path.join(_SPLITS_TEST, 'demand')]
else:
    raise FileNotFoundError(f'missing noise split: {_SPLITS_TEST}')
SNR_LEVELS = [0, 5, 10, 15, 20]
SEED = 42

def load_manifest(path: str, limit: int):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            records.append(json.loads(line))
    return records

def convert_opus_to_wav(opus_path: str, wav_path: str, target_sr: int=16000):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    cmd = ['ffmpeg', '-y', '-i', opus_path, '-ar', str(target_sr), '-ac', '1', '-hide_banner', '-loglevel', 'error', wav_path]
    subprocess.run(cmd, check=True)
    return wav_path

def process_train_record(args):
    record, idx, out_dir = args
    opus_path = os.path.join(GOLOS_ROOT, 'train_opus', record['audio_filepath'])
    wav_path = os.path.join(out_dir, f"{record['id']}.wav")
    convert_opus_to_wav(opus_path, wav_path)
    return {'id': record['id'], 'text': record['text'], 'wav': wav_path, 'duration': record['duration']}

def process_test_record(args):
    record, idx, out_dir = args
    opus_path = os.path.join(GOLOS_ROOT, 'test_opus', 'farfield', record['audio_filepath'])
    wav_path = os.path.join(out_dir, f"{record['id']}.wav")
    convert_opus_to_wav(opus_path, wav_path)
    return {'id': record['id'], 'text': record['text'], 'wav': wav_path, 'duration': record['duration']}

def build_mini_train():
    print('\nbuild mini_train')
    os.makedirs(MINI_TRAIN_DIR, exist_ok=True)
    records = load_manifest(TRAIN_MANIFEST, TRAIN_SIZE)
    print(f'loaded: {len(records)}')
    tasks = [(r, i, MINI_TRAIN_DIR) for i, r in enumerate(records)]
    results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_train_record, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Train conversion'):
            res = future.result()
            results.append(res)
    manifest_path = os.path.join(MINI_TRAIN_DIR, 'manifest.jsonl')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')
    print(f'saved: {len(results)} -> {MINI_TRAIN_DIR}')
    return results

def build_mini_test():
    print('\nbuild mini_test')
    os.makedirs(MINI_TEST_DIR, exist_ok=True)
    records = load_manifest(TEST_MANIFEST, TEST_SIZE)
    print(f'loaded: {len(records)}')
    tasks = [(r, i, MINI_TEST_DIR) for i, r in enumerate(records)]
    results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_test_record, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Test conversion'):
            res = future.result()
            results.append(res)
    manifest_path = os.path.join(MINI_TEST_DIR, 'manifest.jsonl')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')
    print(f'saved: {len(results)} -> {MINI_TEST_DIR}')
    return results

def build_noisy_test_sets():
    print('\nbuild noisy test sets')
    test_records = []
    manifest_path = os.path.join(MINI_TEST_DIR, 'manifest.jsonl')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_records.append(json.loads(line))
    noise_files = []
    for noise_dir in NOISE_POOL:
        for root, _, files in os.walk(noise_dir):
            for fname in files:
                if fname.endswith('.wav'):
                    noise_files.append(os.path.join(root, fname))
    print(f'noise files: {len(noise_files)}')
    clean_dir = os.path.join(EVAL_DIR, 'clean')
    os.makedirs(clean_dir, exist_ok=True)
    for rec in test_records:
        dst = os.path.join(clean_dir, os.path.basename(rec['wav']))
        if not os.path.exists(dst):
            os.link(rec['wav'], dst)
    with open(os.path.join(clean_dir, 'manifest.jsonl'), 'w', encoding='utf-8') as f:
        for rec in test_records:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')
    print(f'  clean: {len(test_records)}')
    for snr in SNR_LEVELS:
        out_dir = os.path.join(EVAL_DIR, f'snr_{snr}db')
        os.makedirs(out_dir, exist_ok=True)
        noisy_records = []
        for idx, rec in enumerate(tqdm(test_records, desc=f'SNR={snr} dB')):
            clean_audio, sr = sf.read(rec['wav'], dtype='float32')
            noise_idx = (SEED + idx) % len(noise_files)
            noise_audio, noise_sr = sf.read(noise_files[noise_idx], dtype='float32')
            if noise_audio.ndim > 1:
                noise_audio = noise_audio.mean(axis=1)
            if noise_sr != sr:
                from scipy import signal
                gcd = np.gcd(sr, noise_sr)
                noise_audio = signal.resample_poly(noise_audio, up=sr // gcd, down=noise_sr // gcd)
            mixed = mix_at_exact_snr(clean_audio, noise_audio, snr, seed=SEED + idx)
            out_wav = os.path.join(out_dir, os.path.basename(rec['wav']))
            sf.write(out_wav, mixed, sr)
            noisy_records.append({'id': rec['id'], 'text': rec['text'], 'wav': out_wav, 'duration': rec['duration'], 'snr_db': snr})
        with open(os.path.join(out_dir, 'manifest.jsonl'), 'w', encoding='utf-8') as f:
            for r in noisy_records:
                json.dump(r, f, ensure_ascii=False)
                f.write('\n')
        print(f'  snr_{snr}db: {len(noisy_records)}')
    print('done')

def main():
    build_mini_train()
    build_mini_test()
    build_noisy_test_sets()
    print('\ndone')
if __name__ == '__main__':
    main()
