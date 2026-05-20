import os
import glob
from typing import Dict, List, Optional, Union
import numpy as np
import soundfile as sf
from src.config.config import AugmentationConfig

class NoiseAugmenter:

    def __init__(self, noise_paths: Union[List[str], Dict[str, str]], config: AugmentationConfig=AugmentationConfig(), seed: Optional[int]=None, target_sr: int=16000):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.target_sr = target_sr
        self.noise_categories: Dict[str, List[str]] = self._collect_noise_files(noise_paths)
        self.all_noise_files: List[str] = []
        for files in self.noise_categories.values():
            self.all_noise_files.extend(files)
        if not self.all_noise_files:
            raise ValueError(f'no noise files: {noise_paths}')
        total = len(self.all_noise_files)
        cats = {k: len(v) for k, v in self.noise_categories.items()}
        print(f'noise files: {total}')
        for cat, cnt in cats.items():
            print(f'  {cat}: {cnt}')

    def _collect_noise_files(self, noise_paths: Union[List[str], Dict[str, str]]) -> Dict[str, List[str]]:
        categories: Dict[str, List[str]] = {}
        items = noise_paths.items() if isinstance(noise_paths, dict) else ((os.path.basename(os.path.normpath(p)), p) for p in noise_paths)
        for cat_name, path in items:
            if not os.path.exists(path):
                raise FileNotFoundError(f'missing noise directory: {path}')
            files = glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)
            if not files:
                raise FileNotFoundError(f'empty noise directory: {path}')
            categories.setdefault(cat_name, []).extend(files)
        return categories

    def _load_random_noise(self, min_length: int) -> np.ndarray:
        max_attempts = 10
        for _ in range(max_attempts):
            cat_names = list(self.noise_categories.keys())
            cat = self.rng.choice(cat_names)
            noise_file = self.rng.choice(self.noise_categories[cat])
            noise, sr = sf.read(noise_file, dtype='float32')
            if len(noise) == 0:
                raise ValueError(f'empty noise file: {noise_file}')
            if sr != self.target_sr:
                from scipy import signal
                gcd = np.gcd(self.target_sr, sr)
                noise = signal.resample_poly(noise, up=self.target_sr // gcd, down=sr // gcd)
            if noise.ndim > 1:
                noise = noise.mean(axis=1)
            if len(noise) < min_length:
                repeats = int(np.ceil(min_length / len(noise)))
                noise = np.tile(noise, repeats)
            if len(noise) > min_length:
                offset = self.rng.randint(0, len(noise) - min_length)
                noise = noise[offset:offset + min_length]
            return noise[:min_length].astype(np.float32)
        raise RuntimeError('noise selection failed')

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.rng.random() > self.config.augmentation_prob:
            return samples
        signal_power = np.mean(samples ** 2)
        noisy = samples.copy()
        if self.rng.random() < self.config.gaussian_prob:
            snr_db = self.rng.uniform(self.config.gaussian_min_snr_db, self.config.gaussian_max_snr_db)
            gaussian = self.rng.randn(len(samples)).astype(np.float32)
            gaussian_power = np.mean(gaussian ** 2)
            if gaussian_power > 0 and signal_power > 0:
                scale = np.sqrt(signal_power / (gaussian_power * 10 ** (snr_db / 10)))
                noisy = noisy + scale * gaussian
        if self.rng.random() < self.config.background_noise_prob:
            noise = self._load_random_noise(len(samples))
            snr_db = self.rng.uniform(self.config.min_snr_db, self.config.max_snr_db)
            noise_power = np.mean(noise ** 2)
            if noise_power > 0 and signal_power > 0:
                scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
                noisy = noisy + scale * noise
        peak = np.max(np.abs(noisy))
        if peak > 1.0:
            noisy *= 0.99 / peak
        if self.rng.random() < self.config.gain_prob:
            gain_db = self.rng.uniform(self.config.gain_min_db, self.config.gain_max_db)
            gain_linear = 10 ** (gain_db / 20)
            noisy = noisy * gain_linear
            peak = np.max(np.abs(noisy))
            if peak > 1.0:
                noisy *= 0.99 / peak
        return noisy.astype(np.float32)

def create_noise_augmentation(noise_paths: Union[List[str], Dict[str, str]], config: AugmentationConfig=AugmentationConfig(), seed: Optional[int]=None, target_sr: int=16000):
    return NoiseAugmenter(noise_paths, config, seed, target_sr)

def mix_at_exact_snr(clean: np.ndarray, noise: np.ndarray, target_snr_db: float, seed: int=42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    offset = rng.randint(0, max(1, len(noise) - len(clean)))
    noise = noise[offset:offset + len(clean)]
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        raise ValueError('zero-power noise')
    scale = np.sqrt(signal_power / (noise_power * 10 ** (target_snr_db / 10)))
    mixed = clean + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed *= 0.99 / peak
    return mixed.astype(np.float32)

def prepare_train_example(example: dict, processor, augmenter) -> dict:
    audio = example['audio']
    waveform = audio['array'].astype('float32')
    sr = audio['sampling_rate']
    augmented = augmenter(samples=waveform, sample_rate=sr)
    example['input_features'] = processor.feature_extractor(augmented, sampling_rate=sr).input_features[0]
    text = example['sentence']
    example['labels'] = processor.tokenizer(text).input_ids
    return example

def prepare_eval_example(example: dict, processor) -> dict:
    audio = example['audio']
    waveform = audio['array'].astype('float32')
    sr = audio['sampling_rate']
    example['input_features'] = processor.feature_extractor(waveform, sampling_rate=sr).input_features[0]
    text = example['sentence']
    example['labels'] = processor.tokenizer(text).input_ids
    return example
