from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    v3_model_id: str = 'openai/whisper-large-v3'
    language: str = 'Russian'
    task: str = 'transcribe'
    torch_dtype: str = 'bfloat16'

@dataclass
class AugmentationConfig:
    augmentation_prob: float = 0.8
    background_noise_prob: float = 0.7
    min_snr_db: float = 0.0
    max_snr_db: float = 20.0
    gaussian_prob: float = 0.3
    gaussian_min_snr_db: float = 15.0
    gaussian_max_snr_db: float = 40.0
    gain_prob: float = 0.3
    gain_min_db: float = -6
    gain_max_db: float = 6

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = 'none'
    target_modules_min: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
