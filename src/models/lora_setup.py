import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.config.config import LoRAConfig, ModelConfig

def create_lora_config(config: LoRAConfig=LoRAConfig(), target_modules: list=None) -> LoraConfig:
    modules = target_modules if target_modules is not None else config.target_modules_min
    lora_config = LoraConfig(r=config.r, lora_alpha=config.lora_alpha, target_modules=modules, lora_dropout=config.lora_dropout, bias=config.bias)
    return lora_config

def load_whisper_model(model_id: str=None, model_path: str=None, torch_dtype: str='bfloat16') -> WhisperForConditionalGeneration:
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map[torch_dtype]
    source = model_path if model_path else model_id
    if source is None:
        source = ModelConfig().v3_model_id
    print(f'loading model: {source}')
    print(f'  dtype: {torch_dtype}')
    model = WhisperForConditionalGeneration.from_pretrained(source, torch_dtype=dtype)
    return model

def load_processor(model_id: str=None, model_path: str=None, language: str='russian', task: str='transcribe') -> WhisperProcessor:
    source = model_path if model_path else model_id
    if source is None:
        source = ModelConfig().v3_model_id
    print(f'loading processor: {source}')
    processor = WhisperProcessor.from_pretrained(source, language=language, task=task)
    return processor

def prepare_model_for_training(model: WhisperForConditionalGeneration, use_gradient_checkpointing: bool=True) -> WhisperForConditionalGeneration:
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.generation_config.language = 'russian'
    model.generation_config.task = 'transcribe'
    model.generation_config.forced_decoder_ids = None
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print('gradient checkpointing enabled')
    return model

def apply_lora(model: WhisperForConditionalGeneration, lora_config: LoraConfig) -> torch.nn.Module:
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def setup_model_for_lora(model_id: str=None, model_path: str=None, torch_dtype: str='bfloat16', lora_config: LoRAConfig=None, target_modules: list=None, use_gradient_checkpointing: bool=True) -> tuple:
    cfg = LoRAConfig() if lora_config is None else lora_config
    model = load_whisper_model(model_id=model_id, model_path=model_path, torch_dtype=torch_dtype)
    processor = load_processor(model_id=model_id, model_path=model_path)
    model = prepare_model_for_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    lora_cfg = create_lora_config(cfg, target_modules)
    model = apply_lora(model, lora_cfg)
    return (model, processor)
