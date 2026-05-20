import hashlib
import json
import os
from typing import Optional
from datasets import load_from_disk
from src.data.augmentation import prepare_eval_example, prepare_train_example
from src.data.dataset_loader import load_manifest_as_dataset
DEFAULT_CACHE_DIR = os.environ.get('DATASET_FILTER_CACHE_DIR', os.path.join(os.getcwd(), '.cache', 'filtered_datasets'))

def _filter_cache_key(manifest_path: str, limit: Optional[int], max_duration: Optional[float], max_label_tokens: Optional[int], filter_empty: bool, tokenizer_name: str) -> str:
    st = os.stat(manifest_path)
    payload = {'manifest': os.path.abspath(manifest_path), 'mtime_ns': st.st_mtime_ns, 'size': st.st_size, 'limit': limit, 'max_duration': max_duration, 'max_label_tokens': max_label_tokens, 'filter_empty': filter_empty, 'tokenizer': tokenizer_name}
    blob = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha1(blob).hexdigest()[:16]

class _AugmentedTransform:

    def __init__(self, processor, augmenter):
        self.processor = processor
        self.augmenter = augmenter

    def __call__(self, examples):
        input_features = []
        labels = []
        batch_size = len(examples['audio'])
        for i in range(batch_size):
            ex = {k: v[i] for k, v in examples.items()}
            out = prepare_train_example(ex, self.processor, self.augmenter)
            input_features.append(out['input_features'])
            labels.append(out['labels'])
        return {'input_features': input_features, 'labels': labels}

class _CleanTransform:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        input_features = []
        labels = []
        batch_size = len(examples['audio'])
        for i in range(batch_size):
            ex = {k: v[i] for k, v in examples.items()}
            out = prepare_eval_example(ex, self.processor)
            input_features.append(out['input_features'])
            labels.append(out['labels'])
        return {'input_features': input_features, 'labels': labels}

def _filter_dataset(ds, processor, max_duration: Optional[float]=30.0, max_label_tokens: Optional[int]=448, filter_empty: bool=True):
    if max_duration is not None:
        ds = ds.filter(lambda x: x['duration'] <= max_duration, desc='filter duration')
    if filter_empty:
        ds = ds.filter(lambda x: len(x['sentence'].strip()) > 0, desc='filter empty text')
    if max_label_tokens is not None and processor is not None:
        ds = ds.filter(lambda x: len(processor.tokenizer(x['sentence']).input_ids) <= max_label_tokens, desc='filter labels')
    return ds

def load_and_filter_dataset(manifest_path: str, processor, limit: Optional[int]=None, max_duration: Optional[float]=30.0, max_label_tokens: Optional[int]=448, filter_empty: bool=True, cache_dir: Optional[str]=None, use_cache: bool=True):
    if os.environ.get('DATASET_FILTER_CACHE', '1') == '0':
        use_cache = False
    if use_cache:
        cache_root = cache_dir or DEFAULT_CACHE_DIR
        tokenizer_name = getattr(getattr(processor, 'tokenizer', None), 'name_or_path', '') if max_label_tokens is not None else ''
        key = _filter_cache_key(manifest_path, limit, max_duration, max_label_tokens, filter_empty, tokenizer_name)
        cache_path = os.path.join(cache_root, key)
        if os.path.isdir(cache_path):
            print(f'[cache] load: {cache_path}')
            return load_from_disk(cache_path)
    ds = load_manifest_as_dataset(manifest_path, limit=limit)
    ds = _filter_dataset(ds, processor, max_duration, max_label_tokens, filter_empty)
    if use_cache:
        import shutil
        os.makedirs(cache_root, exist_ok=True)
        tmp_path = cache_path + '.tmp'
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        ds.save_to_disk(tmp_path)
        os.replace(tmp_path, cache_path)
        print(f'[cache] save: {cache_path}')
        ds = load_from_disk(cache_path)
    return ds

def attach_train_transform(ds, processor, augmenter=None):
    if augmenter is not None:
        ds.set_transform(_AugmentedTransform(processor, augmenter))
    else:
        ds.set_transform(_CleanTransform(processor))
    return ds

def attach_eval_transform(ds, processor):
    ds.set_transform(_CleanTransform(processor))
    return ds

def make_train_dataset(manifest_path: str, processor, augmenter=None, limit: Optional[int]=None, max_duration: Optional[float]=30.0, max_label_tokens: Optional[int]=448, filter_empty: bool=True):
    ds = load_and_filter_dataset(manifest_path, processor, limit, max_duration, max_label_tokens, filter_empty)
    return attach_train_transform(ds, processor, augmenter)

def make_eval_dataset(manifest_path: str, processor, limit: Optional[int]=None, max_duration: Optional[float]=30.0, max_label_tokens: Optional[int]=448, filter_empty: bool=True):
    ds = load_and_filter_dataset(manifest_path, processor, limit, max_duration, max_label_tokens, filter_empty)
    return attach_eval_transform(ds, processor)
