import json
from pathlib import Path
from typing import Optional
from datasets import Audio, Dataset

def _resolve_audio_path(manifest_path: str, record: dict) -> str:
    raw_path = record.get('wav') or record.get('audio_filepath')
    if not raw_path:
        raise KeyError("missing 'wav' or 'audio_filepath'")
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    manifest_dir = Path(manifest_path).resolve().parent
    candidates = [manifest_dir / path, manifest_dir.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f'missing audio file: {raw_path}')

def load_manifest_as_dataset(manifest_path: str, limit: Optional[int]=None) -> Dataset:
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            records.append({'audio': _resolve_audio_path(manifest_path, rec), 'sentence': rec['text'], 'duration': rec['duration']})
    ds = Dataset.from_list(records)
    ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    return ds
