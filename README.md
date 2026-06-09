## Методы и варианты сравнения

| Вариант | Описание | Данные обучения |
|---|---|---|
| `zero_shot` | Whisper large-v3 без дообучения | не обучается |
| `full_ft_clean` | полное дообучение Whisper | Golos Crowd 100h |
| `lora_clean` | LoRA r=16 | Golos Crowd 100h |
| `full_ft_noisy` | полное дообучение + шумовая аугментация | Golos Crowd 100h + шумы |
| `lora_noisy` | LoRA r=16 + шумовая аугментация | Golos Crowd 100h + шумы |
| `lora_noisy_r32` | LoRA r=32 + шумовая аугментация | Golos Crowd 100h + шумы |
| `gigaam_v2_ctc` | внешняя русскоязычная ASR-модель для сравнения | готовая модель |

Шумовая аугментация использует реальные шумовые корпуса MUSAN, ESC-50 и DEMAND. Во время обучения к записи может добавляться гауссовский шум, фоновый шум с заданным SNR и случайное изменение громкости; после смешивания сигнал нормализуется, чтобы избежать клиппинга. Для оценки используются записи без добавленного шума, условия с заданным SNR, Farfield и внешние наборы из других доменов.

## Результаты

WER, %, меньше лучше.

| Модель | без шума | SNR 20 dB | SNR 10 dB | SNR 0 dB | Farfield | Open STT Calls | RuDevices 10% | Common Voice 25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | 20.41 | 21.78 | 25.47 | 41.54 | 17.61 | 30.43 | 17.08 | 5.61 |
| `full_ft_clean` | 4.82 | 5.51 | 7.48 | 18.66 | 7.65 | 24.72 | 14.56 | 5.98 |
| `full_ft_noisy` | 4.75 | 5.31 | 6.94 | 15.23 | 7.42 | 25.16 | 14.89 | 6.10 |
| `lora_noisy_r32` | 3.98 | 4.30 | 5.84 | 12.66 | 7.62 | 30.20 | 17.13 | 8.25 |
| `gigaam_v2_ctc` | 2.79 | 2.83 | 3.74 | 12.55 | 4.37 | 21.10 | 8.40 | 3.26 |

## Статистическая значимость

Для ключевых сравнений используется парный бутстрэп по аудиозаписям и поправка Холма-Бонферрони. Это позволяет сравнивать модели на одних и тех же примерах и оценивать устойчивость разницы WER.

Обозначения в таблицах: `да` - различие статистически значимо после поправки Холма-Бонферрони, `нет` - значимого преимущества не подтверждено.

| Сравнение | без шума | SNR 20 dB | SNR 15 dB | SNR 10 dB | SNR 5 dB | SNR 0 dB | Farfield |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lora_noisy_r32` лучше `zero_shot` | да | да | да | да | да | да | да |
| `lora_noisy_r32` лучше `full_ft_clean` | да | да | да | да | да | да | нет |
| `lora_noisy_r32` лучше `full_ft_noisy` | да | да | да | да | да | да | нет |
| `full_ft_noisy` лучше `full_ft_clean` | нет | нет | нет | да | да | да | нет |

## Как пользоваться репозиторием

### 1. Установить зависимости

```bash
python3 -m pip install -r requirements.txt
```

### 2. Подготовить данные

```bash
python scripts/extract_data.py
python scripts/split_noise_pool.py
python scripts/prepare_quick_run_data.py
```

`extract_data.py` готовит речевые и шумовые данные, `split_noise_pool.py` разделяет шумы для обучения и оценки, `prepare_quick_run_data.py` создаёт малые выборки для быстрой проверки конвейера. Тестовые наборы без шума и с заданным SNR размещаются в `eval/thesis_test_sets/`; их нужно подготовить перед обучением и оценкой, если папка ещё не создана.

Дополнительные внешние наборы готовятся отдельными скриптами:

```bash
python scripts/prepare_cv25_eval.py \
  --archive DataAudiosets/1774215337504-cv-corpus-25.0-2026-03-09-ru.tar.gz \
  --extract-dir data/cv25 \
  --out-manifest data/cv25/test_manifest.jsonl \
  --split test \
  --workers 16

python scripts/prepare_open_stt_eval.py \
  --manifest-csv data/open_stt/asr_calls_2_val.csv \
  --audio-root data/open_stt/asr_calls_2_val \
  --out data/open_stt/asr_calls_2_val/manifest.jsonl \
  --sample-n 13000

python scripts/prepare_rudevices_eval.py \
  --audio-root data/rudevices \
  --out data/rudevices/manifest_10pct.jsonl \
  --fraction 0.1
```

### 3. Обучить варианты сравнения

```bash
python scripts/run_experiment.py \
  --train-manifest data/golos_wav/train_opus/100hours.jsonl \
  --eval-manifest data/mini_test/manifest.jsonl \
  --noise-dir noise_data \
  --eval-root eval/thesis_test_sets \
  --farfield-manifest data/golos_wav/test_opus/farfield/manifest.jsonl \
  --num-epochs 3 \
  --eval-strategy epoch
```

Для запуска отдельных вариантов можно использовать аргумент `--baselines`, например:

```bash
python scripts/run_experiment.py \
  --train-manifest data/golos_wav/train_opus/100hours.jsonl \
  --eval-manifest data/mini_test/manifest.jsonl \
  --noise-dir noise_data \
  --eval-root eval/thesis_test_sets \
  --farfield-manifest data/golos_wav/test_opus/farfield/manifest.jsonl \
  --baselines lora_noisy_r32 \
  --lora-r 32 \
  --num-epochs 3 \
  --eval-strategy epoch
```

### 4. Оценить контрольные точки Whisper

```bash
python scripts/run_eval_only.py \
  --checkpoints-root checkpoints \
  --eval-root eval/thesis_test_sets \
  --farfield-manifest data/golos_wav/test_opus/farfield/manifest.jsonl \
  --results-dir eval/results_thesis \
  --batch-size 160
```

Внешние условия оценки добавляются через `--extra-manifests`:

```bash
python scripts/run_eval_only.py \
  --checkpoints-root checkpoints \
  --eval-root eval/thesis_test_sets \
  --results-dir eval/results_thesis \
  --batch-size 160 \
  --extra-manifests \
    open_stt_calls=data/open_stt/asr_calls_2_val/manifest.jsonl \
    rudevices_10pct=data/rudevices/manifest_10pct.jsonl \
    cv25_test=data/cv25/test_manifest.jsonl
```

### 5. Оценить GigaAM

```bash
python scripts/eval_gigaam.py \
  --eval-root eval/thesis_test_sets \
  --farfield-manifest data/golos_wav/test_opus/farfield/manifest.jsonl \
  --results-dir eval/results_thesis \
  --batch-size 64 \
  --extra-manifests \
    open_stt_calls=data/open_stt/asr_calls_2_val/manifest.jsonl \
    rudevices_10pct=data/rudevices/manifest_10pct.jsonl \
    cv25_test=data/cv25/test_manifest.jsonl
```

### 6. Посчитать статистическую значимость

Сначала сохраняются предсказания и числа ошибок для каждой аудиозаписи:

```bash
python -u scripts/run_eval_predictions.py --batch-size 160
python -u scripts/eval_gigaam_predictions.py --batch-size 64
```

Затем выполняется бутстрэп-анализ:

```bash
python -u scripts/analyze_significance.py
```

Для контролируемых условий без шума и с заданным SNR можно явно указать манифесты:

```bash
python -u scripts/run_eval_predictions.py \
  --output-dir eval/predictions_significance_controlled \
  --conditions clean snr_20db snr_15db snr_10db snr_5db snr_0db farfield \
  --condition-manifests \
    clean=eval/thesis_test_sets/clean/manifest.jsonl \
    snr_20db=eval/thesis_test_sets/snr_20db/manifest.jsonl \
    snr_15db=eval/thesis_test_sets/snr_15db/manifest.jsonl \
    snr_10db=eval/thesis_test_sets/snr_10db/manifest.jsonl \
    snr_5db=eval/thesis_test_sets/snr_5db/manifest.jsonl \
    snr_0db=eval/thesis_test_sets/snr_0db/manifest.jsonl \
    farfield=data/golos_wav/test_opus/farfield/manifest.jsonl \
  --batch-size 160

python -u scripts/analyze_significance.py \
  --predictions-dir eval/predictions_significance_controlled \
  --output-dir eval/significance_controlled_full \
  --conditions clean snr_20db snr_15db snr_10db snr_5db snr_0db farfield \
  --comparisons \
    lora_noisy_r32:zero_shot \
    lora_noisy_r32:full_ft_clean \
    lora_noisy_r32:full_ft_noisy \
    full_ft_noisy:full_ft_clean
```

## Структура проекта

| Путь | Назначение |
|---|---|
| `scripts/extract_data.py` | подготовка речевых и шумовых корпусов |
| `scripts/run_experiment.py` | обучение и оценка вариантов на базе Whisper |
| `scripts/run_eval_only.py` | оценка готовых контрольных точек Whisper |
| `scripts/eval_gigaam.py` | оценка GigaAM-v2-CTC |
| `scripts/run_eval_predictions.py` | сохранение предсказаний Whisper по отдельным аудиозаписям |
| `scripts/eval_gigaam_predictions.py` | сохранение предсказаний GigaAM по отдельным аудиозаписям |
| `scripts/analyze_significance.py` | парный бутстрэп и поправка Холма-Бонферрони |
| `src/data/` | загрузка манифестов, отложенные аудиопреобразования, аугментация, сборка пакетов данных |
| `src/models/` | загрузка Whisper и применение LoRA |
| `src/training/` | настройка `Seq2SeqTrainer` |
| `src/evaluation/` | пакетное распознавание аудио и расчёт метрик |
| `src/utils/metrics.py` | нормализация текста, WER/CER, подсчёт ошибок |

## Данные и артефакты

| Путь | Что хранит | В репозитории |
|---|---|:---:|
| `DataAudiosets/` | исходные архивы датасетов | нет |
| `data/` | подготовленные речевые манифесты и аудио | нет |
| `noise_data/` | подготовленные шумовые корпуса | нет |
| `checkpoints/` | обученные модели и LoRA-адаптеры | нет |
| `eval/thesis_test_sets/` | тестовые наборы без шума и с заданным SNR | нет |
| `eval/results_thesis/` | агрегированные WER/CER результаты | нет |
| `eval/predictions_significance*/` | JSONL для бутстрэп-анализа | нет |

Тяжёлые данные, контрольные точки и промежуточные файлы с предсказаниями намеренно не хранятся в репозитории. Код ожидает, что они будут подготовлены локально описанными выше скриптами.

## Окружение

Минимальная установка:

```bash
python3 -m pip install -r requirements.txt
```

Для GigaAM может потребоваться установка пакета из исходников:

```bash
git clone --depth=1 https://github.com/salute-developers/GigaAM.git /tmp/gigaam_repo
pip install --no-deps /tmp/gigaam_repo
pip install hydra-core omegaconf sentencepiece
```

Полное обучение Whisper large-v3 требует CUDA GPU с достаточным объёмом памяти. Для быстрой проверки конвейера используйте малые выборки и готовые режимы запуска:

```bash
python scripts/run_preset.py smoke
python scripts/run_preset.py quick
```
