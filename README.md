# rap-gpt

QLoRA fine-tuning **Gemma-3-4B** на русских рэп-текстах с Genius.
В комплекте: async-парсер, скрипт обучения, merge LoRA, CLI-чат и Gradio UI.

Целевое железо: ноутбук с **RTX 5070 Mobile (8 GB VRAM)**.

## Что в репо

- `scraper.py` — async-парсер Genius (aiohttp + BeautifulSoup), поддерживает несколько исполнителей/альбомов через запятую
- `finetune.py` — QLoRA-обучение (4-бит NF4 + LoRA)
- `merge_lora.py` — слияние LoRA-адаптера с базовой моделью
- `chat.py` — CLI-чат с моделью (merged или base+adapter)
- `app.py` — Gradio UI со всеми четырьмя шагами пайплайна
- `data/lyrics.jsonl` — готовый датасет (~380 песен)
- `checkpoints/qwen-rap-lora/` — LoRA-адаптер, обученный на этом датасете

## Быстрый старт

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

Gemma-3 — gated-модель, нужен HF-аккаунт + принятие лицензии на
https://huggingface.co/google/gemma-3-4b-pt, затем:

```bash
hf auth login --token <ТВОЙ_ТОКЕН>
```

## Запустить чат сразу (с готовым адаптером)

```bash
python chat.py \
  --base google/gemma-3-4b-pt \
  --adapter checkpoints/qwen-rap-lora \
  --load-4bit
```

Или через UI:

```bash
python app.py
```

## Заново обучить на своём датасете

```bash
# 1) спарсить
python scraper.py --artists "Oxxxymiron, Скриптонит" --output data/lyrics.jsonl

# 2) обучить (настройки под 8GB VRAM)
python finetune.py \
  --dataset data/lyrics.jsonl \
  --model google/gemma-3-4b-pt \
  --output-dir checkpoints/qwen-rap-lora \
  --epochs 6 \
  --max-seq-length 768 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --learning-rate 2e-4

# 3) (опционально) склеить адаптер с базой
python merge_lora.py \
  --base google/gemma-3-4b-pt \
  --adapter checkpoints/qwen-rap-lora \
  --output checkpoints/qwen-rap-merged
```

## Настройки при OOM

Крути по порядку:
1. `--max-seq-length 768 → 512`
2. `--grad-accum` увеличь (8 → 16 → 32)
3. `--lora-r 32 → 16`
4. Возьми модель поменьше: `google/gemma-3-1b-pt`

## Стек

- PyTorch 2.11 + CUDA 12.8 (Blackwell sm_120)
- transformers ≥ 5.5, trl ≥ 1.2, peft ≥ 0.19, bitsandbytes ≥ 0.49
- gradio ≥ 6.0
