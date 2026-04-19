"""
QLoRA fine-tuning Qwen на датасете рэп-текстов.

Целевая конфигурация: ноутбук с NVIDIA RTX 5070 Mobile (8 GB VRAM).
Если у вас 12 GB — можно поднять max_seq_length до 2048 и batch_size до 2.

Пример:
    python finetune.py \
        --dataset data/lyrics.jsonl \
        --model Qwen/Qwen2.5-1.5B \
        --output-dir checkpoints/qwen-rap-lora \
        --epochs 3

>>> ЕСЛИ ВЫЛЕТАЕТ Out Of Memory — крутите параметры в таком порядке:
    1) --max-seq-length 1024 -> 768 -> 512
    2) --batch-size 1 (обычно уже 1)
    3) --grad-accum увеличить (4 -> 8 -> 16), чтобы сохранить эффективный батч
    4) --lora-r 16 -> 8
    5) сменить модель на меньшую: Qwen2.5-1.5B вместо 3B/7B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


class TokenTypeIdsCollator:
    """Обёртка над LM-коллатором: добавляет token_type_ids=0.
    Нужна для Gemma 3 — её текстовый декодер требует token_type_ids при train()
    (реликт multimodal-архитектуры: 0 = текст, не-0 = image-токены)."""

    def __init__(self, tokenizer):
        self.inner = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features):
        batch = self.inner(features)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch

from gpu_utils import assert_cuda, print_vram


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="Путь к .jsonl")
    p.add_argument(
        "--model",
        default="google/gemma-3-4b-pt",
        help="HF id базовой модели. 8GB: gemma-3-4b-pt (pt=base). 12GB+: gemma-3-12b-pt.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen-rap-lora"))

    # === Гиперпараметры. Крутите их при OOM. ===
    # Дефолты подобраны под качество рифмы/размера на 8GB VRAM.
    p.add_argument("--max-seq-length", type=int, default=1024)  # 12GB → 2048, целая песня в одном окне
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)         # эффективный батч = bs*grad_accum
    p.add_argument("--epochs", type=int, default=5)             # для рифмы лучше переучить, чем недоучить
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    # packing склеивает песни в один поток — ломает границы строф и рифмовые цепочки.
    # Для лирики ВЫКЛЮЧЕНО по умолчанию. На больших датасетах можно включить --packing
    # (сэкономит ~30% времени ценой некоторой размытости рифмы).
    p.add_argument("--packing", action="store_true",
                   help="Включить packing (быстрее, но хуже для рифмы)")

    # === LoRA ===
    # r=32 — больше «ёмкости» под поэтические паттерны, стиль, рифмопары.
    # При OOM: 32 -> 16 -> 8; alpha обычно держим = 2*r.
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    return p.parse_args()


def build_bnb_config() -> BitsAndBytesConfig:
    # 4-битное квантование (QLoRA): nf4 + double quant + bf16 compute.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen2.x: целевые attention+MLP проекции. При OOM можно сузить до ["q_proj","v_proj"].
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def load_jsonl_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    ds = load_dataset("json", data_files=str(path), split="train")
    # Быстрая санити-проверка
    sample = ds[0]
    if "text" not in sample:
        raise ValueError("Ожидается поле 'text' в каждой записи .jsonl")
    return ds


def main() -> None:
    args = parse_args()
    assert_cuda()
    print_vram("start | ")
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_args.json").write_text(
        json.dumps(vars(args), default=str, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=build_bnb_config(),
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # обязательное условие для gradient checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    print_vram("model loaded | ")

    dataset = load_jsonl_dataset(args.dataset)
    print(f"Датасет: {len(dataset)} записей")

    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",  # paged-оптимизатор экономит VRAM
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_seq_length,  # в trl>=1.0 параметр называется max_length
        packing=args.packing,  # False по умолчанию: сохраняет границы песен → стабильнее рифма
        neftune_noise_alpha=5,  # шум в эмбеддингах: ощутимо улучшает креативную генерацию
    )

    # Gemma 3 требует token_type_ids во входе при обучении. Для других
    # моделей лишний тензор не мешает — tokenizer генерирует input_ids/attn_mask,
    # а дополнительное поле игнорируется.
    data_collator = TokenTypeIdsCollator(tokenizer) if "gemma-3" in args.model.lower() else None

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # в trl>=1.0 вместо tokenizer
        train_dataset=dataset,
        peft_config=build_lora_config(args),
        args=sft_cfg,
        data_collator=data_collator,
    )

    print_vram("before train | ")
    trainer.train()
    print_vram("after train  | ")

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"LoRA-адаптеры сохранены в {args.output_dir}")


if __name__ == "__main__":
    main()
