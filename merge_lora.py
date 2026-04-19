"""
Merge LoRA-адаптеров с базовой моделью.

Важно: для слияния базовая модель загружается в fp16/bf16 БЕЗ квантования
(иначе merge_and_unload даст деградацию весов). Слияние 4B-модели
выполняется на CPU — это нормально и не требует GPU.

Пример:
    python merge_lora.py \
        --base google/gemma-3-4b-pt \
        --adapter checkpoints/qwen-rap-lora \
        --output checkpoints/qwen-rap-merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="HF id базовой модели")
    p.add_argument("--adapter", type=Path, required=True, help="Путь к LoRA-чекпоинту")
    p.add_argument("--output", type=Path, required=True, help="Куда сохранить merged-модель")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Загружаю базовую модель {args.base} ({args.dtype}, {args.device})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=args.device if args.device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"Применяю адаптер из {args.adapter}")
    model = PeftModel.from_pretrained(base, str(args.adapter))

    print("merge_and_unload...")
    model = model.merge_and_unload()

    print(f"Сохраняю в {args.output}")
    model.save_pretrained(str(args.output), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output))
    print("Готово.")


if __name__ == "__main__":
    main()
