"""
Интерактивное общение с обученной моделью.

Два режима:
    # 1) Merged-модель (после merge_lora.py) — самый быстрый старт
    python chat.py --model checkpoints/qwen-rap-merged

    # 2) Базовая модель + LoRA-адаптер (без merge), 4-битная загрузка
    python chat.py --base google/gemma-3-4b-pt --adapter checkpoints/qwen-rap-lora --load-4bit

Флаги:
    --prompt "..."     одиночная генерация и выход
    --max-new 300      длина ответа в токенах
    --temperature 0.9  креативность (0.7–1.1 для рэпа норм)
    --top-p 0.95
    --repetition-penalty 1.1

В интерактивном режиме:
    введи пустую строку → отправить;  /reset → очистить контекст;  /exit → выход.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

from gpu_utils import print_vram


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, help="Путь к merged-модели (папка)")
    p.add_argument("--base", help="HF id базовой модели (если грузим с адаптером)")
    p.add_argument("--adapter", type=Path, help="Путь к LoRA-адаптеру")
    p.add_argument("--load-4bit", action="store_true",
                   help="Загрузить базовую модель в 4-бит (QLoRA-стиль, экономит VRAM)")
    p.add_argument("--prompt", default=None, help="Одиночный промпт, без REPL")
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--cpu", action="store_true", help="Форсировать CPU")
    args = p.parse_args()
    if not args.model and not (args.base and args.adapter):
        raise SystemExit("Укажите либо --model, либо пару --base + --adapter")
    return args


def load_model_and_tokenizer(args: argparse.Namespace):
    device_map = "cpu" if args.cpu else {"": 0}

    if args.model:
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
        quant = None
        if args.load_4bit and not args.cpu:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model),
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=quant,
            trust_remote_code=True,
        )
    else:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
        quant = None
        if args.load_4bit and not args.cpu:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        base = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=quant,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(args.adapter))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_inputs(tokenizer, prompt: str, history: list[dict] | None):
    """Qwen2.5 имеет chat-template. Используем его, если модель instruct-вариант;
    иначе подаём сырой текст (base-модель, обученная на lyrics, работает как LM)."""
    if history is not None and getattr(tokenizer, "chat_template", None):
        messages = history + [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


@torch.inference_mode()
def generate(model, tokenizer, text: str, args: argparse.Namespace) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    generated = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def repl(model, tokenizer, args: argparse.Namespace) -> None:
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    history: list[dict] = [] if has_chat_template else None
    print("Готов. Пустая строка — отправить; /reset — очистить; /exit — выход.")
    while True:
        try:
            lines = []
            while True:
                line = input("you> " if not lines else "... ")
                if line == "":
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        prompt = "\n".join(lines).strip()
        if not prompt:
            continue
        if prompt == "/exit":
            return
        if prompt == "/reset":
            if history is not None:
                history = []
            print("[контекст очищен]")
            continue

        text = build_inputs(tokenizer, prompt, history)
        print("bot>")
        reply = generate(model, tokenizer, text, args).strip()
        if history is not None:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": reply})
        print()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)
    print_vram("loaded | ")

    if args.prompt:
        text = build_inputs(tokenizer, args.prompt, [] if getattr(tokenizer, "chat_template", None) else None)
        generate(model, tokenizer, text, args)
        print()
    else:
        repl(model, tokenizer, args)


if __name__ == "__main__":
    main()
