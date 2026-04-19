"""
Rap-GPT UI — Gradio-интерфейс для всего пайплайна.

Запуск:
    python app.py

Открой http://127.0.0.1:7860 в браузере. Там 4 вкладки:
    1. Парсер  — вписать артиста/альбом → собрать .jsonl
    2. Обучение — запустить fine-tuning с подобранными гиперпараметрами
    3. Merge    — склеить LoRA с базовой моделью
    4. Чат      — загрузить модель и общаться

Парсер/тренировка/merge запускаются subprocess-ами, лог стримится в UI.
Чат живёт в процессе Gradio, модель грузится один раз по кнопке.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from gpu_utils import vram_report

ROOT = Path(__file__).parent
DEFAULT_MODEL = "google/gemma-3-4b-pt"  # лучший русский в 4B, легко лезет в 8GB QLoRA

# ==================== Chat state ====================

chat_state: dict = {
    "model": None,
    "tokenizer": None,
    "label": None,
}


def _bnb_4bit() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_chat_model(mode: str, merged_path: str, base: str, adapter: str, load_4bit: bool) -> str:
    """Загружает модель в chat_state. Возвращает строку-статус."""
    try:
        if not torch.cuda.is_available():
            device_map = "cpu"
        else:
            device_map = {"": 0}

        quant = _bnb_4bit() if (load_4bit and device_map != "cpu") else None

        if mode == "Merged":
            if not merged_path:
                return "Укажи путь к merged-модели"
            tok = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                merged_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                quantization_config=quant,
                trust_remote_code=True,
            )
            label = f"merged: {merged_path}"
        else:  # Base + LoRA
            from peft import PeftModel

            if not base or not adapter:
                return "Укажи базовую модель и путь к адаптеру"
            tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            bm = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                quantization_config=quant,
                trust_remote_code=True,
            )
            mdl = PeftModel.from_pretrained(bm, adapter)
            label = f"{base} + {adapter}"

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl.eval()

        chat_state.update(model=mdl, tokenizer=tok, label=label)
        return f"Загружено: {label}\n{vram_report()}"
    except Exception as e:  # pragma: no cover
        return f"Ошибка загрузки: {e}"


def chat_generate(
    message: str,
    history: list[dict],
    completion_mode: bool,
    max_new: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> Iterator[str]:
    if chat_state["model"] is None:
        yield "Сначала загрузи модель на панели слева."
        return

    tok = chat_state["tokenizer"]
    mdl = chat_state["model"]

    # completion_mode=True: модель обучалась на сыром тексте песен —
    # подаём твой ввод как затравку и продолжаем, без чат-шаблона.
    # completion_mode=False: нормальный чат с ролями (имеет смысл только для
    # instruct-моделей или если ты дообучал её на диалогах).
    if completion_mode or not getattr(tok, "chat_template", None):
        text = message
    else:
        messages = (history or []) + [{"role": "user", "content": message}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(text, return_tensors="pt").to(mdl.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=50,
        repetition_penalty=float(repetition_penalty),
        do_sample=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=mdl.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial
    thread.join()


# ==================== Subprocess streaming ====================

def stream_cmd(cmd: list[str]) -> Iterator[str]:
    """Запускает команду, yield'ит накапливаемый stdout."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    buf = f"$ {' '.join(cmd)}\n"
    yield buf
    assert proc.stdout is not None
    for line in proc.stdout:
        buf += line
        yield buf
    proc.wait()
    buf += f"\n[exit code: {proc.returncode}]"
    yield buf


# ==================== Tab handlers ====================

def run_scraper(artist: str, album: str, output: str, max_songs: int) -> Iterator[str]:
    artist = (artist or "").strip()
    album = (album or "").strip()
    if bool(artist) == bool(album):
        yield "Заполни ровно одно поле: артист ИЛИ альбом"
        return
    output = (output or "data/lyrics.jsonl").strip()

    cmd = [sys.executable, "scraper.py", "--output", output]
    if artist:
        cmd += ["--artist", artist]
    else:
        cmd += ["--album", album]
    if max_songs and max_songs > 0:
        cmd += ["--max-songs", str(int(max_songs))]
    yield from stream_cmd(cmd)


def run_training(
    dataset: str,
    model_id: str,
    output_dir: str,
    epochs: int,
    max_seq: int,
    batch: int,
    grad_accum: int,
    lora_r: int,
    lr: float,
) -> Iterator[str]:
    if not dataset:
        yield "Укажи путь к датасету (.jsonl)"
        return
    cmd = [
        sys.executable, "finetune.py",
        "--dataset", dataset,
        "--model", model_id or DEFAULT_MODEL,
        "--output-dir", output_dir or "checkpoints/qwen-rap-lora",
        "--epochs", str(int(epochs)),
        "--max-seq-length", str(int(max_seq)),
        "--batch-size", str(int(batch)),
        "--grad-accum", str(int(grad_accum)),
        "--lora-r", str(int(lora_r)),
        "--learning-rate", str(float(lr)),
    ]
    yield from stream_cmd(cmd)


def run_merge(base: str, adapter: str, output: str) -> Iterator[str]:
    if not (base and adapter and output):
        yield "Заполни base / adapter / output"
        return
    cmd = [
        sys.executable, "merge_lora.py",
        "--base", base,
        "--adapter", adapter,
        "--output", output,
    ]
    yield from stream_cmd(cmd)


# ==================== UI ====================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Rap-GPT") as app:
        gr.Markdown(
            "# Rap-GPT\n"
            "Пайплайн: **парсер → обучение → merge → чат**. "
            f"Дефолтная модель: `{DEFAULT_MODEL}`.  \n"
            f"{vram_report()}"
        )

        with gr.Tab("1. Парсер"):
            gr.Markdown(
                "Заполни **артист** ИЛИ **альбом** (не оба). Источник — Genius.  \n"
                "Можно указать **несколько через запятую** — спарсит всех и склеит в один файл."
            )
            with gr.Row():
                artist = gr.Textbox(
                    label="Артист(ы)",
                    placeholder="Oxxxymiron, Face, Miyagi",
                )
                album = gr.Textbox(
                    label="Альбом(ы)",
                    placeholder="Горгород, Pyramid",
                )
            with gr.Row():
                output = gr.Textbox(label="Выходной .jsonl", value="data/lyrics.jsonl")
                max_songs = gr.Number(
                    label="Макс. песен на артиста/альбом (0 = все)",
                    value=0,
                    precision=0,
                )
            scrape_btn = gr.Button("Спарсить", variant="primary")
            scrape_log = gr.Textbox(label="Лог", lines=18, max_lines=18)
            scrape_btn.click(
                run_scraper,
                inputs=[artist, album, output, max_songs],
                outputs=scrape_log,
            )

        with gr.Tab("2. Обучение"):
            gr.Markdown(
                "QLoRA (4-bit) + gradient checkpointing + paged_adamw_32bit.  \n"
                "**При OOM** крути в порядке: max-seq ↓, grad-accum ↑, lora-r ↓, модель меньше."
            )
            with gr.Row():
                tr_dataset = gr.Textbox(label="Датасет .jsonl", value="data/lyrics.jsonl")
                tr_model = gr.Textbox(label="HF id модели", value=DEFAULT_MODEL)
            with gr.Row():
                tr_output = gr.Textbox(label="Куда сохранить LoRA", value="checkpoints/qwen-rap-lora")
                tr_epochs = gr.Number(label="Эпохи", value=5, precision=0)
            with gr.Row():
                tr_max_seq = gr.Slider(256, 4096, value=1024, step=128, label="max_seq_length")
                tr_batch = gr.Slider(1, 8, value=1, step=1, label="batch_size")
                tr_grad_accum = gr.Slider(1, 32, value=8, step=1, label="grad_accum")
            with gr.Row():
                tr_lora_r = gr.Slider(4, 64, value=32, step=4, label="lora_r (32 — лучше рифма)")
                tr_lr = gr.Number(label="learning_rate", value=2e-4)
            tr_btn = gr.Button("Запустить обучение", variant="primary")
            tr_log = gr.Textbox(label="Лог", lines=22, max_lines=22)
            tr_btn.click(
                run_training,
                inputs=[tr_dataset, tr_model, tr_output, tr_epochs, tr_max_seq,
                        tr_batch, tr_grad_accum, tr_lora_r, tr_lr],
                outputs=tr_log,
            )

        with gr.Tab("3. Merge LoRA"):
            gr.Markdown("Склеивает LoRA-адаптеры с базовой моделью. Делается на CPU.")
            with gr.Row():
                m_base = gr.Textbox(label="База (HF id)", value=DEFAULT_MODEL)
                m_adapter = gr.Textbox(label="LoRA-адаптер", value="checkpoints/qwen-rap-lora")
                m_output = gr.Textbox(label="Merged-модель", value="checkpoints/qwen-rap-merged")
            m_btn = gr.Button("Смержить", variant="primary")
            m_log = gr.Textbox(label="Лог", lines=14, max_lines=14)
            m_btn.click(run_merge, inputs=[m_base, m_adapter, m_output], outputs=m_log)

        with gr.Tab("4. Чат"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Загрузка модели")
                    mode = gr.Radio(
                        ["Merged", "Base + LoRA"],
                        value="Base + LoRA",
                        label="Режим",
                    )
                    merged_path = gr.Textbox(label="Merged-модель", value="checkpoints/qwen-rap-merged")
                    base_id = gr.Textbox(label="Базовая модель", value=DEFAULT_MODEL)
                    adapter_path = gr.Textbox(label="LoRA-адаптер", value="checkpoints/qwen-rap-lora")
                    load_4bit = gr.Checkbox(label="Загрузить в 4-bit (экономит VRAM)", value=True)
                    load_btn = gr.Button("Загрузить модель", variant="primary")
                    load_status = gr.Textbox(label="Статус", lines=4)
                    load_btn.click(
                        load_chat_model,
                        inputs=[mode, merged_path, base_id, adapter_path, load_4bit],
                        outputs=load_status,
                    )

                    gr.Markdown("### Параметры генерации")
                    completion_mode = gr.Checkbox(
                        value=True,
                        label="Режим продолжения (выкл. для чата)",
                        info="ВКЛ: модель продолжает твой текст (для LoRA на песнях). "
                             "ВЫКЛ: чат с ролями (для instruct-моделей).",
                    )
                    gr.Markdown(
                        "Дефолты настроены под рифму: умеренная температура, "
                        "активный repetition penalty."
                    )
                    max_new = gr.Slider(32, 1024, value=400, step=16, label="max_new_tokens")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.92, step=0.01, label="top_p")
                    rep_pen = gr.Slider(1.0, 1.5, value=1.18, step=0.02, label="repetition_penalty")

                with gr.Column(scale=2):
                    gr.ChatInterface(
                        fn=chat_generate,
                        additional_inputs=[completion_mode, max_new, temperature, top_p, rep_pen],
                        title="Поболтать с моделью",
                        description=(
                            "В режиме продолжения каждое сообщение — отдельная затравка. "
                            "Напиши 2–4 строки с рифмой — модель подхватит размер."
                        ),
                    )

    return app


if __name__ == "__main__":
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
