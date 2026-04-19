"""Утилиты проверки GPU и VRAM."""

from __future__ import annotations

import torch


def assert_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA не обнаружена. Проверьте драйвер NVIDIA и сборку PyTorch с CUDA."
        )
    return torch.device("cuda")


def vram_report(prefix: str = "") -> str:
    if not torch.cuda.is_available():
        return "CUDA недоступна"
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(idx) / 1024**3
    reserved = torch.cuda.memory_reserved(idx) / 1024**3
    return (
        f"{prefix}GPU[{idx}] {name} | "
        f"total={total:.2f} GB, allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
    )


def print_vram(prefix: str = "") -> None:
    print(vram_report(prefix))
