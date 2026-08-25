"""Shared guard for the GPU-only half of the experiment.

Every script in this directory that needs MiniCPM-o weights calls ``require_gpu``
first. On this CPU machine (torch 2.10.0+cpu, 8 GB VRAM, no checkpoint) it prints
the exact blocker and the exact command to run on a GPU box, then exits non-zero
with status ``BLOCKED``. It NEVER fabricates a metric to look like it ran.

A script that *can* run on CPU (the SILT half: silt_proxy_run.py, export.py,
prepare_dataset.py) does not import this guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "minicpmo45_assamese"
HW = json.loads((EXP / "hardware.json").read_text(encoding="utf-8"))

BLOCKED_MESSAGE = "BLOCKED: requires CUDA torch + >=16 GB VRAM + MiniCPM-o checkpoint. See hardware.json blockers B1-B3."


def cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def minicpmo_checkpoint_present() -> bool:
    # HF cache + any local dir named like the checkpoint.
    hf = Path.home() / ".cache" / "huggingface" / "hub"
    if hf.exists():
        for p in hf.glob("*MiniCPM*"):
            return True
    return any(REPO.glob("**/MiniCPM*"))


def require_gpu(stage: str, gpu_command: str, min_vram_gb: float = 16.0):
    """Print the blocker + the exact GPU-box command, then exit BLOCKED."""
    reasons = []
    if not cuda_available():
        reasons.append("torch.cuda.is_available()==False (installed torch is {})".format(
            HW["torch"]["version"]))
    elif vram_gb() < min_vram_gb:
        reasons.append("VRAM {:.1f} GB < required {:.0f} GB".format(
            vram_gb(), min_vram_gb))
    if not minicpmo_checkpoint_present():
        reasons.append("MiniCPM-o checkpoint not present locally")
    if reasons:
        print("[{}] {}".format(stage, BLOCKED_MESSAGE))
        for r in reasons:
            print("  - {}".format(r))
        print("\nExact command to run on a GPU box:")
        print("  " + gpu_command.replace("\n", "\n  "))
        sys.exit(2)  # 2 = blocked, distinct from a normal error