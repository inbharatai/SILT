"""Arm B/C MiniCPM-o LoRA training launcher — calls an EXTERNAL trainer.

SILT trains no weights. This script hands the budget-matched dataset +
NOT_EXECUTED job spec (produced by prepare_dataset.py / export.py) to an external
trainer (LLaMA-Factory or ms-swift). It does NOT overwrite the base checkpoint
(LoRA only) and does NOT merge an unverified adapter. The trained adapter re-
enters SILT as a NEW receiver module and is re-benchmarked before use.

GPU-only. On CPU it prints the exact command and exits BLOCKED. The command
below is the LLaMA-Factory form; ms-swift equivalent is in training_plan.md.

Pilot-first rule: run the tiny pilot (1 epoch, small slice) and confirm the
held-out A/B shows lift BEFORE spending the full budget. No "loss went down
therefore success" — the gate's held-out improvement is the bar.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from _gpu_guard import require_gpu  # noqa: E402

GPU_CMD = (
    "# Pilot first (tiny slice, 1 epoch) — only proceed if held-out A/B lifts:\n"
    "llamafactory-cli train \\\n"
    "  --model_name_or_path openbmb/MiniCPM-o-2_6 \\\n"
    "  --dataset minicpmo_assamese_<arm> \\\n"
    "  --finetuning_type lora --lora_rank 16 --lora_alpha 32 \\\n"
    "  --learning_rate 1e-4 --num_train_epochs 2 \\\n"
    "  --lora_target q_proj,k_proj,v_proj,o_proj \\\n"
    "  --output_dir experiments/minicpmo45_assamese/adapter_<arm> \\\n"
    "  --do_train --report_to tensorboard\n"
    "# Re-enter the trained adapter as a NEW receiver and re-benchmark via SILT\n"
    "# BEFORE use. Never merge an unverified adapter. Never overwrite the base."
)

require_gpu("train_lora (Arm B/C)", GPU_CMD, min_vram_gb=16)
print("PENDING_EXECUTION_ON_GPU — pilot first, then full LoRA; re-benchmark before use.")