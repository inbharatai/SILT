"""SILT teacher ingestion — register an Assamese specialist as a SILT sender.

On the GPU box this registers the AI4Bharat specialists (IndicConformer for STT,
IndicF5 for TTS, NLLB/strong Indic LLM for text) as SILT sender modules so the
pipeline can measure the capability gap into MiniCPM-o and distill a verified
skill packet. This reuses the existing catalog-addition pattern (catalog._ollama
for ollama teachers; a new HF connector for the AI4Bharat models) — NO core edit.

On this CPU machine it documents the pattern and the GPU download steps, then
exits BLOCKED for the AI4Bharat models (NLLB is already wired as `nllb-teacher`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from _gpu_guard import require_gpu, minicpmo_checkpoint_present  # noqa: E402

GPU_CMD = (
    "huggingface-cli download ai4bharat/IndicConformer \\\n"
    "  && huggingface-cli download ai4bharat/IndicF5 \\\n"
    "  && PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/silt_teacher_ingest.py"
)

# NLLB text teacher is already available on CPU as `nllb-teacher` (catalog.py);
# only the AI4Bharat speech teachers need the GPU box.
if not minicpmo_checkpoint_present():
    print("NLLB text teacher: available now as `nllb-teacher` (CPU, real weights).")
    print("AI4Bharat speech teachers (IndicConformer/IndicF5):")
    require_gpu("silt_teacher_ingest (speech teachers)", GPU_CMD, min_vram_gb=16)
print("PENDING_EXECUTION_ON_GPU — register IndicConformer/IndicF5 as SILT senders.")