"""Arm A — untouched MiniCPM-o 4.5 baseline evaluation.

Measures the untouched checkpoint on every Assamese held-out split the experiment
cares about (text as->en, STT, TTS G2P, STS) AND on the regression split (English/
Chinese/general) so we can detect capability loss later. GPU-only on a CUDA box.

Output: writes ../baseline.json with real per-suite numbers. On this CPU machine
it prints the exact GPU command and exits BLOCKED (never fabricates a score).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from _gpu_guard import require_gpu  # noqa: E402

GPU_CMD = (
    "huggingface-cli download openbmb/MiniCPM-o-2_6 \\\n"
    "  && PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/baseline_eval.py \\\n"
    "       --checkpoint openbmb/MiniCPM-o-2_6 --out experiments/minicpmo45_assamese/baseline.json"
)

require_gpu("baseline_eval (Arm A)", GPU_CMD, min_vram_gb=16)

# --- GPU-box body (only reached when the guard passes) ------------------------
import json  # noqa: E402
import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

# On the GPU box: load MiniCPM-o-2_6 in 4-bit, run each Assamese held-out suite
# via the SILT BenchmarkHarness (reuse src/asea/benchmarks/harness.py), record
# real scores. This body is intentionally a stub until executed on a GPU box —
# it will be filled in from the live run and committed with real numbers. No
# number is written before the checkpoint actually runs.
out = {
    "arm": "A",
    "checkpoint": args.checkpoint,
    "status": "PENDING_EXECUTION_ON_GPU",
    "note": "guard passed on a CUDA box; fill from the live run, never fabricate.",
}
Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
print("PENDING_EXECUTION_ON_GPU — populate from the live checkpoint run.")