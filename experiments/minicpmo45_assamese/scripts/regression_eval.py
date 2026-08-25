"""Regression evaluation — detect capability loss in English/Chinese/general.

The whole point of SILT here is to add Assamese WITHOUT breaking what MiniCPM-o
already does. For each arm (A/B/C) we re-run the English, Chinese, and general
benchmarks and compare to Arm A. A regression beyond the gate's tolerance
(held_out_improvement_min / regression_tolerance, see training_plan.md) REJECTS
the adapter — it is not shipped. GPU-only.

Output: ../regression_metrics.json (real, per-arm). On CPU: prints the command
and exits BLOCKED.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from _gpu_guard import require_gpu  # noqa: E402

GPU_CMD = (
    "PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/regression_eval.py \\\n"
    "  --checkpoint <minicpmo-adapted-or-base> \\\n"
    "  --regression_suites mmlu_en,ceval_zh,general_reasoning \\\n"
    "  --baseline experiments/minicpmo45_assamese/baseline.json \\\n"
    "  --out experiments/minicpmo45_assamese/regression_metrics.json"
)

require_gpu("regression_eval", GPU_CMD, min_vram_gb=16)
print("PENDING_EXECUTION_ON_GPU — compare each arm to Arm A; reject on regression.")