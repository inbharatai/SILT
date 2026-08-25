"""STS (Assamese speech-to-speech) end-to-end evaluation.

Full round-trip: Assamese speech in -> MiniCPM-o -> Assamese speech out. Scored
by native-speaker MOS on the full round-trip AND independent-ASR WER on the
output transcript. GPU-only; never self-eval; no STS claim until native review.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from _gpu_guard import require_gpu  # noqa: E402

GPU_CMD = (
    "PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/sts_eval.py \\\n"
    "  --checkpoint <minicpmo-adapted-or-base> \\\n"
    "  --heldout data/indicvoices_as_sts_heldout.jsonl \\\n"
    "  --independent_asr AI4Bharat/IndicConformer \\\n"
    "  --out experiments/minicpmo45_assamese/sts_<arm>.json \\\n"
    "  --mos_pack experiments/minicpmo45_assamese/human_eval_template.csv"
)

require_gpu("sts_eval", GPU_CMD, min_vram_gb=16)
print("PENDING_EXECUTION_ON_GPU — full round-trip + native-speaker MOS.")