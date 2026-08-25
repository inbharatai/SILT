"""STT (Assamese ASR) evaluation — independent-ASR round-trip, NEVER self-eval.

For Arm A/B/C MiniCPM-o STT we measure Word Error Rate against the held-out
IndicVoices Assamese transcripts, transcribed by an INDEPENDENT external ASR
(AI4Bharat IndicConformer) — never by the model scoring itself. GPU-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from _gpu_guard import require_gpu  # noqa: E402

GPU_CMD = (
    "PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/stt_eval.py \\\n"
    "  --checkpoint <minicpmo-adapted-or-base> \\\n"
    "  --heldout data/indicvoices_as_heldout.jsonl \\\n"
    "  --independent_asr AI4Bharat/IndicConformer \\\n"
    "  --out experiments/minicpmo45_assamese/stt_<arm>.json"
)

require_gpu("stt_eval", GPU_CMD, min_vram_gb=16)

# GPU-box body: transcribe held-out Assamese audio with the model under test,
# score transcripts with the independent IndicConformer ASR WER, record real WER.
# Constraint: never self-eval (the scorer must be a different model).
print("PENDING_EXECUTION_ON_GPU — populate from the live run; never self-eval.")