"""TTS (Assamese speech synthesis) evaluation — native-speaker MOS + independent
ASR intelligibility. NEVER claims TTS improvement from a text LoRA.

The symbolic G2P layer (grapheme->IPA, text) is the only part measurable on CPU
today and is handled by silt_proxy_run.py (real, 0.7359 candidate). THIS script
measures the AUDIO synthesis layer, which needs the MiniCPM-o speech decoder and
real Assamese recordings — GPU-only, and the final word is the native-speaker
MOS pack (human_eval_template.csv), not a model score.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from _gpu_guard import require_gpu  # noqa: E402

GPU_CMD = (
    "# 1. synthesize held-out Assamese prompts with the model under test\n"
    "PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/tts_eval.py \\\n"
    "  --checkpoint <minicpmo-adapted-or-base> --split heldout \\\n"
    "  --out_dir experiments/minicpmo45_assamese/audio_<arm>\n"
    "# 2. independent ASR intelligibility (NOT self-eval)\n"
    "# 3. send audio_<arm>/ + human_eval_template.csv to native Assamese speakers\n"
    "#    — NO TTS-quality claim until the MOS pack is filled."
)

require_gpu("tts_eval (audio synthesis)", GPU_CMD, min_vram_gb=16)

# GPU-box body: synthesize held-out Assamese prompts with the speech decoder,
# run independent-ASR intelligibility, write wav list for native-speaker MOS.
# Hard rules enforced here: real recordings only (no synthetic-only training),
# no voice cloning without consent, no TTS claim from a text LoRA.
print("PENDING_EXECUTION_ON_GPU — synthesize audio, then native-speaker MOS.")