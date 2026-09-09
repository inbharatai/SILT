# FLEURS English pilot attribution

Source: official Google FLEURS, immutable revision `4683b04af03d2d9549064c7d72060a9a94bb6046`.
https://huggingface.co/datasets/google/fleurs/tree/4683b04af03d2d9549064c7d72060a9a94bb6046

Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . Full preserved legal text: `CC-BY-4.0.txt`; upstream license declaration and citation: `upstream-README.md`. No endorsement is implied.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna (2022). *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*. https://arxiv.org/abs/2205.12446

Changes: deterministic 60-distinct-sentence subset of en_us test; source IEEE float32 audio converted to mono 16 kHz PCM16 by rounded/saturated quantization, with no resampling, trimming or normalization. The original dataset transcription is unchanged. Original audio hashes, output hashes and per-file transformations are preserved in manifest.json. Actual audio is external to this source distribution; fetch it with scripts/prepare_speech_pilot.py using the pinned official URLs in provenance.json.

This is a small real-recording ASR pilot, not full FLEURS, not an Indian-English evaluation, and not a speaker-independent or ASR quality claim. No model inference has been performed. See docs/SPEECH_PILOT.md for limits, selection, licensing fetch exception, tests and reproduction.
