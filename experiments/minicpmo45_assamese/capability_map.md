# MiniCPM-o 4.5 → SILT capability map

MiniCPM-o is an **omni** model: a Qwen3 LLM backbone plus a vision encoder, an audio encoder, an
audio-language projector, speech-token generation, a CosyVoice speech decoder, streaming /
full-duplex, a tokenizer, and an audio codec. Each component maps to one SILT capability. The map
below is the receiver capability profile SILT measures gaps against and gates transfers into.

**Convention:** `capability = <task_type> / <modality> / <domain> / <language>` (SILT
`CapabilityKey.as_str()`). Assamese = `as` (ISO 639-1); IPA-out uses `as-ipa` (matches the existing
`_g2p_as()` in `catalog.py`).

**Per-row fields:** `baseline` = MiniCPM-o untouched score on the held-out split (PENDING the
checkpoint; the existing `tts_pronunciation_as` suite gives us a real *symbolic* G2P baseline we
can measure against a proxy receiver today, see `baseline.md`). `teacher` = the specialist model
SILT ingests as the sender. `method` = L0–L5 (L0–L3 = packet conditioning at inference; L4 = PEFT
LoRA; L5 = sequence KD) — SILT only *selects/gates/exports*; the weight update is external.
`regression_risk` = the original capability the arm could damage. `validation` = how we prove it
worked (never self-eval-only).

## Map

| MiniCPM-o component | SILT capability | teacher (Assamese specialist) | method | data requirement | regression_risk | validation | baseline |
|---|---|---|---|---|---|---|---|
| Qwen3 LLM backbone (text) | `translate / text / translation / as->en` + `reason / text / general / as` | NLLB-200 (as↔en); strong Indic LLM for reasoning | L3 packet (glossary/exemplars) → L4 LoRA on the text backbone | IndicVoices transcripts + curated as→en pairs | English/Chinese/general text quality | held-out as→en BLEU/COMET vs A; external-ASR-free for text | PENDING |
| Qwen3 backbone (text reasoning, Assamese) | `reason / text / general / as` | strong Indic LLM | L3 exemplars → L4 LoRA | Assamese reasoning/explanation corpus | English reasoning drop | held-out Assamese QA + native fluency review | PENDING |
| Audio encoder + audio-language projector (ASR / STT) | `transcribe / audio_asr / general / as` | AI4Bharat **IndicConformer** (Assamese ASR) | L3 packet (transcription exemplars) → L4 LoRA on audio encoder/projector (verify framework trains audio-input adapters) | IndicVoices / IndicVoices-R Assamese audio + transcripts | English/Chinese ASR WER; English text quality | **independent-ASR round-trip** (TTS→external-ASR→WER), never self-eval; native-speaker transcription spot-check | PENDING |
| Speech-token generation + CosyVoice decoder (TTS) | `grapheme_to_phoneme / speech_tts / pronunciation / as-ipa` then `synthesize / speech_tts / general / as` | (G2P) GLM/strong LLM; (synthesis) AI4Bharat **IndicF5** Assamese TTS | L3 G2P lexicon packet (text, **already PROVEN**, see `error_analysis.md`) → **verify** the framework can train the speech-output decoder; if not, L4 text backbone + L5 KD only | IndicTTS / IndicF5 Assamese speech; **real recordings only, not synthetic-only** | English/Chinese TTS naturalness; accent contamination | **native-speaker MOS pack** (`human_eval_template.csv`); independent-ASR intelligibility; **never** claim TTS from a text LoRA | PENDING (G2P proxy: real, 0.7359) |
| STS (speech-to-speech, end-to-end) | `sts / speech_tts / general / as` | cascade (IndicConformer STT → LLM → IndicF5 TTS) as the teacher | L3 exemplars → L4 (only if the framework trains speech-in/speech-out; else cascade is the honest baseline) | paired as-speech-in/as-speech-out | English/Chinese STS; latency | native-speaker MOS on full round-trip + independent-ASR WER on the transcript | PENDING |
| Vision encoder + vision-language projector | `describe / image / general / as` | strong Indic VLM / caption-translation teacher | L3 exemplars → L4 LoRA on vision-language projector | Assamese image-caption pairs (curation needed — may be scarce) | English/Chinese vision quality | held-out Assamese caption accuracy + native review | PENDING |
| Tool calling (Qwen3 backbone, function-call head) | `tool_call / text / general / as` | strong function-call LLM | L3 exemplars → L4 LoRA | Assamese tool-call traces (synthetic-from-real only, depth≤2 per SILT rule) | English tool-call accuracy | held-out Assamese tool-call success vs A | PENDING |
| Agentic reasoning (planning + tool use) | `agent_reason / text / general / as` | strong agent LLM | L3 exemplars → L4 LoRA | Assamese agent trajectories | English agent quality | held-out Assamese agent task success vs A | PENDING |

## What this map does NOT do

- It does **not** declare a baseline score. Every `baseline` is `PENDING` until the checkpoint
  runs on a GPU box.
- It does **not** invent a teacher. Teachers are named specialist models (AI4Bharat
  IndicConformer / IndicF5, NLLB, strong Indic LLM/VLM) — cited in `dataset_manifest.jsonl`, to be
  **downloaded under their licences on the GPU box**, never scraped here.
- It does **not** edit SILT. New Assamese speech capabilities reuse `Modality.AUDIO_ASR` and
  `Modality.SPEECH_TTS` + existing `TTSExtractor`/`TTSDistiller`/`ASRExtractor`/`ASRDistiller`
  where present; STS is the one genuinely-new modality and is flagged per `docs/ADDING_A_DOMAIN.md`
  / `connector_authoring.md` as a **new-modality** path to author on the GPU box — no silent
  core edit.

## Provenance of this map

Grounded in the existing Assamese work: `src/asea/studio/catalog.py` (`_g2p_as()`,
`tts-teacher-as`, `tts-learner-zero`), `docs/SILT_TTS_G2P_TEST.md` (the PROMOTED G2P run), and the
`assamese_english` / `assamese_phrases` / `tts_pronunciation_as` benchmark suites. MiniCPM-o
component names from the OpenBMB MiniCPM-o model card (to be re-verified against the current 4.5
checkpoint docs at execution time).