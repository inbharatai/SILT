# MiniCPM-o 4.5 Assamese — training plan (NOT executed on this machine)

This plan is the GPU-box handoff. Nothing here has run (see `hardware.json` blockers
B1-B3). Every command is exact and ready to fire on a CUDA box with ≥16 GB VRAM
(≥24 GB preferred for the omni model). SILT trains no weights; the commands below
are the **external trainer** half. The trained adapter re-enters SILT as a NEW
receiver module and is re-benchmarked before use.

## Staged order (A → F) — pilot-first, never skip the gate

Each stage: (1) measure the gap into MiniCPM-o via SILT, (2) prepare B + C
budget-matched datasets, (3) **pilot** (1 epoch, small slice) and confirm held-out
A/B lifts BEFORE the full budget, (4) full LoRA, (5) re-enter adapter as a new
receiver + re-benchmark + regression eval. A stage that regresses English/Chinese/
general beyond `regression_tolerance` (0.02) is **rejected — not shipped**.

| stage | capability | teacher | what trains (LoRA target) | gate |
|---|---|---|---|---|
| A | TEXT_ASSAMESE (as->en, reasoning) | NLLB + strong Indic LLM | Qwen3 backbone (q,k,v,o proj) | held-out as->en BLEU/COMET + native fluency |
| B | STT_ASSAMESE | AI4Bharat IndicConformer | audio encoder + audio-language projector (**verify the framework trains audio-input adapters**) | independent-ASR WER (never self-eval) |
| C | TTS_ASSAMESE (G2P, then synthesis) | G2P: GLM/LLM; synthesis: AI4Bharat IndicF5 | **VERIFY** the framework can train the speech-output decoder; if not, L4 text backbone + L5 KD only — **never claim TTS from a text LoRA** | native-speaker MOS (`human_eval_template.csv`) + independent-ASR intelligibility |
| D | STS_ASSAMESE | cascade (IndicConformer→LLM→IndicF5) | only if the framework trains speech-in/speech-out; else cascade is the honest baseline | native-speaker MOS on full round-trip |
| E | TOOL_CALLING_ASSAMESE | strong function-call LLM | Qwen3 backbone | held-out Assamese tool-call success |
| F | VISION_ASSAMESE | strong Indic VLM | vision-language projector | held-out Assamese caption accuracy + native review |

## Regression-budget policy (the whole point: add Assamese, keep everything else)

- Before any training: record Arm A baselines on English (MMLU-style), Chinese
  (C-Eval-style), and general reasoning → `baseline.json`.
- After each arm: re-run those suites → `regression_metrics.json`. A drop beyond
  `regression_tolerance` (0.02) on any non-Assamese suite **rejects the adapter**.
- LoRA only — **never overwrite the base checkpoint.** Never merge an unverified
  adapter. Re-enter the trained adapter as a NEW receiver module and re-benchmark
  through SILT before use.

## Exact commands (GPU box)

```bash
# 0. environment (pin versions; verify speech-output training support for B/C/D)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install peft bitsandbytes accelerate datasets trl tensorboard \
            torchaudio soundfile librosa
pip install llamafactory  # OR: pip install ms-swift
huggingface-cli download openbmb/MiniCPM-o-2_6   # verify the current 4.5 id

# 1. SILT gap measurement + B/C dataset prep (CPU-feasible, runs here too):
PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/silt_teacher_ingest.py
PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/prepare_dataset.py

# 2. Arm A baseline (untouched checkpoint):
PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/baseline_eval.py \
  --checkpoint openbmb/MiniCPM-o-2_6 --out experiments/minicpmo45_assamese/baseline.json

# 3. Arm B (conventional) — pilot first, then full:
llamafactory-cli train \
  --model_name_or_path openbmb/MiniCPM-o-2_6 \
  --dataset minicpmo_assamese_B \
  --finetuning_type lora --lora_rank 16 --lora_alpha 32 \
  --learning_rate 1e-4 --num_train_epochs 1 \
  --lora_target q_proj,k_proj,v_proj,o_proj \
  --output_dir experiments/minicpmo45_assamese/adapter_B --do_train

# 4. Arm C (SILT-gated) — same budget (3 epochs on 1/3 the rows):
llamafactory-cli train \
  --model_name_or_path openbmb/MiniCPM-o-2_6 \
  --dataset minicpmo_assamese_C \
  --finetuning_type lora --lora_rank 16 --lora_alpha 32 \
  --learning_rate 1e-4 --num_train_epochs 3 \
  --lora_target q_proj,k_proj,v_proj,o_proj \
  --output_dir experiments/minicpmo45_assamese/adapter_C --do_train

# ms-swift equivalent (alternative trainer):
# swift sft --model_type minicpm-o --dataset minicpmo_assamese_<arm> \
#   --sft_type lora --lora_target_modules q_proj k_proj v_proj o_proj \
#   --num_train_epochs <1 or 3> --output_dir adapter_<arm>

# 5. Re-enter each trained adapter as a NEW receiver, re-benchmark via SILT,
#    then regression eval (reject on regression):
PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/regression_eval.py \
  --checkpoint experiments/minicpmo45_assamese/adapter_B \
  --baseline experiments/minicpmo45_assamese/baseline.json \
  --out experiments/minicpmo45_assamese/regression_metrics.json
# (repeat for adapter_C; then compare A vs B vs C in final_report.md)
```

## Speech-output verification gate (stages B/C/D)

Before training audio-out, **verify** that LLaMA-Factory / ms-swift can train the
MiniCPM-o speech-token + CosyVoice decoder path (not just the text backbone). If
the chosen framework cannot, the plan says so and uses the cleanest supported
alternative (L4 text backbone + L5 sequence-KD, or a cascade). **A text LoRA is
never reported as a TTS improvement.** The `tts_eval.py` audio-synthesis script
and the native-speaker MOS pack are the only valid TTS evidence.

## Hard rules (the user's §23, enforced by the scripts)

No MiniCPM-V/o confusion · no GGUF/Ollama training · no STT claim from a text test
· no TTS claim from a text LoRA · no synthetic-only training (synthetic-from-real
capped at `synthetic_depth ≤ 2`) · no test data in training · no self-eval-only ·
no unlicensed audio · no voice cloning without consent · no overwriting the base
checkpoint · no merging unverified adapters · no big compute before the pilot
passes · no "loss went down therefore success" (the held-out A/B is the bar) · no
native-TTS claim without the human pack · no fabricated metrics.