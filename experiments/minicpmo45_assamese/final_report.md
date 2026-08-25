# Final report — MiniCPM-o 4.5 + SILT → Assamese Omni Voice Agent

**Date:** 2026-08-13
**Honesty rule:** every number below is either **real, measured on this CPU
machine today**, or **PENDING a GPU box**. None is fabricated. If a future run
shows conventional fine-tuning (B) beating SILT-gated (C), the report will say so.

## 1. What the experiment set out to answer

> Does SILT reliably transfer Assamese speech capability into MiniCPM-o 4.5
> while preserving its original capabilities — better than conventional
> fine-tuning at equal compute/data budget?

Design: three arms on the same Assamese data and same compute budget — **A**
untouched baseline, **B** conventional fine-tune, **C** SILT-selected/trust-gated
fine-tune — reported truthfully, even if SILT does not win.

## 2. The one structural fact

**SILT trains no weights** (`src/asea/distill/export.py:1-15`; no
optimizer/backward anywhere in `src/asea`). SILT is a selection / trust-gating /
capability-gap-measurement / dataset-export layer; the weight update is done by
an external trainer (LLaMA-Factory / ms-swift / PEFT) and the trained adapter
re-enters SILT as a new receiver for re-benchmarking. This split the experiment
into a **CPU-feasible SILT half** and a **GPU-required trainer half**.

## 3. What ran for real on THIS machine (CPU)

### 3.1 SILT capability map + receiver registration for MiniCPM-o 4.5
Every omni component → a SILT capability (TEXT/STT/TTS/STS/VISION/TOOL/AGENT ×
Assamese), with teacher / method / data / regression-risk / validation per row
(`capability_map.md`). MiniCPM-o is also registered as an actual SILT **receiver**
in the real `ModuleAdapter` contract (`minicpmo_receiver.py`) — `is_mock=False`,
9 Assamese capabilities, a checkpoint preflight, and an `infer` guard that raises
`BLOCKED` (blockers B1-B3) rather than fabricating output. The exact additive
catalog entry to wire it into `catalog.py` on the GPU box is in
`register_into_catalog()` (same shape as the existing `tts-learner-zero` /
`glm-ollama` additions; no core/policy/gate edit; production `src/asea` untouched).

### 3.1a §4 Assamese tokenization expansion (real, CPU, today)
The Qwen2.5 backbone BPE tokenizer (MiniCPM-o 2.6's family) tokenizes Assamese at
**1.203 tokens/char — 5.19× English expansion** (ratio of mean tokens-per-char:
1.203 ÷ 0.232 = 5.185; English 0.232, Hindi 0.976), near byte-level: no learned
Assamese subword merges. A real Assamese tokenizer (NLLB) achieves 0.442. This is
a tokenizer-level prerequisite diagnosis (independent of weights) predicting a
weak Assamese baseline and token-expensive training; first thing to re-verify
against the actual MiniCPM-o 4.5 `tokenizer.json` on the GPU box. Per-sentence data
in `tokenization.json` (where the mean-of-per-sentence-ratios aggregation, 5.46×,
is recorded separately — not used in any headline claim).

### 3.2 Real SILT-on-Assamese validation (NOT a MiniCPM-o result)

**G2P replay** of the PROMOTED run (`docs/SILT_TTS_G2P_TEST.md` §5.6), re-run
today read-only via `BenchmarkHarness` against the on-disk approved packet:

| | value |
|---|---|
| receiver baseline (qwen3.5, held-out) | **0.6023** |
| receiver + SILT-approved G2P lexicon | **0.7359** |
| improvement | **+0.1336** |
| task_success | 0.0 → 0.333 |
| case regressions | 3/6 (ratio 0.50, within gate limit) |
| gate verdict (original run) | **PROMOTED** — all 13 checks passed |
| exact reproduction of documented run | **yes** |

The two wins that carried the run: `সাত: sat → xat` (+0.477, learned the
Assamese-distinctive স→/x/) and `কিতাপ: kitaap → kitap` (+0.400, exact). Per-case
detail in `error_analysis.md`.

**Text transfer** `nllb-teacher → qwen2.5:1.5b-instruct` on `assamese_english`:

| | value |
|---|---|
| status | done (90s, no error, no mock) |
| measured as->en gaps | none actionable |
| extracted / promoted | 0 / 0 |
| gate verdict | **NO_OP** — correctly refused (no gap to close) |

**What these two prove:** SILT discriminates on Assamese — it transfers when
there is a measured, actionable gap with real held-out lift (G2P +0.1336,
PROMOTED) and refuses when there is not (text: 0 extracted). That discrimination
is the core of "does SILT provide value," validated today on real weights. They
do **not** prove anything about MiniCPM-o.

### 3.3 A/B/C handoff for MiniCPM-o (budget-matched, NOT_EXECUTED)

`scripts/prepare_dataset.py` produced, from the real G2P run:

| arm | rows | epochs | row-epochs | gating | job spec |
|---|---|---|---|---|---|
| B (conventional) | 12 | 1 | 12 | none | `handoff/B_conventional.job.json` (NOT_EXECUTED, LoRA, `openbmb/MiniCPM-o-2_6`) |
| C (SILT-gated) | 4 | 3 | 12 | relevance+safety+trust+regression | `handoff/C_silt_gated.job.json` + bundle zip (NOT_EXECUTED, LoRA, same base) |

Equal training-token budget (12 row-epochs each). Shape demonstration on the real
G2P run; the GPU box regenerates B/C at IndicVoices scale via the same code path.

### 3.4 Reproducible scripts (`scripts/`)
`silt_proxy_run.py` (runs here, real), `prepare_dataset.py` + `export.py` (run
here, real), `trust_gate.py` (read-only, runs here), and the GPU-only half
(`baseline_eval.py`, `stt_eval.py`, `tts_eval.py`, `sts_eval.py`,
`regression_eval.py`, `train_lora.py`, `silt_teacher_ingest.py`) — each prints
the exact GPU command and exits `BLOCKED` on this CPU machine rather than
faking output.

## 4. What did NOT run, and the exact blockers (`hardware.json`)

| id | blocks | cause |
|---|---|---|
| B1 | all MiniCPM-o weight training + omni baseline inference | torch is CPU-only (`2.10.0+cpu`, `cuda.is_available()==False`); 8 GB VRAM < ~14-18 GB LoRA floor for an 8B backbone |
| B2 | any local SFT/LoRA/audio pipeline | peft/bitsandbytes/accelerate/datasets/trl/torchaudio/soundfile/librosa/llamafactory/ms_swift all absent |
| B3 | MiniCPM-o baseline/training without a large download | checkpoint not present locally |
| B4 | git branch | not a git repo (worked in `experiments/` only) |

Consequence: **MiniCPM-o arms A, B, C are all PENDING a GPU box.**
`training_runs.jsonl` is empty — no run is logged until one executes.

## 5. The answer to the question (status, not a conclusion)

The MiniCPM-o A/B/C comparison has **not been executed** — it is blocked on
hardware and handed off with exact commands (`training_plan.md`). What this run
*does* establish, today and for real:

- SILT's selection / trust-gating / held-out-lift / audited-accept-reject
  machinery works on Assamese (G2P PROMOTED +0.1336; text NO_OP), on real
  weights, reproducing the documented result exactly.
- The A/B/C handoff is ready: budget-matched B (raw) and C (SILT-gated) datasets
  + `NOT_EXECUTED` job specs targeting `openbmb/MiniCPM-o-2_6`, with the
  regression-budget policy and the speech-output verification gate in place.

The definitive answer — whether SILT-gated training of MiniCPM-o beats
conventional fine-tuning at equal budget while preserving English/Chinese/general
— is **PENDING the GPU-box run**. The plan, scripts, datasets, and gate are
ready to produce it honestly.

## 6. Discipline audit (the user's §23 "do not" list — all honoured)

No MiniCPM-V/o confusion · no GGUF/Ollama training (ollama is only the labelled
SILT-proxy receiver) · no STT claim from a text test · no TTS claim from a text
LoRA · no synthetic-only training · no test data in training · no self-eval-only
· no unlicensed audio · no voice cloning · no overwriting the base checkpoint ·
no merging unverified adapters · no big compute before a pilot passes · no
"loss went down therefore success" · no native-TTS claim without the human pack
(`human_eval_template.csv` shipped, unfilled) · no fabricated metrics · no
silent SILT architecture changes (production `src/asea` untouched; all work under
`experiments/`).