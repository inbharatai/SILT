# Baseline

## Arm A — MiniCPM-o 4.5 untouched: PENDING (GPU box)

The untouched-checkpoint baseline on every Assamese held-out split (text as->en,
STT, TTS G2P, STS) and on the regression split (English / Chinese / general) has
**not been measured** — it requires the MiniCPM-o checkpoint on a CUDA box with
≥16 GB VRAM (blockers B1-B3). It is produced by `scripts/baseline_eval.py` on the
GPU box and written to `baseline.json`. No number is filled in here until that
run executes.

## Real SILT-proxy baseline (measured on this CPU machine, today)

This is **not** a MiniCPM-o result. It is the SILT-on-Assamese baseline that the
feasible half actually measured, using real connectors available here. It grounds
the claim that SILT's selection / trust-gating / held-out-lift machinery works on
Assamese.

### G2P (symbolic TTS pronunciation layer) — `tts_pronunciation_as` suite

Replay of the existing PROMOTED run (`docs/SILT_TTS_G2P_TEST.md` §5.6), re-run
today via `scripts/silt_proxy_run.py` (read-only `BenchmarkHarness` A/B against
the on-disk approved packet, never `Evaluator.evaluate`):

| | score |
|---|---|
| receiver (qwen3.5:latest) held-out baseline | **0.6023** |
| receiver + SILT-approved G2P lexicon | **0.7359** |
| improvement | **+0.1336** |
| gate verdict (the original run) | **PROMOTED** — all 13 checks passed |

This reproduces the documented run exactly. Per-case detail in `error_analysis.md`.

### Text (as->en) — `assamese_english` suite

A fresh real transfer `nllb-teacher` → `qwen2.5:1.5b-instruct` (ollama):

| | value |
|---|---|
| status | done (93s, no error) |
| involves_mock | false |
| measured_gaps (as->en) | none actionable |
| extracted / promoted | 0 / 0 |

**Honest reading:** the SILT gate measured the as->en gap between NLLB and
qwen2.5:1.5b on the extraction split and found **no actionable headroom** — the
receiver already handles the 12 single-word extraction items well enough that
NLLB has nothing to add on them — so it correctly **refused to push a transfer**
(`extracted=0`, `actionable=0`). This is the gate doing its job: SILT does not
ship a transfer when there is no measured gap to close. It is a real, audited
"no-op" outcome, recorded verbatim in `metrics.json` → `text_transfer.report`.

### What these two baselines tell us (and what they do not)

- **Do:** SILT correctly discriminates on Assamese — it *transfers* when there is
  a measured, actionable gap with real held-out lift (G2P: +0.1336, PROMOTED),
  and *refuses* when there is not (text: no gap, 0 extracted). That
  discrimination is the core of "does SILT provide value."
- **Do not:** neither number is a MiniCPM-o number. The receiver here is qwen3.5 /
  qwen2.5:1.5b, not MiniCPM-o. The MiniCPM-o A/B/C comparison is PENDING a GPU
  box. No claim about MiniCPM-o Assamese quality is made on this page.

## §4 — Assamese tokenization expansion (real, CPU, today)

A MiniCPM-o *text* baseline needs the checkpoint (PENDING). But the prerequisite
diagnosis — **can the backbone tokenizer even represent Assamese efficiently?** —
is measurable now. `scripts/tokenization_analysis.py` loaded the Qwen2.5 BPE
tokenizer (MiniCPM-o 2.6's documented backbone tokenizer family; tokenizer files
only, no weights) and measured tokens/char on a real parallel Assamese/English/
Hindi sentence set, with NLLB-200 (a real Assamese tokenizer) as a reference.

| tokenizer | Assamese tok/char | English tok/char | Hindi tok/char | as/en ratio |
|---|---|---|---|---|
| **Qwen2.5 BPE (MiniCPM-o backbone)** | **1.203** | 0.232 | 0.976 | **5.19×** |
| NLLB-200 (reference, covers asm_Beng) | 0.442 | 0.310 | 0.371 | 1.43× |

Example: "আজি বুধবাৰ আৰু বতৰ বৰষুণৰ" (25 Assamese chars) → **33 Qwen2.5 tokens**
(1.32 tok/char), vs 8 tokens for the 43-char English gloss (0.19 tok/char).

**Method note (the two ratios agree to differ):** the `as/en ratio` column is the
**ratio of the mean tokens-per-char** (1.203 ÷ 0.232 = 5.185 ≈ **5.19×**), which is
what the table columns show. Averaging the *per-sentence* ratios separately gives
5.46× — a legitimately different aggregation (mean of ratios ≠ ratio of means).
This report uses the ratio-of-means as the headline so the ratio matches the cells
above it; the per-sentence average (5.46×) is recorded in `tokenization.json` and is
not used in any headline claim.

**Finding (honest):** Assamese tokenization on the MiniCPM-o backbone is **not
acceptable** — near byte-level (~1 token/char, 5.19× English expansion), meaning the
Qwen2.5 BPE has **no learned Assamese subword merges**. A real Assamese tokenizer
(NLLB) achieves 0.44 tok/char (~2.7× better). Implications for the A/B/C plan:
(a) Assamese training will be token-expensive (longer sequences, more compute);
(b) fidelity risk — single-character tokens carry less semantic compression;
(c) this is a **tokenizer-level diagnosis**, independent of weights — it predicts
the baseline will be weak on Assamese even before any fine-tuning, and it is the
first thing to verify against the *actual* MiniCPM-o 4.5 `tokenizer.json` on the GPU
box (it adds audio/vision special tokens, but the Assamese text BPE is governed by
the Qwen2.5 vocab). Full per-sentence data in `tokenization.json`.