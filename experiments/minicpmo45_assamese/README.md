# MiniCPM-o 4.5 + SILT → Accurate Assamese Omni Voice Agent

**Status:** SILT half executed for real on this machine (CPU). MiniCPM-o weight-training half
**blocked by hardware** and handed off to a GPU box with exact commands. **No MiniCPM-o metric on
this page is real** — every such field is `PENDING` or absent until the checkpoint actually runs.

## The question this experiment answers

> Does SILT reliably transfer Assamese speech capability into MiniCPM-o 4.5 while preserving its
> original capabilities — better than conventional fine-tuning at equal compute/data budget?

To answer it we compare three arms on the **same Assamese data and the same compute budget**:

| arm | what it does | who trains weights |
|---|---|---|
| **A — baseline** | untouched MiniCPM-o 4.5, measured only | nobody |
| **B — conventional** | raw accepted Assamese pairs, fine-tuned with no SILT selection/gating | external trainer (LLaMA-Factory / ms-swift / PEFT) |
| **C — SILT-guided** | the same source volume after SILT relevance + safety + trust-gate + held-out-regression filtering, exported as a gated dataset + `NOT_EXECUTED` job spec | external trainer (same, same budget) |

We report the truth. If B beats C, the report says so. If A is already good enough that the gap is
unmeasurable, the report says so. **No fabricated metrics, ever.**

## The one fact that shapes everything here

**SILT trains no weights.** (`src/asea/distill/export.py:1-15`; grep finds no optimizer/backward
anywhere in `src/asea`). SILT is a **selection / trust-gating / capability-gap-measurement /
dataset-export** layer. It produces a `NOT_EXECUTED` L4 (LoRA) / L5 (sequence-KD) job spec + a
validated JSONL dataset that a **human hands to an external trainer**. So:

- "SILT-guided training of MiniCPM-o" = SILT selects + gates + exports the dataset/spec;
  LLaMA-Factory / ms-swift / PEFT do the weight update; the trained adapter re-enters SILT as a
  **new receiver module** and is re-benchmarked before use.
- This splits the experiment into a **CPU-feasible SILT half** (selection, gating, gap
  measurement, dataset/job-spec export, audited accept/reject, held-out A/B) and a
  **GPU-required trainer half** (the actual MiniCPM-o fine-tune + baseline inference).

## What ran for real on THIS machine (CPU, today)

1. **Capability map** for MiniCPM-o 4.5 (`capability_map.md`) — every omni component → a SILT
   capability (TEXT/STT/TTS/STS/VISION/TOOL/AGENT × Assamese), with teacher / method / data /
   regression-risk / validation per row.
2. **A real, audited SILT Assamese capability transfer** — not MiniCPM-o, an honest proxy on
   available real connectors: `nllb-teacher` → `qwen2.5:1.5b-instruct` on `assamese_english`
   (`POST /api/transfers`), plus a replay/extension of the existing **PROMOTED** Assamese G2P run
   (`e0ebb0553635`, candidate 0.7359) via the new `POST /api/skills/test` read-only A/B. Real
   funnel counts, gate verdicts, and per-case Assamese numbers are recorded in `metrics.json`,
   `trust_gate_results.jsonl`, `error_analysis.md`. **This validates SILT-on-Assamese, not
   MiniCPM-o-on-Assamese — stated plainly.**
3. **The A/B/C handoff artefacts** for MiniCPM-o: budget-matched raw (B) and SILT-gated (C)
   datasets + `NOT_EXECUTED` job specs, plus the exact trainer commands in `training_plan.md`.
4. **Reproducible scripts** under `scripts/` — each runs here (SILT half) or prints the exact GPU
   command and exits `BLOCKED` rather than faking output.

## What did NOT run here (and exactly why)

See `hardware.json` → `blockers`. In short: torch is CPU-only, VRAM is 8 GB (below the LoRA floor
for an 8B backbone and borderline for 4-bit omni inference), the training+audio stack is absent,
and the MiniCPM-o checkpoint is not present. So **A, B, C for MiniCPM-o are all PENDING a GPU
box.** `training_runs.jsonl` is empty — no run is logged until one executes.

## Files

| file | real / pending |
|---|---|
| `hardware.json` | real |
| `capability_map.md` | real (map); per-row baselines PENDING the checkpoint |
| `baseline.md` | MiniCPM-o baseline PENDING; SILT-proxy baseline real |
| `dataset_manifest.jsonl` / `dataset_provenance.jsonl` | real catalogued sources + schema |
| `training_plan.md` | real plan + exact GPU commands (not executed) |
| `training_runs.jsonl` | empty (no fake runs) |
| `metrics.json` / `regression_metrics.json` | SILT-proxy numbers real; MiniCPM-o fields PENDING |
| `trust_gate_results.jsonl` | real (from the live SILT run) |
| `error_analysis.md` | real (replayed G2P per-case diff + new run) |
| `human_eval_template.csv` | real template, unfilled (no TTS claim until native-speaker fill) |
| `final_report.md` | honest status |
| `scripts/*.py` | real (SILT half runs; trainer half prints BLOCKED + commands) |

## Discipline (the user's hard constraints, enforced)

No MiniCPM-V/o confusion · no GGUF/Ollama training (ollama is only the SILT-proxy receiver,
labelled) · no STT claim from a text test · no TTS claim from a text LoRA · no synthetic-only
training · no test data in training · no self-eval-only · no unlicensed audio · no voice cloning ·
no overwriting the base checkpoint · no merging unverified adapters · no big compute before a
pilot passes · no "loss went down therefore success" · no native-TTS claim without the human
pack · no fabricated metrics · no silent SILT architecture changes. Every result is either
real-on-this-machine or explicitly PENDING-on-GPU.