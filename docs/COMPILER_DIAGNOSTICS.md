# Compiler pass-2 research diagnostics

**Research only. No admission, preserved-quality, coding, novelty, training, or upstream-conversion-equivalence claim.** Implemented before real-model execution. Engineering validation used only tiny random native Switch checkpoints below 1 MB. Existing keep4 evidence, source weights, and original references are not rewritten.

Public entry point: `silt-compile` (equivalently `python -m asea.compiler`).

## Commands

```text
silt-compile diagnose --model PATH --suite FILE --output NEW_JSON \
  --dtype float16|bfloat16 [--preserve-router-fp32] [--reference-model PATH] \
  [--max-length 64] [--max-new-tokens 16] [--teacher-tokens 16] \
  [--memory-budget-mib 3584] [--atol 0.001] [--rtol 0.001]

silt-compile roundtrip --model PATH --output NEW_DIRECTORY \
  --dtype float16|bfloat16 [--suite FILE] [--preserve-router-fp32] \
  [--max-length 64] [--max-new-tokens 16] [--teacher-tokens 16] \
  [--memory-budget-mib 3584] [--atol 0.001] [--rtol 0.001]
```

Both commands emit JSON-only stdout, dependency/progress logging on stderr, `status=AUDIT_ONLY`, `admission=UNADMITTED`, `quality_certification=false`. `pruned=false` describes what the audit command does, not whether an input checkpoint was previously pruned. **Exit 0 means the audit completed, NOT that parity, quality, or certification passed.** Inspect every comparison and finite/termination trace.

`diagnose` runs independently loaded native-HF and SILT-wrapper controls in **two sequential fresh Python interpreters**, then compares their greedy token IDs and short teacher-forced logits. With `--reference-model`, a third fresh worker loads that source and compares it to the specified model at the **same dtype and router-precision policy**. No source and candidate are simultaneously resident. The parent coordinator never loads model weights.

`roundtrip` runs three fresh workers:

1. Native HF baseline.
2. SILT source before modification; identity mapping `0..N-1` through the existing expert/router replacement mechanics; another in-memory probe; safetensors save.
3. Fresh reload from the saved checkpoint and the same probes.

It reports native/wrapper, source/identity, saved-in-memory/fresh-reload, and source/fresh-reload comparisons. Parameter count and expert count must remain identical. The new directory contains native standalone weights/tokenizer, `roundtrip_manifest.json`, and `roundtrip_diagnostics.json`. **No `compiler_manifest.json` is created. This is not a prune artifact.** The production `prune` command still requires `1 <= keep_experts < N` and physical reduction. Roundtrip does not change those checks.

Omitting `--suite` for roundtrip uses one explicitly labelled mechanics-only span probe; it is not quality evidence. Supply a fixed research suite for meaningful local comparisons.

## New diagnostic suite contract

```json
{
  "schema": "switch-span-diagnostics-v2",
  "split": "diagnostic",
  "description": "Exploratory source/parity controls; not certification heldout evidence",
  "cases": [
    {
      "id": "capital-debug",
      "group": "target",
      "provenance": "old-debug-prompt; preserve original evidence hash here",
      "prompt": "The capital of France is <extra_id_0>.",
      "target": "<extra_id_0> Paris <extra_id_1>",
      "expected_spans": ["Paris"],
      "references": ["<extra_id_0> Paris <extra_id_1>"]
    },
    {
      "id": "two-span-control",
      "group": "control",
      "prompt": "The <extra_id_0> is <extra_id_1>.",
      "target": "<extra_id_0> sky <extra_id_1> blue <extra_id_2>",
      "expected_spans": ["sky", "blue"],
      "references": ["<extra_id_0> sky <extra_id_1> blue <extra_id_2>"]
    }
  ]
}
```

- 1–32 cases; unique string IDs; nonempty prompt and target, each <=4096 characters.
- `expected_spans`: 1–8 strings, sentinel order starting at 0. A terminal sentinel closes the last expected span.
- `references` retains **original raw strings**, unchanged. If omitted it defaults to `[target]`. `group` is optional, and when supplied is `target` or `control`.
- Preserve exact old prompt/reference text when reusing historical cases and record original evidence hashes/provenance in the suite. The complete supplied suite and file hash are bound into the report. These fields do not independently authenticate provenance.
- A v1 `split=heldout` suite is deliberately not accepted here. A diagnostic suite must declare `split=diagnostic`. Do not replace the old literal metric or use these new span scores in admission.
- Full teacher target including EOS must fit `--teacher-tokens` (1–32). No silent teacher truncation. EOS is appended only if the tokenizer did not already supply it. Overflow returns `DIAGNOSTIC_TARGET_TOO_LONG`.
- Input `--max-length` is 1–64. Input IDs and `input_truncated` are recorded. Greedy `--max-new-tokens` is 1–32; batch size is 1. Larger corpora/lengths need a separately reviewed harness.

## What is measured

Every case retains raw generated IDs, full special-token decode, the original `skip_special_tokens=True` decode, original references and the unchanged **literal** proxy score. Sentinel parsing records spans, missing/misordered/duplicate/extra sentinels, prefix/tail IDs and trailing text. Parsed-span correctness and full-output format correctness are separate: a correct first span does not conceal trailing repetition or punctuation.

The new metric is `switch-span-diagnostics-v2`, parser `t5-token-sentinel-spans-v2`. Normalization is only symmetric whitespace trim/collapse. It does **not** remove punctuation, search substrings, collapse repeated periods, or apply code transformations. Missing or malformed boundaries cannot produce a valid span score. EOS emission/position, effective generation EOS IDs, token count, max-token limit, longest same-token run, repeated 2/3/4-grams and termination reason are explicit. Tokenizer EOS and inherited generation settings are separately recorded.

Teacher-forced outputs include:

- Full-vocabulary logits for at most 32 positions, held temporarily as float32 `.npy` (lossless representation of FP16/BF16 logits) solely for comparison.
- Actual original logit dtype, finite/NaN/Inf counts, finite min/max/mean/L2, token argmax IDs and top1–top2 margins.
- Mean target-token NLL and per-token NLL, including EOS; separate sentinel, EOS and content slices. Cross entropy is computed directly from logits in FP32, **not** the native combined loss containing router auxiliary terms. It is a diagnostic, not quality certification.
- Paired exact generation-ID agreement, teacher next-token agreement, NLL delta, full short-logit max absolute error, RMS error, relative error with denominator floor, and prespecified `atol + rtol * abs(left)` checks. Nonfinite logits never pass numeric parity.

Intermediate logit files are removed after the comparisons; reports keep their hashes and numeric summaries, not the arrays. Cross-dtype full-array comparison is not currently a public command: run the separate FP16/BF16 arms and compare their persisted IDs/NLL/numeric summaries. Direct native controls use an independently implemented native class loader and `model.generate`; wrapper controls call SILT's native loader and `generate(..., return_token_ids=True)`. The normal existing string-returning inference API is unchanged.

Per-layer routing telemetry separates:

- `probability_sum`: analytic FP32 softmax mass across all experts.
- `analytic_fp32_top1_count`: argmax of that analytic distribution.
- `top1_count`: actual native **post-activation-dtype-cast, pre-capacity** argmax, reconstructed with the pinned native softmax dtype/cast expression.
- `dispatched_count`: actual returned native post-capacity mask; drops, drop fraction, and postcast-vs-analytic disagreement counts.
- Actual dispatched expert output counts, output L2 sums, gate-times-output-L2 sums, and their conditional means (EAN/REAP adaptation). Only native dispatched expert outputs before router multiplication are observed.

Generation and teacher-forcing have separate telemetry. Encoder versus decoder is identified by native layer names. Generation follows the serialized native cache setting, recorded in `tokenizer_and_generation.generation_config`; it is normally cached for Switch. Numeric hooks cover lm_head, router and dispatched-expert outputs; **not every hidden intermediate, saturation, or clamp**. These are not per-token paired router-switch traces or full-layer reconstruction errors.

Reports bind source inventory, suite hash, implementation/parser/normalizer hashes, graph/module/parameter-shape and dtype information, loaded unique-parameter counts, actual bytes, effective generation/tokenizer semantics and runtime versions. Tokenizer serialization file hashes can differ after save without tokenization changing: compare vocabulary/sentinel metadata, input IDs, target IDs and output IDs rather than assuming bytewise tokenizer JSON equality implies semantics.

## Explicit router-precision arm

`--preserve-router-fp32` is available on `diagnose`, `roundtrip` and `prune`. It is **not a whole-model FP32 load** and does not cast the entire model. After the normal global HF dtype load, it reads only original `F32` `.router.classifier.*` tensors directly from local safetensors using `safe_open.get_tensor`, restoring their original bits to the small corresponding parameters. Router tensor hashes and mode are recorded. Original non-F32 router tensors retain native behavior. Native router configuration must be FP32; no matching original F32 tensors yields `NO_F32_ROUTERS`. Oversized restoration tensors are bounded.

Without the flag, the baseline remains native global-half-load then native runtime FP32-router upcast; that upcast cannot restore lost original F32 precision. Compare this baseline to the explicitly labelled preserving arm, not to a silently changed default. Prune/roundtrip manifests bind the arm. Loading such an artifact with a mismatched flag fails `ROUTER_PRECISION_POLICY_MISMATCH`. Roundtrip diagnostic loading also checks its dtype and bound checkpoint inventory. The v1 generic inference/evaluation CLI has no precision-restoration flag; use `diagnose` for preserving-arm artifacts rather than silently evaluating a different graph.

New preserving/REAP prune artifacts are labelled `experiment_config.audit_only=true` and explicitly rejected by v1 lineage validation (`AUDIT_ONLY_ARTIFACT`). Existing certification floors, literal evaluator, exact bounds, size requirements, reference identity checks and keep<N constraints are not relaxed.

## Conservative pruning and scorer baselines

The existing probability-mass scorer stays the default:

```text
silt-compile prune --model ORIGINAL_SOURCE --output NEW_KEEP7 \
  --keep-experts 7 --calibration CALIBRATION.json --dtype bfloat16 \
  --max-length 64 --scorer probability_mass [--preserve-router-fp32]
```

Keep7 and keep6 must be generated **independently from the unchanged source**, not sequentially from a previously pruned model. Do not jump to keep4 if source/parity/dtype or conservative controls collapse. Do not relabel the existing keep4 evidence as evidence for a new setup.

Optional scorer:

```text
silt-compile prune --model ORIGINAL_SOURCE --output NEW_KEEP7_REAP \
  --keep-experts 7 --calibration CALIBRATION.json --dtype bfloat16 \
  --max-length 64 --scorer reap_dispatched \
  --minimum-expert-observations 4 --preserve-router-fp32
```

Calibration format is unchanged: `{"samples":[{"prompt":"...", "target":"<extra_id_0> ... <extra_id_1>"}]}` (1–256 examples). Use short, diverse teacher targets; separately fix diagnostic target/control probes. There is no training.

`reap_dispatched` is explicitly **REAP-Switch-dispatched-adaptation**:

```text
score_j = sum(actual_selected_gate * ||actual_expert_output_before_gate||_2)
          / actual_post_capacity_dispatched_output_count_j
```

This is a conditional mean, not a global sum, squared norm, all-expert mass, or double gate multiplication. Dropped tokens never become expert observations. With drops, it is capacity-conditioned, not an unqualified reproduction of REAP. Zero observations mean **undefined/unknown**, represented by JSON null, never zero importance. The scorer blocks the entire selection, before mutation/output creation, if any expert in any layer has fewer than the requested observations; `INSUFFICIENT_EXPERT_COVERAGE` names the layer and indices. Default minimum is 1, merely a mechanical floor, not statistically adequate coverage; explicitly choose a higher pilot threshold. The legacy probability-mass scorer intentionally retains its old selection policy and does not gain a REAP coverage guarantee.

Selection supports `probability_mass` and `reap_dispatched`; actual usage and EAN are recorded comparison baselines, **not additional CLI selection methods**. Prune manifests distinguish analytic fixed-input survivor reweighting from actual postcast counts. Mean survivor mass and removed-original-winner fraction are recorded. Mean gate amplification is deliberately null: it cannot be recovered by taking the reciprocal of mean retained mass. No native capacity policy, gate denominator, routing algorithm, penalty, or repair behavior is changed.

## 4 GB CPU execution protocol (parent/operator only)

The known original has 619,339,008 unique parameters: nominal half weights are 1,238,678,016 bytes (~1.239 GB); keep8 is a full checkpoint, not pruning. Full FP32 source loading remains outside the existing guard. BF16 now uses the same 2-byte preflight accounting and normalized-size accounting as FP16. The default ceiling stays **3584 MiB**, additionally bounded by available/cgroup headroom; a larger user budget cannot bypass it. Estimate remains two target-weight copies plus 512 MiB. A tiny dtype matmul smoke test runs before a diagnostic model load. BF16 is a range control, not a CPU speed or correctness guarantee.

Check free disk/cgroup headroom before authorizing work. Roundtrip needs one additional ~1.24 GB checkpoint plus bounded temporary logits and a 256 MiB disk margin. There is a disk preflight. No model downloads, pickle weights, remote code, shell-executed worker payloads, concurrent source/candidate loads, or writes under the original model directory.

Each successful worker records observed `process_peak_rss_bytes`, PID, fresh-process flag, runtime versions, thread count, wall time, generation time, per-case generated length and seconds/generated token. RSS is **the whole fresh worker's lifetime peak**, not isolated tensor memory. Roundtrip's save worker includes two probe passes and serialization, so do not compare that peak as pure inference savings. The public `prune` result also records observed process-lifetime RSS/PID and requires a fresh CLI invocation for independent peaks. An external profiler is recommended to capture failed/OOM commands too; interrupted workers cannot reliably self-report their peak.

Example operator commands (not executed as part of implementation):

```sh
export PYTHONPATH=/agent/workspace/silt-pass2/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
PY=/agent/workspace/silt-venv/bin/python
SOURCE=/agent/workspace/silt-models/switch-base-8
EVIDENCE=/agent/workspace/silt-pass2-evidence
# SUITE must be a new, fixed schema=v2 diagnostic file outside source trees.
SUITE="$EVIDENCE/span-suite-v2.json"

# One independent command at a time; unique paths, no overwrites.
/usr/bin/time -v -o "$EVIDENCE/source-f16.time.txt" \
  "$PY" -m asea.compiler diagnose --model "$SOURCE" --suite "$SUITE" \
  --output "$EVIDENCE/source-f16.json" --dtype float16 \
  > "$EVIDENCE/source-f16.stdout.json" 2> "$EVIDENCE/source-f16.stderr.log"

/usr/bin/time -v -o "$EVIDENCE/source-bf16-preserved.time.txt" \
  "$PY" -m asea.compiler diagnose --model "$SOURCE" --suite "$SUITE" \
  --output "$EVIDENCE/source-bf16-preserved.json" --dtype bfloat16 --preserve-router-fp32 \
  > "$EVIDENCE/source-bf16-preserved.stdout.json" 2> "$EVIDENCE/source-bf16-preserved.stderr.log"

# Authorize only after source controls; requires free disk for another checkpoint.
/usr/bin/time -v -o "$EVIDENCE/keep8-bf16-preserved.time.txt" \
  "$PY" -m asea.compiler roundtrip --model "$SOURCE" --suite "$SUITE" \
  --output /agent/workspace/silt-models/switch-identity8-bf16-pass2 \
  --dtype bfloat16 --preserve-router-fp32 \
  > "$EVIDENCE/keep8-bf16-preserved.stdout.json" 2> "$EVIDENCE/keep8-bf16-preserved.stderr.log"
```

Also run source BF16 without preservation and FP16 with preservation, using distinct paths, to separate range and router-rounding hypotheses. Repeat an unchanged command into another new output for determinism if needed. Only after controls, run independently profiled keep7 then keep6 from source and `diagnose --reference-model "$SOURCE"` on each, with matching dtype/flag. A diagnostic paired command has independently profiled workers; never attribute the public command's whole-process high-water mark to only one model.

**Stop/escalate to investigation** on native/wrapper or identity/reload token mismatch, nonfinite logits, marked repetition/max-length degeneration, large source-relative NLL/generation changes, inadequate REAP coverage, memory/disk guard failure, or weak absolute source controls. No automatic quality repair, admission, activation, or coding claim follows any diagnostic score.

## Engineering validation log

Read `research-pruning-pass2.md` and installed Transformers 4.51.3 native router/sparse MLP implementation before changes. Native capacity is sequence-order cumsum; actual gate argmax occurs after the activation-dtype probability cast.

```sh
/agent/workspace/silt-venv/bin/python -m pytest -q \
  /agent/workspace/silt-pass2/tests/test_compiler_diagnostics.py \
  /agent/workspace/silt-pass2/tests/test_compiler_v1.py \
  /agent/workspace/silt-pass2/tests/test_compiler_admission.py
```

Final implementation/regression run: **77 passed in 38.87 seconds**. Tests include both half dtypes, real native tiny-checkpoint CLI processes, identity mapping/reload parity, original F32 restoration (not rounded upcast), post-capacity REAP conditional math, parser/termination/repetition behavior, path/output guards, memory ceilings, and existing certification regression tests. The near-tie BF16 postcast-vs-analytic test is also included. All random model tests are **mechanics only**; no real source/candidate inference, download, or pruning experiment was performed by the implementation agent.
