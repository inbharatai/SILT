# Specialist span pilot v1 — controlled Switch → dense T5 mechanics

**Status: DATA_PREPARED; no model, tokenizer, reconstruction, recovery, or inference run performed.** This is a very small locally authored synthetic span-copying pilot for a real native Switch source, a weight-derived dense T5 reconstruction, and existing recovery. It is **not a coding claim, independent research benchmark, full A-to-Z coding demo, quality certificate, or promotion study**. A real checkpoint's ability on these fixtures remains unvalidated.

## Frozen data and allowed use

Paths are relative to the repository root, `/agent/workspace/silt-specialist-core`.

| File under `data/specialist-span-v1/` | Rows | Role |
|---|---:|---|
| `train.json` | 16 | Supervised recovery optimizer rows only |
| `calibration.json` | 16 | Reconstruction activation/router calibration only |
| `dev.json` | 4 | Recovery's read-only pre/post validation and development generation |
| `final.json` | 4 | Quarantined one-shot later mechanism holdout |

Each file has `schema`, `purpose`, provenance and token-policy metadata, plus `samples: [{id, prompt, response}]`. Extra top-level metadata is accepted by the existing reconstruction/recovery data readers; no parser or implementation change was made. Labels always have exact text shape `<extra_id_0> x <extra_id_1>`, with no manual EOS. The reference adjective occurs elsewhere in the input; these are controlled context-redundant reconstruction fixtures, not open-domain knowledge questions. For example, an actual **training** row is:

```json
{"id":"span-v1-train-01","prompt":"The acorn is damp. This acorn is <extra_id_0>.","response":"<extra_id_0> damp <extra_id_1>"}
```

The four shared target words are `damp`, `clean`, `dusty`, and `worn`. Their counts are balanced (four of each in train/calibration; one of each in dev/final). Target words may tokenize to multiple pieces: no single-token assumption is made. Four grammatical templates intentionally recur; this tests a narrow mechanism and cannot establish task-family generalization.

There are **40 unique IDs, 40 unique prompts, and 40 unique entity nouns**. Each row repeats its own entity noun twice; no noun is reused by another row or split. Shared function words and target adjectives are intentional, not noun-group leakage. Membership/order are explicitly fixed, not randomly re-split. Seed/version label: **271828**; the same seed is proposed for later reconstruction/recovery. Noun assignments for the final split are not reproduced here or in the public metadata.

`static-audit.json` records exact prompt and answer-value exclusion against these historical local artifacts, with their file hashes:

- `/agent/workspace/silt-validation/switch-diagnostic.json` (the old four prompts);
- `/agent/workspace/silt-validation/switch-calibration.json` (old six);
- `/agent/workspace/silt-pass2-evidence/switch-span-debug-v2.json` (the reused four diagnostics);
- `/agent/workspace/silt-pass2-evidence/switch-calibration-v2.json` (pass-2 calibration 32).

All checked overlaps are zero. Those old artifacts are **negative exclusion inputs only**, not training/calibration inputs or model-output-derived examples. This is an exact check over named artifacts, not an exhaustive pretraining contamination claim. No old output was used to choose replacement labels. See `ATTRIBUTION.md` for locally authored synthetic origin and explicit **CC0-1.0** fixture permission. These are generated fixtures, not human-reviewed annotations; no legal guarantee is claimed.

## Final quarantine and freeze discipline

The authoring worker alone constructed and statically checked final contents. The parent must **not open, preview, tokenize, sample, infer from, copy into a prompt, or recursively parse `final.json` now**. It receives the count, byte count, whole-file SHA-256, and quarantine policy only. Do not run blanket `sha256sum -c SHA256SUMS` before authorization: that would read final bytes. The stored authoring hash is the current quarantine commitment; re-hash final only at its later authorized release.

Freeze source/store identity, software version, reconstruction method, recovery settings, export representation, all dev decisions, raw-token metric definitions, and selected artifacts before a later final release. Final must never enter reconstruction calibration, training, recovery validation, selection, or model-output-driven revision. If a defect is found, retire this fixture version transparently rather than altering frozen answers. No access-control or cryptographic secrecy guarantee is implied by this procedural quarantine. A four-case final result is descriptive only, not a statistical quality conclusion.

## Fixed admission and no compression

- Full native prompt bound: **256 tokens**. Full native response bound: **256 tokens**. Both include native special tokens/EOS; these are separate seq2seq bounds, not a combined sequence length.
- `max_samples=16` for reconstruction and recovery: all 16 calibration rows, all 16 training rows, all 4 dev rows. Existing readers validate entire supplied files before subsampling or weights.
- **Zero input compression**, no cropping, truncation, elision, prompt shortening, sentinel removal, or reference rewriting. No compress/prune-context command participates in this pilot. Model parameter reduction is separate from input compression.
- Static audit found at most **59 prompt UTF-8 bytes** and **31 response UTF-8 bytes**. These short ASCII fixtures are conservatively designed for the 256 cap, but bytes are **not measured native tokens**. No tokenizer or weight store was opened. Actual complete teacher/student encodings and token-boundary parity remain required admission checks in existing reconstruction/recovery. Reject an incompatible/overlength row rather than changing data or silently shortening it.
- Use native `native_seq2seq_v1`: `add_special_tokens=True`, `truncation=False`, independent full prompt/response encodings, native EOS and native shifted decoder inputs. Do not append a causal newline or chat template. Teacher and student token-to-ID vocabularies and special IDs/complete encodings must agree.

## Proposed bounded mechanics protocol — NOT EXECUTED

Read-only source inspection confirmed public flags in `src/asea/specialist/__main__.py`; no Python source/helper was written and no CLI/model run was executed. Commands below are **later operator commands**, not receipts or assertions of success. Use the existing local environment, a permitted real Switch store, CPU admission, and a new run directory with an existing parent. Output model/receipt paths must not exist and must be disjoint from source/data. Do not use random-weight fixtures as evidence of real Switch behavior. Do not acquire or access weights during this data-preparation task.

For a later approved run, set placeholders to explicit paths:

```sh
export PYTHONPATH=/agent/workspace/silt-specialist-core/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=/agent/workspace/silt-venv/bin/python
DATA=/agent/workspace/silt-specialist-core/data/specialist-span-v1
TEACHER=/local/approved-real-switch-store
RUN=/local/already-created-fresh-span-run
```

1. **Source capability diagnostic first**, on dev only. Preserve raw token traces whether it passes or fails. Do not revise fixtures to match the teacher. Teacher failure does not prevent labeling a later structural run as a mechanics attempt, but it blocks claims of retained demonstrated teacher capability.

```sh
# Illustrative first dev row only, not the whole evaluation.
PROMPT=$(node -e 'process.stdout.write(require(process.argv[1]).samples[0].prompt)' "$DATA/dev.json")
"$PY" -m asea.specialist infer --model "$TEACHER" \
  --prompt "$PROMPT" --dtype bfloat16 --max-new-tokens 32 \
  > "$RUN/source-dev-01.json"
```

Later collect equivalent immutable receipts for **all four** dev prompts for each state. Native generation is greedy; freeze `max_new_tokens=32` separately from the 256 admission caps. EOS absence or cap exhaustion is evidence of an incomplete generation, not permission to trim it. The `infer` command does not expose a `--max-length` flag: its native context admission is separate. Require the pilot's 256 complete-input cap in the approved token audit before inference; do not invent a CLI flag.

2. **Real Switch-to-dense T5 reconstruction**, calibrated only on the new 16 rows:

```sh
"$PY" -m asea.specialist reconstruct \
  --source-dir "$TEACHER" --output-dir "$RUN/dense-initial" \
  --calibration-path "$DATA/calibration.json" \
  --family switch_transformers --method activation --retention 0.75 \
  --dtype bfloat16 --max-length 256 --max-samples 16 --seed 271828 \
  --report "$RUN/reconstruction.receipt.json"
"$PY" -m asea.specialist infer --model "$RUN/dense-initial" \
  --prompt "$PROMPT" --dtype bfloat16 --max-new-tokens 32 \
  > "$RUN/reconstructed-dev-01.json"
```

This branch selects **one expert per sparse layer**, retaining source `d_ff`, and writes native dense T5 weights. For Switch, `retention=0.75` is recorded but **does not control expert count**; actual expert retention is `1/num_experts`, not 75%. Record actual parameter/storage changes and selected source expert provenance. Routing and router probability scaling are removed; do not claim logit parity. A successful reconstruction is `RECONSTRUCTED_UNVALIDATED`, not a recovered-capability result.

3. **Existing supervised-plus-KD recovery**, fixed small budget, train16/dev4 only:

```sh
"$PY" -m asea.specialist recover \
  --teacher-dir "$TEACHER" --student-dir "$RUN/dense-initial" \
  --output-dir "$RUN/dense-recovered" \
  --training-path "$DATA/train.json" --validation-path "$DATA/dev.json" \
  --steps 32 --rank 8 --learning-rate 0.0001 --kd-weight 0.7 \
  --method lora_kd --teacher-mode cached --export-mode native_merged \
  --dtype bfloat16 --max-length 256 --max-samples 16 --seed 271828 \
  --report "$RUN/recovery.receipt.json"
"$PY" -m asea.specialist infer --model "$RUN/dense-recovered" \
  --prompt "$PROMPT" --dtype bfloat16 --max-new-tokens 32 \
  > "$RUN/recovered-dev-01.json"
```

Thirty-two optimizer steps are two cycles through the 16 selected training rows, not evidence of convergence. Existing recovery uses response CE plus full-vocabulary teacher→student KL; dev is read-only. Source teacher remains frozen; only the student's LoRA parameters train. Supervised-only is a separately preregistered control (`--method supervised_lora --kd-weight 0`), not the command above. Such a control still needs teacher metadata validation but does not provide teacher KL. Do not add post-hoc control runs or retune against final.

`native_merged` is an explicit **proposed** deployment choice, not a guarantee it passes BF16 merge/reload checks. A merge parity/resource rejection remains a rejection. Do not silently fall back to `factor_preserving`, loosen tolerances, increase bounds, or report a missing artifact as completed. Any deliberately predeclared factor-preserving branch uses a new output path and bundle-root inference, and must be reported as a different representation. Commands expose optional `--memory-budget-mib`; an operator ceiling never overrides existing observed-headroom guards. Honor cancellation/time budgets in a supervising process; a larger model is not permission to bypass resource admission.

These commands intentionally do not invoke `build`, `evaluate`, `validate`, or `finalize`. Existing specialist `evaluate`/`validate` execute a **function-IO/code oracle**, not a span oracle; the full study workflow is not a drop-in span benchmark. No final-read command is supplied at this stage. Any later final teacher-forced evaluation requires separately reviewed read-only support; do not pass final as recovery validation merely to obtain losses.

## Intended SpanMetrics: token boundaries AND teacher-forced loss

**No SpanMetrics were computed.** A later metric report must link each row/model-state to the same frozen data hash, native encoding audit, raw generation trace, and existing recovery loss receipt. Decoded word-string equality, code pass rate, or pretty output is not a substitute.

| Intended measure | Required evidence and definition | Existing support / limitation |
|---|---|---|
| Native boundary validity | Preserve complete encoder IDs, native decoder sequence including its initial decoder-start ID, continuation IDs, ordered `<extra_id_0>`/`<extra_id_1>` IDs, and EOS position. Require exactly one ordered sentinel pair, a nonempty interior span, and terminal EOS; flag extra content/markers, missing markers, cap exhaustion, or incomplete stop separately. | `infer` returns `result.generation.input_token_ids`, `native_sequence_token_ids`, `generated_token_ids`, `eos_positions`, `cap_reached`, `truncated`, `stop_reason`, and generation config. Confirm seq2seq continuation removes exactly the decoder-start ID. |
| Strict full-response token exact match | Compare all continuation IDs through EOS against **native tokenization of the complete frozen response**, including sentinel IDs and the one terminal EOS. Preserve the original complete trace; extra/missing IDs fail, not trim-to-match. | Raw generated IDs exist. Expected IDs must be produced under the admitted local tokenizer later, without relabeling or assuming fixed numeric sentinel IDs. |
| Interior span token exact match | Compare IDs strictly between verified sentinel boundaries to the corresponding IDs extracted from the **complete encoded reference response**, not separately tokenized bare words. Report alongside boundary validity and strict full-response match. | Intended external token-aware accounting, not an implemented code-oracle metric. Invalid boundaries fail the case and remain in the denominator. |
| Completion/failure accounting | Use all four dev rows in each state's denominator; operational failures, invalid boundaries, and truncations are shown explicitly, never silently excluded. | Preserve raw receipts and stop reasons; a receipt marked `completed` means execution completed, not necessarily valid span/EOS. |
| Teacher-forced supervised loss and recovery change | Report `validation_pre.supervised_ce`, `validation_post.supervised_ce`, their difference, `response_tokens`, `forward_kl`, and `objective` from the existing recovery receipt. CE is response-token-weighted and includes full response markers/EOS, **not span-only CE**. | Existing `recovery.py` calculates these quantities. `validation_post` is trained-adapter **premerge**, not reloaded deployment CE; deployment generation remains separate. |

The CLI `infer` currently decodes with `skip_special_tokens=True`; its `text` can look right while sentinel boundaries are wrong. Never grade that text alone, normalize variants, choose alternate numeric spellings, or strip extra sentinels/EOS into a pass. Pin actual tokenizer/special-token IDs from the approved encoding contract, not assumptions based on a different T5 store.

Recovery's CE is the **student's** teacher-forced loss against frozen ground-truth responses; teacher-forced does not mean teacher-model CE. `forward_kl` compares teacher and student distributions, not teacher accuracy. The current receipt does **not** supply a standalone source-teacher CE baseline, per-case dev CE, final CE, or a reloaded deployment CE. Report them as **not measured / unsupported by these direct commands**, not inferred from generation, training history, or KL. A later stage that requires them needs separately approved read-only evaluation; this task adds no implementation. Mechanism evidence should pair raw-token generation measures with the available genuine recovery loss evidence, while keeping these state/coverage differences explicit.

## Evidence and handoff

`manifest.json`, `static-audit.json`, and `SHA256SUMS` provide metadata commitments without final examples. Static validation checks counts, IDs, prompts, noun disjointness, exact textual sentinel shape, literal in-context reference presence, balanced target vocabulary, and named-history exclusions. **It does not validate native tokenization, teacher skill, model compression, recovery, inference, code quality, or deployment.**

All changes for this task are new files under `data/specialist-span-v1/` and this document only. No existing implementation/locked file, model store, prior frozen fixture, Git state, or prior receipt was modified. No dependency installation, model-weight access, model run, or Python-code addition was performed.
