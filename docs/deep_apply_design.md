# DEEP-APPLY — design (Phase 1)

DEEP-APPLY is an **optional** stage where SILT itself trains a LoRA adapter on
the receiver, using **only packets that already passed the promotion gate
(Gate 1)**, and then gates the trained adapter **again (Gate 2)** with the same
all-or-nothing discipline before it is admitted, with its own rollback and audit
trail. After this, a SILT user never needs an external fine-tuning tool: packet
mode for instant inference-time skills, weights mode for baked-in skills — both
gated, both reversible, both audited.

The load-bearing principle is unchanged: **SILT separates HOW a capability was
produced from WHETHER it is allowed to exist in the receiver.** Gate 2 never
trusts the trainer — it measures the trained artifact's outcome. Streaming and
standard backends are judged identically.

This is a trust product, not a performance product. The trainer is small and
inspectable; the gate is what makes it shippable.

---

## 1. Attachment point

Deep-apply runs **after promotion**, consuming only `PROMOTED` packets from the
approved store (`store.list("approved")`). It is a **separate, explicit
invocation** — never an automatic side effect of a transfer run. The transfer
pipeline (`Pipeline.run`) is unchanged; deep-apply is an additional entry
point that reuses the pipeline's store, harness, evaluator machinery, gate
machinery and audit log.

L4/L5 remain export-only (`distill/export.py`) **unless** the user explicitly
invokes deep-apply — and then only `PROMOTED` content is eligible. The
`applicable_learning_level` check is relaxed *only inside the deep-apply gate*
(see §6) and only for `L4_PEFT_CANDIDATE`; the existing `PromotionGate` is not
touched.

## 2. Training data (provenance flows through)

Built from the existing export path so provenance is identical to the L4/L5
dataset a human would take to an external trainer:

* Reuse `distill.export._rows_from_packet` to flatten each approved packet into
  supervised `{input, output, packet_id, capability, domain, origin_kind,
  synthetic_depth, is_mock}` rows.
* Gate at intake: refuse anything not `PROMOTED` (named error
  `DeepApplyIntakeError`); refuse any packet whose `provenance.is_mock` under
  `strict_no_mock` (a mock cannot launder itself into training data, same guard
  as `export_dataset`).
* Every training row carries its `packet_id`. The `AdapterPacket` records the
  full list of source `packet_id`s, so an adapter is traceable to the packets
  that produced it.
* The dataset is hashed (`dataset_sha256`) and the hash is recorded in
  `AdapterPacket` and in the `train_completed` audit event.

## 3. Trainer

LoRA/PEFT via `peft` + `transformers`, behind a new optional dependency group
`pip install -e ".[deep]"` (adds `torch`, `transformers`, `peft`, `accelerate`,
`sentencepiece`). **Core install must work without it.** Invoking deep-apply
without the extra raises the named error `DeepApplyBlocked` naming the missing
extra and the install command — never a silent mock, never a fabricated result.

`is_mock` stays `False` everywhere real: the real trainer loads real weights and
runs real optimization. Nothing mock enters any real path (test doubles live
in `tests/`, are clearly named, and are never imported by `deep_apply`).

Determinism: fixed seeds (Python, NumPy, torch) are recorded in
`AdapterPacket.training.seed`. Decoding for evaluation is deterministic
(`do_sample=False`), matching `HFCausalConnector`.

## 4. Low-VRAM backend (layer streaming)

A pluggable `TrainerBackend` interface with two real implementations:

* **Standard** — model resident on device; the default. For ample VRAM.
* **Streamed** — low-VRAM training in the style of **Soup**
  (github.com/MakazhanAlpamys/Soup, Apache-2.0): the frozen base is kept in host
  RAM and decoder layers are streamed to the GPU one at a time via forward hooks
  (each layer is moved to CUDA for its compute and back to CPU afterwards), with
  LoRA params resident on device. Their published measured result is an 8B LoRA
  train in ~3.3 GB VRAM.

**Prior-art honesty (binding).** Layer streaming/offloading is third-party
published work. The file header and `NOTICE` record the Apache-2.0 attribution.
The technique is **credited, never claimed as ours**. The backend used and its
version are recorded in `AdapterPacket.backend` / `AdapterPacket.backend_version`.

**Integration route chosen: (ii) a minimal in-house streaming loop**, with
justification:
* (i) depending on `soup-cli` as a runtime dependency couples a trust product to
  an external beta project's CLI surface, version drift, and failure modes we
  cannot audit line-by-line. The deep-apply gate can absorb trainer risk, but a
  hard runtime dependency that may itself be unavailable/uninstallable blocks the
  whole feature for users who could otherwise run the standard backend.
* (ii) a minimal in-house loop implementing the published technique keeps the
  dependency optional, the failure surface auditable, and the behavior
  testable. We implement only what is needed for CausalLM stacks
  (`model.layers` / `transformer.h`); unsupported architectures raise the named
  `DeepApplyBlocked` error suggesting the standard backend — never a silent
  fallback.

**Rules (binding).**
* `AdapterPacket` records which backend and version produced it.
* **Gate 2 contains zero backend-conditional branches.** Streamed and standard
  adapters are judged identically — the gate measures the outcome, never the
  trainer.
* If the streamed backend fails or the architecture is unsupported (e.g.
  non-CausalLM "omni" models), raise the named `DeepApplyBlocked` suggesting the
  standard backend or bigger hardware — **never silently fall back**.

## 5. Hardware ladder

| Condition | Behaviour |
|---|---|
| No CUDA + large model (param count above a CPU ceiling, default ~1.5B) | `DeepApplyBlocked` naming the CUDA requirement. Never mock, never fabricate a training log. |
| Small model (e.g. SmolLM2-135M/360M), CPU | Trains on CPU — slow but real. Standard backend only (streaming targets GPU VRAM, pointless on CPU). |
| Small CUDA GPU (~4–8 GB) + big model | Streamed backend eligible. |
| Ample VRAM | Standard backend. |

The CPU ceiling is a named constant (`CPU_PARAM_CEILING`), not magic, and the
`DeepApplyBlocked` message names both the model size and the missing resource.

## 6. The AdapterPacket

A new typed Pydantic record (`extra="forbid"`, mirroring `SkillPacket`'s
discipline) describing the trained artifact:

* `adapter_id`, `base_model`, `base_model_fingerprint` (config/arch hash, not
  full weights).
* `source_packet_ids: List[str]` — every packet the adapter was trained on.
* `lora_config` (rank, alpha, target_modules), `training_config_hash`,
  `dataset_hash`, `seed`.
* `synthetic_depth = max(p.provenance.synthetic_depth for p in sources)` —
  depth **propagates**: an adapter trained on depth-2 packets is depth-2
  knowledge; the ceiling applies at Gate 2 independently of Gate 1.
* `risk_domain = max_severity(p.domain for p in sources)` — the highest-risk
  domain among the source packets. If any source is medical/legal/finance, the
  adapter is `HIGH` risk.
* `backend`, `backend_version`, `trainable_param_count`, `training_loss`
  (finite; NaN/Inf ⇒ Gate 2 sanity reject), `adapter_artifact_ref` (path into
  the adapter store).
* `provenance` — the merged chain of all source packets' provenance chains.
* `learning_level = LearningLevel.L4_PEFT_CANDIDATE`.
* `scores: EvaluationScores`, `promotion_status`, `rejection_reason`,
  `human_approved_by`, `rollback_token`.

## 7. Gate 2 — check mapping

Gate 2 reuses `Check`, `GateDecision`, `PromotionPolicy` (same thresholds, same
all-or-nothing `all(c.passed)`, same `PENDING_HUMAN` logic) but operates on an
`AdapterPacket` via a `DeepApplyGate`. The human-approval logic is replicated
exactly: `needs_human = risk_tier == HIGH`, **not a policy knob**, not
disableable. For an adapter, `risk_tier` comes from `risk_domain` (max severity
over sources), so **any** high-risk source packet ⇒ `PENDING_HUMAN`.

| # | Existing check | Gate 2 treatment |
|---|---|---|
| 1 | `schema_validation` | **Adapter analogue**: `AdapterPacket` Pydantic-valid (`extra="forbid"`) and `training_loss` finite. |
| 2 | `distilled_payload_present` | **Adapter analogue**: adapter artifact present (`adapter_artifact_ref` resolves, `trainable_param_count > 0`). |
| 3 | `evaluator_threshold` | **Unchanged**: `scores.aggregate >= min_evaluator_score`. |
| 4 | `safety_threshold` | **Unchanged**: `safety_score >= min_safety_score` (safety inherited from source packets' min; high-risk still human-gated). |
| 5 | `benchmark_improvement` | **Unchanged**: held-out `candidate - baseline >= min_improvement` (adapter vs receiver-baseline). |
| 6 | `no_regression` | **Unchanged**: no control-suite regression beyond tolerance. |
| 7 | `case_regression_limit` | **Unchanged**: per-case regression ratio ≤ `max_case_regression_ratio`. |
| 8 | `provenance_present` | **Adapter analogue**: merged source chain non-empty AND `source_packet_ids` non-empty. |
| 9 | `synthetic_depth` | **Adapter analogue**: `synthetic_depth = max(sources) <= max_synthetic_depth` — depth propagates, ceiling applies at Gate 2. |
| 10 | `no_self_transfer` | **Adapter analogue** (`no_self_lineage`): the receiver `module_id` must not appear in any source packet's provenance chain — the receiver cannot be the teacher of its own training data. |
| 11 | `rollback_metadata` | **Unchanged**: `rollback_token` present (issued before admit). |
| 12 | `applicable_learning_level` | **Adapted**: deep-apply admits `L4_PEFT_CANDIDATE` (the whole point); `L5_DISTILL_DATASET` stays export-only and is refused. |
| 13 | `no_mock_provenance` | **Adapter analogue**: no source packet has `provenance.is_mock` (under `strict_no_mock`). |
| 14 | `human_approval` | **Adapted (stricter)**: `needs_human = risk_domain in HIGH_RISK_DOMAINS`. Any high-risk source ⇒ `PENDING_HUMAN`, not disableable. |

**New check (justified).** `training_loss_finite`: the recorded `training_loss`
is finite (not NaN/Inf). This is a **degenerate-artifact sanity guard**, not a
quality endorsement — it does not trust the trainer's claim that training
*worked*, only that it did not numerically blow up. A diverged run produces a
broken adapter; Gate 2 rejects it regardless of held-out numbers. This is the
same category as `schema_validation`, not a relaxation of "the gate never trusts
the trainer".

No existing check is weakened, skipped, or reinterpreted. The `PromotionGate`
code is untouched; `DeepApplyGate` is a separate class that reuses the same
`Check`/`GateDecision`/`PromotionPolicy` machinery.

## 8. Storage & rollback

A new separated `AdapterStore` mirroring `MemoryStore`'s discipline:

```
<workspace>/adapters/
  candidate_adapters/   trained adapters not yet admitted
  approved_adapters/    admitted adapters — the only dir a receiver reads from
  rejected_adapters/    refused adapters, kept for audit
  snapshots/            rollback snapshots of the approved-adapter set
```

* `approve(adapter)` refuses anything not `PROMOTED` (gate decides, store
  records) — same separation as `MemoryStore.approve`.
* Duplicate-content guard on `adapter_hash` (LoRA config + dataset hash +
  source ids), same receiver scope rule as packets.
* `RollbackLayer`-style snapshot before each admission; admission issues a
  `rollback_token`. **Adapters are removable by construction** — v1 never merges
  into base weights. Rollback detaches the adapter and restores the prior
  approved-adapter set, so the receiver returns to its **exact pre-admission
  behavior** (receiver-baseline + any previously-admitted adapters).

## 9. Audit

The **same** `AuditLog` (same hash chain). New events appended in order:
`train_started` (source packet ids, base model, backend),
`train_completed` (dataset_hash, training_loss, trainable_param_count),
`gate2_decision` (full `GateDecision.to_dict()`),
`adapter_admitted` (rollback_token), `adapter_rolled_back` (token, counts).
Tamper detection (`verify` → `broken_at`) is inherited unchanged.

## 10. API surface

* `deep_apply(receiver, packet_ids, config) -> DeepApplyReport` (pipeline-level
  orchestrator in `runner.py`).
* `AdapterStore`, `DeepApplyGate`, `TrainerBackend` (+ `StandardTrainerBackend`,
  `StreamedTrainerBackend`), `AdapterPacket`.
* CLI sketch only (not built in v1): `python -m asea.cli deep-apply
  --workspace . --receiver <id> --packets <id...> --backend standard|streamed`.
* A Studio endpoint stub is intentionally deferred to v1 (no UI).

## 11. Honest limits

* Deep-apply needs the `[deep]` extra; without it, `DeepApplyBlocked`.
* Big models need a CUDA GPU; otherwise `DeepApplyBlocked`.
* A tiny-run rejection at Gate 2 is **normal and expected** — the mechanism
  (verdict + named reasons + audit + rollback) is what is asserted, not a
  promotion.
* Streaming is beta; unsupported architectures block (named error), never fall
  back silently.
* v1 never merges LoRA into base weights; adapters are removable. Weight
  baking is future work.

---

## 12. As-built (2026-08-16)

This section records what was actually built and run, versus the design above.
It is the authoritative state-of-the-world; the design sections are the intent.

### Files

| Path | Role |
|---|---|
| `src/asea/deepapply/errors.py` | `DeepApplyError`, `DeepApplyIntakeError`, `DeepApplyBlocked`, `AdapterNotPromoted` |
| `src/asea/deepapply/adapter_packet.py` | `AdapterPacket` (Pydantic, `extra="forbid"`, terminal-state validators), `max_risk_tier` |
| `src/asea/deepapply/dataset.py` | `TrainingDataset`, `build_training_dataset` (reuses `distill.export._rows_from_packet`), intake guards |
| `src/asea/deepapply/trainer.py` | `TrainerBackend` ABC, `StandardTrainerBackend`, `StreamedTrainerBackend` (BETA), `AdapterArtifact` ABC, `_AdaptedHFModule`, `get_backend`, `CPU_PARAM_CEILING` |
| `src/asea/deepapply/store.py` | `AdapterStore` (candidate/approved/rejected/snapshots), `AdapterRollbackLayer` |
| `src/asea/deepapply/gate2.py` | `DeepApplyPolicy`, `DeepApplyGate` (14-check discipline, human-approval for HIGH risk) |
| `src/asea/deepapply/evaluator.py` | `DeepApplyEvaluator` (reuses `BenchmarkHarness`, baseline vs adapted, regression sweep) |
| `src/asea/deepapply/runner.py` | `DeepApplyConfig`, `DeepApplyReport`, `DeepApplyRunner`, `deep_apply`, `from_pipeline` |
| `src/asea/deepapply/__init__.py` | lazy exports (no eager torch import) |
| `tests/test_deep_apply.py` | 41 tests (gate, runner, blocked, real e2e) |
| `scripts/real_deep_apply_probe.py` | one-off honest-number probe (not a test) |
| `pyproject.toml` | added `[deep]` optional-dependency |
| `NOTICE` | Soup (Apache-2.0) layer-streaming attribution + HF stack |

### Gate 2 checks as built (14 + human-approval)

`schema_validation`, `training_loss_finite` (new, degenerate-artifact sanity
guard — does **not** trust the trainer's claim that training worked, only
that it did not numerically blow up), `adapter_artifact_present`,
`evaluator_threshold`, `safety_threshold`, `benchmark_improvement`,
`no_regression`, `case_regression_limit`, `provenance_present`,
`synthetic_depth` (propagated max over sources), `no_self_lineage`,
`rollback_metadata`, `applicable_learning_level` (admits L4; refuses L5),
`no_mock_provenance`. Human-approval is `needs_human = risk_tier == HIGH`,
driven by `risk_domain = max_severity(sources)` — a module constant over
`{MEDICAL, LEGAL, FINANCE}`, **not a policy knob**, not disableable.

### Runner seam for tests

`DeepApplyRunner.run(..., trainer=None)` accepts an injected `TrainerBackend`.
Real runs leave `trainer=None` and resolve via `get_backend(config.backend)`.
Tests inject a `ScriptedTrainerBackend` so the double never enters a real path.
`DeepApplyBlocked` propagates from the trainer with **no fallback** (the
streamed backend never silently falls back to standard).

### Provenance propagation (as built)

`_merged_provenance`: `chain` = ordered union of source chains; `synthetic_depth`
= max over sources; `is_mock` = OR over sources; `origin_kind` = most synthetic
via an `ORIGIN_RANK`. Depth and risk propagate to the adapter independently of
Gate 1, so Gate 2's ceiling and human-approval apply even if a source packet
passed Gate 1.

### Rollback (as built)

Admission snapshots the approved-adapter set (via `AdapterRollbackLayer`) and
issues a `rollback_token`. `rollback_adapter(adapter_id, token, actor)` sets the
adapter record to `ROLLED_BACK`, moves it to the rejected bucket, and calls
`rollback_layer.rollback(token)`. v1 never merges LoRA into base weights —
adapters are removable by construction.

### Real run (verbatim numbers)

Model `HuggingFaceTB/SmolLM2-135M`, standard backend, 3 LoRA steps, CPU. Gate 2
**rejected** with `evaluator_threshold` (0.270 < 0.60) and
`benchmark_improvement` (+0.0000 < 0.010). `trainable_param_count = 230400`,
`training_loss = 7.464` (finite), audit `ok` (5 entries), `rejected_adapters =
1`, `approved_adapters = 0`. Full figures and check-by-check table in
`docs/deep_apply_real_run_findings.md`.

### Test results

* `python -m pytest tests/test_deep_apply.py -q` → **41 passed, 1 skipped**
  (the skip is the real-e2e test gated on `ASEA_RUN_REAL=1`).
* `ASEA_RUN_REAL=1 python -m pytest tests/test_deep_apply.py -q -k real` →
  **1 passed** (≈61 s).
* `python -m pytest tests/ -q` → **279 passed, 5 skipped** (baseline 238
  passed / 4 skipped unchanged; +41 passed, +1 skipped all from the new file).

### Honest limits (as built)

* Streamed backend runtime on a CUDA GPU is **not verifiable here** (no CUDA
  on this machine). Its blocking behaviour is unit-tested; its streaming loop
  is implemented but marked BETA pending a GPU box.
* No adapter was admitted by a real run; the admit path is covered by scripted
  unit tests only. A real promotion needs a model + dataset + step budget that
  clears all 14 checks — out of reach of a 3-step CPU run.
* `L5_DISTILL_DATASET` remains export-only; deep-apply admits only `L4`.