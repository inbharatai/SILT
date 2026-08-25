# DEEP-APPLY — real run findings

This records **only what actually ran on this machine**, with the exact
numbers the run produced. No figure here is fabricated or projected; every
value below comes from a command executed in this session. Patent pending
(India). (This is the *deep-apply* real run; the packet-mode real run is
documented separately in `real_run_findings.md`.)

## Environment

* Machine: Windows 11, CPU only — **no CUDA GPU available**.
* Model: `HuggingFaceTB/SmolLM2-135M` (≈135M params, English-centric). Chosen
  because it sits well below `CPU_PARAM_CEILING` (1.5B), so the standard
  backend trains it on CPU per the hardware ladder — no `DeepApplyBlocked`.
* Backend: `standard` (streamed backend targets GPU VRAM and is pointless on
  CPU; it would raise `DeepApplyBlocked` demanding CUDA).
* Training: 3 LoRA steps, rank 4, alpha 8, seed 0, on a 4-row glossary packet.
* Optional deps installed: `torch`, `transformers`, `peft`, `accelerate`,
  `sentencepiece` (the `[deep]` extra).

## Commands run and their exact results

### 1. Deep-apply unit + gate + runner suite

```
python -m pytest tests/test_deep_apply.py -q
```
Result: **41 passed, 1 skipped**. The 1 skip is the real-e2e test, which is
gated on `ASEA_RUN_REAL=1`.

### 2. Real LoRA end-to-end (mechanism, not promotion)

```
ASEA_RUN_REAL=1 python -m pytest tests/test_deep_apply.py -q -k real
```
Result: **1 passed** (≈61 s). The test asserts the *mechanism* (trainable
params > 0, finite loss, artifact ref present, provenance not mock, decision
status in {PROMOTED, REJECTED, PENDING_HUMAN}, audit chain ok) — **not** that
the adapter is promoted. A tiny CPU run that Gate 2 rejects is a pass.

### 3. Full suite (existing + new) — "don't break the existing suite"

```
python -m pytest tests/ -q
```
Result: **279 passed, 5 skipped** in 39.06 s.

Baseline before adding `tests/test_deep_apply.py` was **238 passed, 4
skipped**. After: 238 + 41 = 279 passed; 4 + 1 = 5 skipped. **No existing test
broke or newly skipped.** The +1 skip is the real-e2e test (gated on
`ASEA_RUN_REAL=1`, unset in the plain suite run).

### 4. Honest-number probe (the figures below come from this)

```
ASEA_RUN_REAL=1 PYTHONPATH=src python scripts/real_deep_apply_probe.py
```
Exit 0. Full report captured. Scalars extracted verbatim from the run:

| Field | Value |
|---|---|
| `adapter_id` | `58083002-82d1-4253-82d0-f73e34a95ee5` |
| `session_id` | `da-9cd8a1139b0b` |
| `backend` | `standard` |
| `status` | `rejected` |
| `base_model` | `HuggingFaceTB/SmolLM2-135M` |
| `target_module` | `smollm2-receiver` |
| `source_packet_ids` | `["real-p1"]` |
| `source_domains` | `["translation"]` |
| `synthetic_depth` | `0` |
| `risk_tier` | `low` |
| `trainable_param_count` | `230400` |
| `training_loss` | `7.4639573097229` |
| `dataset_hash` | `0f498309e8a86d5afe988e7bb39b6f2b402dc40647ab3c3849cfa767266d7a60` |
| `dataset_rows` | `4` |
| `evaluator_score` | `0.27` |
| `improvement` | `0.0` |
| `case_regressions` | `0` |
| `case_count` | `2` |
| `rollback_token` | `20260816T091532-104ce27d` |
| `gate2.status` | `rejected` |
| `gate2.approved` | `false` |
| `gate2.needs_human` | `false` |
| `gate2.reason` | `evaluator_threshold: evaluator_score 0.270 (need >= 0.60); benchmark_improvement: improvement +0.0000 (need >= 0.010)` |
| `audit.verify()` | `{ok: true, entries: 5}` |
| `store.stats()` | `{candidate_adapters: 0, approved_adapters: 0, rejected_adapters: 1}` |

### Gate 2 check-by-check (from the same probe output)

| # | Check | Passed | Detail |
|---|---|---|---|
| 1 | `schema_validation` | ✅ | adapter identity present |
| 2 | `training_loss_finite` | ✅ | training_loss finite (7.464) |
| 3 | `adapter_artifact_present` | ✅ | artifact_ref=present, trainable_params=230400 (need ≥ 1) |
| 4 | `evaluator_threshold` | ❌ | evaluator_score 0.270 (need ≥ 0.60) |
| 5 | `safety_threshold` | ✅ | safety_score 1.000 (need ≥ 0.70) |
| 6 | `benchmark_improvement` | ❌ | improvement +0.0000 (need ≥ 0.010) |
| 7 | `no_regression` | ✅ | no regression detected |
| 8 | `case_regression_limit` | ✅ | 0/2 held-out cases regressed (ratio 0.00, max 1.00) |
| 9 | `provenance_present` | ✅ | source_packets=1, chain=['sender-real'] |
| 10 | `synthetic_depth` | ✅ | synthetic_depth 0 (max 2) |
| 11 | `no_self_lineage` | ✅ | receiver 'smollm2-receiver' absent in source provenance chain |
| 12 | `rollback_metadata` | ✅ | rollback_token present |
| 13 | `applicable_learning_level` | ✅ | level L4 is applicable (deep-apply) |
| 14 | `no_mock_provenance` | ✅ | source provenance excludes a mock module |

(14 checks total; human-approval is folded in — `needs_human` was `false`
here because the source domain is translation, not a HIGH-risk domain.)

## What this run proves

* **The mechanism is real, end to end.** `trainable_param_count = 230400 > 0`
  confirms a genuine LoRA optimization ran on the receiver's q/k/v/o
  projections — not a mock, not a no-op. `is_mock` stays `False` on the real
  path.
* **Training was numerically healthy.** `training_loss = 7.464` is finite, so
  the new `training_loss_finite` sanity check passes — the run did not blow up.
* **Double-gating works.** Gate 1 promoted the packet (the test seeds a
  `PROMOTED` glossary). Gate 2 then independently evaluated the trained adapter
  and **rejected it with two named reasons** (`evaluator_threshold`,
  `benchmark_improvement`) — exactly the discipline the design requires.
* **Audit and rollback are intact.** The hash-chained audit log verifies
  (`ok: true`, 5 entries) and a `rollback_token` was issued before the gate
  decision, so this rejected adapter is reversible-by-construction even
  though it was never admitted.
* **Honesty of the outcome.** A 3-step CPU LoRA on a 135M English-centric model
  does **not** learn Assamese→English translation. `evaluator_score = 0.27`
  and `improvement = 0.0` reflect that. Gate 2 rejecting is the *correct*
  behaviour — the test asserts the mechanism, not a promotion.

## NOT VERIFIABLE HERE

* **Streamed backend runtime on a CUDA GPU.** This machine has no CUDA. The
  streamed backend's per-layer forward-hook streaming loop is implemented and
  unit-tested for its *blocking* behaviour (it raises `DeepApplyBlocked`
  demanding CUDA when no GPU is present, with no silent fallback), but its
  *runtime* on an actual GPU is **not verifiable on this hardware**. It is
  marked BETA in the design — a GPU box is required to validate the streaming
  loop and reproduce Soup's low-VRAM result.
* **Promotion of an adapter.** No adapter was admitted (`approved_adapters:
  0`). The code path that admits a `PROMOTED` adapter (snapshot → store.approve
  → audit `adapter_admitted`) is covered by unit tests with scripted doubles,
  but a *real* promoted adapter requires a model + dataset + step budget that
  actually clears all 14 checks — not achievable in a 3-step CPU run and not
  asserted here.