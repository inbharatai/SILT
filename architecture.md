# SILT — Architecture

*Skill Interchange Layer with Trust-gating. Package imports as `asea`.*

## The shape of the thing

```
  ┌────────────┐                                            ┌──────────────┐
  │  SENDER    │                                            │   RECEIVER   │
  │ (teacher)  │                                            │  (learner)   │
  └─────┬──────┘                                            └──────▲───────┘
        │ manifest                                    redacted     │
        │                                             skills only  │
  ┌─────▼────────────────────────────────────────────────────┬─────┴───────┐
  │              SILT — SKILL INTERCHANGE LAYER              │             │
  │                                                          │             │
  │  handshake ─► gap negotiation ─► extraction ─► relevance │             │
  │       │                                            │     │             │
  │       │                                            ▼     │             │
  │       │              safety ◄─────────────────────────┐  │             │
  │       │                 │                             │  │             │
  │       │                 ▼                             │  │             │
  │       │            distillation ──► evaluation ──► snapshot ──► GATE   │
  │       │                                  │                       │     │
  │       │                                  │            ┌──────────┴───┐ │
  │       │                                  │            │ promote      │ │
  │       │                                  │            │ pending human│ │
  │       │                                  │            │ reject       │ │
  │       │                                  ▼            └──────┬───────┘ │
  │       └──────────────► AUDIT LOG (hash-chained) ◄────────────┘         │
  └────────────────────────────┬─────────────────────────────────────────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │  MEMORY STORE              │
                 │  candidate/  ← not visible │
                 │  approved/   ← receiver    │
                 │  rejected/   ← audit       │
                 │  snapshots/  ← rollback    │
                 └────────────────────────────┘
```

## What makes it universal (and where that stops)

**Universal core.** `src/asea/core/pipeline.py` contains no modality branching.
It resolves an extractor, a distiller and a metric out of the plugin registry
based on the capability's declared modality, then runs the same nine stages
regardless. `tests/test_conformance.py` asserts this three ways: by inspecting
the pipeline source for `if ... Modality` branches, by running four dissimilar
modality pairs and requiring a byte-identical audit event sequence, and by
registering a brand-new OCR modality with no core edit.

**Not universal.** Extraction, distillation and task-success scoring are
per-modality plugins. This seam is deliberate and documented: a glossary and a
pronunciation lexicon cannot share a compression algorithm. The honest claim is
"one protocol, many mechanisms", not "one mechanism".

## Components

| Component | Module | Responsibility |
|---|---|---|
| Packet protocol | `core/protocol.py` | The typed envelope everything agrees on |
| Capability manifest | `core/protocol.py` | What a module claims it can do |
| Handshake | `core/handshake.py` | Compatibility check, session issuance, level negotiation |
| Gap engine | `core/gap.py` | Measured deficiency detection; refuses to act without evidence |
| Module registry | `registry/registries.py` | Id-keyed store with role validation |
| Sender / Receiver registries | `registry/registries.py` | Role projections over the module registry |
| Adapter registry | `registry/registries.py` | Named sender→receiver bindings |
| Plugin registry | `core/plugins.py` | Modality → extractor / distiller / metric resolution |
| Extraction engine | `extraction/extractors.py` | Probes the sender on the gap set |
| Relevance filter | `filters/relevance.py` | Drops signals that cannot help |
| Safety filter | `filters/safety.py` | Scores and hard-blocks harmful payloads |
| Distillation engine | `distill/strategies.py` | Compresses signals; drops raw output |
| L4/L5 export | `distill/export.py` | Dataset + job spec; explicitly does not train |
| Evaluator | `evaluator/evaluator.py` | Held-out A/B plus regression sweep |
| Metrics | `evaluator/metrics.py`, `metrics_plugins.py` | Schema, similarity, language, hallucination, task success |
| Benchmark harness | `benchmarks/harness.py` | Split discipline enforcement |
| Memory store | `memory/store.py` | Physically separated candidate / approved / rejected |
| Rollback layer | `memory/store.py` | Snapshot and restore the approved set |
| Promotion gate | `promotion/gate.py` | All-or-nothing rule check |
| Audit log | `audit/logger.py` | Append-only hash chain |
| Config | `config.py` | Declarative wiring |
| CLI | `cli.py` | run / report / approve / rollback / audit / export |

## The packet

Every field required by the brief is present in `SkillPacket`: `packet_id`,
`task_type`, `source_module`, `target_module`, `sender_capability`,
`receiver_gap`, `modality`, `language`, `domain`, `raw_input_reference`,
`sender_output`, `distilled_skill`, `confidence_score`, `evaluator_score`,
`safety_score`, `provenance`, `learning_level`, `promotion_status`,
`rejection_reason`, `version`.

Three invariants are enforced by the type system rather than by convention:

- a `REJECTED` packet cannot exist without a `rejection_reason`;
- a `PROMOTED` packet cannot exist without both a `distilled_skill` and a
  `rollback_token`;
- `redacted_for_receiver()` — the only view a receiver ever gets — has no
  `sender_output` field at all. Raw model output cannot reach a learner even
  through a bug, because it is not in the dictionary that crosses the boundary.

## Learning levels

| Level | Meaning | Status here |
|---|---|---|
| L0 | Interaction only, no learning | Supported |
| L1 | Context injection | Supported |
| L2 | Memory / RAG | Supported |
| L3 | Skill packet | **Primary path**, fully implemented |
| L4 | LoRA / PEFT candidate | Export only **by default**; applicable via optional **deep-apply** (see below) |
| L5 | Distillation dataset candidate | Export only — gate refuses live promotion |

`APPLICABLE_LEVELS` in the protocol is the single source of truth, and the gate's
`applicable_learning_level` check reads from it. Through the standard
`PromotionGate`, L4/L5 packets cannot reach a live receiver through any code
path — they are exported as a dataset + a `NOT_EXECUTED` job spec.

### Deep-apply — the optional second apply mode (double-gated)

`src/asea/deepapply/` is an **optional** stage (`pip install -e ".[deep]"` for
torch/transformers/peft/accelerate/sentencepiece) where SILT itself trains a
LoRA adapter on the receiver, using **only packets that already passed Gate 1**,
then gates the trained adapter **again (Gate 2)** before admission. Two apply
modes now coexist, both gated, both reversible, both audited:

* **Packet mode (L3, default)** — the receiver conditions on the approved skill
  packet at inference time. No weights touched. This is the primary path.
* **Weights mode (L4, deep-apply)** — a trained LoRA adapter is admitted to the
  receiver. Adapters are removable by construction; v1 never merges into base
  weights. Admission snapshots the approved-adapter set; rollback restores it.

The load-bearing principle is unchanged: SILT separates HOW a capability was
produced from WHETHER it is allowed in the receiver. **Gate 2 never trusts the
trainer** — it measures the trained artifact's outcome on the held-out split
plus a regression sweep, reusing `BenchmarkHarness`. Standard and streamed
backends are judged identically (Gate 2 has zero backend-conditional branches).

Binding rules: no Gate 1 check is weakened (intake re-enforces "PROMOTED only,
no mock"); high-risk source domains (medical/legal/finance) park the adapter at
`PENDING_HUMAN` at Gate 2 regardless of scores — not disableable; honest
hardware (small models train on CPU, big models without CUDA raise a named
`DeepApplyBlocked`, never a fabricated result); rollback covers weights. See
`docs/deep_apply_design.md` for the full design and check mapping.

## Promotion rules

A packet is promoted only if **every** check passes:

1. `schema_validation` — payload well-formed for its packet type
2. `distilled_payload_present`
3. `evaluator_threshold`
4. `safety_threshold`
5. `benchmark_improvement` — measured gain on the **held-out** split
6. `no_regression` — no control capability dropped beyond tolerance
7. `case_regression_limit` — caps the fraction of held-out cases allowed to
   regress even when the average improves (`max_case_regression_ratio`)
8. `provenance_present` — non-empty lineage
9. `synthetic_depth` — within the model-collapse ceiling
10. `no_self_transfer` — the receiver is not in its own teaching chain
11. `rollback_metadata` — a snapshot exists to undo this
12. `applicable_learning_level` — L0–L3 only
13. `no_mock_provenance` — under the default strict policy
14. `human_approval` — for HIGH-risk domains, **not configurable**

Rule 14 cannot be relaxed. `test_no_configuration_can_disable_human_approval`
constructs a maximally permissive policy and asserts a medical packet still
lands in `PENDING_HUMAN`.

## Split discipline

The harness enforces three disjoint splits and raises if a caller reads the
wrong one:

- `extraction` — the diagnostic set. Gap measurement and sender probing only.
- `heldout` — evaluation only. Never seen by the extractor.
- `regression` — control capabilities the transfer is not trying to improve.

Without this you measure memorisation and call it learning. The shipped Assamese
suite is deliberately structured so extraction is word-level and held-out is
sentence-level, forcing a packet to generalise compositionally rather than
recall.

## Data flow guarantees

1. Raw `sender_output` is set to `None` during distillation and is absent from
   the receiver's view.
2. Candidate and approved data live in different directories; a receiver reads
   only `approved/`.
3. Every state transition appends to a hash-chained log; altering history breaks
   verification at a reported index.
4. Every promotion is preceded by a snapshot, so every promotion is reversible.
