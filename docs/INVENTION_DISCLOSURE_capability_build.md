# Technical invention disclosure — capability footprinting, causal intervention, and minimum-capability search

**Status: DRAFT FOR OWNER / PATENT COUNSEL REVIEW — NOT A CLAIM.** This
document is an internal technical disclosure prepared for the owner and patent
counsel to evaluate. Nothing here is a claim of inventorship, priority,
validity, infringement, or capability. It is deliberately **not** part of
`PATENT.md`, which remains untouched; the existing provisional
(202631101454) already declares the September 2026 reconstruction/recovery
work outside its coverage, and this disclosure describes later work.

## 1. Problem statement

Given a large multi-purpose model (a "teacher") and a *narrowly defined*
capability expressed as a machine-readable specification, existing systems
either (a) distill whole-model behaviour without localising which
computational components participate in the capability, or (b) prune
architectures using usage statistics that are usage evidence, not causal
importance. Neither answers: *what is the minimum computational artifact that
reproduces this capability in a much smaller student, proven on untouched
held-out tests, without materially regressing controls?*

## 2. Candidate inventive elements (for counsel's evaluation only)

1. **Evidence-class-separated teacher footprinting.** Traces are collected
   under one of two declared evidence classes — behavioural (external
   connector, per-run explicit consent, outcomes marked UNJUDGED unless a
   host oracle has graded them) or instrumented open-weight (internal routing
   telemetry from an isolated worker) — and the system *refuses at schema
   level* to mix the two classes within one trace or one footprint.
   Behavioural evidence is structurally incapable of producing internal
   component claims.

2. **Causal intervention with enforced restore-verification.** For
   open-weight teachers, candidate components are evaluated by a protocol of
   verify-unchanged → seeded baseline → temporary mask → measure on identical
   target and control case sets → restore → verify-unchanged again. An
   intervention whose restore cannot be verified (parameter-hash mismatch) is
   *never* recorded as causal evidence — a machine-enforced distinction
   between correlation (routing/usage enrichment) and causation (measured,
   reversible ablation). Teacher weights are read-only throughout.

3. **Admission-separated minimum-capability search.** A search over candidate
   reductions minimises artifact size subject to held-out functional retention
   (target score ≥ retention ratio × teacher score) and bounded control
   regression, while recording every iteration including failures; and a
   candidate artifact remains `CANDIDATE_UNADMITTED` by construction —
   admission and certification can only be granted by a separate, existing
   gated pipeline (Deep-apply/Gate 2, then state certification), never by the
   search itself. A parameter-count decrease is never itself evidence.

4. **Leakage-guarded capability dataset construction.** A five-split
   (training/development/heldout/final/controls) case builder enforcing
   cross-split unique-ID, exact-content-hash, family, and *near-duplicate
   token-shape* disjointness (beyond exact-duplicate guards), with
   quarantined frozen evaluation sets and a selection lock frozen before any
   model sees a case.

5. **Runtime isolation for heterogeneous model runtimes.** A
   protocol-framed isolated worker (own Dockerfile, own pinned dependency
   lock, versioned JSONL frame protocol, memory preflight before load,
   partial-frame preservation on crash — a lost generation is never
   fabricated) allowing a system pinned to one model runtime
   (`transformers==4.51.3`) to instrument a teacher requiring a different,
   incompatible runtime (Transformers 5.x) without changing the host
   environment's pin.

## 3. Explicitly NOT claimed (binding honesty limits)

- No universal capability-extraction claim; the method is scoped to
  operator-defined capabilities under defined workloads.
- No claim that routing frequency or usage enrichment *is* causal importance.
- No claim that an MoE model's "active parameter" count is an identifiable
  subset; no "small GLM" claim without held-out functional evidence.
- No claim that behavioural (external API) evidence reveals internal
  components.
- No claim that any implementation success in this repository constitutes
  model-quality success; no artifact has been admitted or certified.
- This disclosure describes work that is **implemented and minimally
  executed (one behavioural pilot)**, not validated as an end-to-end
  capability-transfer result.

## 4. Supporting artifacts

- Implementation: `src/asea/capability_build/`, `workers/glm53/`
- Description and honest pilot record: `docs/CAPABILITY_BUILD.md`
- Capability catalog entries: `docs/CAPABILITIES.md` (C26–C29)
- Tests: `tests/test_capability_build.py`