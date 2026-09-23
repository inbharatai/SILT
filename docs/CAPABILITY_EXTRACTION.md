# Capability Extraction Program — Ledger and Preregistration

Status vocabulary (binding): `NOT_IMPLEMENTED`, `IMPLEMENTED_UNTESTED`,
`FIXTURE_VERIFIED`, `REAL_MODEL_VERIFIED`, `BLOCKED_RESOURCE`,
`FAILED_HYPOTHESIS`, `PARTIAL_RESULT`, `CERTIFIED`. An item may hold
exactly one status. Fixture evidence never upgrades a real-model claim.

## Section A — Repository integrity (recorded 2026-09-23)

- Audited baseline named by the owner's brief:
  `4d294302391cf8fa8435a5cb0df8f4527ce3b532`.
- `origin/main` at program start: `5cbddf6`. Intervening commits between
  the audited baseline and program start: exactly ONE, `5cbddf6` ("E2
  expansion: 3.7x KD set, 3x student, GPU LoRA — net negative on every
  split"), authored in this workspace on 2026-09-23 and reviewed in full
  by the same operator that authored it. It touches
  `experiments/glm53_flash_e2/**`, `data/capability_v2/**`,
  `src/asea/deepapply/{runner,trainer}.py`, `tests/test_deep_apply.py`,
  `.gitattributes`, `.gitignore`, and docs. It does not touch any
  mechanism this program builds on (`workers/glm53/**`,
  `src/asea/capability_build/**`, `src/asea/compiler/**`,
  `src/asea/specialist/**`). No plan changes were required by it.
- Authoritative CI on the pre-change tree (`5cbddf6`): run 35829147797
  GREEN (sanity + pytest Python 3.9/3.11/3.12) on 2026-09-23. Local
  offline suite: 2905 passed / 20 skipped; the 34 errors are the known
  pre-existing WSL playwright environment gap (system libs), not
  regressions.
- All program work happens on the feature branch
  `feat/glm-extraction-compiled-capability`, never directly on public
  main.

## Preregistered hypothesis (recorded BEFORE any program implementation)

Hypothesis H1, first target capability = repository-level software
engineering:

> A standalone GLM-5.3-Flash-derived CompiledCapabilityModel that
> physically retains the tokenizer, embeddings, dense layers, attention,
> and head, plus a causally selected subset of routed experts with
> rebuilt routers, reproduces at least **90% retention** of the source
> model's functional score on the untouched final evaluation of the
> defined capability while regressing control suites by no more than
> their preregistered tolerances, with at least a **25% reduction in
> stored parameters** relative to the full source checkpoint.

Decision rule: retention is measured as
`CompiledCapabilityModel_score / full_source_score` on the sealed final
split, graded by the external host oracle under authored-denominator
accounting. Control regression is measured per control suite. The
verdict is `CERTIFIED` only if both thresholds hold on the sealed final
split evaluated exactly once. If thresholds are not reached, the record
is `FAILED_HYPOTHESIS` or `PARTIAL_RESULT` as the evidence determines.
**These thresholds will not be weakened after seeing final results.**
If hardware insufficiency prevents the measurement, the record is
`BLOCKED_RESOURCE` with the exact requirement and remedy — never a
substituted estimate.

Known constraint recorded up front (owner's Section G): the present
8 GB-VRAM machine cannot hold a BF16 ~320B-parameter source
(~643 GiB). Full-source measurements (I, J, K, S, T) are therefore
expected to report `BLOCKED_RESOURCE` here. Fixture-level mechanics may
be `FIXTURE_VERIFIED` on this machine; they are never real-model
evidence.

## Implementation ledger

| Item | Description | Status |
|---|---|---|
| A1 | Feature branch, integrity record, preregistration | IMPLEMENTED_UNTESTED |
| C1 | Canonical GLM layer addressing in masking | FIXTURE_VERIFIED |
| C2 | Internal traces functionally judged (no success=null) | FIXTURE_VERIFIED |
| C3 | Exact SourceCheckpointManifest incl. per-shard sha256 | FIXTURE_VERIFIED |
| C4 | Official GLM inference contract (AutoProcessor + apply_chat_template) | FIXTURE_VERIFIED |
| C5 | CLI exit semantics 0/2/3/4/5, subprocess-tested | FIXTURE_VERIFIED |
| C6 | Physically sealed final split | FIXTURE_VERIFIED |
| C7 | Durable failed-attempt evidence records | FIXTURE_VERIFIED |
| C8 | Documentation corrections + CI docs-drift check | NOT_IMPLEMENTED |
| D1 | CapabilityExtractionSpec schema | FIXTURE_VERIFIED |
| D2 | SourceCheckpointManifest schema | FIXTURE_VERIFIED |
| D3 | FunctionalCaseManifest schema | FIXTURE_VERIFIED |
| D4 | RoutingObservation schema | FIXTURE_VERIFIED |
| D5 | ActivationObservation schema | FIXTURE_VERIFIED |
| D6 | InterventionAttempt schema | FIXTURE_VERIFIED |
| D7 | CausalEffectEstimate schema | FIXTURE_VERIFIED |
| D8 | CausalComponentGraph schema | FIXTURE_VERIFIED |
| D9 | ExtractionPlan schema | FIXTURE_VERIFIED |
| D10 | TensorTransformation schema | FIXTURE_VERIFIED |
| D11 | TensorProvenance schema | FIXTURE_VERIFIED |
| D12 | CompiledCapabilityConfig schema | FIXTURE_VERIFIED |
| D13 | CompiledCapabilityManifest schema | FIXTURE_VERIFIED |
| D14 | RecoveryRun schema | FIXTURE_VERIFIED |
| D15 | CapabilityExtractionReceipt schema | FIXTURE_VERIFIED |
| E | Full-source inventory validated against loaded config | NOT_IMPLEMENTED |
| F | Isolated workers/glm53 runtime (exists, carried forward) | FIXTURE_VERIFIED |
| G | Hardware honesty / BLOCKED_RESOURCE discipline | FIXTURE_VERIFIED |
| H | Substantial dataset: 14 task families, controls, power analysis | NOT_IMPLEMENTED |
| I | Full-source baseline | NOT_IMPLEMENTED |
| J | Internal traces joined to functional outcomes | NOT_IMPLEMENTED |
| K | Real causal interventions (mask/restore/verify) | NOT_IMPLEMENTED |
| L | CausalComponentGraph classification | NOT_IMPLEMENTED |
| M | Causal selection vs matched controls | NOT_IMPLEMENTED |
| N | Conservative first extraction (dense+expert subset+routers) | NOT_IMPLEMENTED |
| O | Physical build of the CompiledCapabilityModel | NOT_IMPLEMENTED |
| P | Tensor-level provenance | NOT_IMPLEMENTED |
| Q | Standalone independence tests | NOT_IMPLEMENTED |
| R | Recovery training on the extracted artifact | NOT_IMPLEMENTED |
| S | All-arms evaluation | NOT_IMPLEMENTED |
| T | Statistical proof, sealed final evaluated exactly once | NOT_IMPLEMENTED |
| U | Further reduction (only after first proof) | NOT_IMPLEMENTED |
| V | Immutable CapabilityExtractionReceipt | FIXTURE_VERIFIED |
| W | silt-extract CLI (18 commands, honest stage refusals) | FIXTURE_VERIFIED |
| X | Test list — core set landed: schemas, malformed evidence, exact checkpoint identity, unjudged-trace exclusion, sealed final split, failed-attempt ledger, receipt integrity, CLI exit codes | FIXTURE_VERIFIED |
| Y | Public-claim rules enforced in docs | NOT_IMPLEMENTED |
| Z | Definition of done (24 items) | NOT_IMPLEMENTED |

Build notes (2026-09-23, statuses above reflect exactly this scope):

- FIXTURE_VERIFIED means the MECHANISM is implemented and tested on
  fixtures only; it never upgrades any real-model status and never
  supports an L5 claim.
- C1: canonical decoder-layer ids (0..44, sparse 3..44) validated,
  dense/out-of-range refused with typed errors, round-trip tests
  (`tests/test_capability_build.py`). The remaining C1 sub-item —
  the end-to-end trace → footprint component id → intervention →
  exact layer/expert test — lands with the footprint-integration work
  under J/K.
- C2 is enforced at the schema layer: `FunctionalJoin` (verdict is a
  required bool — `success = null` cannot be represented),
  `judged_only()` excludes unjudged telemetry from all evidence, and
  joins that cross source revisions are rejected. Trace COLLECTION
  (J) still needs the real model.
- C3: the worker `manifest` op (per-file sha256 over the whole tree,
  disk-only, no model load) is subprocess-tested, and the CLI
  `source inventory` (names/sizes only) / `source verify` (sha256
  walk → `silt.extraction.source_manifest.v1` with Merkle-style
  aggregate verified against the tree) are wired.
- C4: `_chat_encode` runs the checkpoint's own chat template with
  `add_generation_prompt=True`; the pinned `clear_thinking=True` /
  `reasoning_effort="low"` template arguments were verified against
  the REAL GLM-5.3-Flash chat template (which declares
  `reasoning_effort`, default 'max', and `clear_thinking`, default
  false — `enable_thinking` does not exist in it), and every
  generate response records the full pinned decoding policy. An
  over-budget templated prompt is refused, never truncated.
- C5: `silt-extract` exit codes 0/2/3/4/5 are tested against ACTUAL
  subprocess exit status.
- C6: `SealedSplit` — physically separate artifact, development
  access always refused and always logged, final evaluation opens
  exactly once (consumption recorded BEFORE cases are returned),
  spec-pin and tamper checks. The CLI's `evaluate --split final`
  refuses NOT_IMPLEMENTED WITHOUT opening the seal and records the
  attempt.
- C7: `FailedAttemptLedger` — append-only, hash-chained (tamper
  detection tested), all nine required fields.
- C8 partials already landed: the false "scoring_func absent" claim
  is corrected everywhere (adapter EXPECTED + detection, CAPABILITIES
  C27, CHANGELOG, test fixture). Still open: /README.md serving or
  removal, stale docs/README.md, CAPABILITY_BUILD.md status
  contradictions, memory estimates, worker descriptions, "gate
  verdict"/"never opened" wording, and the CI docs-drift check.
- W: all 18 commands are wired; `spec validate`, `source
  inventory`, `source verify`, `dataset validate`, `receipt` are
  implemented, and every real-model stage reports an honest
  `NOT_IMPLEMENTED` typed refusal (exit 2) naming what it needs —
  when those stages are implemented, this 8 GB host will report
  `BLOCKED_RESOURCE` (exit 4), never a substituted fixture result.

Naming rule (binding, owner's brief): the artifact is called the
**CompiledCapabilityModel** — "GLM-5.3-Flash-derived Repository
Engineering CompiledCapabilityModel" when the target capability is
repository engineering. It is never called a "student", "adapter",
"small GLM" or "extracted intelligence", and no claim of universal
extraction is made.

## Evidence classes

- `L5 — Structurally Derived Capability Model`: evidence that a
  GLM-derived standalone model retains a defined capability, backed by
  real-model artifacts and an immutable receipt.
- Fixture-derived mechanism evidence is labeled `FIXTURE_VERIFIED` and
  never supports an L5 claim.
- `internal_open_weight` and `behavioural_remote` evidence are never
  mixed in one artifact (carried from the capability-build layer).

A separate private technical disclosure for owner/counsel review is
prepared OUTSIDE the repository and never committed or published.
`PATENT.md` is untouched by this program.