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
| C1 | Canonical GLM layer addressing in masking | NOT_IMPLEMENTED |
| C2 | Internal traces functionally judged (no success=null) | NOT_IMPLEMENTED |
| C3 | Exact SourceCheckpointManifest incl. per-shard sha256 | NOT_IMPLEMENTED |
| C4 | Official GLM inference contract (AutoProcessor + apply_chat_template) | NOT_IMPLEMENTED |
| C5 | CLI exit semantics 0/2/3/4/5, subprocess-tested | NOT_IMPLEMENTED |
| C6 | Physically sealed final split | NOT_IMPLEMENTED |
| C7 | Durable failed-attempt evidence records | NOT_IMPLEMENTED |
| C8 | Documentation corrections + CI docs-drift check | NOT_IMPLEMENTED |
| D1 | CapabilityExtractionSpec schema | NOT_IMPLEMENTED |
| D2 | SourceCheckpointManifest schema | NOT_IMPLEMENTED |
| D3 | FunctionalCaseManifest schema | NOT_IMPLEMENTED |
| D4 | RoutingObservation schema | NOT_IMPLEMENTED |
| D5 | ActivationObservation schema | NOT_IMPLEMENTED |
| D6 | InterventionAttempt schema | NOT_IMPLEMENTED |
| D7 | CausalEffectEstimate schema | NOT_IMPLEMENTED |
| D8 | CausalComponentGraph schema | NOT_IMPLEMENTED |
| D9 | ExtractionPlan schema | NOT_IMPLEMENTED |
| D10 | TensorTransformation schema | NOT_IMPLEMENTED |
| D11 | TensorProvenance schema | NOT_IMPLEMENTED |
| D12 | CompiledCapabilityConfig schema | NOT_IMPLEMENTED |
| D13 | CompiledCapabilityManifest schema | NOT_IMPLEMENTED |
| D14 | RecoveryRun schema | NOT_IMPLEMENTED |
| D15 | CapabilityExtractionReceipt schema | NOT_IMPLEMENTED |
| E | Full-source inventory validated against loaded config | NOT_IMPLEMENTED |
| F | Isolated workers/glm53 runtime (exists, carried forward) | IMPLEMENTED_UNTESTED |
| G | Hardware honesty / BLOCKED_RESOURCE discipline | NOT_IMPLEMENTED |
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
| V | Immutable CapabilityExtractionReceipt | NOT_IMPLEMENTED |
| W | silt-extract CLI (16 commands) | NOT_IMPLEMENTED |
| X | Test list incl. exit codes, sealed final, provenance, source immutability | NOT_IMPLEMENTED |
| Y | Public-claim rules enforced in docs | NOT_IMPLEMENTED |
| Z | Definition of done (24 items) | NOT_IMPLEMENTED |

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