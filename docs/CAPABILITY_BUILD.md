# Capability build — teacher footprinting, causal intervention, minimum-capability search

**Status: implemented / source_code_verified, with one small executed behavioural
pilot recorded below. This layer is NOT enabled by default, NOT auto-activated,
and confers no capability on any model by existing.**

This is a research/build layer on top of the existing SILT mechanisms. It answers
one engineering question, stated in the operator's spec, never implicitly:

> Given a teacher (GLM-5.3-Flash class) and a narrowly defined capability
> (`python_repo_debugging_v1`), what is the minimum computational artifact that
> reproduces that capability in a much smaller student — *proven on untouched
> held-out tests* — *without materially regressing controls*?

It does not replace, weaken or shortcut any existing mechanism: the Pipeline,
Gates 1/2, DeepApply, SiltSpring, the compiler, the packet distillers and the
specialist path are all reused as-is (see "Boundary rules" below).

## Vocabulary (same discipline as CAPABILITIES.md)

Every statement below uses the catalog's five-state discipline; no state implies
the next:

| Term | Meaning here |
|---|---|
| **Implemented** | Source path + inspected contract exists (`src/asea/capability_build/`). Not installed-verified, not executed, not quality-validated. |
| **Enabled** | An operator wrote a spec and selected a teacher connector per run. Nothing here is ever enabled by default. |
| **Executed** | A real command ran in a stated environment (the pilot below). An executed trace is NOT a judged outcome; UNJUDGED is recorded whenever the host oracle has not spoken. |
| **Admitted** | A named subsystem accepted evidence under its own policy. Here only `CANDIDATE_UNADMITTED` exists: admission is DeepApply/Gate 2's to give, never this package's. |
| **Activated** | Nothing in this package ever activates anything. No deployment pointer, no auto-load, no background work. |

Boundary states this layer adds: `BLOCKED_RESOURCE` (a typed refusal naming
requirement + remedy — the honest outcome on insufficient hardware, missing
daemons, or non-Linux sandbox use), `NOT_MEASURED` (a value that was never
measured, materialised at receipt time, never estimated), `UNJUDGED` (a teacher
response the host oracle has not graded — the teacher never grades itself),
`CANDIDATE_UNADMITTED` (a reduction/search candidate; only functional held-out
evidence plus DeepApply/Gate 2 admission could ever change that).

## What is implemented (source_code_verified)

* `CapabilitySpec` (`silt.capability_spec.v1`): machine-readable capability
  definition — teacher pin, target metrics, controls, `minimum_retention_ratio`,
  `maximum_control_regression`, four evaluation splits. Thresholds are operator
  configuration, never guarantees.
* Two teacher evidence classes, **never mixed in one trace or one footprint**:
  * `behavioural_remote` — Ollama connector (default `http://localhost:11434`),
    per-run explicit remote consent (`--allow-remote` or
    `remote_connector_selected` in the spec; never carries over between runs;
    no silent fallback if the remote path is unavailable).
  * `internal_open_weight` — instrumented router telemetry via the isolated GLM
    worker only (see below). Refused on quantized checkpoints (the main HF repo
    is FP8; the remedy names the BF16 variant).
* `CapabilityTrace` (`silt.capability_trace.v1`): per-case records bound by
  `sample_id` + `prompt_hash`; schema-level rejection of evidence-class mixing.
* `CapabilityFootprint` (`silt.capability_footprint.v1`): "computational
  components associated with and experimentally important to a defined
  capability under a defined workload" — **never** "these neurons contain coding
  knowledge". Behavioural evidence yields NO internal component claims at all.
  Every footprint carries the compiler's discipline verbatim: *routing and
  usage enrichment is usage evidence, not causal expert importance.*
* Causal intervention protocol (net-new; the compiler only prunes physically):
  verify-unchanged → seeded baseline → temporary mask → measure → restore →
  verify-unchanged again. An intervention whose restore cannot be verified is
  **never** causal evidence. The teacher's weights are read-only throughout.
* MoE adapter layer (`adapters/`): a `MoEArchitectureAdapter` contract with a
  Switch implementation (wrapping compiler telemetry, byte-exact parameter-hash
  verification) and a GLM-5.3-Flash implementation pinned to the published
  architecture (45 layers, 288 routed + 1 shared expert, 8 experts/token,
  sigmoid router scoring, first-3-MLP dense, hidden 4096, 1M positions). The
  ~18B-active figure is recorded as a per-token compute statement about the FULL
  model; it is not an identifiable 18B subset, and no small-GLM claim is made.
* Isolated GLM worker (`workers/glm53/`): own Dockerfile, own
  `requirements.lock` (Transformers 5.x per the model card; the core
  `transformers==4.51.3` pin is untouched), versioned JSONL frame protocol,
  memory preflight before load (a small host is honestly `BLOCKED`, which is
  the correct outcome, not a bug), partial-frame preservation on crash —
  a lost generation is never fabricated.
* Fresh dataset builder (`dataset.py`, CLI `dataset build`): five splits
  (training/development/heldout/final/controls), per-case unique ID, content
  hash, family ID, provenance, license; cross-split ID/content/family
  disjointness; a NEW near-duplicate token-shape guard (the existing repo only
  guards exact duplicates); the frozen September sets are quarantined as
  inputs; manifest + selection lock frozen before any model sees a case.
* Student baselines: LOCAL connectors only (a non-localhost student host is a
  typed `BLOCKED_RESOURCE`; no remote student path exists). The measured gap
  `Gap = TeacherScore − StudentScore` is the only justification for any
  transfer work.
* Sequence-level KD pair builder + DeepApply hand-off: pairs only from
  host-judged successful traces (failures kept as labelled negatives; UNJUDGED
  traces contribute nothing); leakage collisions poison the whole set; training
  itself is DeepApply's (LoRA via Gate 2) — the trainer never certifies itself.
  Cross-family (GLM→Qwen) pairs are text-only: no vocabulary or logit
  alignment is assumed.
* Minimum-capability search (`search.py`, library surface): the objective
  `min Size(M)` s.t. `TargetScore(M) ≥ retention_ratio × teacher` and control
  regression ≤ tolerance; every iteration recorded including failures; a
  parameter decrease is never itself evidence. Candidates are
  `CANDIDATE_UNADMITTED` by construction.
* Functional evaluation routes to the existing external host oracle
  (`asea.certification.function_oracle`) and Linux-only sandbox; verdicts are
  host-owned; the teacher never grades itself. On Windows/macOS the gate fails
  closed with `BLOCKED_RESOURCE` naming the WSL2/Linux remedy.
* `CapabilityBuildReceipt` (`silt.capability_receipt.v1`) signed with the
  existing local HMAC signer, `receipt`/`receipt-verify` CLI pair; unmeasured
  fields are `NOT_MEASURED`, never estimated. The signature is a local HMAC —
  not a portable attestation, not authorship proof, not a quality certificate.
* CLI `silt-capability` (console script; JSON-only stdout; exit codes 0/2/3/4
  mirroring `silt-compile`).

## What is NOT in this build (honest refusals, not gaps hidden)

* `reduce`, `search`, `certify` CLI commands are registered surfaces that
  **refuse with `status: rejected`** and the scheduled phase — they emit no
  placeholder results. Their library implementations exist (`search.py`;
  reduction candidates and SiltSpring certification are Phase 8 live work that
  needs admitted candidate models).
* The open-weight intervention loop is wired up to worker `hello`, then refuses
  honestly that the generation+judging stage inside the worker is not in this
  build.
* No live student baseline, no DeepApply training run, no SiltSpring state
  certification has been executed. Nothing has been admitted anywhere.

## Executed pilot (recorded 2026-09-17; small, honest, labelled)

One behavioural pilot of `python_repo_debugging_v1` on the operator-selected
Ollama cloud connector (`glm-5.3-flash:cloud`, remote consent given per run):

* 8 authored cases (4 target python-repair, 4 control math/QA), CC0, provenance
  recorded; **not** the frozen September sets.
* `teacher-baseline`: 8 cases asked, stored with outcome UNJUDGED (the teacher
  cannot grade itself).
* `trace --mode behavioural`: 8 traces written, `evidence_class:
  behavioural_remote`.
* `footprint`: built with **zero internal component claims** (behavioural
  evidence only), correlation-only limitation carried verbatim.
* Host-oracle judgment of the four teacher repairs (Linux sandbox, 2026-09-18):
  all six oracle checks PASSED (`pass_rate 1.0`, `status: PASSED`) — flatten
  (recursive, also on `[]`), median (sorted-center on two unsorted odd-length
  lists), slugify (strips trailing hyphen), chunk (full remainder). One honest
  scope note: the teacher's even-length median branch returns a float
  (`(a+b)/2`), and the host oracle's exact-type value language (null, bool,
  int, string, list, dict — no floats) cannot express it, so that branch is
  **not oracle-graded** and no claim is made about it. This is a **judged
  functional outcome on 4 target cases only**; it is not a capability
  certificate, not a quality claim about GLM-5.3-Flash in general, and
  involves no student at all.

The open-weight teacher path on this laptop reports `BLOCKED_RESOURCE`
honestly: the worker's memory preflight cannot admit a ~320B-parameter teacher
on this hardware, and Docker-on-Windows is itself unverified — both are the
correct outcomes, never silently skipped.

## Boundary rules (binding, all verified in review)

Package stays `asea`; `asea run` unchanged; Pipeline called, never mutated;
Gates 1/2 untouched and reused; no bypass flags anywhere (the CI anti-bypass
grep stays clean); `CapabilityDiff` keeps its meaning (the new object is
`CapabilityFootprint`); existing packet distillers remain the L3 mechanism;
DeepApply and SiltSpring reused, never replaced; frozen September results and
the consumed 16-task final set untouched and quarantined as inputs;
`transformers==4.51.3` pin untouched (GLM runtime isolated in the worker);
teacher weights read-only; routing frequency never called causal; no
"small GLM" claim without evidence; no universal-extraction claim; no
auto-activation anywhere; no silent cloud fallback; evidence classes never
mixed; implementation success is never labelled model-quality success; failed
experiments stay visible.

## Tests

`tests/test_capability_build.py` — 57 tests: schema validation, evidence-class
separation, consent gating (no network without per-run consent), enrichment
scoring with correlation limitation, intervention protocol (mask-measure-
restore-verify on a fake adapter; unrestorable never causal), store/receipt
integrity + tamper detection, worker IPC crash/blocked frames, dataset
leakage/near-duplicate/quarantine guards, student gap bookkeeping, KD pair
leakage poisoning, search accept/reject/failure visibility, CLI exit codes,
and two REAL Linux-sandbox oracle integration tests (skipped on Windows where
the sandbox fails closed). Every test is a MECHANISM test unless its docstring
says REAL; no test asserts anything about any teacher's behaviour.

## Commands

```
silt-capability spec validate --spec spec.json
silt-capability dataset build --cases cases.jsonl --out data/capability_v1
silt-capability dataset validate --dir data/capability_v1
silt-capability teacher-baseline --spec spec.json --cases cases.json [--allow-remote]
silt-capability trace --spec spec.json --cases cases.json [--mode behavioural|internal] [--allow-remote]
silt-capability footprint --spec spec.json
silt-capability intervene --spec spec.json --component expert:3/7 [--checkpoint DIR]
silt-capability evaluate --spec spec.json --cases cases.json --source candidate.py
silt-capability student-baseline --spec spec.json --cases cases.json [--model qwen2.5-coder:0.5b]
silt-capability distill --spec spec.json
silt-capability receipt --receipt receipt.json
silt-capability receipt-verify --receipt receipt.json
```

All artifacts land under a workspace (default `.capability-build/`) with
content hashes; receipts are signed against the workspace-local key.