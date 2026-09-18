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
    no silent fallback if the remote path is unavailable). Both the teacher
    and the student talk through the SAME shared transport
    (`modules/real/ollama.py`), which refuses HTTP redirects outright
    (review round 2: urllib's default handler silently follows a 3xx to any
    location, so a "local" host answering with a redirect could have
    smuggled the request off-host; the transport now raises with the target
    named and the request is never re-sent — tested through the actual
    connector against throwaway loopback servers).
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
  verify-unchanged again. The mask is a **router-output forward hook**: a
  masked expert's routing logit is set to −1e9 on the cloned router output,
  independent of the hidden state. No weight row is ever written — writing a
  constant into a router Linear's row makes the masked expert's logit
  `−c·Σ(hidden)`, which is strongly *positive* for negative-sum hidden states
  and would have made the masked expert MORE likely to be routed. An
  intervention whose restore cannot be verified is **never** causal evidence.
  The teacher's weights are read-only throughout (and `verify_unchanged` holds
  even during the mask window, because nothing was modified).
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
  a lost generation is never fabricated. The client **enforces** the request
  deadline over the WHOLE exchange, not just the read (review round 2): the
  watchdog timer is armed BEFORE the request is transmitted, so a worker that
  stops reading its stdin cannot stall the write unboundedly — it is killed
  at the deadline and the failure is reported as a typed `WorkerTimeout`
  ("the request was never accepted") carrying every preserved frame plus the
  worker's stderr tail (drained from a temp file — an undrained stderr PIPE
  both deadlocks a chatty worker and throws the crash diagnostic away). The
  Docker build context copies `build_support.py` (the single root-level module
  `pyproject.toml` declares under `[tool.setuptools]`) AND every file in
  `src/asea/_package_resources.py::AUDIT_ORIGINS` the build hook snapshots
  into the wheel (`README.md`, `docs/SPECIALIST_WORKFLOW.md`, the specialist
  experiment script, the four audit test files) — the hook fails the build if
  any input is missing, so `pip install /silt` inside the worker image
  succeeds with the full audit trail embedded. A clean image build was
  verified on this machine on 2026-09-18 (wheel built inside the image with
  every audit snapshot included; image tagged `silt-glm53-worker`).
* Fresh dataset builder (`dataset.py`, CLI `dataset build --spec`): five splits
  (training/development/heldout/final/controls), per-case unique ID, content
  hash, family ID, provenance, license; cross-split ID/content/family
  disjointness; a NEW near-duplicate token-shape guard (the existing repo only
  guards exact duplicates); the frozen September sets are quarantined as
  inputs; manifest + selection lock frozen before any model sees a case. The
  spec is REQUIRED: the manifest binds capability ID, teacher provider/model/
  revision/access and the spec fingerprint, and `dataset validate` refuses a
  manifest whose identity fields are missing or altered. Review round 2:
  `dataset validate` also REQUIRES the `selection-lock.json` and verifies the
  manifest's sha256 against it — a dataset directory without its lock was
  never frozen by a build (or the lock was removed), and a manifest edited
  after the dataset was frozen no longer matches the lock; both are refusals,
  no matter how internally consistent the remaining files look.
* Student baselines: LOCAL connectors only — the student host URL is PARSED
  and its actual hostname validated (`localhost`, `127.0.0.1`, `::1`,
  `0.0.0.0` only; non-http schemes, userinfo, query strings, fragments,
  non-root paths and lookalike hosts such as
  `http://localhost.example.invalid:11434` are typed `BLOCKED_RESOURCE`
  refusals; and redirects are refused at the TRANSPORT level (review round 2:
  the shared `urllib` opener rejects 3xx instead of following them, tested
  through the actual connector — a redirecting local daemon can never smuggle
  a request off-host, and the redirect target is provably never contacted));
  no remote student path exists. The
  measured gap `Gap = TeacherScore − StudentScore` is the only justification
  for any transfer work.
* Sequence-level KD pair builder + DeepApply hand-off: `distill` REQUIRES a
  built dataset (`--dataset`), verifies the dataset's identity against the
  spec (capability ID, teacher model + revision, spec fingerprint), builds
  the protected sets (protected sample IDs, content hashes, prompt hashes and
  families from every non-training split) and refuses any trace outside the
  training split. Review round 2 closes the content gap: a training sample ID
  is an IDENTITY CLAIM, not a free pass — each behavioural trace's prompt must
  equal its approved training row byte for byte (a trace filed under a
  training ID with a rewritten prompt was never approved teaching material),
  and the pair builder additionally enforces protected PROMPT TOKEN SHAPES, so
  a protected prompt rewritten only in capitalisation or punctuation still
  poisons the whole set. Pairs come only from host-judged successful traces
  (failures
  kept as labelled negatives; UNJUDGED traces contribute nothing); a leakage
  collision — protected family, protected prompt or protected content —
  poisons the whole set; training itself is DeepApply's (LoRA via Gate 2) —
  the trainer never certifies itself. Cross-family (GLM→Qwen) pairs are
  text-only: no vocabulary or logit alignment is assumed.
* Minimum-capability search (`search.py`, library surface): the objective
  `min Size(M)` s.t. `TargetScore(M) ≥ retention_ratio × teacher` and control
  regression ≤ tolerance. The search enforces its own advertised contract:
  measured `teacher_control_baselines` are REQUIRED (a control regression
  against an assumed 1.0 baseline is a fabrication and the search refuses to
  run on one), every candidate's target score, control results and
  `size_bytes` must be MEASURED and FINITE (review round 2: a NaN control
  score passes every regression comparison — i.e. it showed "zero
  regression" — and an infinite target score satisfies any threshold, so all
  non-finite teacher scores, baselines, candidate targets and control scores
  are refused before any arithmetic; an unmeasured control result, a missing
  or non-positive size, or a size over
  `hardware_budget.max_model_storage_bytes` is a recorded rejection), the
  ENTIRE candidate plan is evaluated (never
  stops at the first passing candidate), and the smallest passing candidate
  by measured `size_bytes` is selected. Every iteration is recorded including
  failures and explicit rejection reasons; a parameter decrease is never
  itself evidence. Candidates are `CANDIDATE_UNADMITTED` by construction.
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

## Judged capability pilot (recorded 2026-09-18; real end-to-end)

The full measured chain, every stage a real command in this repository tree
(`experiments/glm53_flash_pilot/`; evidence committed alongside):

* **Spec + frozen dataset.** `python_repo_debugging_v1` spec pinned to the
  exact teacher revision; 19 authored cases / 54 oracle checks across five
  splits (training 6, development 3, heldout 3, final 3, controls 4), CC0,
  self-verified before any model ran (`authoring_selfcheck.py`: every seeded
  bug is real — the buggy function fails ≥1 check — and every check set is
  satisfiable by a reference implementation, args-arity asserted). Built
  with `dataset build` into `data/capability_v1/` with selection-lock frozen
  before model generation. **The final split was never traced, never judged,
  never used — `final_opened` stays false for this pilot.**
* **Teacher traces.** 16 non-final cases asked of `glm-5.3-flash:cloud`
  (per-run `--allow-remote` consent), stored UNJUDGED by design; behavioural
  footprint built with zero internal-component claims.
* **Host-oracle judgment (both models, identical checks).** Judged through
  the real Linux code sandbox (`silt-capability evaluate` in WSL; the sandbox
  fails closed on native Windows). Strict deterministic extraction: last
  fenced ```python block, else the whole response when it is itself a
  def-containing source, else `no_code_block` failure — never repaired,
  never re-asked. Measured check pass rates:

  | Split (target checks) | teacher `glm-5.3-flash:cloud` | student `qwen2.5:0.5b` |
  |---|---|---|
  | training | 1.0000 (19/19) | 0.5263 |
  | development | 0.7778 | 0.4444 |
  | heldout | 0.6667 | 0.3333 |
  | controls (utility writing) | 1.0000 | 0.7500 |

  Two genuine teacher failures stayed visible: `capitalize_words_v1`
  answered in reasoning prose with no code block (0/3), and `rotate_list_v1`
  returned the same buggy left-rotation (1/3). The measured gap on
  identical target checks is 0.8649 − 0.4595 = **0.4054** — a measurement of
  this case set only, not a quality claim about any model.
* **Distill.** The 6 judged training-split traces (all success) became
  sequence-level KD pairs — 6 positives, 0 negatives — bound byte-exactly to
  the frozen dataset, protected splits enforced, DeepApply handoff descriptor
  emitted (`autoactivated: false`; Gate 2 is the intake; the trainer never
  certifies itself). No DeepApply training was run, so capability retention
  is `NOT_MEASURED`.
* **SiltSpring compression proof (REAL).** `certify_hf_states` — real
  per-layer int8/int4/int2 quantization, one layer resident at a time,
  full-precision re-expand on exit — on `Qwen/Qwen2.5-0.5B-Instruct`
  (`7ae5576…`, 494M parameters; the base-0.5B snapshot in the shared cache
  has no weight files), with certification suites built from the pilot's own
  heldout / development / controls prompts (final never touched):
  int8 packed 359,325,696 bytes (certified development; **revoked** heldout
  and control at 2.1–4.3% loss degradation), int4 180,412,416 (all three
  certified), int2 90,955,776 (all revoked, 117–148%). Weights proved
  read-only: sha256 over every decoder-layer tensor before certification
  equals after, byte-exact. **The int4 negative degradation was investigated
  before any claim** (`diag_quant.py`): the in-place round-trip path with no
  streamer reproduces the streamed losses to 4 decimals, so it is a real
  property of loss-as-proxy on template-less prompts (coarse int4 zeroes
  ~25% of layer weights), not a pipeline defect — and it is exactly why the
  oracle, not suite loss, is SILT's truth signal. No "compression improves
  the model" claim is made anywhere.
* **Receipt.** Signed and stored through the real CLI (`receipt` +
  `receipt-verify`): valid HMAC, unmeasured fields materialised as
  `NOT_MEASURED`, failure history visible. Local tamper-evidence only — not a
  portable attestation and not a quality certificate.
* **A real run caught a real production bug.** `teacher.py`'s lazy import
  used three dots (`from ...modules.real.ollama`), which escapes the
  top-level package and raises `ImportError` at CALL time — the unit tests
  mock the transport, so only this real network run exposed it. Fixed to two
  dots with a live-loopback regression test that starts a real HTTP server
  and exercises health + chat through the real transport (verified to fail
  on the old code).

The open-weight teacher path on this laptop reports `BLOCKED_RESOURCE`
honestly: the worker's memory preflight cannot admit a ~320B-parameter teacher
on this hardware, and no worker RUN has been executed here — both are the
correct outcomes, never silently skipped. (The worker IMAGE itself has been
built cleanly on this machine, 2026-09-18: `docker build` of
`workers/glm53/Dockerfile` completed with the wheel — including every
`AUDIT_ORIGINS` audit snapshot — built and installed inside the image. A
verified image build is not a verified worker run, and no checkpoint has ever
been mounted here.)

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

`tests/test_capability_build.py` — 78 tests: schema validation, evidence-class
separation, consent gating (no network without per-run consent), enrichment
scoring with correlation limitation, intervention protocol (mask-measure-
restore-verify on a fake adapter; unrestorable never causal), store/receipt
integrity + tamper detection, worker IPC crash/blocked frames and deadline
enforcement (a hanging fake worker is killed at the deadline with a typed
`WorkerTimeout` carrying preserved frames + the drained stderr tail; the
deadline also covers REQUEST TRANSMISSION — a worker that never reads stdin
cannot stall the write, and the request is reported as never accepted), dataset
identity binding, leakage/near-duplicate/quarantine guards (including the
selection lock: a dataset without `selection-lock.json`, and a manifest edited
after freezing, are both refusals), distill dataset/training-split/protected-
set enforcement (including trace-content binding: a trace filed under a
training ID whose prompt does not equal the frozen training row byte for byte
is refused, and a protected prompt rewritten only in capitalisation or
punctuation still poisons the set through the token-shape guard), student
host URL-parse validation (lookalike hosts, userinfo, query/fragment,
non-http schemes all refused), transport-level redirect refusal through the
ACTUAL connector (a 302 from a loopback server is refused with the target
named and the redirect target provably never contacted), router-output-hook
masking on both a fake GLM stack and a real tiny
Switch model (a negative-sum hidden state — where the old write-−30-rows
mechanism would have produced a POSITIVE masked-expert logit — proves which
expert was suppressed and that restoration succeeds), student gap bookkeeping,
KD pair leakage poisoning, search contract enforcement (measured baselines,
full-plan evaluation, smallest-passing selection, size/budget/measurement
rejections, and FINITENESS: NaN control scores, infinite target scores and
non-finite teacher scores/baselines are all refused before arithmetic),
CLI exit codes, and two REAL Linux-sandbox oracle integration
tests (they probe for Linux containment and skip where it is unavailable —
GitHub CI containers and Windows both honestly skip; the skip is a probe
result, not an assumption). Every test is a MECHANISM test unless its docstring
says REAL; no test asserts anything about any teacher's behaviour.

Test counts are per-environment facts, not project claims: at commit `a37c2cf`
the CI Python-3.12 log recorded 2,791 passed / 100 skipped for the full suite
(including the two REAL oracle tests skipped there); a local WSL run of the
same tree recorded 2,872 passed / 19 skipped. Neither number is a guarantee
about any other machine.

## Commands

```
silt-capability spec validate --spec spec.json
silt-capability dataset build --spec spec.json --cases cases.jsonl --out data/capability_v1
silt-capability dataset validate --dir data/capability_v1
silt-capability teacher-baseline --spec spec.json --cases cases.json [--allow-remote]
silt-capability trace --spec spec.json --cases cases.json [--mode behavioural|internal] [--allow-remote]
silt-capability footprint --spec spec.json
silt-capability intervene --spec spec.json --component expert:3/7 [--checkpoint DIR]
silt-capability evaluate --spec spec.json --cases cases.json --source candidate.py
silt-capability student-baseline --spec spec.json --cases cases.json [--model qwen2.5-coder:0.5b]
silt-capability distill --spec spec.json --dataset data/capability_v1
silt-capability receipt --receipt receipt.json
silt-capability receipt-verify --receipt receipt.json
```

All artifacts land under a workspace (default `.capability-build/`) with
content hashes; receipts are signed against the workspace-local key.