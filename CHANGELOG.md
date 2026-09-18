# Changelog

All notable changes to SILT (Skill Interchange Layer with Trust-gating) are
recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Older in-repo audit docs (`docs/audit_2026-08-13.md`, `docs/deep_apply_real_run_findings.md`)
quote test counts from the day they were written (238 passed, 279 passed). Those
are historical snapshots. CI counts describe their own run and environment — see
the CI badge in the README. The September experimental evidence separately records
the verified V5 prepublication snapshot; it is not an all-machine CI guarantee.

## Capability-build review fixes — 2026-09-18

An external review of the 2026-09-17 capability-build layer found five
defects that the then-green tests did not catch. All five are fixed, each with
new tests that fail on the old behaviour. No frozen result, Gate or existing
mechanism was weakened; the module suite grew from 57 to 71 tests.

### Fixed
- **Student "local-only" check accepted remote hosts.** The check matched a
  string prefix, so `http://localhost.example.invalid:11434` passed. The
  student host URL is now parsed and its actual hostname validated (literal
  loopback only; non-http schemes, userinfo, query strings, fragments,
  non-root paths and lookalike hosts are typed refusals; redirects are not
  followed).
- **Protected splits were not connected to distillation.** `distill` now
  REQUIRES a built dataset (`--dataset`), verifies the dataset's identity
  against the spec (capability ID, teacher model + revision, spec
  fingerprint), builds protected sets — sample IDs, content hashes, prompt
  hashes and families — from every non-training split, and refuses any trace
  outside the training split. `dataset build` requires `--spec` and binds the
  identity into the manifest; `dataset validate` refuses an altered identity.
- **Search did not enforce its advertised contract.** `search_with_reference`
  now REQUIRES measured `teacher_control_baselines` (refuses an empty or
  assumed-1.0 baseline set as a fabrication), requires a measured target
  score, measured control results and a positive integer `size_bytes` for
  every candidate, enforces `hardware_budget.max_model_storage_bytes`,
  evaluates the ENTIRE candidate plan (never stops at the first passing
  candidate) and selects the smallest passing candidate by measured size;
  rejects carry explicit reasons.
- **The GLM expert mask was mathematically unreliable.** Writing −30 into
  router weight rows makes the masked expert's logit `−30·Σ(hidden)` —
  strongly POSITIVE for negative-sum hidden states, i.e. masking made the
  expert MORE likely to be routed. Masking is now a router-output forward
  hook that sets the masked expert's logit to −1e9 on the cloned router
  output (no weight is written; both the GLM and Switch adapters). Tests use
  a negative-sum hidden state — where the old mechanism demonstrably produced
  a positive logit — and prove which expert was suppressed and that
  restoration succeeds.
- **The isolated worker had software blockers.** The worker Dockerfile
  copied a `build_support/` directory that does not exist (`build_support.py`
  is a single root-level module), so `pip install /silt` inside the image
  failed; the worker client accepted a `timeout` it never enforced and read
  with an unbounded blocking `readline()`; stderr was never drained. Fixed:
  the Docker build context ships `build_support.py`, request deadlines are
  enforced by a watchdog that kills the worker process group at the deadline
  (typed `WorkerTimeout` carrying preserved frames + the drained stderr
  tail, read from a temp file instead of an undrained PIPE).
- **The invention disclosure was published to the public repo.** The
  2026-09-17 entry described `docs/INVENTION_DISCLOSURE_capability_build.md`
  as an internal document for owner/patent-counsel review, but `docs/` is
  publicly served, so the file should never have been in the repo. It has
  been removed from the tree and handed to the owner outside the repo.
  Honesty note: the file remains in the pushed git history; removing it from
  history requires a rewrite of public history, which was NOT done
  unilaterally.

## Capability-build research layer — 2026-09-17

A new research/build layer (`src/asea/capability_build/`, console script
`silt-capability`) asking an operator-defined question: what is the minimum
computational artifact that reproduces a narrowly defined capability in a much
smaller student, proven on untouched held-out tests without materially
regressing controls? It is experimental opt-in, never auto-activated, and
admits nothing — admission stays with the existing Deep-apply/Gate 2 path.
See [`docs/CAPABILITY_BUILD.md`](docs/CAPABILITY_BUILD.md).

### Added
- `CapabilitySpec` / `CapabilityTrace` / `CapabilityFootprint` /
  `CapabilityBuildReceipt` schemas; two teacher evidence classes
  (`behavioural_remote` via Ollama with per-run explicit `--allow-remote`
  consent, `internal_open_weight` via an isolated GLM worker) that are never
  mixed in one trace or footprint.
- Causal intervention protocol (temporary mask → measure → restore →
  verify-unchanged) for open-weight teachers; an unrestorable intervention is
  never recorded as causal evidence. Teacher weights stay read-only.
- Isolated GLM-5.3-Flash worker (`workers/glm53/`) with its own Dockerfile and
  pinned Transformers-5.x runtime lock — the core `transformers==4.51.3` pin
  is untouched. MoE adapter layer for Switch and GLM architectures.
- Fresh five-split dataset builder with cross-split ID/content/family
  disjointness, a near-duplicate token-shape guard, and quarantine of the
  frozen September sets as inputs; local-only student baselines; sequence-level
  KD hand-off to DeepApply (the trainer never certifies itself); a
  minimum-capability search surface (`CANDIDATE_UNADMITTED` by construction).
- `silt-capability` CLI (JSON-only stdout, typed refusals). `reduce`, `search`
  and `certify` are registered surfaces that refuse honestly with the
  scheduled phase rather than emit placeholder results.
- One executed behavioural pilot (8 authored cases against the operator-
  selected `glm-5.3-flash:cloud` connector): teacher baseline + traces stored
  UNJUDGED, footprint built with zero internal component claims, and the four
  target python repairs judged by the host oracle in the Linux sandbox.
  Judged functional outcome on four target cases only — not a capability
  certificate and not a quality claim about the teacher.
- Landing-page research section (`docs/index.html`) and README/CAPABILITIES
  registrations. `PATENT.md` is untouched; a separate invention disclosure
  (capability footprinting + causal intervention + minimum-capability
  compilation) was drafted for owner/patent-counsel review and makes no
  claim. It was originally placed in `docs/` and removed from the public tree
  on 2026-09-18 (see the review-fixes entry below) because that brief called
  for private owner/counsel review.

### Unchanged
- Pipeline, Gates 1/2, DeepApply, SiltSpring, the compiler and the packet
  distillers are reused as-is; `asea run`, `CapabilityDiff` semantics, the
  frozen September results and the consumed 16-task final set are untouched.

## Integrated release snapshot — 2026-09-12

This entry describes the combined source snapshot, not a claim of a completed
push, remote CI run or model deployment. See
[`docs/INTEGRATED_RELEASE_2026_09_12.md`](docs/INTEGRATED_RELEASE_2026_09_12.md).

### Added
- Hardware-adaptive specialist feasibility planning, native checkpoint loading,
  phase-lifetime memory/cache estimates and explicit CPU/single-NVIDIA-CUDA device
  dispatch. Linux CPU is supported and exercised; CUDA remains `GPU_UNVERIFIED`.
  Reconstruction is CPU-only; CPU staging for CUDA is not out-of-core execution.
- Header-only planning and separate authorized TRAIN/DEV profiling with unseen
  FINAL quarantine. `READY` is not approval, reservation, execution or quality.
- Bound execution recipes/configs and mandatory reviewed dataset sidecars in the
  controller; Linux worker containment, bounded capture and exclusive report
  publication with separate terminal/summary fsync acknowledgement.
- [Hardware contract](docs/HARDWARE_ADAPTIVE.md) and updated
  [experiment/device-local runbook](docs/SPECIALIST_QUALITY_EXPERIMENT.md).

### Fixed
- Earlier unmerged wrapper issues in executed recipe/config binding, process
  containment, report admission and outcome/durability classification are remedied
  in this source package. Missing acknowledgements are not claimed as success;
  arbitrary power-loss survival and escaped-descendant containment are not promised.
- Real-HF and toy Spring finite-real numeric guards reject invalid tolerance/loss
  values, malformed suite mappings and derived overflow. Toy recertification
  clears stale/partial certificates; valid finite tolerances and the existing
  full-suite/heldout limitation are unchanged.
- Streamer exit restores original banked parameters and persistent buffers with
  native dtypes, pre-offload devices and registered object ties. Live LoRA,
  nonpersistent caches, failed entry and storage failure are outside that contract.
- Studio tab-scoped navigation, mobile form layout, README table rendering and
  health/readiness language. No model load, probe or job is implied by discovery.

### Evidence and unchanged limits
- **Integrated local full-suite verification:** **2,695 passed / 18 skipped**,
  92 warnings, zero failures/errors in the pinned Python 3.9 CPU environment.
  Ten skips require GPU hardware, six are pretrained opt-ins, and two are pending
  host-enforcement implementations. This is not a clean-install or remote CI claim.
  Historical hardware **2,447 / 14** and product **2,325 / 12** results remain
  separate snapshots, not additive totals. GPU skips are not GPU passes.
- Separate pretrained Linux CPU engineering smoke: **two optimizer steps, one
  TRAIN row, one supervised DEV row, two calibration samples, 32-token generation
  cap**, strict reload and fresh inference. No final consumed; no 3B quality result.
- Historical consumed final remains **source 12/16; recovered 8/16**; recovery did
  not improve the aggregate. No case-retuning, score change or new quality claim.
- Wrapper baseline remains `BASELINE_NOT_RUN`; final needs an explicit reviewed
  engineering-only waiver and separate exercised deployment receipt, not automatic
  four-arm quality approval. Native Windows, macOS/MPS, ROCm/HIP, multi-GPU and
  offload remain unsupported for the adaptive specialist path.
- No model weights included in the public source repository; no model or endpoint
  automatically activated. Existing core gates/defaults and legal notices remain
  unchanged; no new patent coverage is claimed.

## Experimental reconstruction/recovery checkpoint — 2026-09-09

This entry records local frozen-V5 evidence, not certified model quality or a
claim that models have been publicly deployed. See
[`docs/EXPERIMENTAL_RELEASE_2026_09.md`](docs/EXPERIMENTAL_RELEASE_2026_09.md).

### Added
- Optional real source-weight calibration, coherent Qwen MLP reconstruction,
  native Switch-to-dense-T5 conversion, teacher-guided recovery, factor-preserving
  teacher-independent export, strict registered loader and once-only final
  comparison. Existing SILT transfer mechanisms and defaults remain alongside it.
- Qwen evidence: **494,032,768 → 415,586,176 base + 1,413,120 factors =
  416,999,296 parameters**, with **64 real optimizer steps**.
- Switch evidence: **619,339,008 → 224,525,568 base-plus-factor parameters**,
  with **32 real recovery steps**; synthetic-span mechanics, not coding quality.
- Bounded public report with six final arms, actual runtime/verification scope,
  a Git-independent frozen-source digest and read-only loss/win interpretation.

### Recorded results and limits
- Final local 16-task totals: **source 12; activation-unrepaired 8; recovered 8;
  random-MLP control 0; uniform-unrepaired control 0; SmolLM2-360M-Instruct 8**.
  No missing or operationally blocked tasks. No recovery aggregate improvement,
  teacher-level quality preservation or compact-model advantage established.
- Prepublication V5 verification: **2,039 passed, eight expected skips and
  75 warnings**, locally recorded on its stated environment. Older counts below
  remain historical; this result is not a guarantee about later CI or platforms.
- Five Studio construction stages completed as `BUILT_UNCERTIFIED`; successful
  engineering and bounded standalone reload/inference are not quality admission.
- Failed BF16 native merge retained; factor-preserving export selected explicitly
  without relaxed tolerances. The registered SILT factor bundle is not an ordinary
  HF-root/GGUF checkpoint and needs its matching runtime.
- Read-only case analysis distinguishes semantics, API/executability and output
  termination. No case-hardcoded source fix or score change; future improvement
  requires fresh data and a new governed evaluation, not consumed-final tuning.
- LoRA and teacher-guided training are established prior art. No patent novelty
  or coverage of these additions by the existing provisional is claimed. Existing
  legal notices remain unchanged; model weights are separate from source.

## [0.1.0] — 2026-08-25

First public release of the SILT core. Indian provisional patent application
**No. 202631101454** filed 2026-08-21 (ref `TEMP/E-1/111242/2026-KOL`,
assignee Uni Guru Technologies LLP). See `PATENT.md`.

### Added
- Public repository on `github.com/inbharatai/SILT`.
- `LICENSE` (Apache-2.0 full text), `PATENT.md` (consolidated IP notice,
  9 inventive families), `CITATION.cff`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `CONTRIBUTING.md`, this `CHANGELOG.md`, and `.github/` PR + issue templates.
- GitHub Actions CI (`.github/workflows/ci.yml`): pytest matrix on Python
  3.9 / 3.11 / 3.12, offline (real-weight tests skip via `ASEA_RUN_REAL` unset),
  plus a sanity job that asserts no `bypass` parameter was sneaked onto a gate.
- `[project.urls]` and PyPI classifiers in `pyproject.toml`; `MANIFEST.in` for
  sdist inclusion of legal + docs + data.
- README: world-class hero with badges, the public tagline, a prominent patent
  callout (application number `202631101454`), quick-links, and a Mermaid
  architecture diagram alongside the existing ASCII one.
- `docs/teaser.html` — the public brand teaser, cleaned of hosting-injection
  scripts, with the verified patent number and GitHub links; kicker corrected
  to "One principle, six gates" to match the six feature cards.

### Changed
- Test-count claims reconciled to the CI truth: **420+ passing offline**
  (with `[dev,studio]` extras; the count wobbles by one due to an
  order-sensitive test, so the live CI badge is the source of truth). The
  earlier "419 / 8" figure in the README was stale.
- ~27 scattered "pre-patent, confidential, local only" posture markers across
  source docstrings, scripts, and the Studio UI were updated to
  "patent pending (India)" now that the provisional is filed and the code is
  public. The architectural guarantees they carried ("local only",
  "never uploaded", "B1b portable attestation is not built") are preserved
  unchanged — only the publication-confidentiality posture changed.
- README redesigned: coloured GitHub callout blocks (patent NOTE, measured-
  admission IMPORTANT, river-silt TIP), a six-guarantees emoji table, larger
  logo. Prose uses "training" (not "learning") for the AI-acquisition concept;
  the `LearningLevel` enum and `applicable_learning_level` field keep their
  code names. `docs/teaser.html` tagline/body aligned to "training" for
  consistency.
- README leads with an "innovation portfolio" table directly under the patent
  notice — the eight mechanisms SILT claims as novel (double gate, trainer-
  independent admission, SiltStream parity-gated streaming, ZeroForge,
  SiltSpring, asymmetric SPRT, signed capability diff, verified unlearning)
  plus honest refusal, each linked to its proof file/test/doc with on-repo
  honest limits. "Get started" moved up so it stays within the first screens;
  the six-guarantees table now sits after the portfolio and de-duplicates the
  SPRT mention. No honesty artifact (mock warnings, "no % of knowledge
  transferred", "what it is not", risk-report pointers) was removed.

### Fixed
- The "deliberately not built until the patent is filed" wording on the B1b
  portable asymmetric attestation is reworded to "out of scope of this
  release" to reflect that the provisional is now filed.
- **CI green without the heavy `[deep]` extras.** The test job now invokes
  `python -m pytest -q` (not bare `pytest`) so the `tests` package is
  importable for cross-test fixtures (`tests.conftest`, `tests.test_promotion_gate`).
  Torch-dependent modules (`test_siltspring_certification`, `test_streamed_backend`)
  and the torch-reached Studio spring test skip via `pytest.importorskip("torch")`
  — the same pattern `test_studio` already uses for `fastapi` — so
  `pip install -e ".[dev,studio]"` + `pytest` is green without installing torch.
  Verified: 397 passed / 8 skipped with torch absent (the CI state); 420 passed
  / 8 skipped with torch present (no regression).
- `HFSeq2SeqTranslator.infer` now validates the `src->tgt` language tag **before**
  loading the torch backend, so the bad-capability rejection path is torch-free
  and fails fast (a caller bug shouldn't need the model on the host to reject).

### Known limitations (unchanged, by design)
- B1b portable asymmetric third-party attestation is not built; Capability
  Diff and Verified Unlearning use a local HMAC key that never leaves the host.
- The audit log is tamper-**evident**, not tamper-**proof**.
- SILT is a trust layer, not a trainer (in core), not AGI, not weight copying.
  See `docs/feasibility_review.md`.