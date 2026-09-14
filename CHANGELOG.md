# Changelog

All notable changes to SILT (Skill Interchange Layer with Trust-gating) are
recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Older in-repo audit docs (`docs/audit_2026-08-13.md`, `docs/deep_apply_real_run_findings.md`)
quote test counts from the day they were written (238 passed, 279 passed). Those
are historical snapshots. CI counts describe their own run and environment — see
the CI badge in the README. The September experimental evidence separately records
the verified V5 prepublication snapshot; it is not an all-machine CI guarantee.

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