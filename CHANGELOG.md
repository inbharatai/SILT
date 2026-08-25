# Changelog

All notable changes to SILT (Skill Interchange Layer with Trust-gating) are
recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Older in-repo audit docs (`docs/audit_2026-08-13.md`, `docs/deep_apply_real_run_findings.md`)
quote test counts from the day they were written (238 passed, 279 passed). Those
are historical snapshots. From this release onward the canonical count is the one
CI produces — see the CI badge in the README.

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