# Contributing to SILT

SILT is a trust gate, not a feature factory. Contributions that *weaken* a gate,
*add* a bypass, or *over-claim* a result will be refused even if the code is
correct. Contributions that *honestly widen* what SILT can check, *add* a
modality/connector, or *pin* a new adversarial case with a test are very welcome.

## Set up

```bash
git clone https://github.com/inbharatai/SILT.git
cd SILT
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev,studio]"                    # core + tests + Studio
pytest -q                                         # offline, deterministic
```

Real-weight tests (touching torch / transformers) are skipped unless
`ASEA_RUN_REAL=1` is set. CI never sets it — keep your PR green offline.

## The norms

1. **Record rejections.** A gate that says `REJECTED` with named reasons is the
   system working. If your change makes a previously-rejected packet promote,
   say *why* explicitly in the PR and add a test that proves the new behaviour
   is correct, not just convenient.
2. **No bypass argument.** Never add a `bypass`, `force`, or `ignore` parameter
   to `PromotionGate` or `DeepApplyGate`. The only way to act on a refusal is
   `except PromotionBlocked`. If you genuinely need to escalate, route it
   through a *named human* approval, never a flag.
3. **Honest scope.** SILT is a trust layer, not a trainer, not AGI, not weight
   copying. Don't add language that claims otherwise. The honest scope
   statement lives in `docs/feasibility_review.md`.
4. **Split discipline.** Extraction / held-out / regression splits are disjoint;
   cross-reads raise. Don't relax this — it is what makes "held-out proof" real.
5. **Prior art honesty.** If you use a third-party technique (e.g. the Soup
   layer-streaming), credit it in `NOTICE` and the module docstring, and record
   the backend name + version in every produced packet. SILT never claims
   third-party work as its own.

## Where to add things

| You want to… | Read this first | Tests next to |
|---|---|---|
| Add a modality / connector | `docs/connector_authoring.md` | `tests/test_real_connectors.py` |
| Add a domain (e.g. medical) | `docs/ADDING_A_DOMAIN.md` | `tests/test_extraction_and_filters.py` |
| Pin a new attack | `docs/loophole_audit.md` | `tests/test_adversarial.py` |
| Add a gate check | `src/asea/promotion/gate.py` | `tests/test_promotion_gate.py` |
| Extend the Studio UI | `src/asea/studio/` | `tests/test_studio.py` |

## PR checklist

- [ ] `pytest -q` passes (no real weights needed).
- [ ] No `bypass` / `force` / `ignore` gate parameter added.
- [ ] If a number is claimed, it came from a real run (or is marked `is_mock`).
- [ ] `PATENT.md` / `NOTICE` updated if you touched something patentable-novel
      or used a new third-party technique.
- [ ] No secrets, keys, or local paths committed.
- [ ] Commit message explains *what* and *why*; rejections are mentioned, not
      hidden.

## Code style

Match the surrounding code. No type-checker is enforced yet — do not add
`py.typed` or a mypy config in a feature PR without coordinating. Line length is
soft at 100; don't reflow unrelated lines.

## Licensing

By contributing you agree your contribution is licensed under Apache-2.0 and,
if it is necessarily infringed by a claim of the pending patent, is licensed
under that patent's implicit grant per Apache-2.0 §3. See `PATENT.md`.