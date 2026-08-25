## What & why

<!-- One paragraph: what this changes and why. If it makes a previously-rejected packet promote, say so explicitly. -->

## Gate / trust impact

- [ ] No new `bypass` / `force` / `ignore` parameter on a gate.
- [ ] If a gate verdict changed, a test pins the new (correct, not convenient) behaviour.
- [ ] Human sign-off for high-risk domains is still non-configurable.

## Evidence

- [ ] `pytest -q` passes locally (offline, no `ASEA_RUN_REAL`).
- [ ] Any claimed number came from a real run, or is marked `is_mock`.
- [ ] `NOTICE` / `PATENT.md` updated if a third-party technique or patentable-novel feature was touched.

## Scope honesty

<!-- Does this over-claim? SILT is a trust layer, not a trainer / AGI / weight copier. If it edges toward that, explain why it doesn't. -->