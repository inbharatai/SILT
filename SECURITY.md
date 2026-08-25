# Security Policy

SILT's whole purpose is trust: it decides whether an AI is *allowed to keep*
what it learned. Security here is therefore not an afterthought — it is the
product. This file says how to read the security posture and how to report.

## Posture at a glance

- **Local-first by design.** SILT runs on your machine. No model, packet, key or
  audit log leaves the host unless you explicitly export a bundle. There is no
  SILT-operated server in the trust path.
- **Non-bypassable gate.** `PromotionGate` has no `bypass` argument; one hard
  check failure rejects with named reasons. Human sign-off for medical / legal /
  finance is structural — no configuration can disable it. See
  `src/asea/promotion/gate.py`.
- **Tamper-evident, not tamper-proof.** The audit log is hash-chained and
  append-only; tampering is *detected* at the exact entry. It does not prevent a
  privileged attacker with filesystem access from deleting the log — it makes
  that deletion visible.
- **Local HMAC signing keys.** `Capability Diff` and `Verified Unlearning` are
  signed with a local symmetric key that **never** leaves the host. They are
  holder-only attestations, **not** portable third-party proofs (asymmetric
  portable attestation — B1b — is deliberately not built; see `PATENT.md`).
- **Threat model.** A *trusted author* runs SILT on their own hardware. The
  documented accepted risks (domain mislabelling, provenance self-report
  forgery) sit inside that model — see `risk_report.md`.

## Where the security reasoning lives

These documents are the primary security record and are kept current:

- `risk_report.md` — 10 enumerated risks, each with what the code does / does
  NOT do.
- `docs/loophole_audit.md` — 13 adversarial attacks (A1–A13); 3 patched, 2
  accepted risks under the trusted-author model; each pinned to a test in
  `tests/test_adversarial.py`.
- `docs/audit_2026-08-13.md` — the 8-dimension deep ethical audit; 44 confirmed
  findings, 32 fixed, 12 held for sign-off.
- `tests/test_adversarial.py` — the executable attack suite.

## Reporting a vulnerability

Email `hello@inbharat.ai` with `[SILT security]` in the subject. Please include:

1. A minimal reproduction (command, config, expected vs. actual).
2. The threat model you assume (e.g. trusted author on local host vs. remote).
3. Whether it weakens a gate, leaks a key, or breaks tamper-evidence.

Do **not** open a public issue for security vulnerabilities. We will
acknowledge within 5 business days and coordinate a fix + disclosure.

## Scope

In scope: anything in `src/asea/` that touches the gate, audit log, signing
keys, rollback, or the Studio HTTP surface. Out of scope: vulnerabilities in
optional third-party dependencies (torch, transformers, PEFT, FastAPI) — report
those upstream.