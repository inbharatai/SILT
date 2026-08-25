# Patent & Intellectual Property Notice

> **Status: Patent pending (India).** This project is the subject of an Indian
> provisional patent application. Public disclosure of this source code is made
> **after** the provisional filing date, and does not prejudice novelty.

## Filing

| Field | Value |
|---|---|
| **Application number** | `202631101454` |
| **Reference number** | `TEMP/E-1/111242/2026-KOL` |
| **Docket number** | 25537 |
| **CBR number** | 12000 |
| **Filing type** | Provisional application (FORM 1) |
| **Filing date** | 2026-08-21 (21:34:53 IST) |
| **Jurisdiction** | India — Office of the Controller General of Patents, Designs & Trade Marks |
| **Applicant / inventor** | Reeturaj Goswami |
| **Assignee** | Uni Guru Technologies LLP |
| **Title** | Trust-Gated Skill Packet Transfer and Hardware-Aware Adaptation Across Heterogeneous Artificial Intelligence Systems |

A 12-month priority window runs from the filing date during which a complete
application may be filed claiming this provisional's priority.

## What is claimed as inventive

The following inventive families are reflected in this codebase. They are
stated here so the public record of what SILT claims as novel is unambiguous.
Each maps to concrete code in this repository.

1. **Trust-gated skill packet transfer.** A capability moves between two AI
   modules as an *inspectable packet* (glossary / lexicon / rules / exemplars),
   admitted to the receiver only after it proves itself on a held-out split the
   teacher never saw. The receiver *conditions on* the packet at inference; no
   weights are copied. (`src/asea/core/protocol.py`, `core/pipeline.py`,
   `distill/`, `modules/real/prompting.py`)

2. **All-or-nothing promotion gate (Gate 1).** Up to 16 independent checks;
   a single hard failure rejects with named reasons. No aggregate score can
   drown out one check. There is no `bypass` argument — the only way to act on a
   refusal is to catch `PromotionBlocked`. (`src/asea/promotion/gate.py`)

3. **Structural (non-bypassable) human sign-off for high-risk domains.** For
   medical, legal and finance domains the gate emits `PENDING_HUMAN`
   regardless of scores. The `human_approval` check is **not configurable**; a
   maximally permissive policy still parks a medical packet. The only way past
   is a named human re-running the full gate. (`src/asea/promotion/gate.py`,
   `tests/test_promotion_gate.py`)

4. **Asymmetric SPRT early-stop.** A sequential probability-ratio test may
   **early-reject** at 95% confidence but may **never** early-promote —
   rejection is allowed to be cheap, admission never is. (`src/asea/sprt.py`,
   `tests/test_sprt.py`)

5. **Tamper-evident audit + per-skill rollback.** Every extraction, drop,
   verdict, approval and rollback appends to a hash-chained, append-only,
   thread-locked, fsync'd log; tampering is detected at the exact entry. A
   rollback snapshot is taken *before* the gate, bound to a per-skill rollback
   token. (`src/asea/audit/logger.py`, `memory/store.py`)

6. **Layer-streamed low-VRAM training with verified bitwise parity.** A LoRA
   adapter is trained on consumer hardware by streaming decoder layers one at
   time, keeping the frozen base in host RAM. The streamed result is verified
   **bit-for-bit identical** to the resident (expensive) execution. *The
   streaming technique itself is third-party published work (Soup,
   Apache-2.0 — see `NOTICE`); what is claimed here is the trust-gating of it.*
   (`src/asea/deepapply/backends/siltstream_vendor/`, `tests/test_streamed_backend.py`)

7. **Zero-backward-pass training (zeroforge).** A zeroth-order (SPSA /
   MeZO-spirit) trainer achieves `backward_passes == 0` — a forward-only mode
   that trains where backpropagation physically cannot go (no GPU).
   (`src/asea/deepapply/backends/`, `tests/test_deep_apply.py`)

8. **Per-skill, per-state compression certification (SiltSpring).** Compressing
   a model damages specific skills unevenly. SiltSpring quantizes to
   int8/int4/int2 and certifies each (state, skill) pair against held-out
   suites, **revoking** a state for a skill on degradation, with certificate
   staleness bound to the LoRA fingerprint. It refuses to serve a skill its
   current state lost. (`src/asea/spring/`, `tests/test_siltspring_certification.py`)

9. **Signed Capability Diff and Verified Unlearning.** A locally HMAC-signed
   diff of what a transfer added, and a signed `ErasureCertificate` of what an
   unlearning removed — both honest that they are holder-only attestations,
   not portable third-party proofs. (`src/asea/capability_diff.py`,
   `src/asea/unlearning.py`, `src/asea/_signing.py`)

### Hardware-aware technical-effect framing

The application is framed as a **technical-effect** invention (India §3(k)-aware):
the invention produces a concrete hardware effect — enabling trust-gated skill
transfer, training and certified compression on **consumer-grade, CPU-only or
low-VRAM hardware** (e.g. a 7B-parameter model 4-bit on a consumer laptop GPU,
or forward-only training with no GPU at all) — rather than a business method or
marketplace arrangement. There is no marketplace language in the claims.

## What is deliberately NOT built / NOT claimed

- **B1b portable asymmetric attestation.** The signing path uses a local
  symmetric HMAC key that never leaves the host. Portable, third-party-verifiable
  asymmetric attestation is **out of scope of this release** and is not claimed.
- **No weight-level copying.** SILT does not copy a teacher's weights into a
  learner. See `docs/feasibility_review.md` for the honest scope statement.
- **No autonomous self-improvement / AGI.** See `docs/feasibility_review.md`.

## License vs. patent grant

This project is released under the **Apache License 2.0** (see `LICENSE`).
Under Apache-2.0 §3, redistribution grants an **implicit patent license** from
each contributor for the patents they hold that are necessarily infringed by
their contributions. Nothing in this notice expands or restricts that grant
beyond what Apache-2.0 provides.

## Reporting

For IP/patent questions: `hello@inbharat.ai`. For security, see `SECURITY.md`.