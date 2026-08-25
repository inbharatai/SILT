# SILT — Verification prompt for any coding agent

Paste everything below the line into any coding agent with shell access
(Codex, Claude Code, an Ollama-hosted model with tools, etc.), with the
`adaptive-skill-extraction-adapter/` folder present in its workspace.

---

You are a rigorous software test engineer. Your job is to verify a project
called SILT (Skill Interchange Layer with Trust-gating), located in the folder
adaptive-skill-extraction-adapter/ in the current workspace. If that folder is
not present, STOP and say so — do not recreate the project from imagination.

===========================================================================
1. WHAT THIS PROJECT IS (context you must understand before testing)
===========================================================================
SILT is a universal adapter that connects two AI modules — one Sender/Teacher,
one Receiver/Learner — and moves distilled, inspectable "skill packets"
between them ONLY after evaluation proves they help. It is domain-general:
the same core has moved Assamese translation skill (NLLB → Qwen), and medical
triage skill (Qwen → SmolLM2), with no core changes.

Pipeline stages, in order:
  handshake → gap negotiation → extraction → relevance filter → safety filter
  → distillation → held-out evaluation + regression sweep → snapshot
  → promotion gate (14 checks) → separated storage → hash-chained audit log

Non-negotiable invariants the code enforces (and you will verify):
  I1. Raw sender output NEVER reaches the receiver; only distilled_skill does.
  I2. Candidate and approved packets live in physically separate directories;
      receivers read only approved/.
  I3. Medical/legal/finance packets can NEVER auto-promote — a named human
      approver is required, and no policy configuration can disable this.
  I4. Mock-derived packets are rejected by the default strict policy.
  I5. Every state change appends to a hash-chained audit log; tampering with
      any historical line is detected.
  I6. Every promotion has a rollback snapshot taken BEFORE it happens.
  I7. Learning levels L4/L5 (LoRA / distillation datasets) are export-only:
      nothing in this codebase trains a model, ever.
  I8. Identical content cannot be approved twice for the same receiver.
  I9. Instruction-shaped strings in a payload (prompt injection) are blocked
      by the safety filter.
  I10. The core pipeline contains NO branching on modality — universality is
      asserted by tests, not by comments.

Honesty rules of the project (respect them in your report):
  - The four flow_a..flow_d examples use MOCKS (lookup tables) and exist to
    demonstrate plumbing. Their improvement numbers say NOTHING about model
    quality. They deliberately disable the anti-mock gate check to run.
  - flow_real_assamese.py and flow_real_medical.py use REAL model weights.
  - The bundled lexical similarity metric is a proxy, not semantic; the
    embedding backend (ASEA_SIMILARITY=embedding) is the honest option.
  - The Python package imports as `asea`; SILT is the brand name.

===========================================================================
2. ENVIRONMENT SETUP
===========================================================================
Requires Python 3.9+. From the project root:

    python3 -m pip install -r requirements.txt      # pydantic + pytest only

===========================================================================
3. TIER 0 — OFFLINE VERIFICATION (always run this; ~2 minutes, no network)
===========================================================================
3a. Full test suite:

    python3 -m pytest tests/ -q

    EXPECTED: all pass; 4 skipped (the real-weight tests, opt-in via
    ASEA_RUN_REAL=1). The exact pass count grows with the suite, so do not
    hard-assert a number — assert "0 failed, 4 skipped". ANY failure is a
    finding — report the test name and full failure output verbatim.

3b. Mock pipeline demos:

    cd examples && python3 run_all.py && cd ..

    EXPECTED: four flows complete. Flow A shows two relevance drops
    (one "receiver_competent", one "sender_incorrect"), Hindi regression
    stays 1.0000 → 1.0000, and packets promote. Flow B refuses an ASR→TTS
    binding at handshake ("no shared modality"). Flow D parks its packet in
    PENDING_HUMAN and promotes only after a named approver acts.
    A final line reminds you every module was a MOCK — that is intentional.

3c. Invariant spot-checks (write and run a short script for each):
    - Construct a SkillPacket with promotion_status=PROMOTED but no
      rollback_token: pydantic must raise a ValidationError.        (I1/I6)
    - Call redacted_for_receiver() on a packet whose sender_output is a
      marker string: the marker must be absent from the result.      (I1)
    - Build a PromotionGate with a maximally permissive PromotionPolicy and
      apply it to a Domain.MEDICAL packet with perfect scores and no
      approver: status must be PENDING_HUMAN, never PROMOTED.        (I3)
    - Append 3 entries to an AuditLog, manually edit entry 0 in the file,
      call verify(): it must return ok=False with broken_at=0.       (I5)
    - grep the file src/asea/core/pipeline.py for "if" lines containing
      "Modality." — there must be none.                              (I10)

===========================================================================
4. TIER 1 — REAL MODEL VERIFICATION (needs internet OR a local Ollama;
   skip gracefully if neither is available, and SAY you skipped it)
===========================================================================
Option A (HuggingFace weights, ~2.5 GB download, CPU-friendly):

    pip install torch transformers sentencepiece
    cd examples
    ASEA_SIMILARITY=embedding python3 flow_real_assamese.py

    EXPECTED VERDICT (verdicts must reproduce; decimals may vary slightly):
    - the gap engine finds an actionable as->en gap,
    - several extraction signals are dropped with named reasons,
    - ONE glossary packet is distilled and PROMOTED with positive delta,
    - the Hindi regression check stays within tolerance,
    - audit chain verifies ok.

    cd examples && python3 flow_real_medical.py

    EXPECTED VERDICT:
    - Qwen2.5-0.5B (sender) → SmolLM2-360M (receiver), medical triage,
    - at least one signal dropped as sender_incorrect (the weak
      infant-fever answer),
    - the distilled rules packet lands in PENDING_HUMAN (never
      auto-promoted), then promotes after the scripted named approval,
    - the taught rule texts equal the benchmark reference texts, NOT the
      sender's verbose prose (verify by reading the printed rules).

Option B (Ollama, for machines with ≥16 GB RAM):

    ollama serve && ollama pull qwen2.5:7b-instruct
    ASEA_RECEIVER=ollama ASEA_RECEIVER_MODEL=qwen2.5:7b-instruct \
      ASEA_SIMILARITY=embedding python3 examples/flow_real_assamese.py

    This is the genuinely open question (a 7B receiver); report whatever
    the gate decides, including rejection. A rejection with named reasons
    is the system working, not a bug.

===========================================================================
5. ADVERSARIAL CHECKS (already encoded — confirm they hold)
===========================================================================
    python3 -m pytest tests/test_adversarial.py -v

    EXPECTED: all pass. Note two of them PIN accepted risks rather than
    fixes: A8 (domain mislabelling bypasses the medical gate — documented
    residual risk) and A9 homoglyphs (Cyrillic lookalikes evade dedup).
    Do not "fix" these; they are deliberately pinned behaviour.

===========================================================================
6. WHAT TO REPORT (exact format)
===========================================================================
Produce a report with these sections:
  1. Environment: OS, Python version, RAM, GPU (if any).
  2. Tier 0 results: pytest counts verbatim; per-flow one-line outcomes;
     invariant spot-check pass/fail table.
  3. Tier 1 results: which option ran, each flow's gate VERDICT
     (promoted / rejected / pending_human→approved), the delta, and the
     per-case diff table printed by the script. If skipped, say why.
  4. Discrepancies: anything that differs from the EXPECTED lines above,
     with verbatim output. A verdict mismatch matters; a 0.01 score drift
     on real models does not.
  5. Your assessment: is the gate enforcing its rules? Cite evidence only
     from what you ran.

RULES FOR YOUR REPORT:
  - Never invent numbers. Every figure must come from output you produced.
  - Do not soften failures. A failing test is a finding, not an
    embarrassment.
  - Do not claim the system "makes models smarter" — the correct claim, if
    your runs support it, is: "the adapter correctly discriminates between
    transfers that help and transfers that harm, and enforces human
    approval for high-risk domains."
  - The medical sample data is clinically unreviewed. Never present it as
    medical advice.
