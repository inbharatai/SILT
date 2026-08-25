# SILT — Deploy-and-test prompt for Claude Code

Paste everything below the line into Claude Code, with the
`adaptive-skill-extraction-adapter/` folder present in the workspace (unzip
`silt.zip` first if needed). Written for a machine that runs Ollama with a
GLM model available (works identically for any ≥7B instruct model).

---

You are deploying and testing SILT (Skill Interchange Layer with Trust-gating):
a system where one AI model teaches another through evaluated, gated "skill
packets", plus its web platform, SILT Studio. The project lives in
adaptive-skill-extraction-adapter/ in this workspace. If that folder is absent,
STOP and say so — never recreate it from imagination.

Your job has five phases. Do them in order. Never fake an output, never soften
a failure, and never claim a step succeeded without pasting the command output
that proves it.

CONTEXT YOU NEED (60 seconds of reading):
- SILT moves skills teacher→learner through: handshake → gap negotiation →
  extraction → relevance filter → safety filter → distillation → held-out
  evaluation → a promotion gate (up to 14 checks; `human_approval` fires only
  for high-risk domains) → audited storage with rollback.
- The Studio (src/asea/studio/) is a FastAPI + SSE web UI over that pipeline.
  Its catalog serves REAL connectors only; it structurally refuses mocks.
- Medical/legal/finance packets can NEVER auto-promote — a named human
  approver is mandatory, and no configuration can disable that.
- A REJECTED transfer with named gate reasons is the system working correctly,
  not a bug. Report it as a finding.
- Deep references, only if needed: README.md, LOCAL_SETUP.md,
  docs/loophole_audit.md, docs/real_run_findings.md.

===========================================================================
PHASE 0 — ENVIRONMENT AUDIT (report all of this)
===========================================================================
Run and record:
    python3 --version              # need >= 3.9
    ollama list                    # note the EXACT tag of the GLM model
    ollama ps                      # is the server running?
    free -h 2>/dev/null || vm_stat # RAM
    lsof -i :8377 2>/dev/null      # port must be free (else use 8378)

CRITICAL — GLM tag detection: do NOT assume a tag. Read `ollama list` and use
whatever GLM tag actually exists (could be glm4, glm-4.5, glm-4.6:cloud, etc.).
Cloud-hosted tags (":cloud") also need `ollama signin` to have been done.
Verify the model actually answers before proceeding:
    ollama run <GLM_TAG> "Say OK"
If no GLM tag exists, use any >=7B instruct model from `ollama list` instead,
and state the substitution prominently in your report.

===========================================================================
PHASE 1 — INSTALL + OFFLINE VERIFICATION (no models, ~3 minutes)
===========================================================================
    cd adaptive-skill-extraction-adapter
    python3 -m pip install -r requirements.txt
    python3 -m pip install fastapi uvicorn httpx
    python3 -m pytest tests/ -q

EXPECTED: all pass; 4 skipped (the opt-in real-weight tests). The exact pass
count grows with the suite, so do not hard-assert a number — assert "0 failed,
4 skipped".
ANY failure: paste the test name and full output verbatim in your report, and
do NOT silently patch the code to make it pass — a failing test is a finding.

Then run the mock pipeline demos (these use clearly-labelled mocks BY DESIGN,
to exercise the plumbing offline — their scores mean nothing about models):
    cd examples && python3 run_all.py && cd ..
EXPECTED: four flows complete; flow D parks its packet in PENDING_HUMAN and
promotes only after the scripted named approval.

===========================================================================
PHASE 2 — DEPLOY SILT STUDIO
===========================================================================
    PYTHONPATH=src python3 -m uvicorn asea.studio.server:app --port 8377 &
    sleep 5
    curl -s localhost:8377/api/health      # expect {"ok":true,...,"mock_free":true}
    curl -s localhost:8377/api/catalog     # 6 real connectors, zero mocks
    curl -s localhost:8377/api/suites      # 6 suites; medical_triage high_risk:true
    curl -s -o /dev/null -w "%{http_code}\n" localhost:8377/            # 200
    curl -s -o /dev/null -w "%{http_code}\n" localhost:8377/logo.svg    # 200
    curl -s -o /dev/null -w "%{http_code}\n" localhost:8377/favicon.svg # 200

Tell the user the UI is live at http://localhost:8377 — they can watch
everything you do next in the browser.

===========================================================================
PHASE 3 — REGISTER THE LOCAL GLM AS A RECEIVER
===========================================================================
GLM via Ollama needs NO connector code — the existing OllamaConnector speaks
Ollama's /api/chat with deterministic options. It only needs a catalog entry.
Append EXACTLY this to src/asea/studio/catalog.py, replacing <GLM_TAG> with
the tag you detected in Phase 0 (this is a catalog ADDITION — do not modify
any existing code, any core module, or any policy/threshold):

    CATALOG["glm-ollama"] = {
        "factory": _ollama("<GLM_TAG>"),
        "roles": ["sender", "receiver"],
        "description": "GLM via local Ollama (user's model, real weights)",
        "requires": "ollama serve + the tag from `ollama list`",
    }

Then:
    python3 -m pytest tests/ -q      # must STILL pass; 4 skipped (real-weight tests)
Restart the server, and confirm:
    curl -s localhost:8377/api/catalog | grep glm-ollama

Health-check the connector before any long run (fail fast, not after 10 min):
    PYTHONPATH=src python3 -c "
    from asea.studio import catalog
    print(catalog.build('glm-ollama').health())"
EXPECTED: model_present: True. If False, run the printed `ollama pull` hint.

===========================================================================
PHASE 4 — REAL TRANSFERS THROUGH THE HTTP API
===========================================================================
4a. Language transfer — NLLB teaches YOUR GLM Assamese
(NLLB-200-distilled-600M downloads ~2.5 GB on first use via transformers;
install if needed: pip install torch transformers sentencepiece)

    curl -s -X POST localhost:8377/api/transfers \
      -H "Content-Type: application/json" \
      -d '{"sender":"nllb-teacher","receiver":"glm-ollama",
           "suites":["assamese_english","hindi_english"],
           "similarity":"embedding"}'

Note the job_id. Watch the live audit stream (this is what the UI renders):
    curl -N localhost:8377/api/transfers/<job_id>/events
Expected event order: module_registered ×2, adapter_bound, session_opened,
gap_negotiated, extracted, relevance_filtered, safety_filtered, distilled,
evaluated, gate_decision, then promoted OR rejected, run_complete.

When done, GET /api/transfers/<job_id> and record VERBATIM: the counts
funnel, baseline→candidate scores, delta, the Hindi regression line, every
gate check with its value, and the final verdict.

INTERPRETATION RULES:
- "no actionable gap" is a legitimate outcome: GLM may already score ≥0.85
  on Assamese. Report the measured gap numbers; nothing transfers by design.
- PROMOTED with positive delta and flat Hindi = transfer helped. REJECTED
  with named reasons = the gate protected the model. Both are valid results.
- Your GLM is a full-size receiver — unlike the recorded 0.5B runs, skill
  consumption is genuinely plausible here. This is the interesting result.

4b. Medical transfer — the human gate, demonstrated not skipped
    POST /api/transfers with {"sender":"triage-corpus","receiver":"glm-ollama",
      "suites":["medical_triage"],"similarity":"embedding","relevance_floor":0.35}
Assert from the response report: promoted == [] (medical NEVER auto-promotes).
If pending_human is non-empty, approve with a named identity:
    POST /api/transfers/<job_id>/approve
      {"packet_id":"<id>","approver":"<the user's actual name/email>"}
Then confirm the packet moved to approved and the audit chain verifies:
    GET /api/transfers/<job_id>/packets     GET /api/transfers/<job_id>/audit

4c. Playground A/B (uses the identical inference path the evaluator measured):
    POST /api/playground twice — use_skills:false then use_skills:true — with
    module glm-ollama, prompt "মই ভাত খাওঁ", capability
    {"task_type":"translate","modality":"text","domain":"translation","language":"as->en"}
Paste both outputs side by side. skills_active must be >0 in the second call
only if 4a promoted.

4d. Rollback: take the rollback_token of an approved packet,
    POST /api/transfers/<job_id>/rollback {"token":"<token>"},
    confirm approved count drops and GET .../audit still says ok:true.

===========================================================================
PHASE 5 — REPORT (this exact structure)
===========================================================================
1. Environment: OS, Python, RAM, the GLM tag actually used, Ollama status.
2. Phase 1: pytest counts verbatim; demo flows one-liners.
3. Phase 2: each endpoint's status code.
4. Phase 3: catalog diff you made, post-edit pytest count, health() output.
5. Phase 4: per-transfer — verdict, funnel counts, baseline→candidate→delta,
   regression result, gate checks that failed (if any), playground A/B
   outputs, rollback result, audit integrity. All numbers copied from output.
6. Discrepancies vs the EXPECTED lines above (verdict mismatches matter;
   ±0.02 score drift on real models does not).
7. One-paragraph honest assessment. The only claim the evidence can support
   is of the form: "the adapter correctly discriminated between transfers
   that help and transfers that harm, and enforced human approval for the
   medical domain." Never claim a % of knowledge transferred (no such
   measurement exists), never present the medical sample data as clinical
   advice, and never report a number you did not produce.

HARD RULES THROUGHOUT:
- Real connectors only; do not touch src/asea/modules/mock/ or relax
  strict_no_mock anywhere.
- Never edit policies, thresholds (except the documented relevance_floor
  parameter in 4b's request body), gate checks, or core modules.
- If anything fails persistently, report the failure with output attached
  and stop that phase — do not improvise around the safety architecture.
