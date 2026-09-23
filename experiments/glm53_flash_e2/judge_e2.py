"""E2 judge driver for the sixteen NEW teacher traces (REAL run, 2026-09-19).

Same contract as the v1 pilot judge (experiments/glm53_flash_pilot/
judge_pilot.py), applied to the E2 expansion: the teacher traces recorded
by ``silt-capability trace`` are UNJUDGED by design (a teacher never
grades itself); this driver is the HOST side of that contract:

  1. reads every behavioural trace from the E2 tracing workspace (the 16
     newly authored training cases only -- the six v1 training traces are
     already receipted and judged in the copied E1 artifacts),
  2. extracts the candidate source with the SAME strict rule (imported
     from the v1 judge, not re-implemented: last fenced ```python block,
     else whole def-response, else no_code_block),
  3. grades the candidate through the REAL host oracle + Linux code
     sandbox (``silt-capability evaluate`` run in WSL; the same oracle
     refuses honestly on native Windows),
  4. writes JUDGED traces (same prompt and response bytes, judged
     outcome) into the E2 distillation workspace, whose store already
     contains the six receipted v1 judged traces,
  5. writes ``judgment-teacher-e2-new.json`` with every verdict under
     AUTHORED-denominator accounting (2026-09-19 correction): checks the
     oracle could not judge count as FAILED, never dropped.

MECHANISM notes (binding): no response is ever repaired or re-asked; a
blocked oracle aborts the whole run honestly instead of grading locally.

Usage (Windows, repo root):
  python experiments/glm53_flash_e2/judge_e2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "glm53_flash_pilot"))
import judge_pilot  # noqa: E402
# The WSL siltvenv resolves ``asea`` from a STALE tree at ~/SILT (observed at
# commit ec5777e, whose evaluate result predates the 2026-09-19 authored-
# denominator correction and lacks the ``status`` field the shared judge
# reads). Pin the oracle to THIS checkout's src via the venv-activation
# prefix -- the same PYTHONPATH isolation the intervention loop uses. The
# frozen v1 judge file is not modified; only the E2 run's invocation is.
judge_pilot.WSL_VENV = (
    "source ~/siltvenv/bin/activate && "
    "export PYTHONPATH=/mnt/c/Users/reetu/Desktop/SILT/src"
)
from judge_pilot import (  # noqa: E402
    CAP, REVISION, extract_candidate, judge_source,
)

CASES = {
    c["sample_id"]: c
    for c in (
        json.loads(line)
        for line in (HERE / "cases-new-training.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    )
}
TRACES_DIR = HERE / "workspace" / "traces"


def main() -> int:
    rows = []
    for path in sorted(TRACES_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        sid = raw["sample_id"]
        case = CASES.get(sid)
        if case is None:
            raise SystemExit(
                "trace %s is not one of the sixteen new authored cases -- "
                "the E2 expansion traces exactly the new training rows"
                % sid)
        response = raw["behavioural"]["response"]
        source, how = extract_candidate(response)
        if source is None:
            rows.append({
                "sample_id": sid, "split": case["split"],
                "group": case["group"],
                "passed": 0, "total": len(case["oracle"]),
                "judged_cases": 0, "unjudged_checks": len(case["oracle"]),
                "per_check": {}, "extraction": how,
                "candidate_sha256": None,
                "behavioural": raw["behavioural"],
                "oracle_status": "no_candidate",
            })
            print("judged %s: NO CANDIDATE (%s)" % (sid, how),
                  file=sys.stderr)
            continue
        verdict = judge_source(source, case)
        rows.append({
            "sample_id": sid, "split": case["split"], "group": case["group"],
            "passed": verdict["passed"], "total": verdict["total"],
            "judged_cases": verdict["judged_cases"],
            "unjudged_checks": verdict["unjudged_checks"],
            "per_check": verdict["per_check"], "extraction": how,
            "candidate_sha256": hashlib.sha256(
                source.encode("utf-8")).hexdigest(),
            "behavioural": raw["behavioural"],
            "oracle_status": verdict["oracle_status"],
            "diagnostic_event": verdict["diagnostic_event"],
            "candidate_error": verdict["candidate_error"],
        })
        print("judged %s: %d/%d checks (%s)" % (
            sid, verdict["passed"], verdict["total"], how), file=sys.stderr)

    if not rows:
        raise SystemExit("no traces found in %s" % TRACES_DIR)

    # Write judged traces for the new training rows into the E2 distill
    # workspace (new artifacts; the unjudged originals stay untouched).
    from asea.capability_build.schema import BehaviouralRecord, TraceOutcome
    from asea.capability_build.store import CapabilityStore
    from asea.capability_build.trace import make_behavioural_trace

    store = CapabilityStore(HERE / "workspace-judged")
    written = []
    for row in rows:
        behavioural = BehaviouralRecord.model_validate(row["behavioural"])
        outcome = TraceOutcome(
            success=bool(row["passed"] == row["total"] and row["total"] > 0),
            tests_passed=row["passed"],
            tests_total=row["total"],
            metrics={
                "judge": "host_oracle_function_io_linux_sandbox",
                "extraction": row["extraction"],
                "per_check": row["per_check"],
                "candidate_sha256": row["candidate_sha256"],
            },
        )
        trace = make_behavioural_trace(
            capability_id=CAP,
            sample_id=row["sample_id"],
            model_revision=REVISION,
            prompt=behavioural.prompt,
            group=row["group"],
            outcome=outcome,
            behavioural=behavioural,
        )
        name = "%s-%s" % (CAP, row["sample_id"])
        written.append(store.put(
            "traces", name, trace.model_dump(mode="json", by_alias=True)))

    positives = sum(1 for r in rows
                    if r["passed"] == r["total"] and r["total"] > 0)
    summary = {
        "run": "teacher glm-5.3-flash:cloud (behavioural_remote), E2 new "
               "training cases",
        "capability_id": CAP,
        "judge": "host_oracle_function_io_linux_sandbox",
        "extraction_rule": "last fenced ```python block; whole response if it "
                           "is itself a def-containing source; else "
                           "no_code_block failure",
        "cases": rows,
        "authored_denominator": "the graded total per case is the frozen "
                                "authored oracle count; unjudged checks "
                                "(candidate errors) count as FAILED",
        "cases_fully_passed": positives,
        "cases_total": len(rows),
        "judged_traces_written_to_distill_workspace": [
            w["path"] for w in written],
        "honesty": [
            "A pass rate on these authored cases is a measurement of THIS "
            "case set only; it is not a quality claim about the model.",
            "Implementation success is not model-quality success.",
            "The final split was never opened; no measurement exists for it.",
        ],
    }
    out = HERE / "judgment-teacher-e2-new.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "fully_passed": positives,
        "summary": str(out),
        "judged_traces": len(written),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())