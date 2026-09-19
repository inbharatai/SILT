"""GLM-5.3-Flash pilot judge driver (REAL run, 2026-09-18).

This is the pilot's host-judgment stage. The teacher traces recorded by
``silt-capability trace`` are UNJUDGED by design (a teacher never grades
itself); this driver is the HOST side of that contract:

  1. reads every behavioural trace from the tracing workspace,
  2. extracts the candidate source the model actually returned (the LAST
     ```python fenced block; if the response has no fenced block but is
     itself a def-containing source, the whole response is the candidate;
     otherwise the case is judged FAILED as ``no_code_block`` -- a strict,
     deterministic rule, recorded, never repaired),
  3. grades the candidate against the case's authored oracle checks through
     the REAL host oracle + Linux code sandbox (``silt-capability evaluate``
     run in WSL on this machine; the same oracle refuses honestly on native
     Windows),
  4. writes JUDGED traces (same prompt and response bytes, judged outcome)
     into a SEPARATE distillation workspace containing ONLY training-split
     samples -- the store is append-only, so a judged trace is a new
     artifact, never an edit of the unjudged one, and non-training traces
     are protected material that may never become teaching pairs,
  5. writes ``judgment-summary.json`` with every verdict, pass rate per
     split and group, and the extraction rule.

MECHANISM notes (binding): no response is ever repaired or re-asked; a
blocked oracle aborts the whole run honestly instead of grading locally;
implementation success is never model-quality success.

Usage (Windows, repo root):
  python experiments/glm53_flash_pilot/judge_pilot.py judge-teacher
  python experiments/glm53_flash_pilot/judge_pilot.py judge-student --model qwen2.5:0.5b
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAP = "python_repo_debugging_v1"
WSL_VENV = "source ~/siltvenv/bin/activate"
WSL_REPO = "/mnt/c/Users/reetu/Desktop/SILT"

CASES = {
    c["sample_id"]: c
    for c in (
        json.loads(line)
        for line in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
}
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
REVISION = SPEC["teacher"]["revision"]

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
JUDGE = "host_oracle_function_io_linux_sandbox"


def wsl_path(path: Path) -> str:
    return "/mnt/c" + str(path).replace("\\", "/")[2:]


def extract_candidate(response: str):
    """Last fenced python block, else whole response when it is itself a
    def-containing source, else None (judged no_code_block)."""
    blocks = FENCE_RE.findall(response)
    if blocks:
        return blocks[-1].strip() + "\n", "last_fenced_block"
    text = response.strip()
    if re.search(r"^def |^(\s*)def ", text, re.MULTILINE):
        return text + "\n", "whole_response_no_fence"
    return None, "no_code_block"


def run_oracle(candidate_path: Path, oracle_path: Path):
    """Invoke the real evaluate CLI inside WSL (Linux sandbox). Returns the
    parsed stdout payload; raises SystemExit on any non-clean exit."""
    cmd = (
        "%s && cd %s && python -m asea.capability_build evaluate "
        "--cases %s --source %s"
        % (WSL_VENV, WSL_REPO, wsl_path(oracle_path), wsl_path(candidate_path))
    )
    proc = subprocess.run(
        ["wsl.exe", "bash", "-c", cmd], capture_output=True, text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "oracle run FAILED for %s (exit %d)\nstdout: %s\nstderr: %s"
            % (candidate_path.stem, proc.returncode, proc.stdout[-2000:],
               proc.stderr[-2000:])
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def judge_source(source: str, case: dict):
    """Grade one candidate source against its case's oracle checks."""
    judged_dir = HERE / "judging"
    candidates = judged_dir / "candidates"
    oracles = judged_dir / "oracle"
    candidates.mkdir(parents=True, exist_ok=True)
    oracles.mkdir(parents=True, exist_ok=True)
    sid = case["sample_id"]
    candidate_path = candidates / ("%s.py" % sid)
    candidate_path.write_text(source, encoding="utf-8")
    oracle_payload = [
        {
            "sample_id": sid,
            "group": case["group"],
            "prompt": case["prompt"],
            "id": check["id"],
            "function": check["function"],
            "args": check["args"],
            "kwargs": check.get("kwargs", {}),
            "expected": check["expected"],
        }
        for check in case["oracle"]
    ]
    oracle_path = oracles / ("%s.json" % sid)
    oracle_path.write_text(json.dumps(oracle_payload, indent=1), encoding="utf-8")
    payload = run_oracle(candidate_path, oracle_path)
    result = payload["result"]
    if result.get("blocked"):
        raise SystemExit(
            "oracle BLOCKED for %s: %r -- refusing to grade locally"
            % (sid, result)
        )
    # Per-check verdicts live on the embedded oracle payload:
    # result["oracle"]["cases"] carries {'id', 'matched'} entries.
    per_check = {
        c["id"]: c.get("matched")
        for c in (result.get("oracle") or {}).get("cases", [])
    }
    # AUTHORED denominator (2026-09-19 correction): checks the oracle could
    # not judge (candidate import/call error, timeout, resource kill) count
    # as FAILED, never dropped -- the graded total is the frozen authored
    # count, and the reported judged count rides along for transparency.
    authored = len(case["oracle"])
    diagnostic = (result.get("oracle") or {}).get("diagnostic") or {}
    return {
        "passed": int(result.get("passed_cases", 0)),
        "total": authored,
        "judged_cases": int(result.get("judged_cases", 0)),
        "unjudged_checks": int(result.get("unjudged_cases", authored)),
        "per_check": per_check,
        "oracle_status": result["status"],
        "diagnostic_event": diagnostic.get("event"),
        "candidate_error": diagnostic.get("candidate_error"),
    }


def ollama_chat(model: str, prompt: str, host: str = "http://localhost:11434"):
    """One deterministic local chat turn (student models only -- LOCAL, no
    consent dimension; never used for the cloud teacher)."""
    request = urllib.request.Request(
        "%s/api/chat" % host,
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "top_p": 1, "seed": 0,
                        "num_predict": 1024},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload, round((time.monotonic() - started) * 1000.0, 3)


def summarise(rows, run_name):
    def rate(subset):
        if not subset:
            return None
        return round(
            sum(r["passed"] for r in subset) / max(1, sum(r["total"] for r in subset)),
            4,
        )

    # Accounting guard (audit 2026-09-19): the graded total per case must be
    # the frozen authored oracle count. A row that grades fewer checks than
    # the case defines (candidate-error cases reporting judged_cases=0 under
    # the old accounting) silently inflates every rate by shrinking the
    # denominator -- refuse the whole report instead of publishing it.
    for r in rows:
        if r["total"] <= 0:
            raise SystemExit(
                "case %s graded %d checks; every case defines at least one "
                "-- refusing the report" % (r.get("sample_id"), r["total"])
            )

    by = lambda key, value: [r for r in rows if r[key] == value]
    summary = {
        "run": run_name,
        "capability_id": CAP,
        "judge": JUDGE,
        "extraction_rule": "last fenced ```python block; whole response if it "
                           "is itself a def-containing source; else "
                           "no_code_block failure",
        "cases": rows,
        "pass_rates": {
            "overall_checks": rate(rows),
            "cases_fully_passed": round(
                sum(1 for r in rows if r["passed"] == r["total"] and r["total"] > 0)
                / len(rows), 4) if rows else None,
            "target_checks": rate([r for r in rows if r["group"] == "target"]),
            "control_checks": rate([r for r in rows if r["group"] == "control"]),
            "training_checks": rate(by("split", "training")),
            "development_checks": rate(by("split", "development")),
            "heldout_checks": rate(by("split", "heldout")),
            "controls_split_checks": rate(by("split", "controls")),
        },
        "honesty": [
            "A pass rate on these authored cases is a measurement of THIS "
            "case set only; it is not a quality claim about the model.",
            "Implementation success is not model-quality success.",
            "The final split was never opened; no measurement exists for it.",
        ],
    }
    return summary


def write_judged_traces(rows, workspace_name):
    """Write judged traces for TRAINING-split rows only into the distill
    workspace (new artifacts; the unjudged originals stay untouched)."""
    from asea.capability_build.schema import BehaviouralRecord, TraceOutcome
    from asea.capability_build.store import CapabilityStore
    from asea.capability_build.trace import make_behavioural_trace

    workspace = HERE / workspace_name
    store = CapabilityStore(workspace)
    written = []
    for row in rows:
        if row["split"] != "training":
            continue
        behavioural = BehaviouralRecord.model_validate(row["behavioural"])
        outcome = TraceOutcome(
            success=bool(row["passed"] == row["total"] and row["total"] > 0),
            tests_passed=row["passed"],
            tests_total=row["total"],
            metrics={
                "judge": JUDGE,
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
    return written


def judge_traced_responses():
    """Judge the teacher's recorded traces (the cloud run is already stored;
    this stage touches no network)."""
    rows = []
    traces_dir = HERE / "workspace" / "traces"
    for path in sorted(traces_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        sid = raw["sample_id"]
        case = CASES[sid]
        response = raw["behavioural"]["response"]
        source, how = extract_candidate(response)
        if source is None:
            rows.append({
                "sample_id": sid, "split": case["split"], "group": case["group"],
                "passed": 0, "total": len(case["oracle"]),
                "per_check": {}, "extraction": how,
                "candidate_sha256": None,
                "behavioural": raw["behavioural"],
                "oracle_status": "no_candidate",
            })
            continue
        verdict = judge_source(source, case)
        rows.append({
            "sample_id": sid, "split": case["split"], "group": case["group"],
            "passed": verdict["passed"], "total": verdict["total"],
            "judged_cases": verdict["judged_cases"],
            "unjudged_checks": verdict["unjudged_checks"],
            "per_check": verdict["per_check"], "extraction": how,
            "candidate_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "behavioural": raw["behavioural"],
            "oracle_status": verdict["oracle_status"],
        })
        print("judged %s: %d/%d checks (%s)" % (
            sid, verdict["passed"], verdict["total"], how), file=sys.stderr)
    return rows


def judge_student_live(model: str):
    """Ask the LOCAL student the same traced cases and judge identically --
    the measured Gap = TeacherScore - StudentScore on identical checks."""
    rows = []
    for sid, case in sorted(CASES.items()):
        if case["split"] == "final":
            continue  # never opened
        payload, wall_ms = ollama_chat(model, case["prompt"])
        content = (payload.get("message") or {}).get("content", "") or ""
        source, how = extract_candidate(content)
        if source is None:
            rows.append({
                "sample_id": sid, "split": case["split"], "group": case["group"],
                "passed": 0, "total": len(case["oracle"]),
                "per_check": {}, "extraction": how,
                "candidate_sha256": None, "latency_ms": wall_ms,
                "response_tokens": payload.get("eval_count"),
                "oracle_status": "no_candidate",
            })
            continue
        verdict = judge_source(source, case)
        rows.append({
            "sample_id": sid, "split": case["split"], "group": case["group"],
            "passed": verdict["passed"], "total": verdict["total"],
            "judged_cases": verdict["judged_cases"],
            "unjudged_checks": verdict["unjudged_checks"],
            "per_check": verdict["per_check"], "extraction": how,
            "candidate_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "latency_ms": wall_ms,
            "response_tokens": payload.get("eval_count"),
            "oracle_status": verdict["oracle_status"],
        })
        print("student %s: %d/%d checks (%s)" % (
            sid, verdict["passed"], verdict["total"], how), file=sys.stderr)
    return rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "judge-teacher":
        rows = judge_traced_responses()
        summary = summarise(rows, "teacher glm-5.3-flash:cloud (behavioural_remote)")
        written = write_judged_traces(rows, "workspace-judged")
        summary["judged_traces_written_to_distill_workspace"] = [
            w["path"] for w in written
        ]
        out = HERE / "judgment-teacher.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(json.dumps({
            "rows": len(rows),
            "pass_rates": summary["pass_rates"],
            "judged_traces": len(written),
            "summary": str(out),
        }))
    elif mode == "judge-student":
        model = "qwen2.5:0.5b"
        if "--model" in sys.argv:
            model = sys.argv[sys.argv.index("--model") + 1]
        rows = judge_student_live(model)
        summary = summarise(rows, "student %s (local)" % model)
        out = HERE / ("judgment-student-%s.json" % model.replace(":", "_"))
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(json.dumps({"model": model, "rows": len(rows),
                          "pass_rates": summary["pass_rates"],
                          "summary": str(out)}))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()