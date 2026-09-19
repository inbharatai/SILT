"""Correct the A/B scoring of the DeepApply training stage (2026-09-19).

An external read-only audit of main @091637e found a denominator defect in
the functional-evaluation accounting: checks the oracle could not judge
(candidate import/call errors) were reported as 0/0 and then silently
dropped from every rate's denominator, inflating the rates. Five cases were
affected (base: greeting_format_v1, price_total_v1; adapter:
capitalize_words_v1, chunk_v1, transpose_rows_v1) and the published
"held-out improved 0.8889 -> 1.0" was false -- the corrected result is a
held-out regression 0.8889 -> 0.6667 and an aggregate target regression
0.8108 -> 0.5405.

This script recomputes the reports from the EXISTING IMMUTABLE responses
(no new model generation; the final split stays sealed): the frozen
responses-{base,adapter}.jsonl are re-extracted with the same strict rule
and re-graded through the real host oracle under the corrected accounting
(authored denominator; unjudged checks count as FAILED). For every case the
committed judgment graded in full, the recomputed per-check verdicts MUST
match byte-for-byte -- a reproducibility proof, not an assumption.

Outputs (new files; the committed 2026-09-19 records are frozen as
invalidated history and never edited in place):
  deepapply-run/judgment-ab-{base,adapter}-corrected.json
  deepapply-run/training-report-corrected.json
  pilot-receipt-correction.unsigned.json   (for CLI signing; supersedes
        pilot-receipt-training.unsigned.json in the append-only store)

Usage (WSL, repo root, siltvenv, PYTHONPATH shadowing the stale editable
asea install -- see train_deepapply.py):
  export PYTHONPATH=/mnt/c/Users/reetu/Desktop/SILT/src
  python experiments/glm53_flash_pilot/correct_ab_scoring.py
Then sign + store:
  python -m asea.capability_build receipt --receipt \
    experiments/glm53_flash_pilot/pilot-receipt-correction.unsigned.json \
    --workspace experiments/glm53_flash_pilot/workspace
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_deepapply as td  # noqa: E402 (same-dir driver; no torch at module level)
from judge_pilot import extract_candidate, summarise  # noqa: E402


def rejudge_arm(arm: str, cases: dict) -> dict:
    """Re-extract every frozen response and re-grade it through the real
    oracle under authored-denominator accounting. Self-contained on purpose:
    reusing train_deepapply._judge_arm would overwrite the committed
    judgment files in place, and those are frozen invalidated history."""
    committed = json.loads(
        (td.RUN_DIR / ("judgment-ab-%s.json" % arm)).read_text(encoding="utf-8"))
    committed_rows = {r["sample_id"]: r for r in committed["cases"]}
    rows = []
    candidates_dir = td.RUN_DIR / "judging" / "candidates"
    oracles_dir = td.RUN_DIR / "judging" / "oracle"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    oracles_dir.mkdir(parents=True, exist_ok=True)
    for line in (td.RUN_DIR / ("responses-%s.jsonl" % arm)).read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sid = record["sample_id"]
        case = cases[sid]
        authored = len(case["oracle"])
        source, how = extract_candidate(record["response"])
        if source is None:
            rows.append({
                "sample_id": sid, "split": case["split"], "group": case["group"],
                "passed": 0, "total": authored,
                "judged_cases": 0, "unjudged_checks": authored,
                "per_check": {}, "extraction": how,
                "candidate_sha256": None,
                "oracle_status": "no_candidate",
                "diagnostic_event": None, "candidate_error": None,
                "latency_ms": record["latency_ms"],
                "new_tokens": record["new_tokens"],
            })
            continue
        candidate_path = candidates_dir / ("%s-%s.py" % (sid, arm))
        candidate_path.write_text(source, encoding="utf-8")
        oracle_payload = [
            {
                "sample_id": sid, "group": case["group"], "prompt": case["prompt"],
                "id": check["id"], "function": check["function"],
                "args": check["args"], "kwargs": check.get("kwargs", {}),
                "expected": check["expected"],
            }
            for check in case["oracle"]
        ]
        oracle_path = oracles_dir / ("%s-%s.json" % (sid, arm))
        oracle_path.write_text(json.dumps(oracle_payload, indent=1), encoding="utf-8")
        payload = td._run_oracle(candidate_path, oracle_path)
        result = payload["result"]
        if result.get("blocked"):
            raise SystemExit(
                "oracle BLOCKED for %s (%s): %r -- refusing to grade locally"
                % (sid, arm, result))
        if int(result.get("authored_cases", -1)) != authored:
            raise SystemExit(
                "oracle accounting mismatch for %s (%s): reported %s authored "
                "checks, the frozen case defines %d -- refusing the report"
                % (sid, arm, result.get("authored_cases"), authored))
        per_check = {
            c["id"]: c.get("matched")
            for c in (result.get("oracle") or {}).get("cases", [])
        }
        diagnostic = (result.get("oracle") or {}).get("diagnostic") or {}
        rows.append({
            "sample_id": sid, "split": case["split"], "group": case["group"],
            "passed": int(result.get("passed_cases", 0)),
            "total": authored,
            "judged_cases": int(result.get("judged_cases", 0)),
            "unjudged_checks": int(result.get("unjudged_cases", 0)),
            "per_check": per_check, "extraction": how,
            "candidate_sha256": td.hashlib.sha256(
                source.encode("utf-8")).hexdigest(),
            "oracle_status": result["status"],
            "diagnostic_event": diagnostic.get("event"),
            "candidate_error": diagnostic.get("candidate_error"),
            "latency_ms": record["latency_ms"],
            "new_tokens": record["new_tokens"],
        })

    # Reproducibility cross-check against the committed record: every case the
    # committed judgment graded in full must reproduce byte-for-byte; every
    # case it dropped (total 0 under the defective accounting) is recorded as
    # a discrepancy entry with its true verdict.
    dropped = []
    for row in rows:
        old = committed_rows[row["sample_id"]]
        if old["total"] == row["total"]:
            if old["per_check"] != row["per_check"] or old["passed"] != row["passed"]:
                raise SystemExit(
                    "recomputation mismatch for %s (%s): committed %r vs "
                    "recomputed %r -- refusing the correction"
                    % (row["sample_id"], arm, old["per_check"], row["per_check"])
                )
        else:
            dropped.append({
                "sample_id": row["sample_id"],
                "authored_checks": row["total"],
                "committed_total": old["total"],
                "committed_passed": old["passed"],
                "corrected_total": row["total"],
                "corrected_passed": row["passed"],
                "corrected_oracle_status": row["oracle_status"],
                "corrected_diagnostic_event": row["diagnostic_event"],
                "corrected_candidate_error": row["candidate_error"],
            })
    arm_total = sum(r["total"] for r in rows)
    if arm_total != 45:
        raise SystemExit(
            "corrected %s arm grades %d authored checks; the frozen non-final "
            "dataset defines 45 -- refusing the report" % (arm, arm_total))
    return {
        "rows": rows,
        "dropped": dropped,
        "committed_pass_rates": committed["pass_rates"],
    }


def main() -> None:
    cases = td.load_cases()
    names = {
        "base": "student Qwen/Qwen2.5-0.5B-Instruct, no adapter (A/B base arm) "
                "-- CORRECTED authored-denominator accounting",
        "adapter": "student Qwen/Qwen2.5-0.5B-Instruct + trained LoRA adapter "
                   "(A/B adapter arm) -- CORRECTED authored-denominator "
                   "accounting",
    }
    corrected = {}
    for arm in ("base", "adapter"):
        outcome = rejudge_arm(arm, cases)
        summary = summarise(outcome["rows"], names[arm])
        summary["correction"] = {
            "corrects": "judgment-ab-%s.json" % arm,
            "reason": "the committed report dropped candidate-error checks "
                      "from every denominator (reported 0/0 rows vanished "
                      "from the sums); corrected accounting uses the frozen "
                      "authored oracle count and counts unjudged checks as "
                      "FAILED",
            "committed_pass_rates": outcome["committed_pass_rates"],
            "dropped_cases": outcome["dropped"],
            "reproducibility": "every case the committed report graded in "
                               "full reproduced byte-for-byte under "
                               "recomputation from the same frozen response",
        }
        out = td.RUN_DIR / ("judgment-ab-%s-corrected.json" % arm)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        corrected[arm] = summary
        print(json.dumps({"arm": arm, "corrected_pass_rates": summary["pass_rates"],
                          "dropped_cases": [d["sample_id"] for d in outcome["dropped"]]}))

    base_rates = corrected["base"]["pass_rates"]
    adapter_rates = corrected["adapter"]["pass_rates"]
    keys = ("target_checks", "control_checks", "training_checks",
            "development_checks", "heldout_checks", "controls_split_checks",
            "overall_checks")
    deltas = {
        key: round(adapter_rates[key] - base_rates[key], 4)
        for key in keys
        if base_rates.get(key) is not None and adapter_rates.get(key) is not None
    }
    verdict = {
        "run": "deepapply training + independent oracle A/B (REAL) -- "
               "CORRECTED scoring (supersedes training-report.json)",
        "capability_id": td.CAP,
        "base_model": td.MODEL_ID,
        "supersedes": "deepapply-run/training-report.json",
        "supersession_reason": "the superseded report's rates divided by the "
                               "reported judged totals, so candidate-error "
                               "cases (0/0 rows) silently vanished from the "
                               "denominators and inflated every rate",
        "comparison": corrected["base"]["correction"]["reason"],
        "base_pass_rates": base_rates,
        "adapter_pass_rates": adapter_rates,
        "deltas_adapter_minus_base": deltas,
        "heldout_improved": (deltas.get("heldout_checks") or 0) > 0,
        "control_regressed": (deltas.get("control_checks") or 0) < 0,
        "dropped_cases_reinstated": {
            arm: [d["sample_id"] for d in corrected[arm]["correction"]["dropped_cases"]]
            for arm in ("base", "adapter")
        },
        "honesty": [
            "A pass rate on these authored cases measures THIS case set only; "
            "it is not a quality claim about the model.",
            "6 KD pairs is a mechanism-scale training set; a null or negative "
            "held-out delta is a real result, not a failure to hide.",
            "The trainer never certified itself: every number here comes from "
            "the independent host oracle in the Linux sandbox.",
            "The final split was never generated or judged.",
            "Production admission (Gate 1 PROMOTED packets -> DeepApplyRunner "
            "-> Gate 2 -> AdapterStore) was NOT sought; this adapter is a "
            "research artifact and autoactivates nothing.",
            "Corrected verdict: the adapter REGRESSED the measured target "
            "capability overall (training and held-out both regressed; "
            "development and the control split improved). The earlier "
            "'held-out improved' claim was an artifact of the scoring "
            "defect and is withdrawn.",
        ],
    }
    out = td.RUN_DIR / "training-report-corrected.json"
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"report": str(out), "deltas": deltas}))


if __name__ == "__main__":
    main()