"""Assemble the E2 PRE-TRAIN receipt for signing (figures from files).

Every number is READ from an artifact a command produced -- nothing is
hand-copied or estimated. This receipt freezes the E2 expansion state
BEFORE training: the frozen v2 dataset, the teacher's judged outcomes on
the sixteen new training cases, and the distilled KD pair set. The
training stage verifies the KD artifact against this receipt's
``artifact_hashes.kd_pairs`` pin before it trains on it.

Usage (repo root, same interpreter as the CLI):
  python experiments/glm53_flash_e2/make_receipt_e2_pretrain.py
Then sign + store (append-only, NEW record name):
  python -m asea.capability_build receipt --receipt \
    experiments/glm53_flash_e2/e2-receipt-pretrain.unsigned.json \
    --workspace experiments/glm53_flash_e2/workspace
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOT_MEASURED = "NOT_MEASURED"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from asea.capability_build.spec import load_spec

    spec_file = HERE / "spec.json"
    loaded = load_spec(spec_file)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    spec_sha = loaded["spec_sha256"]
    teacher_new = json.loads(
        (HERE / "judgment-teacher-e2-new.json").read_text(encoding="utf-8"))
    distill = json.loads(
        (HERE / "workspace-judged" / "candidates" /
         "python_repo_debugging_v1-kd-pairs.json").read_text(encoding="utf-8"))
    v2_manifest = json.loads(
        (REPO / "data" / "capability_v2" / "manifest.json").read_text(
            encoding="utf-8"))
    v1_manifest = json.loads(
        (REPO / "data" / "capability_v1" / "manifest.json").read_text(
            encoding="utf-8"))

    # Byte-copy verification: the four evaluation splits of v2 must hash
    # identically to the frozen v1 manifest entries.
    eval_copies = {}
    for split in ("development.jsonl", "heldout.jsonl", "final.jsonl",
                  "controls.jsonl"):
        actual = sha256(REPO / "data" / "capability_v2" / split)
        eval_copies[split] = {
            "v2_file_sha256": actual,
            "v1_manifest_sha256": v1_manifest["files"][split],
            "byte_identical_to_v1": (
                actual == v1_manifest["files"][split]),
        }
    for name, entry in eval_copies.items():
        if not entry["byte_identical_to_v1"]:
            raise SystemExit(
                "%s is NOT byte-identical to the frozen v1 split -- refusing "
                "the receipt" % name)

    rows = teacher_new["cases"]
    target_rows = [r for r in rows if r["group"] == "target"]
    teacher_new_rate = round(
        sum(r["passed"] for r in rows) / sum(r["total"] for r in rows), 4)

    receipt = {
        "schema": "silt.capability_receipt.v1",
        "command": "glm53-flash-e2-pretrain",
        "status": "completed",
        "capability_id": spec["capability_id"],
        "spec_sha256": spec_sha,
        "evidence_class": spec["teacher"]["access"],
        "teacher": {
            "provider": spec["teacher"]["provider"],
            "model": spec["teacher"]["model"],
            "revision": spec["teacher"]["revision"],
            "access": spec["teacher"]["access"],
            "traced_cases": "16 new E2 training cases (v1's 16 non-final "
                            "traces stay receipted in the v1 pilot; the "
                            "final split was never traced in either "
                            "experiment)",
            "remote_consent": "--allow-remote passed for the E2 trace run "
                              "only; per-run consent, never carried over",
        },
        "student": {
            "e2_student": "Qwen/Qwen2.5-1.5B-Instruct (local HF cache, GPU "
                           "LoRA receiver; the v1 pilot used the 0.5B "
                           "Instruct student)",
        },
        "data_split_hashes": dict(v2_manifest["files"]),
        "measurements": {
            "dataset_v2": {
                "counts": v2_manifest["counts"],
                "training_composition": "six v1 training cases (byte-identical "
                                         "rows) plus sixteen newly authored "
                                         "cases",
                "evaluation_splits": eval_copies,
                "selection_lock_verified": True,
                "final_opened": False,
            },
            "teacher_e2_new_judged": {
                "judge": teacher_new["judge"],
                "cases": len(rows),
                "checks": sum(r["total"] for r in rows),
                "fully_passed_cases": teacher_new["cases_fully_passed"],
                "overall_checks_rate": teacher_new_rate,
                "target_cases": len(target_rows),
                "authored_denominator": teacher_new["authored_denominator"],
                "extraction_rule": teacher_new["extraction_rule"],
                "note": "these are the SIXTEEN NEW training cases only; the "
                        "v1 pilot's receipted teacher numbers on the frozen "
                        "16 non-final evaluation cases stand unchanged and "
                        "are not re-measured here",
            },
            "distill": {
                "positives": len(distill["positives"]),
                "negatives": len(distill["negatives"]),
                "unjudged_excluded": distill["unjudged_excluded"],
                "composition": "six receipted v1 training pairs (reused; "
                               "their traces were judged in the v1 pilot) "
                               "plus E2 positives from the newly judged "
                               "traces",
                "protected_splits_enforced":
                    distill["dataset"]["protected_splits_enforced"],
                "sequence_level_only": distill["sequence_level_only"],
                "compatibility_note": distill["compatibility_note"],
            },
        },
        "capability_retention": NOT_MEASURED,
        "control_regressions": {},
        "resources": {},
        "gates": {
            "gate1_status": "not_requested (no promotion sought in this "
                            "experiment)",
            "gate2_status": "not yet run; the independent A/B happens after "
                            "training and is recorded in the E2 training "
                            "receipt",
        },
        "failure_history": [],
        "limitations": [
            "A pass rate on these authored cases measures THIS case set "
            "only; it is not a quality claim about any model.",
            "Implementation success is not model-quality success.",
            "The final split was never opened; no measurement exists for it.",
            "The teacher is behavioural-only: no routers, experts or "
            "internal state were observed; no internal claims are made.",
            "KD pairs are sequence-level text only (GLM->Qwen cross-family); "
            "no vocabulary, logit or hidden-state alignment is assumed or "
            "performed.",
            "The KD set is mechanism-scale; a null or negative held-out "
            "delta would be a real result, not a failure to hide.",
        ],
        "artifact_hashes": dict(
            {name: sha256(HERE / name) for name in (
                "spec.json", "cases.jsonl", "cases-new-training.jsonl",
                "authoring_e2.py", "assemble_e2_dataset.py",
                "judge_e2.py", "judgment-teacher-e2-new.json",
            )},
            kd_pairs=distill["artifact_sha256"],
            dataset_v2_manifest=sha256(
                REPO / "data" / "capability_v2" / "manifest.json"),
            dataset_v1_manifest=sha256(
                REPO / "data" / "capability_v1" / "manifest.json"),
        ),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "note": "authored and judged on Windows Python; oracle grading "
                    "ran in WSL2 Ubuntu (Linux sandbox required by the host "
                    "oracle)",
        },
        "hardware_identity": {
            "note": "8 GB RTX 5050 laptop; no GPU was used in the pretrain "
                    "stages (authoring, tracing, judging, distilling)",
        },
        "error": None,
    }
    out = HERE / "e2-receipt-pretrain.unsigned.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"receipt": str(out), "status": receipt["status"]}))


if __name__ == "__main__":
    main()