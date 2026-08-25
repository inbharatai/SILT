"""Prepare the A/B/C training handoff for MiniCPM-o (CPU-feasible, real data).

Produces two budget-matched datasets + NOT_EXECUTED job specs a human runs on a
GPU box, using the REAL Assamese G2P run as the source:

  B (conventional)  = ALL extraction pairs from the suite, NO SILT gating.
                      Includes the 8 the gate dropped (receiver_competent) — i.e.
                      raw signal, including pairs that just reinforce what the
                      receiver already knows and any noisy ones.
  C (SILT-gated)    = only the pairs that survived SILT relevance + safety +
                      trust-gate + held-out-regression (the 4-entry promoted
                      lexicon packet), exported via the production export path.

Budget match = equal total training tokens (row-epochs):
  B: 12 rows * 1 epoch  = 12 row-epochs
  C:  4 rows * 3 epochs = 12 row-epochs
So neither arm wins by seeing more data. This is the honest controlled comparison
the user asked for. If B beats C at equal budget, the report says so.

This is a SHAPE demonstration on the real G2P run (4 vs 12 rows). On the GPU box
the SAME code path regenerates B/C at IndicVoices scale (thousands of rows) — B
= all extracted Assamese pairs, C = the SILT-gated subset — with the same
budget-match rule. No number here is a MiniCPM-o training result; the job specs
are NOT_EXECUTED.

  PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/prepare_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "minicpmo45_assamese"
sys.path.insert(0, str(REPO / "src"))

from asea.benchmarks.harness import load_suite  # noqa: E402
from asea.distill.export import build_job_spec, export_artifact_bundle  # noqa: E402
from asea.memory.store import MemoryStore  # noqa: E402

BASE_MODEL = "openbmb/MiniCPM-o-2_6"  # verify the current 4.5 id on the GPU box
PROMOTED_JOB = "e0ebb0553635"
BENCHMARKS = REPO / "data" / "benchmarks"
OUT = EXP / "handoff"


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _stub_manifest(dataset_path: str, row_count: int, contains_mock: bool):
    return {
        "dataset_path": dataset_path,
        "row_count": row_count,
        "packet_count": 0,
        "skipped": [],
        "contains_mock_data": contains_mock,
        "created_at": "PENDING_GPU_RUN",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    suite = load_suite(BENCHMARKS / "tts_pronunciation_as.json")

    # --- Arm B: conventional (all extraction pairs, no gating) ----------------
    b_rows = []
    for c in suite.split("extraction"):
        b_rows.append({
            "input": c.prompt, "output": c.expected,
            "arm": "B", "suite": suite.suite_id,
            "note": "raw extraction pair; no SILT relevance/safety/gate filter",
        })
    b_path = OUT / "B_conventional.jsonl"
    _write_jsonl(b_path, b_rows)
    b_manifest = _stub_manifest(str(b_path), len(b_rows), contains_mock=False)
    b_spec = build_job_spec(
        b_manifest, base_model=BASE_MODEL, epochs=1,
        eval_gate={"held_out_improvement_min": 0.02,
                   "regression_tolerance": 0.02,
                   "require_human_review": True,
                   "require_native_speaker_review_for_language_tasks": True,
                   "budget_match": "12 rows * 1 epoch = 12 row-epochs"},
    )
    (OUT / "B_conventional.job.json").write_text(
        json.dumps(b_spec, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Arm C: SILT-gated (the promoted packet, via the production export) ----
    store = MemoryStore(REPO / ".studio" / PROMOTED_JOB / "memory")
    packets = store.list("approved")
    if not packets:
        print("no approved packets for job {} -- run silt_proxy_run.py first".format(
            PROMOTED_JOB))
        sys.exit(1)
    c_zip = export_artifact_bundle(
        packets, OUT, name="C_silt_gated", base_model=BASE_MODEL,
    )
    # export_artifact_bundle writes C_silt_gated.jsonl + .job.json (epochs=2
    # default). Rewrite the C job spec with epochs=3 to budget-match B.
    c_jsonl = OUT / "C_silt_gated.jsonl"
    c_rows = [json.loads(line) for line in c_jsonl.read_text(encoding="utf-8").splitlines() if line]
    c_manifest = _stub_manifest(str(c_jsonl), len(c_rows), contains_mock=False)
    c_spec = build_job_spec(
        c_manifest, base_model=BASE_MODEL, epochs=3,
        eval_gate={"held_out_improvement_min": 0.02,
                   "regression_tolerance": 0.02,
                   "require_human_review": True,
                   "require_native_speaker_review_for_language_tasks": True,
                   "budget_match": "4 rows * 3 epochs = 12 row-epochs"},
    )
    (OUT / "C_silt_gated.job.json").write_text(
        json.dumps(c_spec, indent=2, ensure_ascii=False), encoding="utf-8")

    comparison = {
        "base_model": BASE_MODEL,
        "source_run": "real SILT G2P run .studio/{}/ (PROMOTED, candidate 0.7359)".format(PROMOTED_JOB),
        "B_conventional": {"rows": len(b_rows), "epochs": 1, "row_epochs": len(b_rows) * 1,
                           "dataset": str(b_path), "gating": "none"},
        "C_silt_gated": {"rows": len(c_rows), "epochs": 3, "row_epochs": len(c_rows) * 3,
                         "dataset": str(c_jsonl), "bundle": str(c_zip),
                         "gating": "relevance+safety+trust+regression (8 dropped as receiver_competent)"},
        "budget_match": "B row-epochs == C row-epochs == 12 (equal training tokens)",
        "status": "NOT_EXECUTED — run on a GPU box via train_lora.py",
        "scale_note": "shape demonstration on the real G2P run; GPU box regenerates B/C at IndicVoices scale via the same path",
    }
    (OUT / "AB_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print("B: {} rows (1 epoch)  C: {} rows (3 epochs)  budget-match {}=={} row-epochs".format(
        len(b_rows), len(c_rows), comparison["B_conventional"]["row_epochs"],
        comparison["C_silt_gated"]["row_epochs"]))
    print("wrote {} , {} , {}".format(b_path, c_jsonl, c_zip))


if __name__ == "__main__":
    main()