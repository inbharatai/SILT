"""SILT-proxy Assamese experiment — the CPU-feasible half (runs for real).

This is NOT a MiniCPM-o result. It is a real, audited SILT capability transfer on
Assamese using the connectors available on this CPU machine, which validates the
SILT selection / trust-gating / held-out-lift machinery on Assamese — the core of
the "does SILT provide value" question — while the MiniCPM-o weight-training half
is blocked on a GPU box (see ../hardware.json, blockers B1-B3).

Two real measurements:

  1. G2P replay: load the existing PROMOTED Assamese G2P packet from
     .studio/e0ebb0553635/memory/approved, build the real `tts-learner-zero`
     receiver (qwen3.5:latest via ollama, think=False), and run the SAME
     baseline-vs-candidate held-out A/B that POST /api/skills/test runs — directly
     via BenchmarkHarness + MemoryStore.approved_skills, never Evaluator.evaluate
     (so it is read-only w.r.t. the approved store). Reproduces the per-case
     numbers in docs/SILT_TTS_G2P_TEST.md §5.6.

  2. Fresh text transfer (best-effort, may be slow / may need the NLLB download):
     nllb-teacher -> qwen2.5:1.5b-instruct on assamese_english, run in-process via
     TransferJob. Real funnel counts + gate verdict. If it fails, the failure is
     recorded verbatim and the G2P replay still stands.

Outputs (real numbers only) to ../metrics.json, ../trust_gate_results.jsonl,
../error_analysis.md. Never invents a number; never touches src/asea.

Run:
  PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/silt_proxy_run.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "minicpmo45_assamese"
sys.path.insert(0, str(REPO / "src"))
# IPA / Assamese glyphs crash the default cp1252 console on Windows; write UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

from asea.benchmarks.harness import BenchmarkHarness, load_suite  # noqa: E402
from asea.core.plugins import default_registry  # noqa: E402
from asea.memory.store import MemoryStore  # noqa: E402
from asea.studio import catalog  # noqa: E402
from asea.studio.jobs import TransferJob, JobManager, _similarity  # noqa: E402

BENCHMARKS = REPO / "data" / "benchmarks"
PROMOTED_JOB = "e0ebb0553635"
RESULTS = {
    "experiment": "minicpmo45_assamese_silt_proxy",
    "captured_at": "2026-08-13",
    "framing": "SILT-on-Assamese validation (real), NOT a MiniCPM-o result. "
               "MiniCPM-o arms A/B/C are PENDING a GPU box (see hardware.json).",
    "g2p_replay": None,
    "text_transfer": None,
}


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def g2p_replay():
    """Reproduce docs/SILT_TTS_G2P_TEST.md §5.6 directly against the on-disk
    approved packet (read-only; never Evaluator.evaluate, never the gate)."""
    print("== G2P replay: tts_pronunciation_as, receiver qwen3.5:latest ==")
    suite = load_suite(BENCHMARKS / "tts_pronunciation_as.json")
    receiver = catalog.build("tts-learner-zero")  # ollama-qwen3.5-latest, think=False
    store = MemoryStore(REPO / ".studio" / PROMOTED_JOB / "memory")
    skills = store.approved_skills(receiver.module_id)
    if not skills:
        raise RuntimeError(
            "no approved packet for '{}' in .studio/{}".format(
                receiver.module_id, PROMOTED_JOB
            )
        )
    harness = BenchmarkHarness(
        plugins=default_registry(), similarity=_similarity("embedding")
    )
    baseline = harness.run(receiver, suite, split="heldout")
    candidate = harness.run(receiver, suite, split="heldout", skills=skills)

    by_id = {c.case_id: c for c in candidate.case_results}
    cases = []
    for base in baseline.case_results:
        cand = by_id.get(base.case_id)
        if cand is None:
            continue
        cases.append({
            "case_id": base.case_id,
            "expected": base.expected,
            "baseline_output": base.actual,
            "candidate_output": cand.actual,
            "baseline_score": round(base.score, 4),
            "candidate_score": round(cand.score, 4),
            "delta": round(cand.score - base.score, 4),
            "regressed": cand.score < base.score,
        })
    result = {
        "packet_id": skills[0].get("packet_id"),
        "suite_id": suite.suite_id,
        "module": receiver.module_id,
        "is_mock": receiver.is_mock,
        "skills_active": len(skills),
        "similarity_is_semantic": candidate.similarity_is_semantic,
        "baseline": {"score": round(baseline.score, 4),
                     "task_success": round(baseline.task_success, 4),
                     "case_count": len(baseline.case_results)},
        "candidate": {"score": round(candidate.score, 4),
                      "task_success": round(candidate.task_success, 4),
                      "case_count": len(candidate.case_results)},
        "improvement": round(candidate.score - baseline.score, 4),
        "cases": cases,
        "reference_doc": "docs/SILT_TTS_G2P_TEST.md §5.6 (baseline 0.6023, "
                         "candidate 0.7359, improvement +0.1336)",
    }
    RESULTS["g2p_replay"] = result
    _write_json(EXP / "metrics.json", RESULTS)  # persist BEFORE any printing
    print("  baseline {}  candidate {}  improvement {:+}".format(
        result["baseline"]["score"], result["candidate"]["score"],
        result["improvement"]))
    for c in cases:
        print("    {:8} exp={:6} base={:8} cand={:8} delta={:+.3f}{}".format(
            c["case_id"], str(c["expected"]), str(c["baseline_output"]),
            str(c["candidate_output"]), c["delta"], " <-learned" if c["delta"] > 0.1 else ""))
    return result


def text_transfer():
    """Fresh real transfer nllb-teacher -> qwen2.5:1.5b on assamese_english.
    Best-effort: records the outcome verbatim, including failure."""
    print("== Text transfer: nllb-teacher -> qwen2.5:1.5b-instruct, assamese_english ==")
    manager = JobManager(REPO / ".studio")
    job = TransferJob(
        {"sender": "nllb-teacher", "receiver": "qwen2.5-1.5b-instruct-runtime",
         "suites": ["assamese_english"], "similarity": "lexical",
         "relevance_floor": 0.30,
         "description": "SILT-proxy Assamese text transfer (exp/minicpmo45_assamese)"},
        manager.workspace_root,
    )
    # The receiver qwen2.5:1.5b-instruct is not a built-in catalog id; add it at
    # runtime via the same _ollama factory the /api/catalog endpoint uses, so the
    # job's catalog.build() resolves it. Real weights, think unset (non-reasoning).
    from asea.studio.catalog import _ollama, _translation
    from asea.core.protocol import CapabilityKey, Domain, Modality
    cap = CapabilityKey(task_type="translate", modality=Modality.TEXT,
                        domain=Domain.TRANSLATION, language="as->en")
    catalog.CATALOG["qwen2.5-1.5b-instruct-runtime"] = {
        "factory": _ollama("qwen2.5:1.5b-instruct", roles=["receiver"],
                           capabilities=[cap]),
        "roles": ["receiver"],
        "description": "Qwen2.5 1.5B Instruct via ollama (runtime receiver, real weights)",
        "requires": "ollama serve + ollama pull qwen2.5:1.5b-instruct",
    }
    catalog._cache.pop("qwen2.5-1.5b-instruct-runtime", None)
    manager.jobs[job.job_id] = job
    job.start()
    t0 = time.time()
    while time.time() - t0 < 1800:  # 30 min ceiling
        if job.status in ("done", "failed"):
            break
        time.sleep(3)
    elapsed = round(time.time() - t0, 1)
    entry = {
        "job_id": job.job_id,
        "status": job.status,
        "elapsed_seconds": elapsed,
        "error": job.error,
        "config": job.config,
    }
    if job.status == "done" and job.report is not None:
        entry["report"] = job.report
        entry["promoted_count"] = len(job.report.get("promoted", []))
        entry["pending_human_count"] = len(job.report.get("pending_human", []))
        entry["rejected_count"] = len(job.report.get("rejected", []))
        print("  status=done elapsed={}s promoted={} pending_human={} rejected={}".format(
            elapsed, entry["promoted_count"], entry["pending_human_count"],
            entry["rejected_count"]))
    else:
        print("  status={} elapsed={}s error={}".format(
            job.status, elapsed, job.error))
    RESULTS["text_transfer"] = entry
    _write_json(EXP / "metrics.json", RESULTS)
    return entry


def main():
    try:
        g2p_replay()
    except Exception:
        traceback.print_exc()
        RESULTS["g2p_replay"] = {"status": "failed",
                                 "error": traceback.format_exc(limit=3)}
        _write_json(EXP / "metrics.json", RESULTS)
    try:
        text_transfer()
    except Exception:
        traceback.print_exc()
        RESULTS["text_transfer"] = {"status": "failed",
                                    "error": traceback.format_exc(limit=3)}
        _write_json(EXP / "metrics.json", RESULTS)
    print("done -> experiments/minicpmo45_assamese/metrics.json")


if __name__ == "__main__":
    main()