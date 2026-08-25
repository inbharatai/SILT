"""Shared helpers for the demonstration flows.

=====================  DEMO POLICY WARNING  =====================
These examples construct the pipeline with ``strict_no_mock=False``.

Under the DEFAULT policy, any packet whose provenance touches a mock module is
rejected outright -- which is correct, and which would make every example here
terminate at "rejected: provenance includes a mock module". Disabling the check
is the only way to exercise the full path end-to-end in an environment with no
model weights.

Never do this with real learning data. tests/test_promotion_gate.py asserts
that the default strict policy DOES block mock-derived packets, so the
protection is verified even though the demos bypass it.
=================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from asea.benchmarks.harness import BenchmarkSuite, load_suite  # noqa: E402
from asea.core.pipeline import Pipeline  # noqa: E402
from asea.promotion.gate import PromotionGate, PromotionPolicy  # noqa: E402

DATA = ROOT / "data" / "benchmarks"


def suite(name: str) -> BenchmarkSuite:
    return load_suite(DATA / "{}.json".format(name))


def knowledge_from(
    suites: Iterable[BenchmarkSuite],
    splits: Iterable[str] = ("extraction", "heldout"),
    only_cases: Optional[Iterable[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build a mock knowledge table {capability: {prompt: answer}} from suites.

    ``overrides`` lets a demo seed a deliberately WRONG answer so the relevance
    filter's sender_incorrect path is exercised on real data rather than only in
    a unit test.
    """
    allow = set(only_cases) if only_cases is not None else None
    table: Dict[str, Dict[str, Any]] = {}
    for s in suites:
        cap = s.capability().as_str()
        bucket = table.setdefault(cap, {})
        for case in s.cases:
            if case.split not in splits:
                continue
            if allow is not None and case.case_id not in allow:
                continue
            bucket[str(case.prompt).strip()] = case.expected
        for prompt, answer in (overrides or {}).items():
            if prompt in bucket:
                bucket[prompt] = answer
    return table


def demo_pipeline(workspace: Path) -> Pipeline:
    """Pipeline with mock containment DISABLED. See the warning above."""
    gate = PromotionGate(PromotionPolicy(strict_no_mock=False))
    return Pipeline(workspace=workspace, gate=gate)


def show(report, title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    data = report.to_dict()
    counts = data["counts"]
    print("session      : {} -> {}".format(
        data["session"]["sender"], data["session"]["receiver"]))
    print("negotiation  : {} actionable gap(s)".format(
        data["negotiation"].get("actionable", 0)))
    for g in data["negotiation"].get("measured_gaps", []):
        print("   gap {}  receiver={:.3f} sender={:.3f} headroom={:.3f}".format(
            g["capability"], g["receiver_score"], g["sender_score"], g["headroom"]))
    print("counts       : {}".format(json.dumps(counts)))

    for drop in data["dropped_relevance"]:
        print("   dropped(relevance) {}".format(drop["reason"][:100]))
    for drop in data["dropped_safety"]:
        print("   dropped(safety)    {}".format(drop["reason"][:100]))

    for ev in data["evaluations"]:
        print("   eval {}  baseline={:.4f} candidate={:.4f} delta={:+.4f}".format(
            ev["capability"], ev["baseline"]["score"],
            ev["candidate"]["score"], ev["improvement"]))
        for reg in ev["regressions"]:
            print("        regression {} {:.4f} -> {:.4f} ({})".format(
                reg["suite_id"], reg["baseline"], reg["candidate"],
                "REGRESSED" if reg["regressed"] else "ok"))

    for dec in data["decisions"]:
        print("   gate {} -> {}".format(dec["packet_id"][:8], dec["status"]))
        for check in dec["checks"]:
            if not check["passed"]:
                print("        FAILED {}: {}".format(check["name"], check["detail"]))
    if data["mock_warning"]:
        print("NOTE: {}".format(data["mock_warning"]))
