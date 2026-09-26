"""Dataset-integrity tests for data/extraction_v1 (Section H).

These verify the COMMITTED dataset artifacts -- not model claims:

  * every manifest validates through the real ``silt-extract dataset
    validate`` CLI;
  * every manifest's content/prompt hashes match the committed case
    files (no silent drift between builder output and manifests);
  * every TARGET case's buggy code actually FAILS at least one of its
    oracle checks -- a repair case whose "buggy" code already passes is
    an incoherent case, not a repair task;
  * the power analysis is present and states the seed-scale gap.

Nothing here is GLM evidence; the dataset itself is synthetic CC0
content authored without any model output.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "extraction_v1"
_SPLITS = ("training", "development", "heldout", "controls")


def _load(split: str) -> list:
    return [json.loads(line)
            for line in (DATA / ("%s.jsonl" % split)).read_text(
                encoding="utf-8").splitlines() if line.strip()]


class TestExtractionDatasetIntegrity:
    def test_all_manifests_validate_through_the_cli(self):
        for split in _SPLITS:
            completed = subprocess.run(
                [sys.executable, "-m", "asea.extraction", "dataset",
                 "validate", "--dir",
                 str(DATA / ("%s-manifest.json" % split))],
                capture_output=True, text=True, timeout=120)
            assert completed.returncode == 0, split
            payload = json.loads(completed.stdout)
            assert payload["ok"] is True
            assert payload["split"] == split

    def test_manifest_hashes_match_the_case_files(self):
        for split in _SPLITS:
            manifest = json.loads(
                (DATA / ("%s-manifest.json" % split))
                .read_text(encoding="utf-8"))
            by_id = {case["case_id"]: case
                     for case in json.loads(json.dumps(
                         manifest["cases"]))}
            for case in _load(split):
                entry = by_id[case["case_id"]]
                content = json.dumps(
                    {"prompt": case["prompt"], "oracle": case["oracle"]},
                    sort_keys=True)
                assert entry["content_sha256"] == hashlib.sha256(
                    content.encode("utf-8")).hexdigest()
                assert entry["prompt_sha256"] == hashlib.sha256(
                    case["prompt"].encode("utf-8")).hexdigest()

    def test_target_and_control_groups_are_correct(self):
        for split in _SPLITS:
            for case in _load(split):
                if split == "controls":
                    assert case["group"] == "control"
                else:
                    assert case["group"] == "target"

    def test_fourteen_target_families_present(self):
        families = {case["family"] for case in _load("training")}
        families |= {case["family"] for case in _load("development")}
        families |= {case["family"] for case in _load("heldout")}
        assert len(families) == 14

    def test_power_analysis_states_the_seed_scale_gap(self):
        power = json.loads((DATA / "power-analysis.json").read_text(
            encoding="utf-8"))
        assert power["required_n_per_family"] >= 10
        assert power["families_below_required_n"] == sorted(
            set(power["present_n_per_family"]))
        assert "SEED-SCALE" in power["verdict"]

    def test_final_split_is_not_generated(self):
        lock = json.loads((DATA / "selection-lock.json").read_text(
            encoding="utf-8"))
        assert lock["frozen_before_model_generation"] is True
        assert lock["final_generated"] is False
        assert not (DATA / "final.jsonl").exists()


_RUN_ORACLE = r"""
import json, sys
payload = json.load(sys.stdin)
namespace = {}
exec(compile(payload["code"], "<buggy>", "exec"), namespace)
function = namespace[payload["function"]]
results = []
for check in payload["checks"]:
    try:
        actual = function(*check["args"], **check.get("kwargs", {}))
        results.append(actual == check["expected"])
    except Exception:
        results.append(False)
print(json.dumps(results))
"""


class TestBuggyCodeIsActuallyBuggy:
    def test_every_target_buggy_fails_at_least_one_check(self):
        total = 0
        for split in ("training", "development", "heldout"):
            for case in _load(split):
                total += 1
                code = _buggy_source(case["prompt"])
                completed = subprocess.run(
                    [sys.executable, "-c", _RUN_ORACLE],
                    input=json.dumps({
                        "code": code,
                        "function": case["oracle"]["function"],
                        "checks": case["oracle"]["checks"],
                    }),
                    capture_output=True, text=True, timeout=30)
                assert completed.returncode == 0, case["case_id"]
                verdicts = json.loads(completed.stdout.strip())
                assert len(verdicts) == len(case["oracle"]["checks"])
                assert not all(verdicts), (
                    "case %s: the 'buggy' code passes every oracle check; "
                    "this is not a repair task" % case["case_id"])
        assert total >= 40


def _buggy_source(prompt: str) -> str:
    marker = "```python\n"
    start = prompt.index(marker) + len(marker)
    return prompt[start:prompt.index("```", start)]