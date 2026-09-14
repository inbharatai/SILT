"""Regression coverage for the public legacy unlearn dispatch (mock-only).

All before/after snapshots and certificates are created by python -m asea.cli,
not by private helpers or API fixture fallbacks. The explicit mock policy is
fixture-only; provenance, human gates and skill-layer claims remain unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from asea.benchmarks.harness import BenchmarkSuite
from asea.config import build_pipeline, load_config


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def operational_workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli-operational-mock")
    data = root / "data"
    data.mkdir()
    for name in ("assamese_english", "hindi_english"):
        shutil.copyfile(ROOT / "data" / "benchmarks" / (name + ".json"),
                        data / (name + ".json"))
    config = json.loads((ROOT / "configs" / "assamese_transfer.json").read_text())
    assert config["promotion_policy"]["strict_no_mock"] is False
    assert all(m.get("preset", "implicit_mock") in ("implicit_mock", "qwen_mock")
               for m in config["modules"])
    config_path = root / "mock-config.json"
    config_path.write_text(json.dumps(config))
    for name in ("home", "tmp", "cache"):
        (root / name).mkdir()
    # Deliberately do not inherit API keys, user site packages or model caches.
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        "HOME": str(root / "home"), "TMPDIR": str(root / "tmp"),
        "XDG_CACHE_HOME": str(root / "cache"), "HF_HOME": str(root / "cache" / "hf"),
        "PYTHONPATH": str(ROOT / "src"), "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    workspace = root / "workspace"

    def invoke(*args, expected=0):
        process = subprocess.run(
            [sys.executable, "-m", "asea.cli", "--data-dir", str(data),
             *map(str, args)], cwd=root, env=env, capture_output=True, text=True,
            timeout=35,
        )
        assert process.returncode == expected, process.stdout + process.stderr
        assert "Traceback" not in process.stderr, process.stderr
        return json.loads(process.stdout)

    first = invoke("run", "--config", config_path, "--workspace", workspace)
    assert first["counts"]["promoted"] == 1
    initial = invoke("report", "--workspace", workspace)
    assert initial["approved_packets"][0]["is_mock"] is True
    empty = initial["approved_packets"][0]["rollback_token"]
    # Duplicate refusal still captures the populated approved-set snapshot.
    second = invoke("run", "--config", config_path, "--workspace", workspace)
    assert second["counts"]["promoted"] == 0
    populated = invoke("report", "--workspace", workspace)
    before = next(s["token"] for s in populated["snapshots"] if s["packet_count"] == 1)
    rollback = invoke("rollback", "--workspace", workspace, "--token", empty)
    assert rollback["removed"] == 1
    after = invoke("report", "--workspace", workspace)
    assert after["store"]["approved"] == 0
    assert not list((workspace / "memory" / "approved").glob("*.json"))
    return root, data, config_path, workspace, before, empty, invoke


def _unlearn(fixture, suite_id, out, config=None):
    _, _, config_path, workspace, before, after, _ = fixture
    return ("unlearn", "--config", config or config_path, "--workspace", workspace,
            "--suite", suite_id, "--token-before", before, "--token-after", after,
            "--out", out)


@pytest.mark.parametrize("substantive", [True, False])
def test_unlearn_cli_selects_suite_id_and_signs(operational_workspace, substantive):
    root, _, _, workspace, before, after, invoke = operational_workspace
    suite_id = "assamese_english_v1"
    out = root / (suite_id + "-certificate.json")
    args = list(_unlearn(operational_workspace, suite_id, out))
    if not substantive:
        # Empty -> empty removes nothing: emit an honest unverified certificate.
        before = after
        args[args.index("--token-before") + 1] = before
    cert = invoke(*args)
    assert json.loads(out.read_text()) == cert
    assert cert["token_before"] == before and cert["token_after"] == after
    assert cert["verified"] is substantive
    assert cert["substantive"] is substantive
    assert "skill-layer" in cert["honesty_note"].lower()
    assert "weight" in cert["honesty_note"].lower()
    assert invoke("unlearn-verify", "--workspace", workspace, "--report", out)["valid"] is True
    tampered = dict(cert, receiver="TAMPERED")
    tamper_path = root / (suite_id + "-tampered.json")
    tamper_path.write_text(json.dumps(tampered))
    assert invoke("unlearn-verify", "--workspace", workspace, "--report", tamper_path,
                  expected=1)["valid"] is False


@pytest.mark.parametrize("suite_id", ["missing_suite", "assamese_english"])
def test_unlearn_cli_unknown_suite_is_actionable(operational_workspace, suite_id):
    root, _, _, _, _, _, invoke = operational_workspace
    out = root / (suite_id + "-must-not-exist.json")
    error = invoke(*_unlearn(operational_workspace, suite_id, out), expected=1)
    assert error["error"] == "unknown suite '{}'".format(suite_id)
    assert error["available"] == ["assamese_english_v1", "hindi_english_v1"]
    assert not out.exists()


def test_unlearn_cli_rejects_duplicate_embedded_suite_ids(operational_workspace):
    root, data, config_path, _, _, _, invoke = operational_workspace
    # Distinct file stems can carry the same suite_id: config does not reject it.
    duplicate = json.loads((data / "assamese_english.json").read_text())
    (data / "duplicate.json").write_text(json.dumps(duplicate))
    config = load_config(config_path)
    config["suites"].append("duplicate")
    duplicate_config = root / "duplicate-config.json"
    duplicate_config.write_text(json.dumps(config))
    out = root / "duplicate-must-not-exist.json"
    error = invoke(*_unlearn(operational_workspace, "assamese_english_v1", out,
                            config=duplicate_config), expected=1)
    assert "duplicate suite_id 'assamese_english_v1'" in error["error"]
    assert "unique" in error["error"]
    assert not out.exists()


@pytest.mark.parametrize("repeat_filename", [False, True])
def test_actual_build_pipeline_list_contract(operational_workspace, tmp_path, repeat_filename):
    _, data, config_path, _, _, _, _ = operational_workspace
    config = load_config(config_path)
    if repeat_filename:
        config["suites"].append("assamese_english")
    _, suites, _ = build_pipeline(config, tmp_path, data)
    assert isinstance(suites, list)
    assert all(isinstance(suite, BenchmarkSuite) for suite in suites)
    assert [suite.suite_id for suite in suites] == ["assamese_english_v1", "hindi_english_v1"]
    # load_suites deduplicates identical file stems, NOT embedded suite_id values.
