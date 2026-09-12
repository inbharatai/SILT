import json
from pathlib import Path
import pytest

from scripts import run_specialist_quality_experiment as runner


def test_public_cli_argv_shapes_are_specialist_namespace(tmp_path):
    recipe = tmp_path / "recipe.json"
    workspace = tmp_path / "study"
    final_suite = tmp_path / "fresh-final-suite.json"
    final_output = tmp_path / "final-result.json"

    assert runner.build_argv("python", recipe, workspace) == [
        "python", "-m", "asea.specialist", "build",
        "--recipe", str(recipe), "--workspace", str(workspace),
    ]
    assert runner.finalize_argv("python", workspace, final_suite, final_output) == [
        "python", "-m", "asea.specialist", "finalize",
        "--study", str(workspace), "--suite", str(final_suite),
        "--output", str(final_output),
    ]


def test_status_from_cli_preserves_blocked_and_rejected():
    assert runner.status_from_cli({"returncode": 0, "stdout": '{"completed":true}'}) == "COMPLETED"
    assert runner.status_from_cli({"returncode": 1, "stdout": '{"status":"BLOCKED","completed":false}'}) == "BLOCKED"
    assert runner.status_from_cli({"returncode": 1, "stdout": '{"status":"REJECTED","completed":false}'}) == "REJECTED"
    assert runner.status_from_cli({"returncode": 1, "stdout": ""}) == "BLOCKED"
    missing_checkpoint = {
        "command": "build",
        "completed": False,
        "status": "REJECTED",
        "result": {
            "qualified_source": False,
            "error": "not a regular local file: /local/models/qwen/config.json",
        },
    }
    assert runner.status_from_cli({"returncode": 1, "stdout": json.dumps(missing_checkpoint)}) == "BLOCKED"


def test_acceptance_criteria_names_candidate_and_final_guard():
    criteria = runner.acceptance_criteria("Qwen/Qwen2.5-Coder-3B-Instruct", "SmolLM2")
    joined = "\n".join(criteria["criteria"])
    assert criteria["candidate"] == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert criteria["compact_baseline"] == "SmolLM2"
    assert "Do not train on consumed final tasks" in joined
    assert "Freeze the candidate" in joined


def test_write_report_is_exclusive(tmp_path):
    report = tmp_path / "receipt.json"
    runner.write_report(report, {"status": "PREFLIGHT_ONLY"})
    with pytest.raises(FileExistsError):
        runner.write_report(report, {"status": "second"})


def test_main_preflight_does_not_invoke_build(monkeypatch, tmp_path):
    report = tmp_path / "preflight.json"
    monkeypatch.setattr(runner, "environment_report", lambda *args: {"env": "ok"})
    monkeypatch.setattr(runner, "specialist_help", lambda *args: {"root": {"returncode": 0}})
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: pytest.fail("preflight invoked CLI build"))

    code = runner.main(["--repo-root", str(tmp_path), "--preflight-only", "--report", str(report)])

    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "PREFLIGHT_ONLY"
    assert payload["commands"] == []


def test_main_missing_local_checkpoint_blocks_and_does_not_finalize(monkeypatch, tmp_path):
    report = tmp_path / "blocked.json"
    recipe = tmp_path / "recipe.json"
    final_suite = tmp_path / "final-suite.json"
    final_output = tmp_path / "final-output.json"
    recipe.write_text("{}", encoding="utf-8")
    final_suite.write_text("[]", encoding="utf-8")
    calls = []

    def fake_run(argv, *args, **kwargs):
        calls.append(argv)
        receipt = {
            "command": "build",
            "completed": False,
            "status": "REJECTED",
            "result": {
                "qualified_source": False,
                "error": "not a regular local file: /local/models/qwen/config.json",
            },
        }
        return {"returncode": 1, "stdout": json.dumps(receipt), "stderr": "", "elapsed_seconds": 0.01}

    monkeypatch.setattr(runner, "environment_report", lambda *args: {"env": "ok"})
    monkeypatch.setattr(runner, "specialist_help", lambda *args: {"root": {"returncode": 0}})
    monkeypatch.setattr(runner, "run", fake_run)

    code = runner.main([
        "--repo-root", str(tmp_path),
        "--recipe", str(recipe),
        "--workspace", str(tmp_path / "study"),
        "--final-suite", str(final_suite),
        "--final-output", str(final_output),
        "--run-final",
        "--report", str(report),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["status"] == "BLOCKED"
    assert payload["commands"][0]["cli_status"] == "REJECTED"
    assert payload["commands"][0]["stage_status"] == "BLOCKED"
    assert len(calls) == 1


def test_main_quality_failed_build_does_not_consume_final(monkeypatch, tmp_path):
    report = tmp_path / "quality-failed.json"
    recipe = tmp_path / "recipe.json"
    final_suite = tmp_path / "final-suite.json"
    final_output = tmp_path / "final-output.json"
    recipe.write_text("{}", encoding="utf-8")
    final_suite.write_text("[]", encoding="utf-8")
    calls = []

    def fake_run(argv, *args, **kwargs):
        calls.append(argv)
        receipt = {
            "command": "build",
            "completed": True,
            "status": "COMPLETED",
            "result": {"engineering_complete": True, "quality_pass": False},
        }
        return {"returncode": 0, "stdout": json.dumps(receipt), "stderr": "", "elapsed_seconds": 0.01}

    monkeypatch.setattr(runner, "environment_report", lambda *args: {"env": "ok"})
    monkeypatch.setattr(runner, "specialist_help", lambda *args: {"root": {"returncode": 0}})
    monkeypatch.setattr(runner, "run", fake_run)

    code = runner.main([
        "--repo-root", str(tmp_path),
        "--recipe", str(recipe),
        "--workspace", str(tmp_path / "study"),
        "--final-suite", str(final_suite),
        "--final-output", str(final_output),
        "--run-final",
        "--report", str(report),
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["status"] == "REJECTED"
    assert payload["finalization"].startswith("not run")
    assert payload["pre_final_gate"]["required_quality_pass"] is True
    assert len(calls) == 1
