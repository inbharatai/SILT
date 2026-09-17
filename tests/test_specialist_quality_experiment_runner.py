import json
from pathlib import Path
import pytest

from scripts import run_specialist_quality_experiment as runner
from asea.specialist import workflow as w
from asea.artifacts import digest


def bound_plan(recipe_path, device="cpu"):
    cfg = w.recipe_config(recipe_path)
    raw = json.loads(recipe_path.read_text())
    # Synthetic transport fixture, not actual data or model evidence.
    data = {name: {'path': cfg[name], 'sha256': '0'*64, 'size': 1,
                   'identity': {'device': 1, 'inode': i+1}}
            for i, name in enumerate(('training', 'validation_data', 'data_manifest', 'selection_lock', 'validation_suite'))}
    return {"schema_version": 1, "status": "READY", "execution_device": device,
            "model": {"files": {"fixture": {"size": 1, "sha256": "synthetic"}}, "inventory_sha256": "synthetic"},
            "bindings": {"recipe_sha256": digest(cfg), "workspace_parent": str(recipe_path.parent),
                         "requested_device": raw.get("execution_device", "auto"), "inventory_sha256": "synthetic", "data_metadata": data}}


def built_manifest(argv, plan, frozen="synthetic-frozen"):
    path = Path(argv[argv.index("--recipe") + 1])
    cfg = w.recipe_config(path)
    binding = json.loads(Path(argv[argv.index('--expected-data-binding-file') + 1]).read_text())
    return {'data': {'hashes': binding},
            'expected_data_binding_sha256': argv[argv.index('--expected-data-binding-sha256') + 1],
            "config": cfg, "config_sha256": w.json_hash(cfg), "execution_device": cfg["execution_device"],
            "recipe_binding": {"path": str(path), "sha256": argv[argv.index("--expected-recipe-sha256") + 1]},
            "source": {"path": cfg["source_path"], "files": plan["model"]["files"]}, "frozen_models_sha256": frozen}



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


# Fixture-only transport/orchestration tests, not model quality evidence.
def receipt(**changes):
    r = {"schema_version": 1, "command": "build", "status": "BUILT_UNCERTIFIED", "completed": True,
         "workspace": "/fixture/study", "manifest": "/fixture/study/manifest.json",
         "result": {"engineering_complete": True, "quality_pass": True, "qualified_source": True}}
    r.update(changes)
    return r


def command(value=None, **changes):
    r = {"returncode": 0, "stdout": json.dumps(value if value is not None else receipt()), "stderr": ""}
    r.update(changes)
    return r


@pytest.mark.parametrize("changes", [{"returncode": 1}, {"returncode": -9}, {"timeout": True}, {"launch_error": True},
                                     {"stdout": "broken"}, {"stdout": '{"completed":true}'}])
def test_bad_transport_never_completes(changes):
    assert runner.status_from_cli(command(**changes)) != "COMPLETED"


@pytest.mark.parametrize("changes", [{"status": "COMPLETED"}, {"status": "BLOCKED"}, {"status": "REJECTED"},
                                     {"schema_version": True}, {"completed": 1}, {"result": {}}, {"error": "corrupt"},
                                     {"command": "evaluate"}])
def test_inconsistent_receipts_never_complete(changes):
    assert runner.status_from_cli(command(receipt(**changes))) != "COMPLETED"


def test_engineering_completion_is_not_quality_and_raw_status_preserved():
    assert runner.status_from_cli(command()) == "COMPLETED"
    r = receipt()
    r["result"]["quality_pass"] = False
    assert runner.status_from_cli(command(r)) == "COMPLETED"
    assert not runner.build_allows_final(command(r))
    for error in ("missing tensor keys", "config.json corruption", "model not found in tensor index"):
        r = receipt(status="REJECTED", completed=False, result={"qualified_source": False, "error": error})
        assert runner.status_from_cli(command(r, returncode=1)) == "REJECTED"
    r["result"]["error"] = "not a regular local file: /models/config.json"
    assert runner.classify_cli(command(r, returncode=1))[1] == "LOCAL_INPUT_UNAVAILABLE"
    assert r["status"] == "REJECTED"


def test_duplicate_nonfinite_invalid_json():
    for text in ('{"completed":true,"completed":false}', '{"v":NaN}', '[1]'):
        assert runner.cli_receipt({"stdout": text}) is None


def test_owned_report_cannot_overwrite_replacement(tmp_path):
    p = tmp_path / "report"
    reserved = runner.ReservedReport(p)
    assert json.loads(p.read_text())["status"] == "RESERVED"
    p.rename(tmp_path / "original")
    p.write_text("user data")
    try:
        with pytest.raises(FileExistsError):
            reserved.write({"status": "completed"})
    finally:
        reserved.close()
    assert p.read_text() == "user data"


def test_exclusive_report_missing_parent(tmp_path):
    p = tmp_path / "report"
    runner.write_report(p, {})
    with pytest.raises(FileExistsError):
        runner.write_report(p, {})
    with pytest.raises(ValueError):
        runner.write_report(tmp_path / "new-parent" / "report", {})
    assert not (tmp_path / "new-parent").exists()


@pytest.fixture
def wrapped(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "environment_report", lambda *a: {"fixture": True})
    monkeypatch.setattr(runner, "specialist_help", lambda *a: {})
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"source_path": str(tmp_path / "teacher"), "source_metadata": {"canonical_id": "fixture/model"},
        "training": str(tmp_path / "train.json"), "calibration": str(tmp_path / "calibration.json"),
        "validation_data": str(tmp_path / "validation.json"), "validation_suite": str(tmp_path / "validation-suite.json")}))
    report = tmp_path / "report.json"
    args = ["--repo-root", str(tmp_path), "--recipe", str(recipe), "--workspace", str(tmp_path / "study"), "--report", str(report)]
    return args, report


@pytest.mark.parametrize("alias", ["existing", "workspace", "final", "nested", "symlink", "parent-symlink", "unwritable"])
def test_bad_output_paths_stop_before_work(monkeypatch, tmp_path, alias):
    monkeypatch.setattr(runner, "environment_report", lambda *a: pytest.fail("work before reservation"))
    report, study, output = tmp_path / "report", tmp_path / "study", tmp_path / "final"
    if alias == "existing":
        report.write_text("user")
    elif alias == "workspace":
        study = report
    elif alias == "final":
        output = report
    elif alias == "nested":
        report = study / "report"
    elif alias == "symlink":
        report.symlink_to(tmp_path / "target")
    elif alias == "parent-symlink":
        link = tmp_path / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        report = link / "report"
    else:
        directory = tmp_path / "readonly"
        directory.mkdir(mode=0o500)
        report = directory / "report"
    assert runner.main(["--report", str(report), "--workspace", str(study), "--final-output", str(output)]) == 2
    assert not study.exists()


def test_inventory_without_recipe_is_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "environment_report", lambda *a: {})
    monkeypatch.setattr(runner, "specialist_help", lambda *a: {})
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("build without recipe"))
    report = tmp_path / "out"
    assert runner.main(["--preflight-only", "--report", str(report)]) == 0
    assert json.loads(report.read_text())["execution_readiness"] == "NOT_READY_FOR_EXECUTION"


def test_preflight_real_plan_selected_python_not_build(wrapped, monkeypatch):
    args, report = wrapped
    calls = []
    def fake(argv, *a, **k):
        calls.append(argv)
        assert json.loads(report.read_text())["status"] == "RESERVED"
        return command({"schema_version": 1, "status": "PLANNING_ONLY", "execution_device": None})
    monkeypatch.setattr(runner, "run", fake)
    assert runner.main(args + ["--preflight-only", "--python", "selected-python", "--run-final", "--final-suite", "/unreadable-final"]) == 0
    assert len(calls) == 1
    assert calls[0][:4] == ["selected-python", "-m", "asea.hardware", "plan"]
    assert not (report.parent / "study").exists()


def test_candidate_mismatch_before_planning(wrapped, monkeypatch):
    args, report = wrapped
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("planning mismatched source"))
    assert runner.main(args + ["--candidate", "other/model"]) == 2
    assert "does not match" in json.loads(report.read_text())["error"]


def test_final_approval_before_build(wrapped, monkeypatch):
    args, report = wrapped
    monkeypatch.setattr(runner, "hardware_plan", lambda *a: ({"schema_version": 1, "status": "READY", "execution_device": "cpu", "plan_sha256": "a" * 64}, {}))
    monkeypatch.setattr(runner, "run", lambda *a, **k: pytest.fail("unapproved final started build"))
    assert runner.main(args + ["--run-final", "--final-suite", str(report.parent / "suite"), "--final-output", str(report.parent / "final")]) == 2
    assert "reviewed" in json.loads(report.read_text())["error"]


def test_launch_error_durable(wrapped, monkeypatch):
    args, report = wrapped
    monkeypatch.setattr(runner, "hardware_plan", lambda *a: (bound_plan(Path(args[args.index("--recipe") + 1])), {}))
    monkeypatch.setattr(runner, "run", lambda *a, **k: {"returncode": None, "stdout": "", "stderr": "", "launch_error": True})
    assert runner.main(args) == 1
    result = json.loads(report.read_text())
    assert result["status"] == "BLOCKED" and len(result["commands"]) == 1


def test_real_subprocess_bounded_capture(tmp_path):
    import sys
    result = runner.run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x'*1100000); sys.stderr.buffer.write(bytes([255]))"], tmp_path)
    assert result["returncode"] == 0
    assert len(result["stdout"]) <= runner.CAPTURE_LIMIT
    assert result["stdout_capture"]["bytes"] == 1100000 and result["stdout_capture"]["truncated"]
    assert result["stderr_capture"]["raw_sample_base64"] == "/w=="
    json.dumps(result)


def test_timeout_invalid_bytes_and_launch_failure(monkeypatch, tmp_path):
    import subprocess
    def expired(*a, **k):
        raise subprocess.TimeoutExpired("fixture", 1, output=bytes([255]), stderr=bytes([254]))
    monkeypatch.setattr(subprocess, "Popen", expired)
    result = runner.run(["fixture"], tmp_path)
    assert result["timeout"] and result["stdout_capture"]["invalid_utf8"]
    json.dumps(result)
    def failed(*a, **k):
        raise PermissionError("fixture executable")
    monkeypatch.setattr(subprocess, "Popen", failed)
    assert runner.run(["fixture"], tmp_path)["launch_error"]


def test_calibration_criteria_train_subset():
    criteria = runner.acceptance_criteria("fixture", "not run")
    assert criteria["status"] == "proposed_not_approved_by_code"
    assert "TRAIN-only subset" in " ".join(criteria["criteria"])


def test_atomic_terminal_publication_no_overwrite(tmp_path):
    path = tmp_path / "terminal.json"
    runner.publish_terminal(path, {"status": "BLOCKED"})
    assert json.loads(path.read_text())["status"] == "BLOCKED"
    with pytest.raises(FileExistsError):
        runner.publish_terminal(path, {"status": "COMPLETED"})
    assert not list(tmp_path.glob(".*"))


def test_wrapper_freezes_only_selected_device_and_preserves_budget(wrapped, monkeypatch):
    args, report = wrapped
    recipe_path = Path(args[args.index("--recipe") + 1])
    data = json.loads(recipe_path.read_text())
    data.update(execution_device="auto", output_budget_bytes=1234567, recovery={"steps": 2}, training="train.json")
    recipe_path.write_text(json.dumps(data))
    monkeypatch.setattr(runner, "hardware_plan", lambda *a: (bound_plan(Path(args[args.index("--recipe") + 1])), {}))
    def fake(argv, *a, **k):
        assert argv[3] == "build"
        effective = json.loads(Path(argv[argv.index("--recipe") + 1]).read_text())
        assert effective["execution_device"] == "cpu"
        assert effective["output_budget_bytes"] == 1234567 and effective["recovery"] == {"steps": 2}
        assert effective["training"] == str(report.parent / "train.json")
        study = report.parent / "study"
        study.mkdir()
        (study / "manifest.json").write_text(json.dumps(built_manifest(argv, bound_plan(recipe_path))))
        return command()
    monkeypatch.setattr(runner, "run", fake)
    assert runner.main(args) == 0
    summary = json.loads(report.read_text())
    assert summary["status"] == "BUILT_NOT_FINALIZED" and summary["certificate"] is False
    assert json.loads(Path(summary["terminal_report"]).read_text()) == summary
    assert json.loads(recipe_path.read_text())["execution_device"] == "auto"


def test_final_specific_required_flags():
    value = receipt(command="finalize")
    assert runner.classify_cli(command(value), "finalize")[0] == "REJECTED"
    value["result"].update(completed=True, status="BUILT_UNCERTIFIED", final_consumed=True, training_on_final=False)
    assert runner.classify_cli(command(value), "finalize")[0] == "COMPLETED"
    value["result"]["status"] = "REJECTED"
    assert runner.classify_cli(command(value), "finalize")[0] == "REJECTED"


def test_reviewed_policy_ledger_and_plan_bindings(tmp_path):
    import hashlib
    def put(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path
    lock = put("selection-lock.json", {"selection": [{"id": "fresh", "family": "fresh-family", "split": "final"}]})
    recipe = {"selection_lock": str(lock)}
    recipe_path = put("recipe.json", recipe)
    ledger = put("consumed-ledger.json", {"schema_version": 1, "authoritative": True,
                 "consumed": [{"id": "old", "family": "old-family"}]})
    reviewed = {"schema_version": 1, "status": "READY", "execution_device": "cpu",
                "bindings": {"recipe_sha256": "fixture-normalized", "inventory_sha256": "fixture-inventory", "hardware_sha256": "before"}}
    reviewed["plan_sha256"] = hashlib.sha256(json.dumps(reviewed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan_file = put("reviewed.json", reviewed)
    acceptance = {"schema_version": 1, "status": "APPROVED", "policy": "engineering_all_pass_v1",
                  "plan_sha256": reviewed["plan_sha256"], "recipe_sha256": runner.sha256_file(recipe_path)["sha256"],
                  "compact_baseline": "not_run_engineering_only", "exclude_ledger_sha256": runner.sha256_file(ledger)["sha256"]}
    record = put("acceptance.json", acceptance)
    args = runner.parse_args(["--recipe", str(recipe_path), "--reviewed-plan", str(plan_file),
                             "--reviewed-plan-sha256", reviewed["plan_sha256"], "--acceptance-record", str(record),
                             "--exclude-ledger", str(ledger)])
    live = dict(reviewed, bindings=dict(reviewed["bindings"], hardware_sha256="after"))
    assert runner.reviewed_acceptance(args, live, recipe) == acceptance
    drift = dict(live, execution_device="cuda:0")
    with pytest.raises(ValueError, match="binding"):
        runner.reviewed_acceptance(args, drift, recipe)
    put("selection-lock.json", {"selection": [{"id": "old", "family": "fresh-family"}]})
    with pytest.raises(ValueError, match="overlaps"):
        runner.reviewed_acceptance(args, live, recipe)


def test_known_final_path_cannot_be_pre_final_recipe(tmp_path):
    path = tmp_path / "final-suite.json"
    args = runner.parse_args(["--report", str(tmp_path / "report"), "--recipe", str(path), "--final-suite", str(path)])
    with pytest.raises(ValueError, match="final suite"):
        runner.check_output_paths(args)
