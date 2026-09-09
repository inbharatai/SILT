"""Transport/selector tests, not model-quality evidence. No training or model loads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from asea.studio import experimental as ex


@pytest.fixture
def service(tmp_path):
    instance = ex.JobManager(tmp_path / "studio")
    yield instance
    instance.close()


@pytest.fixture
def client(monkeypatch, service):
    monkeypatch.setenv("SILT_ENABLE_EXPERIMENTAL", "1")
    monkeypatch.setattr(ex, "manager", lambda: service)
    app = FastAPI()
    app.include_router(ex.router)
    with TestClient(app, base_url="http://127.0.0.1") as browser:
        yield browser


@pytest.fixture
def recipe(tmp_path):
    # Parser-only local inputs. Deliberately not runnable model or training data.
    (tmp_path / "source").mkdir()
    config = {"source_path": "source", "source_metadata": {"canonical_id": "test-only/parser-fixture"}}
    for key in ("calibration", "training", "validation_data", "validation_suite", "data_manifest", "selection_lock"):
        name = key + ".json"
        (tmp_path / name).write_text("{}")
        config[key] = name
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(config))
    return path


def req(operation="infer", **changes):
    return ex.JobRequest(mode="specialist", operation=operation, operator="Test operator",
                         local_files_confirmed=True, **changes)


def auth(service, reveal=False):
    return {"X-SILT-Experimental-Token": service.token, "X-SILT-Reveal-Sensitive": str(reveal).lower()}


def wait(service, job_id, running=False):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = service.get(job_id)
        if job["status"] in ex.TERMINAL or (running and job["status"] == "running"):
            return job
        time.sleep(.02)
    pytest.fail("transport did not settle")


def fake_process(monkeypatch, code):
    real = subprocess.Popen
    calls = []
    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        return real([sys.executable, "-c", code], **kwargs)
    monkeypatch.setattr(ex.subprocess, "Popen", launch)
    return calls


def receipt(command, **changes):
    result = dict(schema_version=1, command=command,
                  status="BUILT_UNCERTIFIED" if command in ("build", "finalize") else "completed", completed=True,
                  result={"engineering_complete": True, "quality_pass": False, "certificate": False})
    result.update(changes)
    return result


def test_fixed_argv_selectors_exact_prompt_and_cli_defaults(service, tmp_path, recipe):
    artifacts = service.root / "jobs" / "a" / "artifacts"
    prompt = "--help; $(echo no)\nReturn raw Python."
    args, files = ex.build_argv(req(model=str(tmp_path / "missing"), input_text=prompt), service.root, artifacts)
    assert args == [sys.executable, "-m", "asea.specialist", "infer", "--model=" + str(tmp_path / "missing"),
                    "--prompt=" + prompt, "--dtype=bfloat16", "--max-new-tokens=256"]
    assert not files
    args, files = ex.build_argv(req("build", recipe=str(recipe), study_name="new-study", timeout_seconds=3600), service.root, artifacts)
    assert args[:4] == [sys.executable, "-m", "asea.specialist", "build"]
    assert args[-1] == "--workspace=" + str(artifacts.parent / "studies" / "new-study")
    assert files == [(artifacts.parent / "studies" / "new-study", "specialist_study")]
    args, files = ex.build_argv(req("evaluate", model=str(tmp_path), suite=str(recipe), output="report.json"), service.root, artifacts)
    assert "--trace-policy=digest" in args
    assert "--output=" + str(artifacts / "report.json") in args
    assert files == [(artifacts / "report.json", "specialist_report")]


@pytest.mark.parametrize("operation", ["reconstruct", "recover", "validate", "inspect", "noop", "--help", "shell", "asea.compiler"])
def test_initial_ui_action_allowlist(service, operation):
    with pytest.raises(ex.StudioError, match="allowlisted"):
        ex.build_argv(req(operation), service.root, service.root / "artifacts")


@pytest.mark.parametrize("field,value", [("keep_experts", 4), ("trace_policy", "none"), ("confirmation", ""),
    ("approve_high_risk", False), ("recipe", None), ("resource_profile", "observe_only"), ("output", "unused.json"),
    ("case_timeout", 1), ("memory_mib", 100), ("max_length", 128)])
def test_known_but_inapplicable_fields_refused_even_defaults(service, tmp_path, field, value):
    with pytest.raises(ex.StudioError, match="do not apply"):
        ex.build_argv(req(model=str(tmp_path), input_text="hi", **{field: value}), service.root, service.root / "artifacts")


def test_unknown_request_fields_and_api_budgets(client, service, recipe):
    headers = auth(service)
    body = req("build", recipe=str(recipe), study_name="new", timeout_seconds=3600).model_dump(exclude_unset=True)
    for extra in ({"command": "touch anything"}, {"teacher_mode": "resident"}, {"retention": .5}, {"arbitrary_flags": []}):
        response = client.post("/api/experimental/jobs", json={**body, **extra}, headers=headers)
        assert response.status_code == 422
    for timeout in (3601, True, 0, -1, "3600"):
        assert client.post("/api/experimental/jobs", json={**body, "timeout_seconds": timeout}, headers=headers).status_code == 422
    for mode, op in (("specialist", "infer"), ("specialist", "evaluate"), ("specialist", "finalize"),
                     ("compiler", "inspect"), ("compose", "list"), ("execution", "probe"), ("validation", "voice-check")):
        with pytest.raises(ValidationError, match="900"):
            ex.JobRequest(mode=mode, operation=op, operator="test", timeout_seconds=901)
    with pytest.raises(ValidationError):
        ex.JobRequest(mode="compiler", operation="infer", operator="test", max_new_tokens=129)
    assert req(max_new_tokens=384).max_new_tokens == 384
    with pytest.raises(ValidationError):
        req(max_new_tokens=385)
    assert client.post("/api/experimental/jobs", content="x" * (ex.MAX_BODY + 1), headers=headers).status_code == 413
    assert not service.jobs


def test_specialist_consent_before_any_recipe_reads(service, monkeypatch):
    def unexpected_read(path):
        pytest.fail("recipe read before operator authorization")
    monkeypatch.setattr(ex, "approved_recipe", unexpected_read)
    with pytest.raises(ex.StudioError, match="confirm permission"):
        service.submit(ex.JobRequest(mode="specialist", operation="build", operator="test",
                                     recipe="/unread/recipe.json", study_name="new"))
    assert not service.jobs


def test_recipe_commands_paths_and_study_name_rejected(service, recipe, tmp_path):
    for name in ("../escape", "/absolute", "nested/study", "--help", ""):
        with pytest.raises(ex.StudioError):
            ex.build_argv(req("build", recipe=str(recipe), study_name=name), service.root, service.root / "artifacts")
    value = json.loads(recipe.read_text())
    value["command"] = "arbitrary executable"
    recipe.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unknown recipe"):
        service.submit(req("build", recipe=str(recipe), study_name="new"))
    value.pop("command")
    value["source_path"] = str(tmp_path / "missing-source")
    recipe.write_text(json.dumps(value))
    with pytest.raises(ex.StudioError, match="source_path"):
        service.submit(req("build", recipe=str(recipe), study_name="new"))


def test_output_path_type_and_symlink_rejected(service, tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text("{}")
    for output in ("../outside.json", "/tmp/out.json", "directory", "--help.json", "x/y.json"):
        with pytest.raises(ex.StudioError):
            ex.build_argv(req("evaluate", model=str(tmp_path), suite=str(suite), output=output), service.root, service.root / "artifacts")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ex.StudioError):
        ex.build_argv(req(model=str(alias), input_text="hi"), service.root, service.root / "artifacts")
    with pytest.raises(ex.StudioError):
        ex.build_argv(req(model=str(suite), input_text="hi"), service.root, service.root / "artifacts")
    with pytest.raises(ex.StudioError):
        ex.build_argv(req(model=str(tmp_path), input_text="\x00"), service.root, service.root / "artifacts")


def test_final_confirmation_persisted_before_launch(service, tmp_path, monkeypatch):
    study = tmp_path / "frozen-study"
    study.mkdir()
    (study / "manifest.json").write_text("{}")
    suite = tmp_path / "final-suite.json"
    suite.write_text("{}")
    for confirmation in ("", "FINALIZE ", "ACTIVATE", "finalize"):
        with pytest.raises(ex.StudioError, match="FINALIZE"):
            service.submit(req("finalize", study=str(study), suite=str(suite), confirmation=confirmation))
    calls = fake_process(monkeypatch, "import json;print(json.dumps(" + repr(receipt("finalize")) + "))")
    job = wait(service, service.submit(req("finalize", study=str(study), suite=str(suite), confirmation="FINALIZE"))["id"])
    saved = json.loads((service.root / "jobs" / job["id"] / "evidence.json").read_text())
    assert saved["specialist_authorization"]["confirmation"] == "FINALIZE"
    assert saved["specialist_authorization"]["request"]["suite"] == str(suite)
    assert saved["specialist_authorization"]["local_files_confirmed"] is True
    assert "--study=" + str(study) in calls[0][0]
    assert "--confirmation" not in " ".join(calls[0][0])
    assert not (study / "final-consumed.json").exists()  # Studio never consumes itself.


def test_build_snapshot_stage_status_and_no_checkpoint_download(service, recipe, monkeypatch):
    calls = fake_process(monkeypatch, "import time,json;time.sleep(.15);print(json.dumps(" + repr(receipt("build")) + "))")
    submitted = service.submit(req("build", recipe=str(recipe), study_name="new-study", timeout_seconds=3600))
    directory = service.root / "jobs" / submitted["id"]
    snapshot = directory / "recipe.approved.json"
    config = json.loads(snapshot.read_text())
    assert Path(config["source_path"]).is_absolute()
    assert config["recovery"]["teacher_mode"] == "cached"
    assert submitted["specialist_authorization"]["source_metadata"]["canonical_id"] == "test-only/parser-fixture"
    recipe.write_text('{"command":"changed after approval"}')
    assert json.loads(snapshot.read_text()) == config
    study = directory / "studies" / "new-study"
    study.mkdir()
    (study / "recovered").mkdir()
    (study / "recovered" / "huge.safetensors").write_bytes(b"not a real model")
    (study / "manifest.json").write_text('{"status":"BUILT_UNCERTIFIED"}')
    (study / "unlisted.json").write_text("{}")
    (study / "source-validation.started.json").write_text('{"status":"RUNNING"}')
    job = wait(service, submitted["id"])
    assert job["study_status"]["stages"][0]["started_record_exists"] is True
    assert job["study_status"]["stages"][0]["result_record_exists"] is False
    assert job["timeout_seconds"] == 3600
    assert "--recipe=" + str(snapshot) in calls[0][0]
    assert calls[0][1]["shell"] is False and calls[0][1]["start_new_session"] is True
    assert {a["name"] for a in job["artifacts"]} == {"manifest.json", "source-validation.started.json"}
    assert all(a["download_available"] is False for a in job["directory_artifacts"])
    assert any(a["role"] == "recovered_candidate_not_certified" for a in job["directory_artifacts"])
    assert job["summary"]["completed"] is True and job["summary"]["engineering_complete"] is True
    assert job["summary"]["quality_pass"] is False and job["summary"]["certificate"] is False
    assert job["summary"]["diagnosis"] == "completed_quality_not_passed"


@pytest.mark.parametrize("completed,status,exit_code,diagnosis", [(True, "completed", 0, "completed_quality_not_passed"),
    (False, "BLOCKED", 1, "infrastructure_blocked"), (False, "REJECTED", 1, "rejected_or_incomplete")])
def test_receipt_completion_not_quality(completed, status, exit_code, diagnosis):
    job = {"mode": "specialist", "operation": "evaluate", "status": "succeeded" if completed else "failed",
           "exit_code": exit_code, "error": None if completed else {"type": "CLI_FAILED"}}
    payload, actual = ex.decode_cli(job, json.dumps(receipt("evaluate", completed=completed, status=status)), "", exit_code)
    assert actual is completed
    job["result"] = payload
    assert ex.completion_summary(job)["diagnosis"] == diagnosis
    for bad in ({"ok": True}, receipt("build"), receipt("evaluate", completed="true")):
        with pytest.raises(ValueError):
            ex.decode_cli(job, json.dumps(bad), "", 0)


def test_specific_bounded_reports_only_and_reveal(client, service, tmp_path, monkeypatch):
    suite = tmp_path / "suite.json"
    suite.write_text("{}")
    # Use a transport stub then register its known output; no model loading.
    fake_process(monkeypatch, "import json;print(json.dumps(" + repr(receipt("evaluate")) + "))")
    job = wait(service, service.submit(req("evaluate", model=str(tmp_path), suite=str(suite)))["id"])
    output = service.root / "jobs" / job["id"] / "artifacts" / "evaluation.json"
    output.write_text('{"quality_pass":false}')
    live = service.jobs[job["id"]]
    ex.JobManager._register_artifacts(service, live, [(output, "specialist_report")])
    item = live["artifacts"][0]
    route = "/api/experimental/jobs/" + job["id"] + "/artifacts/" + item["id"]
    assert client.get(route, headers=auth(service)).status_code == 403
    response = client.get(route, headers=auth(service, True))
    assert response.status_code == 200 and response.json()["quality_pass"] is False
    assert response.headers["cache-control"] == "no-store"
    assert client.get(route + "/arbitrary-path", headers=auth(service, True)).status_code == 404
    output.write_text("longer changed content")
    assert client.get(route, headers=auth(service, True)).status_code == 409
    other = service.root / "other.json"
    other.write_text("{}")
    live["artifacts"][0]["path"] = "other.json"
    assert client.get(route, headers=auth(service, True)).status_code >= 400
    with pytest.raises(ex.StudioError):
        service._register_artifacts(live, [(other, "specialist_report")])
    # A huge report is not registered, even though the checkpoint directory may be huge.
    monkeypatch.setattr(ex, "SPECIALIST_REPORT_LIMIT", 8)
    live["artifacts"] = []
    service._register_artifacts(live, [(output, "specialist_report")])
    assert not live["artifacts"]


def test_global_serial_queue_all_modes(service, recipe, tmp_path, monkeypatch):
    calls = fake_process(monkeypatch, "import time;time.sleep(30)")
    first = service.submit(req("build", recipe=str(recipe), study_name="new", timeout_seconds=3600))["id"]
    wait(service, first, running=True)
    pending = [service.submit(ex.JobRequest(mode="compose", operation="list", operator="test", local_files_confirmed=True))["id"],
               service.submit(req(model=str(tmp_path), input_text="hi"))["id"],
               service.submit(req("build", recipe=str(recipe), study_name="another"))["id"]]
    with pytest.raises(ex.StudioError, match="three jobs"):
        service.submit(req(model=str(tmp_path), input_text="hi"))
    assert len(calls) <= 1
    for job_id in pending + [first]:
        service.cancel(job_id)
    assert wait(service, first)["status"] == "cancelled"


def test_actual_tiny_cli_missing_model_no_optional_ml(service, tmp_path, monkeypatch):
    source = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv("PYTHONPATH", source)
    job = wait(service, service.submit(req(model=str(tmp_path / "does-not-exist"), input_text="no model loaded"))["id"])
    assert job["status"] == "failed" and job["exit_code"] == 1
    assert job["result"]["command"] == "infer" and job["result"]["completed"] is False
    assert job["error"]["type"] == "CLI_FAILED"
    code = """import importlib.abc,sys
class DenyML(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','peft','numpy'}: raise AssertionError('optional ML imported: '+fullname)
sys.meta_path.insert(0,DenyML())
import asea.studio.experimental
from asea.specialist.__main__ import main
assert main(['infer','--model','/definitely-missing-specialist-model','--prompt','test']) == 1
assert not ({'torch','transformers','peft','numpy'} & set(sys.modules))
"""
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": source}, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_specialist_native_tokens_need_reveal_without_changing_old_projection():
    original = {"mode": "specialist", "result": {"result": {"generation": {
        "input_token_ids": [1, 2], "native_sequence_token_ids": [1, 2, 3], "generated_token_ids": [3]}}}}
    hidden = ex.project_job(original)["result"]["result"]["generation"]
    assert set(hidden.values()) == {ex.REDACTED}
    shown = ex.project_job(original, reveal=True)["result"]["result"]["generation"]
    assert shown["native_sequence_token_ids"] == [1, 2, 3]
    assert original["result"]["result"]["generation"]["input_token_ids"] == [1, 2]
    legacy = dict(original, mode="compiler")
    assert ex.project_job(legacy)["result"]["result"]["generation"]["input_token_ids"] == [1, 2]


def test_trusted_resource_dispatch_only():
    from asea.execution import controls
    from asea.execution.__main__ import parser
    parsed = parser().parse_args(["run", "--profile", "process_as", "--memory-mib", "256", "--timeout", "10", "--operation", "specialist", "--", "--help"])
    assert parsed.operation == "specialist"
    assert controls.run("os", ["--help"])["launched"] is False
    # Same actual bootstrap / parent-death / per-process-AS contract as old modules.
    result = controls.run("specialist", ["--help"], memory_mib=256, timeout=10)
    if not controls.probe()["process_as"]["execution_supported"]:
        assert result["launched"] is False
        return
    assert result["status"] == "OK", result
    assert result["parent_death_established"] is True
    assert result["memory_metric"] == "virtual_address_space_per_process"
    assert result["hard_rss_enforced"] is False and result["aggregate_memory_enforced"] is False
    assert "reconstruct" in result["stdout"] and "finalize" in result["stdout"]


def test_ui_schema_specialist_controls(client, service):
    page = client.get("/experimental").text
    assert '<option value="specialist">' in page
    assert "specialist:['build','infer','evaluate','finalize']" in page
    assert "Type FINALIZE" in page and "including failure" in page
    assert "BUILT_UNCERTIFIED is NOT quality PASS" in page
    assert "download_available" not in page  # Directory info displayed, not fabricated download links.
    assert "directory_artifacts:job.directory_artifacts" in page
    schema = client.get("/api/experimental/schema", headers=auth(service)).json()
    assert schema["worker_count"] == 1 and schema["queue_limit"] == 3
    assert schema["max_timeout_seconds"] == 900 and schema["specialist_build_max_timeout_seconds"] == 3600
