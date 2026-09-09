"""Studio privacy/CLI glue tests: fresh synthetic metadata, no model/dataset runs."""
import copy
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
from asea.compose.__main__ import parser

ROOT = Path(__file__).resolve().parents[1]
SECRET = "fresh-private-return-<img src=x onerror=alert(1)>"


def req(**values):
    return ex.JobRequest(**{**dict(mode="compose", operation="list", operator="Local tester",
                                   local_files_confirmed=True), **values})


def command(tmp_path, **values):
    return ex.build_argv(req(**values), tmp_path, tmp_path / "artifacts")


def spec_file(tmp_path):
    spec = {"name": "Fresh preview only", "input_type": "text", "nodes": [{
        "id": "out", "input": "$input", "prompt_prefix": "Keep this legacy prose exactly.\n",
        "component": {"kind": "hf_text", "task": "causal", "model_path": str(tmp_path / "ABSENT_MODEL"),
                      "risk": "low", "provenance": [{"source": "synthetic test", "license": "test-only", "risk": "low"}]}}],
        "output_node": "out"}
    path = tmp_path / "new-preview-spec.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.mark.parametrize("fmt", ["raw_python", "fenced_python"])
@pytest.mark.parametrize("with_workspace", [False, True])
def test_public_preview_no_workspace_writes(tmp_path, fmt, with_workspace):
    path = spec_file(tmp_path)
    before = path.read_bytes()
    workspace = tmp_path / "must-not-be-created"
    argv = [sys.executable, "-m", "asea.compose"]
    if with_workspace:
        argv += ["--workspace", str(workspace)]
    argv += ["preview", "--spec", str(path), "--output-format", fmt]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}, cwd=tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)["result"]
    assert payload["read_only"] is True and payload["applied"] is False
    assert payload["output_format"] == fmt
    assert payload["nodes"][0]["existing_prompt_prefix"] == "Keep this legacy prose exactly.\n"
    assert payload["nodes"][0]["requires_prefix_review"] is True
    assert path.read_bytes() == before
    assert not workspace.exists() and not (tmp_path / "ABSENT_MODEL").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_preview_argv_and_required_format(tmp_path):
    argv, produced = command(tmp_path, operation="preview", spec="/missing/spec", output_format="raw_python")
    assert argv[3:] == ["preview", "--spec=/missing/spec", "--output-format=raw_python"]
    assert produced == [] and parser().parse_args(argv[3:]).workspace is None
    with pytest.raises(ex.StudioError, match="--output-format"):
        command(tmp_path, operation="preview", spec="/missing/spec")
    with pytest.raises(ex.StudioError, match="does not apply"):
        command(tmp_path, operation="preview", spec="/missing/spec", output_format="raw_python", output="rewrite.json")


@pytest.mark.parametrize("policy", ["none", "digest", "value"])
def test_retention_exact_flags_and_separate_consent(tmp_path, policy):
    values = dict(operation="evaluate", spec="/missing/spec", suite="/missing/suite", trace_policy=policy)
    if policy == "value":
        with pytest.raises(ex.StudioError, match="contain return values"):
            command(tmp_path, **values)
        values["value_trace_consent"] = True
    argv, _ = command(tmp_path, **values)
    assert parser().parse_args(argv[3:]).trace_policy == policy
    assert "--trace-policy=" + policy in argv
    assert not any("consent" in arg for arg in argv)
    assert "--include-sensitive-traces" not in argv


def test_export_separate_boolean_confirmation(tmp_path):
    base = dict(operation="export", deployment="dep")
    argv, _ = command(tmp_path, **base)
    assert not parser().parse_args(argv[3:]).include_sensitive_traces
    with pytest.raises(ex.StudioError, match="separate"):
        command(tmp_path, **base, include_sensitive_traces=True)
    argv, _ = command(tmp_path, **base, include_sensitive_traces=True, sensitive_export_confirmed=True)
    assert parser().parse_args(argv[3:]).include_sensitive_traces
    assert not any("confirmed" in arg for arg in argv)


@pytest.mark.parametrize("values", [
    {"trace_policy": "--help"}, {"trace_policy": True}, {"value_trace_consent": "true"},
    {"value_trace_consent": 1}, {"include_sensitive_traces": "true"}, {"sensitive_export_confirmed": 1},
    {"output_format": "python"}, {"cache_policy": "enabled"}, {"output_contract": {}}, {"arbitrary_flags": []},
])
def test_strict_unsupported_fields(values):
    with pytest.raises(ValidationError):
        req(**values)


@pytest.mark.parametrize("values", [
    {"trace_policy": "digest"}, {"value_trace_consent": True}, {"include_sensitive_traces": True},
    {"sensitive_export_confirmed": True}, {"output_format": "raw_python"},
    {"mode": "compiler", "operation": "evaluate", "trace_policy": "value", "value_trace_consent": True},
])
def test_wrong_operation_typed_rejection(tmp_path, values):
    with pytest.raises(ex.StudioError) as caught:
        command(tmp_path, **values)
    assert caught.value.kind == "INVALID_ARGUMENT"


def evidence_job(flagged=True):
    oracle = {"return_trace": {"schema_version": 1, "policy": {"schema_version": 1, "return_retention": "value"},
              "returns": [{"id": "fresh", "phase": "response", "retention": "value", "reason": "retained",
                           "value_type": "string", "sha256": "a" * 64, "value": SECRET}]},
              "evidence_flags": ["sensitive_actual_value_trace"] if flagged else [],
              "diagnostic": {"host_phase": "response", "candidate_error": {"trust": "untrusted_candidate", "type_code": "MemoryError"}},
              "termination": {"observed_exit": None, "observed_exit_state": "unknown", "final_exit": -9,
                              "cleanup_signal": "SIGKILL", "oom_proven": False}, "stdout": SECRET}
    payload = {"ok": False, "result": {"rows": [{"oracle_evidence": oracle}],
        "generation_trace_v1": [{"node": "out", "requested": {"cache_policy": "model_default"},
            "forwarded": {"generate_kwargs": {}}, "resolved": {"config": {"use_cache": True}, "cache_internal_state": "not_observed"},
            "generated_token_ids": [10, 20], "cap_reached": None, "stop_reason": "unknown",
            "media": {"processor_outputs": {"pixel_values": {"shape": [1, 2, 3, 8, 8]}}}}]}}
    return {"id": "a" * 32, "mode": "compose", "operation": "evaluate", "operator": "historical local operator",
            "status": "failed", "created_at": 1, "exit_code": 2, "artifacts": [], "argv": [], "result": payload,
            "stdout": json.dumps(payload), "stderr": SECRET, "error": {"type": "CLI_FAILED", "message": SECRET}}


@pytest.mark.parametrize("flagged", [True, False])
def test_projection_no_secret_alternate_path_and_no_mutation(flagged):
    original = evidence_job(flagged)
    before = copy.deepcopy(original)
    projected = ex.project_job(original)
    assert SECRET not in json.dumps(projected)
    assert projected["privacy"]["storage_authorization"] == ex.UNAVAILABLE
    assert original == before
    revealed = ex.project_job(original, True)
    if flagged:
        assert revealed["stdout"] == original["stdout"]
        assert SECRET in json.dumps(revealed)
    else:
        assert SECRET not in json.dumps(revealed)
        assert revealed["privacy"]["values_without_evidence_flag_withheld"] is True
    oracle = projected["observability"]["oracle"][0]
    assert oracle["HOST_original_exit_and_separate_cleanup_termination"]["observed_exit"] is None
    assert oracle["HOST_original_exit_and_separate_cleanup_termination"]["final_exit"] == -9
    assert oracle["HOST_diagnostic_with_UNTRUSTED_candidate_claims"]["candidate_error"]["type_code"] == "MemoryError"
    gen = projected["observability"]["generation_requested_forwarded_resolved_tokens_media"][0]["nodes"][0]
    assert gen["requested"]["cache_policy"] == "model_default"
    assert gen["forwarded"]["generate_kwargs"] == {}
    assert gen["cap_reached"] is None and gen["token_ids_truncated"] == ex.UNAVAILABLE


def test_unflagged_malformed_value_cannot_escape_by_missing_retention():
    job = evidence_job(False)
    del job["result"]["result"]["rows"][0]["oracle_evidence"]["return_trace"]["returns"][0]["retention"]
    assert SECRET not in json.dumps(ex.project_job(job, True))


def test_legacy_unavailable_not_false_and_invalid_raw_withheld():
    job = evidence_job()
    job.update(result={"ok": True, "result": {"legacy": True}}, stdout=SECRET)
    projection = ex.project_job(job)
    assert projection["observability"]["oracle"] == ex.UNAVAILABLE
    assert projection["observability"]["generation_requested_forwarded_resolved_tokens_media"] == ex.UNAVAILABLE
    job["result"] = None
    assert ex.project_job(job)["stdout"] == ex.REDACTED


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("SILT_ENABLE_EXPERIMENTAL", "1")
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    manager = ex.JobManager(tmp_path / "studio")
    monkeypatch.setattr(ex, "manager", lambda: manager)
    yield manager
    manager.close()


def test_api_all_projection_paths_auth_and_explicit_reveal(service):
    original = evidence_job()
    service.jobs[original["id"]] = original
    app = FastAPI()
    app.include_router(ex.router)
    path = "/api/experimental/jobs/" + original["id"]
    token = {"X-SILT-Experimental-Token": service.token}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for route in ("/api/experimental/jobs", path):
            response = client.get(route, headers=token)
            assert response.status_code == 200 and SECRET not in response.text
            assert response.headers["cache-control"] == "no-store"
            assert client.get(route).status_code == 403
        assert SECRET not in client.post(path + "/cancel", headers=token).text
        reveal = {**token, "X-SILT-Reveal-Sensitive": "true"}
        assert client.get(path, headers=reveal).json()["job"]["stdout"] == original["stdout"]
        for headers in ({**reveal, "Origin": "https://evil.invalid"}, {**reveal, "Host": "evil.invalid"},
                        {"X-SILT-Reveal-Sensitive": "true"}):
            assert client.get(path, headers=headers).status_code == 403
        assert client.get(path, headers={**token, "X-SILT-Reveal-Sensitive": "1"}).status_code == 422
        assert SECRET not in client.get(path, headers=token).text  # reveal is not sticky server state
        body = req(operation="evaluate", spec="/missing/spec", suite="/missing/suite", trace_policy="value").model_dump()
        denied = client.post("/api/experimental/jobs", headers=token, json=body)
        assert denied.json()["error"]["type"] == "CONFIRM_VALUE_TRACE"
        assert len(service.jobs) == 1


def test_studio_preview_only_outer_private_job_log(service, tmp_path):
    spec = spec_file(tmp_path)
    job_id = service.submit(req(operation="preview", spec=str(spec), output_format="fenced_python"))["id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = service.get(job_id)
        if job["status"] in ex.TERMINAL:
            break
        time.sleep(.02)
    assert job["status"] == "succeeded", job
    assert job["result"]["result"]["applied"] is False
    assert not (service.root / "compose").exists()
    evidence = service.root / "jobs" / job_id / "evidence.json"
    assert evidence.is_file() and evidence.stat().st_mode & 0o777 == 0o600
    assert job["observability_authorization"]["source"] == "studio_operator"
    assert job["observability_authorization"]["value_storage_consent"] is False


def test_preview_success_must_be_readonly_applied_false():
    job = {"mode": "compose", "operation": "preview"}
    for result in ({}, {"read_only": True, "applied": True}):
        with pytest.raises(ValueError, match="applied=false"):
            ex.decode_cli(job, json.dumps({"ok": True, "result": result}), "", 0)


def test_export_artifact_requires_independent_viewer_consent(service):
    job = evidence_job()
    artifact_dir = service.root / "jobs" / job["id"] / "artifacts"
    artifact_dir.mkdir(parents=True)
    path = artifact_dir / "original.zip"
    path.write_bytes(b"synthetic archive bytes only")
    service.jobs[job["id"]] = job
    service._register_artifacts(job, [(path, "deployment_export")])
    app = FastAPI()
    app.include_router(ex.router)
    token = {"X-SILT-Experimental-Token": service.token}
    url = "/api/experimental/jobs/" + job["id"] + "/artifacts/0"
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get(url, headers=token).status_code == 403
        assert client.get(url, headers={**token, "X-SILT-Reveal-Sensitive": "true"}).content == path.read_bytes()
        assert client.get(url, headers={"X-SILT-Reveal-Sensitive": "true"}).status_code == 403


def test_ui_selectors_textcontent_and_reset_contract():
    page = (ROOT / "src/asea/studio/static/experimental.html").read_text()
    for selector in ("output-format", "trace-policy", "value-trace-consent", "include-sensitive-traces",
                     "sensitive-export-confirmed", "reveal-sensitive", "generation-observations", "oracle-observations",
                     "observation-bindings", "projected-result", "privacy-status"):
        assert 'id="' + selector + '"' in page
    assert "innerHTML" not in page and "textContent" in page
    assert "contain return values" in page and "applied: false" in page
    assert "UNTRUSTED candidate" in page and "HOST original exit" in page
    assert "resetConsents()" in page and "resetReveal()" in page
    assert "X-SILT-Reveal-Sensitive" in page
    assert "document.querySelector('#reveal-sensitive').checked!==reveal" in page
