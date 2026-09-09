"""Bridge tests: doubles test orchestration, NOT model quality. One real CLI failure."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from asea.studio import experimental as ex


@pytest.fixture
def service(tmp_path):
    instance = ex.JobManager(tmp_path / "experimental")
    yield instance
    instance.close()


@pytest.fixture
def client(monkeypatch, service):
    monkeypatch.setenv("SILT_ENABLE_EXPERIMENTAL", "1")
    monkeypatch.setattr(ex, "manager", lambda: service)
    app = FastAPI()
    app.include_router(ex.router)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


def req(**changes):
    values = dict(mode="compose", operation="list", operator="Test operator", local_files_confirmed=True)
    values.update(changes)
    return ex.JobRequest(**values)


def auth(service):
    return {"X-SILT-Experimental-Token": service.token}


def wait(service, job_id, status=None):
    until = time.monotonic() + 10
    while time.monotonic() < until:
        job = service.get(job_id)
        if (status and job["status"] == status) or (not status and job["status"] in ex.TERMINAL):
            return job
        time.sleep(.02)
    pytest.fail("job did not settle: " + str(service.get(job_id)))


def fake_process(monkeypatch, code):
    """Test-only transport double launches small Python, never exists in production."""
    real = subprocess.Popen
    seen = []
    def popen(argv, **kwargs):
        seen.append((argv, kwargs))
        return real([sys.executable, "-c", code], **kwargs)
    monkeypatch.setattr(ex.subprocess, "Popen", popen)
    return seen


def test_auth_gate_ui_and_schema(client, service, monkeypatch):
    page = client.get("/experimental")
    assert page.status_code == 200
    assert service.token in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert page.headers["cache-control"] == "no-store"
    assert client.get("/api/experimental/jobs").status_code == 403
    assert client.post("/api/experimental/jobs", json={}).status_code == 403
    assert client.get("/api/experimental/schema", headers=auth(service)).json()["actions"] == ex.ACTIONS
    for path in ("/experimental", "/api/experimental/jobs"):
        assert client.get(path, headers={**auth(service), "Origin": "https://silt.inbharat.ai"}).status_code == 403
        assert client.get(path, headers={**auth(service), "Sec-Fetch-Site": "cross-site"}).status_code == 403
        assert client.get(path, headers={**auth(service), "Host": "evil.example"}).status_code == 403
    monkeypatch.delenv("SILT_ENABLE_EXPERIMENTAL")
    assert client.get("/experimental").status_code == 404
    assert client.get("/api/experimental/jobs", headers=auth(service)).status_code == 404


def test_validation_typed_and_bounded(client, service):
    headers = auth(service)
    cases = [({}, "INVALID_REQUEST"),
             ({**req().model_dump(), "operation": "shell"}, "INVALID_ACTION"),
             ({**req().model_dump(), "local_files_confirmed": False}, "CONFIRM_LOCAL_FILES"),
             ({**req().model_dump(), "arbitrary_flags": "--help"}, "INVALID_REQUEST")]
    for body, error in cases:
        response = client.post("/api/experimental/jobs", json=body, headers=headers)
        assert response.status_code >= 400
        assert response.json()["error"]["type"] == error
    assert client.post("/api/experimental/jobs", content="{", headers=headers).json()["error"]["type"] == "INVALID_REQUEST"
    assert client.post("/api/experimental/jobs", content="x" * (ex.MAX_BODY + 1), headers=headers).status_code == 413
    assert client.get("/api/experimental/jobs/missing", headers=headers).json()["error"]["type"] == "JOB_NOT_FOUND"


def test_allowlisted_argv_no_flag_injection(service, tmp_path):
    args, _ = ex.build_argv(req(mode="compiler", operation="infer", model=str(tmp_path), input_text="--help; echo nope"), service.root, service.root / "artifacts")
    assert args[:4] == [sys.executable, "-m", "asea.compiler", "infer"]
    assert "--prompt=--help; echo nope" in args
    assert "--memory-budget-mib=2048" in args
    assert "--help" not in args
    args, _ = ex.build_argv(req(operation="list"), service.root, service.root / "artifacts")
    assert args == [sys.executable, "-m", "asea.compose", "--workspace", str(service.root / "compose"), "list"]


def test_explicit_named_activation_and_rollback(service):
    with pytest.raises(ex.StudioError, match="ACTIVATE"):
        ex.build_argv(req(operation="activate", evaluation="eval", approve_high_risk=True), service.root, service.root)
    with pytest.raises(ex.StudioError, match="Name the operator"):
        ex.build_argv(req(operation="activate", evaluation="eval", confirmation="ACTIVATE", operator=" "), service.root, service.root)
    args, _ = ex.build_argv(req(operation="activate", evaluation="eval", confirmation="ACTIVATE", approve_high_risk=True), service.root, service.root)
    assert "--approve-high-risk" in args
    assert "--evaluation=eval" in args
    args, _ = ex.build_argv(req(operation="activate", evaluation="eval", confirmation="ACTIVATE"), service.root, service.root)
    assert "--approve-high-risk" not in args
    with pytest.raises(ex.StudioError, match="ROLLBACK"):
        ex.build_argv(req(operation="rollback", deployment="dep"), service.root, service.root)


def test_output_containment_and_symlinks(service, tmp_path):
    for output in ("../escape", "/tmp/escape", "nested/escape", "--flags"):
        with pytest.raises(ex.StudioError):
            ex.build_argv(req(operation="export", deployment="dep", output=output), service.root, service.root / "artifacts")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "target")
    with pytest.raises(ex.StudioError):
        ex.build_argv(req(operation="inspect", spec=str(link)), service.root, service.root)
    with pytest.raises(ex.StudioError):
        ex.JobManager(tmp_path / ".studio" / "experimental")
    with pytest.raises(ex.StudioError, match="one Studio"):
        ex.JobManager(service.root)


def test_real_cli_failure(client, service, monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    body = req(mode="compiler", operation="inspect", model=str(tmp_path / "missing-model")).model_dump()
    response = client.post("/api/experimental/jobs", json=body, headers=auth(service))
    assert response.status_code == 202
    job = wait(service, response.json()["job"]["id"])
    assert job["status"] == "failed"
    assert job["exit_code"] != 0
    assert job["error"]["type"] == "CLI_FAILED"
    assert job["result"]["ok"] is False
    assert job["result"]["error"]["type"]
    evidence = service.root / "jobs" / job["id"] / "evidence.json"
    assert json.loads(evidence.read_text())["exit_code"] == job["exit_code"]
    assert (evidence.parent / "stderr.log").exists()


def test_success_stderr_and_no_autoactivation(service, monkeypatch):
    seen = fake_process(monkeypatch, "import sys,json;print('dependency log',file=sys.stderr);print(json.dumps({'ok':True,'result':{'candidate':'only'}}))")
    job = wait(service, service.submit(req())["id"])
    assert job["status"] == "succeeded"
    assert job["stderr"] == "dependency log\n"
    assert len(seen) == 1
    assert seen[0][0][-1] == "list"
    assert seen[0][1]["shell"] is False
    assert seen[0][1]["start_new_session"] is True
    assert not (service.root / "compose" / "active.json").exists()


@pytest.mark.parametrize("code,error", [("print('not json')", "INVALID_CLI_JSON"),
                                       ("import json;print(json.dumps({'ok':False,'error':{'type':'BLOCKED'}}))", "CLI_FAILED")])
def test_bad_cli_evidence(service, monkeypatch, code, error):
    fake_process(monkeypatch, code)
    job = wait(service, service.submit(req())["id"])
    assert job["error"]["type"] == error


def test_output_flood_bounded(service, monkeypatch):
    monkeypatch.setattr(ex, "MAX_OUTPUT", 4096)
    fake_process(monkeypatch, "import os;os.write(1,b'x'*200000);os.write(2,b'y'*200000)")
    job = wait(service, service.submit(req())["id"])
    assert job["status"] == "failed"
    assert job["error"]["type"] == "OUTPUT_LIMIT"
    assert job["output_bytes"] == 4096
    assert len(job["stdout"]) + len(job["stderr"]) <= 4096


def test_timeout(service, monkeypatch):
    fake_process(monkeypatch, "import time;time.sleep(30)")
    job = wait(service, service.submit(req(timeout_seconds=1))["id"])
    assert job["status"] == "timed_out"
    assert job["error"]["type"] == "TIMEOUT"


def test_serial_queue_and_cancel(service, monkeypatch):
    seen = fake_process(monkeypatch, "import time;time.sleep(30)")
    first = service.submit(req())["id"]
    wait(service, first, "running")
    while not seen:
        time.sleep(.01)
    pending = [service.submit(req())["id"] for _ in range(ex.MAX_QUEUE)]
    with pytest.raises(ex.StudioError, match="three jobs"):
        service.submit(req())
    assert len(seen) == 1
    for job_id in pending:
        assert service.cancel(job_id)["status"] == "cancelled"
    service.cancel(first)
    assert wait(service, first)["status"] == "cancelled"
    time.sleep(.1)
    assert len(seen) == 1


def test_process_group_cancels_descendants(service, monkeypatch, tmp_path):
    marker = tmp_path / "descendant-survived"
    child = "import time,pathlib;time.sleep(2);pathlib.Path(%r).write_text('bad')" % str(marker)
    code = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',%r]);print('started',flush=True);time.sleep(30)" % child
    fake_process(monkeypatch, code)
    job_id = service.submit(req())["id"]
    wait(service, job_id, "running")
    time.sleep(.25)
    service.cancel(job_id)
    assert wait(service, job_id)["status"] == "cancelled"
    time.sleep(2.1)
    assert not marker.exists()


def test_artifact_only_confirmed_roles_contained(client, service, monkeypatch, tmp_path):
    fake_process(monkeypatch, "import json;print(json.dumps({'ok':True}))")
    job = wait(service, service.submit(req())["id"])
    path = service.root / "jobs" / job["id"] / "artifacts" / "evidence.json"
    path.write_text('{"real_produced_role":true}')
    with service.lock:
        service._register_artifacts(service.jobs[job["id"]], [(path, "compiler_evidence")])
    url = "/api/experimental/jobs/" + job["id"] + "/artifacts/0"
    assert client.get(url).status_code == 403
    result = client.get(url, headers=auth(service))
    assert result.status_code == 200
    assert result.headers["content-disposition"].startswith("attachment")
    assert result.json()["real_produced_role"]
    assert client.get(url + "0", headers=auth(service)).status_code == 404
    path.unlink()
    path.symlink_to(tmp_path / "outside")
    assert client.get(url, headers=auth(service)).json()["error"]["type"] == "UNSAFE_PATH"
    assert client.get("/api/experimental/files?path=/etc/passwd", headers=auth(service)).status_code == 404


def test_restart_and_bounded_retention(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "MAX_JOBS", 2)
    fake_process(monkeypatch, "import json;print(json.dumps({'ok':True}))")
    service = ex.JobManager(tmp_path / "root")
    ids = []
    for _ in range(3):
        ids.append(service.submit(req())["id"])
        wait(service, ids[-1])
    assert len(service.list()) == 2
    assert not (service.root / "jobs" / ids[0] / "evidence.json").exists()
    service.close()
    restored = ex.JobManager(tmp_path / "root")
    try:
        assert len(restored.list()) == 2
        assert restored.get(ids[-1])["status"] == "succeeded"
    finally:
        restored.close()


def test_import_does_not_import_torch():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-c", "import sys;import asea.studio.experimental;assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules"], env=env, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode()
