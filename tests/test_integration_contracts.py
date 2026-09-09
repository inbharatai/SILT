"""Integration contracts only: no downloaded weights or model-quality claims.

Adapter doubles below exercise evidence binding, not vision inference. Public CLI
and legacy-server integration checks run real entry points in isolated processes.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from pydantic import ValidationError

from asea.artifacts import Blocked, Workspace, digest
import asea.certification as cert
from asea.compose import runtime
from asea.compose.__main__ import parser
from asea.compose.schema import CompositionSpec, EvaluationCase, EvaluationSuite
from asea.studio import experimental as ex

ROOT = Path(__file__).resolve().parents[1]


def request(**changes):
    values = dict(mode="compose", operation="list", operator="Integration reviewer", local_files_confirmed=True)
    values.update(changes)
    return ex.JobRequest(**values)


def spec_for(tmp_path, input_type="image"):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{}')
    (model / "model.safetensors").write_bytes(b"NOT WEIGHTS: adapter-double contract only")
    kind, task = {"image": ("hf_vision", "smolvlm"), "text": ("hf_text", "causal"), "audio": ("hf_asr", "whisper")}[input_type]
    return CompositionSpec.model_validate({"name": "CONTRACT DOUBLE ONLY", "input_type": input_type,
        "nodes": [{"id": "answer", "input": "$input", "component": {"kind": kind, "task": task,
            "model_path": str(model), "risk": "high", "provenance": [{"source": "unit-test double", "risk": "high", "license": "test-only"}]}}], "output_node": "answer"})


def suite_for(**inputs):
    return EvaluationSuite(name="independent contract references", reference_source="test literals", cases=[
        EvaluationCase(id=group, group=group, reference="correct", metric="text_exact", threshold=1.0, **inputs)
        for group in ("target", "control")])


@pytest.fixture
def image_contract(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image = tmp_path / "input.png"
    Image.new("RGB", (2, 3), "red").save(image)
    spec = spec_for(tmp_path)
    monkeypatch.setattr(runtime, "_adapter", lambda *a, **kw: {"type": "text", "text": "correct"})
    return Workspace(tmp_path / "workspace"), spec, image


@pytest.mark.parametrize("prompt", [None, "What color is this?"])
def test_image_evaluation_and_activation_bind_bytes_and_question(image_contract, prompt):
    ws, spec, image = image_contract
    suite = suite_for(input=prompt, input_file=str(image))
    result = cert.evaluate(ws, spec, suite)
    assert result["status"] == "admitted", result  # double, NOT quality evidence
    expected = {"type": "image", **runtime._image_info(str(image), spec.limits),
                "text": runtime.DEFAULT_IMAGE_PROMPT if prompt is None else prompt}
    for row in result["case_evidence"]:
        assert ws.read_record("runs", row["run_id"])["input_digest"] == digest(expected)
    candidate = ws.read_record("candidates", result["candidate_id"])
    assert candidate["runtime_version"] == runtime.RUNTIME_VERSION
    assert candidate["graph_hash"] == digest({"spec": spec.model_dump(mode="json"),
        "artifacts": candidate["artifacts"], "runtime_version": runtime.RUNTIME_VERSION})
    selected = cert.select_measured(ws, [result["id"]], "image", "text", 1000.0, 10000.0, result["suite_hash"])
    assert selected["status"] == "selected" and selected["activated"] is False
    assert not (ws.root / "active.json").exists()
    assert cert.activate(ws, result["id"], approver="Contract reviewer")["approver"] == "Contract reviewer"


@pytest.mark.parametrize("change", ["bytes", "question", "run_digest", "runtime"])
def test_image_tampering_blocks_activation(image_contract, monkeypatch, change):
    ws, spec, image = image_contract
    result = cert.evaluate(ws, spec, suite_for(input="Original question", input_file=str(image)))
    assert result["status"] == "admitted"
    if change == "bytes":
        from PIL import Image
        Image.new("RGB", (2, 3), "blue").save(image)
    elif change == "runtime":
        monkeypatch.setattr(runtime, "RUNTIME_VERSION", runtime.RUNTIME_VERSION + "-changed")
    else:
        original = ws.read_record
        def changed(collection, identifier):
            record = original(collection, identifier)
            if collection == "evaluations" and change == "question":
                for case in record["suite"]["cases"]:
                    case["input"] = "Changed question"
                record["suite_hash"] = digest(record["suite"])
            if collection == "runs" and change == "run_digest":
                record["input_digest"] = "0" * 64
            return record
        monkeypatch.setattr(ws, "read_record", changed)
    with pytest.raises((Blocked, ValueError)):
        cert.activate(ws, result["id"], approver="Contract reviewer")
    assert not (ws.root / "active.json").exists()


def test_evaluator_checks_input_digest_before_admission(image_contract, monkeypatch):
    ws, spec, image = image_contract
    original = runtime._run
    def wrong_digest(*args, **kwargs):
        run = original(*args, **kwargs)
        run["input_digest"] = "0" * 64
        return run
    monkeypatch.setattr(runtime, "_run", wrong_digest)
    result = cert.evaluate(ws, spec, suite_for(input="Question", input_file=str(image)))
    assert result["status"] == "blocked" and not result["case_evidence"]
    assert "bind reference fixture input" in result["reason"]


@pytest.mark.parametrize("input_type,inputs", [
    ("text", {"input": "question", "input_file": "missing.png"}),
    ("audio", {"input": "question", "input_file": "missing.wav"}),
    ("text", {"input_file": "missing.png"}), ("audio", {"input": "question"}),
    ("image", {"input": "question"})])
def test_invalid_case_graph_combinations_block_before_runs(tmp_path, input_type, inputs):
    ws = Workspace(tmp_path / "workspace")
    result = cert.evaluate(ws, spec_for(tmp_path, input_type), suite_for(**inputs))
    assert result["status"] == "blocked"
    assert "evaluation requires" in result["reason"]
    assert not list((ws.root / "runs").iterdir())


@pytest.mark.parametrize("inputs", [{}, {"input": " "}, {"input_file": " "},
    {"input": "question", "input_file": ""}, {"input": "", "input_file": "image.png"}])
def test_case_still_rejects_empty_input_fields(inputs):
    with pytest.raises(ValidationError):
        suite_for(**inputs)


@pytest.mark.parametrize("input_type,inputs", [("text", {"input": "question"}), ("audio", {"input_file": "audio.wav"}),
    ("image", {"input_file": "image.png"}), ("image", {"input_file": "image.png", "input": "question"})])
def test_valid_case_shapes_preserve_text_audio(tmp_path, input_type, inputs):
    assert suite_for(**inputs).cases[0].validate_input_type(spec_for(tmp_path, input_type))


def test_select_and_plan_use_fixed_cli_argv(tmp_path):
    select = request(operation="select", evaluations=["evaluation-one", "evaluation-two"], input_type="image",
        output_type="text", max_wall_seconds=10.0, max_peak_rss_mb=128.0,
        max_disk_bytes=1024, suite_hash="a"*64, graph_hash="b"*64)
    argv, produced = ex.build_argv(select, tmp_path, tmp_path / "artifacts")
    args = parser().parse_args(argv[3:])
    assert args.command == "select" and args.evaluations == select.evaluations
    assert args.input_type == "image" and args.max_disk_bytes == 1024
    assert args.suite_hash == "a"*64 and args.graph_hash == "b"*64
    assert not produced and "activate" not in argv
    argv, produced = ex.build_argv(request(operation="plan", spec=str(tmp_path / "spec.json")), tmp_path, tmp_path)
    assert parser().parse_args(argv[3:]).command == "plan" and not produced
    argv, _ = ex.build_argv(request(operation="activate", evaluation="evaluation-one", confirmation="ACTIVATE", approve_high_risk=True), tmp_path, tmp_path)
    assert parser().parse_args(argv[3:]).approver == "Integration reviewer"


@pytest.mark.parametrize("identifier", ["--help", "--workspace=/tmp/escape", "one two", "one\n--help", "a\x00b", "x"*129])
def test_select_rejects_nargs_flag_injection(tmp_path, identifier):
    with pytest.raises(ex.StudioError, match="identifiers"):
        ex.build_argv(request(operation="select", evaluations=[identifier]), tmp_path, tmp_path)


@pytest.mark.parametrize("changes", [{"max_wall_seconds": float("nan")}, {"max_peak_rss_mb": float("inf")},
    {"max_disk_bytes": True}, {"evaluations": "evaluation-one"}, {"suite_hash": "bad"}])
def test_select_request_schema_is_strict(changes):
    with pytest.raises(ValidationError):
        request(operation="select", **changes)


def test_ui_action_contract_and_labels():
    page = (ROOT / "src/asea/studio/static/experimental.html").read_text()
    assert "compose:" + str(ex.ACTIONS["compose"]).replace(" ", "") in page
    for field in ("evaluations", "input_type", "output_type", "max_wall_seconds", "max_peak_rss_mb", "max_disk_bytes", "suite_hash", "graph_hash"):
        assert 'name="' + field + '"' in page
    assert "PNG/JPEG/WebP" in page and "--approver" in page
    assert "innerHTML" not in page and "textContent" in page


def test_public_cli_help_and_errors_are_one_json(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
    for command in (["--help"], ["--workspace", str(tmp_path / "store"), "select", "--evaluations", "evaluation-missing",
        "--input-type", "image", "--output-type", "text", "--max-wall-seconds", "1", "--max-peak-rss-mb", "128"],
        ["--workspace", str(tmp_path / "store"), "run"]):
        result = subprocess.run([sys.executable, "-m", "asea.compose", *command], env=env, capture_output=True, text=True, timeout=20)
        payload = json.loads(result.stdout)
        if command == ["--help"]:
            assert result.returncode == 0 and "image" in str(payload)
        else:
            assert result.returncode != 0 and payload["ok"] is False
    assert not (tmp_path / "store/active.json").exists()


def test_legacy_server_flag_contract_and_real_new_endpoint(tmp_path):
    # Fresh interpreter each time: module caches cannot hide accidental opt-in.
    # Only legacy workspace ROOT is redirected; actual app/routes/managers run.
    code = r'''
import json, os, sys, time
from pathlib import Path
from fastapi.testclient import TestClient
import asea.studio.jobs as jobs
jobs.ROOT = Path(os.environ["CONTRACT_ROOT"])
import asea.studio.server as server
on = os.environ.get("SILT_ENABLE_EXPERIMENTAL") == "1"
assert ("asea.studio.experimental" in sys.modules) is on
root = Path(os.environ["SILT_EXPERIMENTAL_ROOT"])
assert not root.exists()
# Snapshot the public HTTP contract, not FastAPI's version-specific route
# storage (newer versions can retain nested _IncludedRouter entries).
http_methods = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
legacy = sorted(
    (path, sorted(method.upper() for method in operations if method in http_methods),
     sorted((method, operation.get("operationId"))
            for method, operation in operations.items() if method in http_methods))
    for path, operations in server.app.openapi()["paths"].items()
    if not path.startswith(("/experimental", "/api/experimental"))
)
with TestClient(server.app, base_url="http://127.0.0.1") as client:
    health = client.get("/api/health").json()
    assert health["ok"] is True and health["mock_free"] is True
    assert client.get("/").content == (server.STATIC / "index.html").read_bytes()
    assert client.get("/api/readme").content == server.README_PATH.read_bytes()
    page = client.get("/experimental")
    if not on:
        assert page.status_code == 404
        assert client.get("/api/experimental/jobs").status_code == 404
        assert not root.exists()
    else:
        from asea.studio import experimental as ex
        assert page.status_code == 200
        token = ex.manager().token
        headers = {"X-SILT-Experimental-Token": token}
        assert token in page.text
        for method, path in [("GET", "/api/experimental/schema"), ("GET", "/api/experimental/jobs"),
            ("POST", "/api/experimental/jobs"), ("GET", "/api/experimental/jobs/missing"),
            ("POST", "/api/experimental/jobs/missing/cancel"), ("GET", "/api/experimental/jobs/missing/artifacts/0")]:
            assert client.request(method, path).status_code == 403
            assert client.request(method, path, headers={"X-SILT-Experimental-Token": "wrong"}).status_code == 403
            assert client.request(method, path, headers={**headers, "Origin": "https://silt.inbharat.ai"}).status_code == 403
            assert client.request(method, path, headers={**headers, "Host": "public.example"}).status_code == 403
        assert client.get("/experimental", headers={"Origin": "https://silt.inbharat.ai"}).status_code == 403
        assert client.get("/api/experimental/schema", headers=headers).json()["actions"] == ex.ACTIONS
        response = client.post("/api/experimental/jobs", headers=headers, json={"mode":"compose", "operation":"plan",
            "operator":"Contract reviewer", "local_files_confirmed":True, "spec":str(root / "missing.json")})
        assert response.status_code == 202
        identifier = response.json()["job"]["id"]
        until = time.monotonic() + 15
        while time.monotonic() < until:
            job = client.get("/api/experimental/jobs/" + identifier, headers=headers).json()["job"]
            if job["status"] in ex.TERMINAL:
                break
            time.sleep(.02)
        assert job["status"] == "failed" and job["error"]["type"] == "CLI_FAILED", job
        assert job["exit_code"] != 0 and job["result"]["ok"] is False
        assert job["result"]["error"]["type"]
        assert not (root / "compose/active.json").exists()
print(json.dumps(legacy))
'''
    snapshots = []
    for flag in (None, "0", "true", "1"):
        location = tmp_path / str(flag)
        location.mkdir()
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CONTRACT_ROOT": str(location),
            "SILT_EXPERIMENTAL_ROOT": str(location / "experimental"), "CUDA_VISIBLE_DEVICES": ""}
        env.pop("SILT_ENABLE_EXPERIMENTAL", None)
        if flag is not None:
            env["SILT_ENABLE_EXPERIMENTAL"] = flag
        result = subprocess.run([sys.executable, "-c", code], env=env, cwd=location, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        snapshots.append(json.loads(result.stdout))
    assert snapshots and all(routes == snapshots[0] for routes in snapshots)
    assert any(route[0] == "/api/health" for route in snapshots[0])
