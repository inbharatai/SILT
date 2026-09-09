"""NEWPASS Studio transport/negative proofs. No model quality or human ratings."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from asea.studio import experimental as ex


ROOT = Path(__file__).resolve().parents[1]


def request(**values):
    return ex.JobRequest(**{**dict(mode="compose", operation="list", operator="Test operator",
                                   local_files_confirmed=True), **values})


def argv(tmp_path, **values):
    return ex.build_argv(request(**values), tmp_path, tmp_path / "artifacts")


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    manager = ex.JobManager(tmp_path / "studio")
    yield manager
    manager.close()


def settled(service, req):
    job_id = service.submit(req)["id"]
    until = time.monotonic() + 20
    while time.monotonic() < until:
        job = service.get(job_id)
        if job["status"] in ex.TERMINAL:
            return job
        time.sleep(.02)
    pytest.fail("CLI did not settle: " + str(service.get(job_id)))


def test_default_old_argv_unchanged(tmp_path):
    command, produced = argv(tmp_path)
    assert command == [sys.executable, "-m", "asea.compose", "--workspace", str(tmp_path / "compose"), "list"]
    assert produced == []
    command, _ = argv(tmp_path, operation="evaluate", spec="/missing/spec", suite="/missing/suite")
    assert command[-2:] == ["--spec=/missing/spec", "--suite=/missing/suite"]
    command, _ = argv(tmp_path, mode="compiler", operation="infer", model="/missing/model", input_text="hello")
    assert command[3:] == ["infer", "--model=/missing/model", "--dtype=float32", "--max-length=128",
                           "--memory-budget-mib=2048", "--prompt=hello", "--max-new-tokens=32"]


@pytest.mark.parametrize("profile", ["observe_only", "process_as", "cgroup_v2"])
def test_compose_profiles_exact_no_downgrade(tmp_path, profile):
    command, _ = argv(tmp_path, operation="evaluate", spec="/missing/spec", suite="/missing/suite",
                      resource_profile=profile, memory_mib=1024, case_timeout=12.5)
    assert command[-3:] == ["--resource-profile=" + profile, "--memory-mib=1024", "--case-timeout=12.5"]


@pytest.mark.parametrize("operation", ["diagnose", "roundtrip"])
def test_diagnostic_exact_public_parser(tmp_path, operation):
    from asea.compiler.__main__ import parser
    command, produced = argv(tmp_path, mode="compiler", operation=operation, model="/missing/model",
                             suite="/missing/suite", dtype="bfloat16", preserve_router_fp32=True,
                             worker_timeout_seconds=45.0)
    parsed = parser().parse_args(command[3:])
    assert parsed.dtype == "bfloat16" and parsed.preserve_router_fp32
    assert parsed.worker_timeout_seconds == 45 and parsed.max_length == 64
    assert parsed.max_new_tokens == 16 and parsed.suite == "/missing/suite"
    assert produced[0][1] == "compiler_audit"
    assert Path(parsed.output).is_relative_to(tmp_path)
    assert not any(flag.startswith("--scorer") for flag in command)


def test_prune_and_roundtrip_options(tmp_path):
    from asea.compiler.__main__ import parser
    command, _ = argv(tmp_path, mode="compiler", operation="prune", model="/missing/model",
                      calibration="/missing/calibration", scorer="reap_dispatched", minimum_expert_observations=4,
                      dtype="bfloat16", preserve_router_fp32=True)
    parsed = parser().parse_args(command[3:])
    assert parsed.scorer == "reap_dispatched" and parsed.minimum_expert_observations == 4
    command, _ = argv(tmp_path, mode="compiler", operation="roundtrip", model="/missing/model")
    assert not any(flag.startswith("--suite") for flag in command)
    assert "--dtype=float16" in command and "--worker-timeout-seconds=120" in command


@pytest.mark.parametrize("values", [
    {"memory_mib": 0}, {"memory_mib": True}, {"resource_profile": "auto"},
    {"case_timeout": float("nan")}, {"case_timeout": 86401},
    {"worker_timeout_seconds": float("inf")}, {"worker_timeout_seconds": 901},
    {"minimum_expert_observations": 0}, {"scorer": "usage"}, {"arbitrary_flags": "--help"},
    {"use_purpose": "approved"}, {"listening_approval": True}, {"module": "os"},
])
def test_strict_bounded_fields(values):
    with pytest.raises(ValidationError):
        request(**values)


@pytest.mark.parametrize("values", [
    {"resource_profile": "process_as"}, {"mode": "compiler", "operation": "infer", "scorer": "probability_mass"},
    {"mode": "compiler", "operation": "evaluate", "preserve_router_fp32": True},
    {"mode": "compiler", "operation": "diagnose", "dtype": "float32"},
    {"mode": "compiler", "operation": "diagnose", "max_length": 128},
    {"mode": "compiler", "operation": "diagnose", "worker_timeout_seconds": 301},
    {"mode": "compiler", "operation": "roundtrip", "reference_model": "/missing/reference"},
    {"mode": "compiler", "operation": "certify", "dtype": "bfloat16"},
    {"mode": "execution", "operation": "run"}, {"mode": "execution", "operation": "probe", "memory_mib": 128},
    {"mode": "execution", "operation": "probe", "model": "/missing/model"},
    {"mode": "validation", "operation": "check-review"},
    {"mode": "validation", "operation": "voice-check", "approve_high_risk": True},
])
def test_wrong_target_fails_before_spawn(tmp_path, values):
    with pytest.raises(ex.StudioError):
        argv(tmp_path, **values)


def test_voice_prefix_and_literal_text_no_injection(tmp_path):
    command, produced = argv(tmp_path, mode="validation", operation="voice-check", audio_file="/missing/audio.wav",
                             text="--help; touch should-not-exist", language="en", model_manifest="/missing/voice.json",
                             use_purpose="commercial")
    assert command[:6] == [sys.executable, "-m", "asea.validation", "--workspace", str(tmp_path / "validation"), "voice-check"]
    assert "--text=--help; touch should-not-exist" in command
    assert "--audio=/missing/audio.wav" in command
    assert not any(flag.startswith(("--use", "--purpose", "--approve", "--sign")) for flag in command)
    assert produced[0][1] == "validation_evidence"
    with pytest.raises(ex.StudioError, match="filename"):
        argv(tmp_path, mode="validation", operation="export-listen-batch", manifest="/missing/batch.json", output="../escape")


@pytest.mark.parametrize("mode,operation,values", [
    ("compose", "evaluate", {"spec": "/missing/spec", "suite": "/missing/suite", "resource_profile": "cgroup_v2"}),
    ("compiler", "diagnose", {"model": "/missing/model", "suite": "/missing/suite", "dtype": "bfloat16"}),
    ("compiler", "roundtrip", {"model": "/missing/model", "dtype": "float16"}),
    ("validation", "voice-check", {"audio_file": "/missing/audio", "text": "--help", "language": "en", "model_manifest": "/missing/manifest"}),
    ("validation", "export-listen-batch", {"manifest": "/missing/manifest"}),
])
def test_real_newpass_cli_negative_proof(service, mode, operation, values):
    job = settled(service, request(mode=mode, operation=operation, **values))
    assert job["status"] == "failed", job
    assert job["error"]["type"] == "CLI_FAILED", job
    assert job["exit_code"] != 0 and job["artifacts"] == []
    assert job["result"] is not None
    if mode == "validation":
        assert job["result"]["status"] == "blocked"
        assert job["stdout"] == "" and job["stderr"]
    else:
        assert job["result"]["ok"] is False
    assert not (service.root / "compose" / "active.json").exists()


def test_real_public_readonly_probe(service):
    job = settled(service, request(mode="execution", operation="probe"))
    assert job["argv"] == [sys.executable, "-m", "asea.execution", "probe"]
    assert job["status"] == "succeeded", job
    assert "ok" not in job["result"]  # Never rewrite the public envelope.
    assert job["result"]["cgroup_v2"]["enforcement_tested"] is False
    assert job["result"]["cgroup_v2"]["delegation_verified"] is False
    assert job["artifacts"] == []


def test_real_synthetic_pcm_and_batch_remain_pending_human(service, tmp_path):
    # Signal fixture only: zeros have no perceptual correctness evidence.
    audio = tmp_path / "silence.wav"
    with wave.open(str(audio), "wb") as out:
        out.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        out.writeframes(b"\x00\x00" * 800)
    manifest = tmp_path / "voice.json"
    manifest.write_text(json.dumps({"model_id": "unit-fixture", "model_sha256": "a" * 64, "licenses": [{
        "spdx": "CC-BY-NC-4.0", "source": "https://example.invalid/fixture", "revision": "fixture-v1",
        "license_text_sha256": "b" * 64, "attribution": "Synthetic unit fixture, no rights grant",
        "usage_restrictions": [], "consent_privacy": "not_applicable", "redistribution": "prohibited"}]}))
    job = settled(service, request(mode="validation", operation="voice-check", audio_file=str(audio),
                                  text="Synthetic silence is not speech", language="en", model_manifest=str(manifest),
                                  use_purpose="noncommercial_experimental"))
    assert job["status"] == "succeeded", job
    assert job["result"]["status"] == "pending_human" and job["result"]["quality_admission"] is False
    assert job["result"]["dimensions"]["naturalness"] == "pending_human"
    assert job["use_purpose"] == "noncommercial_experimental"
    artifact = job["artifacts"][0]
    path, item = service.artifact(job["id"], artifact["id"])
    assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({"protocol_id": "unit-fixture", "clips": [{"clip_id": "silence", "audio": str(audio),
        "text": "Not real speech", "language": "en", "model_manifest": str(manifest)}]}))
    job = settled(service, request(mode="validation", operation="export-listen-batch", manifest=str(batch)))
    assert job["status"] == "succeeded", job
    assert job["result"]["status"] == "pending_human" and job["result"]["human_collection_available"] is False
    assert job["result"]["review_template"]["trials"][0]["listener_id"] is None
    assert not (service.root / "validation").exists()  # Voice path never opens registry/workspace.


def test_legacy_or_fake_green_audit_envelope_refused():
    job = {"mode": "compiler", "operation": "diagnose"}
    for payload in ({"ok": True}, {"ok": True, "status": "PASS", "admission": "ADMITTED"}):
        with pytest.raises(ValueError):
            ex.decode_cli(job, json.dumps(payload), "", 0)
    receipt = {"ok": True, "status": "AUDIT_ONLY", "admission": "UNADMITTED", "quality_certification": False,
               "stdout_contract": "bounded_receipt_v1", "report": "/report.json", "report_sha256": "a" * 64}
    parsed, done = ex.decode_cli(job, json.dumps(receipt), "", 0)
    assert done and parsed == receipt  # Transport double only, no audit performed.
    with pytest.raises(ValueError):
        ex.decode_cli({"mode": "validation", "operation": "voice-check"}, '{"ok":true,"status":"PASS"}', "", 0)


def test_all_new_writes_keep_auth_origin_loopback(service, monkeypatch):
    monkeypatch.setenv("SILT_ENABLE_EXPERIMENTAL", "1")
    monkeypatch.setattr(ex, "manager", lambda: service)
    app = FastAPI()
    app.include_router(ex.router)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = {"X-SILT-Experimental-Token": service.token}
        for mode, operation in (("compiler", "diagnose"), ("compiler", "roundtrip"), ("execution", "probe"),
                                ("validation", "voice-check"), ("validation", "export-listen-batch")):
            body = request(mode=mode, operation=operation).model_dump()
            for headers in ({}, {**token, "Origin": "https://evil.example"}, {**token, "Host": "evil.example"}):
                assert client.post("/api/experimental/jobs", json=body, headers=headers).status_code == 403
        assert not service.jobs
        page = client.get("/experimental").text
        for selector in ('id="guidance"', 'name="resource_profile"', 'name="audio_file"', 'name="worker_timeout_seconds"'):
            assert selector in page
        assert "clearEvidence()" in page and "Historical envelope" in page
        assert "No ASR, automatic playback/listening" in page
        schema = client.get("/api/experimental/schema", headers=token).json()
        assert schema["actions"]["execution"] == ["probe"]
        assert schema["actions"]["validation"] == ["voice-check", "export-listen-batch"]
