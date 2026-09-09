"""Experimental integration regression tests, NOT model quality evidence.

Mock adapters/workers below test orchestration only. Actual OS oracle CLI tests
use tiny source-controlled functions, not model weights or generated verdicts.
"""
import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from asea.artifacts import Blocked, Workspace, atomic_json, digest
import asea.certification as cert
from asea.certification import sandbox
from asea.compose.schema import CompositionSpec, EvaluationCase, EvaluationSuite
from asea.compose.__main__ import parser


FUNCTION_CASES = [{"id": "sum", "function": "add", "args": [2, 3], "kwargs": {}, "expected": 5}]


def suite(metric="text_exact"):
    cases = []
    for group in ("target", "control"):
        row = {"id": group, "group": group, "input": group, "reference": "correct",
               "metric": metric, "threshold": 1.0}
        if metric == "function_io":
            row.update(reference="Addition on the named examples; descriptive text, not executable tests.",
                       function_cases=copy.deepcopy(FUNCTION_CASES))
        cases.append(row)
    return EvaluationSuite.model_validate({"name": "MOCK integration references", "reference_source": "unit tests only",
                                          "claims": ["coding"] if metric == "function_io" else ["reference_text"], "cases": cases})


@pytest.fixture
def harness(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"INVALID: MOCK ADAPTER ONLY; NEVER LOAD")
    spec = CompositionSpec.model_validate({"name": "MOCK ONLY", "input_type": "text", "nodes": [
        {"id": "text", "input": "$input", "component": {"kind": "hf_text", "task": "causal",
         "model_path": str(model), "risk": "low", "provenance": [{"source": "mock", "risk": "low", "license": "test"}]}}],
         "output_node": "text"})
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a, **k: {"type": "text", "text": "correct"})
    return Workspace(tmp_path / "store"), spec


def mock_worker(monkeypatch, ws, spec, mutate=None):
    """Unit double only: does NOT establish kernel or public-CLI evidence."""
    from asea.compose.runtime import run_spec
    calls = []
    def run(operation, arguments, *, profile, memory_mib, timeout, workspace):
        assert not ws._owns_writer(), "parent writer must be released before worker"
        assert operation == "compose" and workspace == str(ws.root)
        assert arguments[:2] == ["run", "--spec"]
        assert json.loads(Path(arguments[2]).read_text()) == spec.model_dump(mode="json")
        calls.append(arguments)
        result = run_spec(Workspace(workspace), spec, arguments[arguments.index("--input") + 1])
        pid = 100000 + len(calls)
        cpu = __import__("math").ceil(timeout)
        external = {"schema": "silt.resource-controls.v1", "operation": operation, "profile": profile,
                    "status": "OK", "returncode": 0, "worker_pid": pid, "supported": True, "launched": True,
                    "enforcement_established": True, "group_kill_succeeded": True,
                    "memory_metric": "virtual_address_space_per_process",
                    "requested_limits": {"address_space_bytes": memory_mib * 1048576, "wall_seconds": timeout,
                                         "cpu_soft_seconds": cpu, "cpu_hard_seconds": cpu + 1},
                    "effective_limits": {"as": [memory_mib * 1048576] * 2, "cpu": [cpu, cpu + 1]},
                    "stdout": json.dumps({"ok": True, "command": "run", "execution_pid": pid, "result": result})}
        if mutate:
            mutate(external)
        return external
    monkeypatch.setattr("asea.execution.run", run)
    return calls


@pytest.mark.parametrize("change", [
    {"expected": 1.5}, {"args": (2, 3)}, {"id": True}, {"function": "x.y"}, {"passed": True},
])
def test_function_schema_uses_oracle_before_coercion(change):
    raw = suite("function_io").model_dump()
    raw["cases"][1]["function_cases"][0].update(change)
    with pytest.raises((ValidationError, ValueError)):
        EvaluationSuite.model_validate(raw)


def test_function_cases_metric_exclusive_and_exact_threshold():
    raw = suite().model_dump()
    raw["cases"][0]["function_cases"] = FUNCTION_CASES
    with pytest.raises(ValidationError, match="only valid"):
        EvaluationSuite.model_validate(raw)
    raw = suite("function_io").model_dump()
    raw["cases"][0]["threshold"] = .99
    with pytest.raises(ValidationError, match="threshold 1.0"):
        EvaluationSuite.model_validate(raw)
    assert cert._check_policy(suite("function_io")).policy_version == "composition-admission-v2"


def test_cli_defaults_and_explicit_profile():
    base = ["--workspace", "/tmp/ws", "evaluate", "--spec", "s.json", "--suite", "t.json"]
    assert parser().parse_args(base).resource_profile == "observe_only"
    args = parser().parse_args(base + ["--resource-profile", "process_as", "--memory-mib", "1024", "--case-timeout", "12"])
    assert (args.memory_mib, args.case_timeout) == (1024, 12)


def test_invalid_entire_suite_before_generation(harness, monkeypatch):
    ws, spec = harness
    good = suite("function_io")
    badcase = good.cases[1].model_copy(update={"function_cases": [{**FUNCTION_CASES[0], "expected": 1.5}]})
    bad = good.model_copy(update={"cases": [good.cases[0], badcase]})
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a: pytest.fail("must validate all cases first"))
    result = cert.evaluate(ws, spec, bad)
    assert result["status"] == "blocked" and result["case_evidence"] == []
    assert not list((ws.root / "runs").iterdir())


def test_cgroup_no_launch_no_generation(harness, monkeypatch):
    ws, spec = harness
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a: pytest.fail("must not generate"))
    monkeypatch.setattr("asea.execution.run", lambda *a, **k: pytest.fail("unavailable profile must not launch"))
    result = cert.evaluate(ws, spec, suite(), resource_profile="cgroup_v2")
    assert result["status"] == "blocked"
    assert "before model generation; no fallback" in result["reason"]
    assert result["worker_attempts"] == []


def test_observe_only_retains_path_and_fingerprint(harness, monkeypatch):
    ws, spec = harness
    monkeypatch.setattr("asea.execution.run", lambda *a, **k: pytest.fail("legacy path changed"))
    result = cert.evaluate(ws, spec, suite())
    assert result["status"] == "admitted"  # mock control flow, not model admission evidence
    assert result["execution_config"]["resource_profile"] == "observe_only"
    assert result["implementation_fingerprint"]["environment"]["dependencies"]["pydantic"]
    assert result["worker_attempts"] == []
    cert.activate(ws, result["id"])


def test_protected_serial_unlocked_binding_and_activation(harness, monkeypatch):
    ws, spec = harness
    observed = cert.evaluate(ws, spec, suite())
    calls = mock_worker(monkeypatch, ws, spec)
    result = cert.evaluate(ws, spec, suite(), resource_profile="process_as", memory_mib=128, case_timeout=8)
    assert result["status"] == "admitted", result.get("reason")  # mocked workers only
    assert len(calls) == 2
    assert result["graph_hash"] == observed["graph_hash"]
    assert result["execution_config_hash"] != observed["execution_config_hash"]
    assert result["execution_config"]["memory_semantics"] == "virtual_address_space_per_process"
    assert len({r["worker_receipt"]["resource_evidence"]["worker_pid"] for r in result["case_evidence"]}) == 2
    cert.activate(ws, result["id"])


@pytest.mark.parametrize("fault", ["pid", "output", "limits", "exit"])
def test_forged_worker_result_cannot_admit(harness, monkeypatch, fault):
    ws, spec = harness
    def mutate(external):
        if fault == "exit":
            external["returncode"] = 7
        elif fault == "limits":
            external["effective_limits"]["as"] = [1, 1]
        else:
            envelope = json.loads(external["stdout"])
            if fault == "pid":
                envelope["execution_pid"] += 1
            else:
                envelope["result"]["output"]["text"] = "forged correct"
            external["stdout"] = json.dumps(envelope)
    mock_worker(monkeypatch, ws, spec, mutate)
    result = cert.evaluate(ws, spec, suite(), resource_profile="process_as")
    assert result["status"] == "blocked"
    assert result["case_evidence"] == [] and len(result["worker_attempts"]) == 1
    with pytest.raises(Blocked):
        cert.activate(ws, result["id"])


def test_old_signed_run_replay_is_not_fresh(harness, monkeypatch):
    ws, spec = harness
    mock_worker(monkeypatch, ws, spec)
    first = cert.evaluate(ws, spec, suite(), resource_profile="process_as")
    stale = first["case_evidence"][0]["worker_receipt"]["resource_evidence"]
    monkeypatch.setattr("asea.execution.run", lambda *a, **k: stale)
    replay = cert.evaluate(ws, spec, suite(), resource_profile="process_as")
    assert replay["status"] == "blocked" and "stale child run replay" in replay["reason"]


@pytest.mark.parametrize("cancel", [False, True, "sigterm"])
def test_timeout_cancel_keeps_recovery_and_no_retry(harness, monkeypatch, cancel):
    ws, spec = harness
    ids = []
    def interrupted(*a, **k):
        assert not ws._owns_writer()
        child = Workspace(ws.root)
        with child.writer():
            ident = child.new_id("run")
            ids.append(ident)
            out = child.root / "outputs" / ident
            out.mkdir()
            atomic_json(out / "pending.json", {"status": "running"})
        if cancel == "sigterm":
            assert callable(signal.getsignal(signal.SIGTERM)), "parent must install cleanup handler"
            os.kill(os.getpid(), signal.SIGTERM)
            pytest.fail("SIGTERM should unwind")
        if cancel:
            raise KeyboardInterrupt("unit cancellation")
        return {"status": "TIMEOUT", "returncode": -9, "reason": "unit deadline", "worker_pid": 100000}
    monkeypatch.setattr("asea.execution.run", interrupted)
    if cancel:
        with pytest.raises(SystemExit if cancel == "sigterm" else KeyboardInterrupt):
            cert.evaluate(ws, spec, suite(), resource_profile="process_as")
        result = ws.list_records()["evaluations"][0]
    else:
        result = cert.evaluate(ws, spec, suite(), resource_profile="process_as")
    assert result["status"] == "blocked" and len(result["worker_attempts"]) == 1
    assert len(ids) == 1 and ws.run_status(ids[0])["status"] == "interrupted"
    with ws.writer():
        pass  # no abandoned lock/deadlock after child cancellation
    assert not (ws.root / "active.json").exists()


@pytest.mark.parametrize("fault", ["missing", "changed", "receipt", "profile"])
def test_activation_stale_implementation_and_receipt(harness, monkeypatch, fault):
    ws, spec = harness
    mock_worker(monkeypatch, ws, spec)
    result = cert.evaluate(ws, spec, suite(), resource_profile="process_as")
    original = ws.read_record
    def changed(collection, identifier):
        value = original(collection, identifier)
        if collection == "evaluations":
            if fault == "missing":
                value.pop("implementation_fingerprint")
            elif fault == "changed":
                value["implementation_fingerprint"]["environment"]["python"] = "changed"
            elif fault == "receipt":
                value["case_evidence"][0]["worker_receipt_hash"] = "0" * 64
            else:
                value["execution_config"]["resource_profile"] = "observe_only"
        return value
    monkeypatch.setattr(ws, "read_record", changed)
    with pytest.raises(Blocked, match="stale_|binding mismatch|receipt hash"):
        cert.activate(ws, result["id"])
    assert not (ws.root / "active.json").exists()


def test_candidate_pass_flags_without_comparisons_are_not_score():
    assert not cert._function_passed({"oracle_mode": "host_data_only_functions_v1", "status": "PASSED",
        "passed": True, "supported": True, "returncode": 0, "resource_flags": [], "tests_total": 1,
        "tests_run": 0, "tests_passed": 0, "cases": [], "resources": {"limits_established": True}}, FUNCTION_CASES)


@pytest.mark.parametrize("source,passes", [
    ("def add(a,b): return a+b\n", True),
    ('print(\'{"passed":true,"tests_run":999}\')\ndef add(a,b): return 9\n', False),
    ('def add(a,b): return {"passed":True,"tests_run":999}\n', False),
    ('import os\nos._exit(0)\n', False),
])
def test_actual_os_oracle_public_cli(tmp_path, source, passes):
    candidate = tmp_path / "candidate.py"
    cases = tmp_path / "cases.json"
    candidate.write_text(source)
    cases.write_text(json.dumps(FUNCTION_CASES))
    proc = subprocess.run([sys.executable, "-m", "asea.certification.function_oracle", "--source", str(candidate),
                           "--cases", str(cases)], capture_output=True, text=True, timeout=45)
    result = json.loads(proc.stdout)
    if result["status"] == "BLOCKED":
        assert result["tests_run"] == 0 and not result["passed"] and proc.returncode != 0
        pytest.skip("actual containment unavailable (fail-closed): " + result["reason"])
    assert result["supported"] is True
    assert result["passed"] is passes, result
    assert proc.returncode == (0 if passes else 1)
    assert result["tests_passed"] == (1 if passes else 0)


def test_function_io_parent_oracle_and_activation(harness, monkeypatch):
    probe = sandbox.probe_code_sandbox()
    if not probe.supported or not probe.passed:
        pytest.skip("actual OS oracle unavailable: " + probe.reason)
    ws, spec = harness
    monkeypatch.setattr("asea.compose.runtime._adapter", lambda *a: {"type": "text", "text": "Answer\n```python\ndef add(a,b): return a+b\n```"})
    result = cert.evaluate(ws, spec, suite("function_io"))
    assert result["status"] == "admitted", result.get("reason")  # mock generation, actual OS oracle
    assert result["oracle_modes"]["function_io"]["mode"] == "host_data_only_functions_v1"
    assert all(r["oracle_evidence"]["tests_passed"] == 1 for r in result["case_evidence"])
    cert.activate(ws, result["id"])


def test_actual_resource_worker_rejects_public_fixture_before_generation(tmp_path):
    from asea.execution import probe
    if not probe()["process_as"]["execution_supported"]:
        pytest.skip("native process_as unavailable")
    model = tmp_path / "model"
    model.mkdir()
    (model / "fixture.json").write_text('{"prefix":""}')
    spec = CompositionSpec.model_validate({"name": "fixture cannot use public CLI", "input_type": "text", "nodes": [
        {"id": "echo", "input": "$input", "component": {"kind": "fixture_text", "task": "fixture", "model_path": str(model),
         "risk": "low", "provenance": [{"source": "unit", "risk": "low", "license": "test"}]}}], "output_node": "echo"})
    ws = Workspace(tmp_path / "store")
    result = cert.evaluate(ws, spec, suite(), allow_fixtures=True, resource_profile="process_as", memory_mib=256, case_timeout=10)
    assert result["status"] == "blocked" and result["case_evidence"] == []
    external = result["worker_attempts"][0]["resource_evidence"]
    assert external["worker_pid"] != os.getpid()
    assert external["enforcement_established"] is True
    assert external["returncode"] != 0
    assert "unit-test-only" in external["stdout"]
