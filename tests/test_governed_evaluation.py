"""UNIT evidence only: static pilot data and fake CLI; never real model promotion."""
import copy
import json
from pathlib import Path
import sys

import pytest

from asea.artifacts import Blocked, Workspace, digest
from asea.validation import load_json
from asea.validation import governed as g
from asea.validation.__main__ import main

PILOT = Path(__file__).resolve().parents[1] / "data" / "pilot-v2"


@pytest.fixture
def spec(tmp_path):
    model = tmp_path / "unit-model"
    model.mkdir()
    (model / "unit.safetensors").write_bytes(b"NOT REAL WEIGHTS; UNIT INVENTORY ONLY")
    (model / "config.json").write_text('{}')
    value = {"name": "unit-only", "input_type": "text", "output_node": "text", "nodes": [
        {"id": "text", "input": "$input", "component": {"kind": "hf_text", "task": "causal",
         "model_path": str(model), "risk": "low", "provenance": [{"source": "unit fixture", "risk": "low", "license": "MIT"}]}}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(value))
    return path


def options(tmp_path, spec, purpose="final"):
    return dict(workspace=tmp_path / "validation", compose_workspace=tmp_path / "compose", spec=spec,
                suite=PILOT / (purpose + ".json"), purpose=purpose, memory_mib=256,
                case_timeout=1.0, max_seconds=5.0, dataset_manifest=PILOT / "manifest.json")


def ledger(tmp_path):
    return g.GovernedRegistry(tmp_path / "validation.identity", tmp_path / "validation")


def fake_error(argv, directory, seconds):
    # Fake subprocess seam: assert begin committed BEFORE an attempted CLI launch.
    events = load_json(sorted((directory.parent / "events").glob("*.json"))[-1])
    assert events["kind"] == "function_io_begin"
    assert argv[:3] == [sys.executable, "-m", "asea.compose"]
    assert "--resource-profile" in argv and "process_as" in argv
    assert "--memory-mib" in argv and "--case-timeout" in argv
    suite = load_json(argv[argv.index("--suite") + 1])
    assert all(c["metric"] == "function_io" and c["reference"] for c in suite["cases"])
    out, err = directory / "stdout.txt", directory / "stderr.txt"
    out.write_text(json.dumps({"ok": False, "error": {"message": "unit backend blocked"}}))
    err.write_text("")
    return {"argv": argv, "stdout": str(out), "stderr": str(err), "exit_code": 2, "termination_cause": None}


def test_pilot_adapter_real_fields_and_distinct_families():
    dev, dr = g.adapt_pilot(PILOT / "development.json", "development")
    final, fr = g.adapt_pilot(PILOT / "final.json", "final")
    assert (len(dev.cases), len(final.cases)) == (8, 24)
    assert {m["family_id"] for m in dr["members"]}.isdisjoint({m["family_id"] for m in fr["members"]})
    assert {m["license"] for m in fr["members"]} == {"MIT", "CC-BY-4.0"}
    assert all(m["source_task_id"] and m["source_record_sha256"] for m in fr["members"])
    assert dr["legal_authorization_granted"] is False


def test_development_does_not_read_final_or_provenance(monkeypatch):
    original = g.load_json
    seen = []
    def guarded(path):
        seen.append(Path(path).name)
        assert Path(path).name not in {"final.json", "provenance.json"}
        return original(path)
    monkeypatch.setattr(g, "load_json", guarded)
    g.adapt_pilot(PILOT / "development.json", "development")
    with pytest.raises(Blocked, match="exact split"):
        g.adapt_pilot(PILOT / "final.json", "development")
    assert "development.json" in seen


def test_missing_manifest_blocks(tmp_path):
    path = tmp_path / "development.json"
    path.write_bytes((PILOT / "development.json").read_bytes())
    with pytest.raises(Blocked):
        g.adapt_pilot(path, "development")


def test_inventory_hash_changes_on_weights_and_spec(tmp_path, spec):
    before = g.candidate_binding(spec)
    weights = tmp_path / "unit-model" / "unit.safetensors"
    weights.write_bytes(b"OTHER UNIT WEIGHTS")
    after = g.candidate_binding(spec)
    assert before["graph_hash"] != after["graph_hash"]
    assert before["implementation"] == after["implementation"]
    value = load_json(spec)
    value["nodes"][0]["max_new_tokens"] = 9
    spec.write_text(json.dumps(value))
    assert after["graph_hash"] != g.candidate_binding(spec)["graph_hash"]


def test_final_backend_block_consumes_and_logs_observed(tmp_path, spec, monkeypatch):
    monkeypatch.setattr(g, "_public_cli", fake_error)
    result = g.governed_evaluate(**options(tmp_path, spec))
    assert result["state"] == "consumed"
    assert result["trust"] == "observed_public_cli"
    assert result["outcome"]["quality_admission"] is False
    assert result["outcome"]["counts"]["missing"] == 24
    events = ledger(tmp_path).history()["events"]
    assert [e["kind"] for e in events] == ["function_io_register", "function_io_freeze", "function_io_begin", "function_io_finish"]
    assert events[1]["data"]["reservation_hash"] == digest(events[1]["data"]["binding"])
    changed = options(tmp_path, spec)
    changed["compose_workspace"] = tmp_path / "compose-new"
    with pytest.raises(Blocked, match="already reserved"):
        g.governed_evaluate(**changed)


def test_final_cannot_reset_by_renaming_suite_or_candidate(tmp_path):
    _, reg = g.adapt_pilot(PILOT / "final.json", "final")
    store = ledger(tmp_path)
    store.register_function_io(reg)
    store.freeze_function_io(reg, {"candidate": "unit-a"})
    renamed = {**reg, "suite_id": "different-name"}
    store.register_function_io(renamed)
    with pytest.raises(Blocked, match="already reserved"):
        store.freeze_function_io(renamed, {"candidate": "unit-b"})


def test_freeze_immutable_dependencies_and_begin_once(tmp_path):
    _, reg = g.adapt_pilot(PILOT / "final.json", "final")
    store = ledger(tmp_path)
    store.register_function_io(reg)
    binding = {"threshold": 1.0, "resources": {"memory": 256}, "candidate": "unit"}
    reservation = store.freeze_function_io(reg, binding)
    with pytest.raises(Blocked, match="dependency"):
        store.begin_function_io(reservation, {**binding, "threshold": 0.1})
    run = store.begin_function_io(reservation, binding)
    with pytest.raises(Blocked, match="consumed"):
        store.begin_function_io(reservation, binding)
    store.finish_observed(run, {"status": "blocked"})
    with pytest.raises(Blocked, match="already finished"):
        store.finish_observed(run, {"status": "admitted"})


def test_source_mapping_is_immutable(tmp_path):
    _, reg = g.adapt_pilot(PILOT / "development.json", "development")
    store = ledger(tmp_path)
    store.register_function_io(reg)
    changed = copy.deepcopy(reg)
    changed["suite_id"] = "renamed"
    changed["source_mapping"][0]["source_record_sha256"] = "f" * 64
    with pytest.raises(Blocked, match="source mapping"):
        store.register_function_io(changed)


def test_launch_exception_consumes(tmp_path, spec, monkeypatch):
    def fail(*args):
        raise OSError("unit launch failure")
    monkeypatch.setattr(g, "_public_cli", fail)
    result = g.governed_evaluate(**options(tmp_path, spec))
    assert result["state"] == "consumed"
    assert result["receipt"]["exit_code"] is None
    assert ledger(tmp_path).history()["events"][-1]["kind"] == "function_io_finish"


def test_preflight_failure_does_not_consume(tmp_path, spec):
    value = options(tmp_path, spec)
    value["suite"] = PILOT / "development.json"  # purpose final mismatch
    with pytest.raises(Blocked):
        g.governed_evaluate(**value)
    assert not any(e["kind"] == "function_io_begin" for e in ledger(tmp_path).history()["events"])


@pytest.mark.parametrize("field,value", [("case_timeout", float("nan")), ("max_seconds", float("inf")),
                                        ("resource_profile", "observe_only"), ("memory_mib", 0)])
def test_unbounded_or_unprotected_resource_request_blocks(tmp_path, spec, field, value):
    opts = options(tmp_path, spec)
    opts[field] = value
    with pytest.raises(Blocked):
        g.governed_evaluate(**opts)


def test_zero_exit_alone_never_admits(tmp_path, spec, monkeypatch):
    def fake_zero(*args):
        receipt = fake_error(*args)
        receipt["exit_code"] = 0
        Path(receipt["stdout"]).write_text('{"ok":true,"result":{"status":"admitted"}}')
        return receipt
    monkeypatch.setattr(g, "_public_cli", fake_zero)
    result = g.governed_evaluate(**options(tmp_path, spec))
    assert not result["outcome"]["quality_admission"]
    assert not result["certificate_issued"]


def test_signed_rejected_is_done_not_backend_blocked(tmp_path, spec, monkeypatch):
    """Fake signed CLI records exercise transport only; not model-quality evidence."""
    def fake_signed(argv, directory, seconds):
        receipt = fake_error(argv, directory, seconds)
        workspace = Workspace(argv[argv.index("--workspace") + 1])
        from asea.compose.runtime import _prepare
        from asea.compose.schema import CompositionSpec
        from asea.certification import implementation_fingerprint
        suite = load_json(argv[argv.index("--suite") + 1])
        with workspace.writer():
            candidate = _prepare(workspace, CompositionSpec.model_validate(load_json(argv[argv.index("--spec") + 1])))
            rows = []
            for index, case in enumerate(suite["cases"]):
                run_id = workspace.new_id("run")
                workspace.write_record("runs", run_id, {"id": run_id, "candidate_id": candidate["id"], "graph_hash": candidate["graph_hash"]})
                rows.append({"case_id": case["id"], "run_id": run_id, "passed": False})
            evaluation = {"id": workspace.new_id("evaluation"), "candidate_id": candidate["id"], "graph_hash": candidate["graph_hash"],
                          "suite": suite, "suite_hash": digest(suite), "status": "rejected", "fixture_only": False,
                          "case_evidence": rows, "execution_config": {"resource_profile": "process_as", "memory_mib": 256, "case_timeout": 1.0},
                          "implementation_fingerprint": implementation_fingerprint()}
            workspace.write_record("evaluations", evaluation["id"], evaluation)
        Path(receipt["stdout"]).write_text(json.dumps({"command": "evaluate", "ok": False, "result": evaluation}))
        return receipt
    monkeypatch.setattr(g, "_public_cli", fake_signed)
    result = g.governed_evaluate(**options(tmp_path, spec, "development"))
    assert result["receipt"]["exit_code"] == 2
    assert result["outcome"]["status"] == "rejected", result
    assert result["outcome"]["completion_status"] == "done"
    assert result["outcome"]["counts"] == {"cases": 8, "measured": 8, "passed": 0, "failed": 8, "missing": 0}
    assert not result["outcome"]["quality_admission"]
    assert Path(result["outcome"]["signed_records_archive"]["path"]).is_file()


@pytest.mark.parametrize('case', ['sleep', 'leave_descendant'])
def test_public_subprocess_timeout_is_finite_unit_only(tmp_path, monkeypatch, case):
    import subprocess
    from asea.execution import _worker
    original = subprocess.Popen
    def fixed_fixture(argv, **kwargs):
        assert argv[:3] == [sys.executable, "-m", "asea.compose"]
        import os
        config = {"control_fd": 1, "expected_parent_pid": os.getpid(),
                  "memory_bytes": 96 * 1024 * 1024, "cpu_seconds": 5, "test_case": case}
        return original([sys.executable, "-I", "-S", _worker.__file__, json.dumps(config)], **kwargs)
    monkeypatch.setattr(subprocess, "Popen", fixed_fixture)
    result = g._public_cli([sys.executable, "-m", "asea.compose", "--help"], tmp_path, 0.2)
    if case == 'sleep':
        assert result["termination_cause"] == "overall_timeout"
        assert result["exit_code"] != 0
    else:
        import time
        assert result['exit_code'] == 0 and result['termination_cause'] is None
        rows = [json.loads(line) for line in Path(result['stdout']).read_text().splitlines()]
        child = next(row['descendant_pid'] for row in rows if 'descendant_pid' in row)
        status = Path('/proc', str(child), 'status')
        end = time.monotonic() + 2
        while status.exists() and 'State:\tZ' not in status.read_text() and time.monotonic() < end:
            time.sleep(0.01)
        assert not status.exists() or 'State:\tZ' in status.read_text()
    assert result["elapsed_seconds"] < 11


def test_public_cli_rejects_arbitrary_program_before_launch(tmp_path):
    with pytest.raises(Blocked, match="fixed public compose"):
        g._public_cli([sys.executable, "-c", "raise SystemExit(0)"], tmp_path, 1)
    assert not list(tmp_path.iterdir())


def test_timeout_final_stays_consumed(tmp_path, spec, monkeypatch):
    def timeout(*args):
        result = fake_error(*args)
        result.update(exit_code=-15, termination_cause="overall_timeout")
        return result
    monkeypatch.setattr(g, "_public_cli", timeout)
    result = g.governed_evaluate(**options(tmp_path, spec))
    assert result["state"] == "consumed"
    assert result["receipt"]["exit_code"] == -15
    assert result["outcome"]["quality_admission"] is False


def test_changed_dependencies_after_freeze_never_launch(tmp_path, spec, monkeypatch):
    original = g.candidate_binding
    count = []
    def changed(path):
        value = original(path)
        count.append(1)
        if len(count) > 1:
            value["graph_hash"] = "f" * 64
        return value
    monkeypatch.setattr(g, "candidate_binding", changed)
    monkeypatch.setattr(g, "_public_cli", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(Blocked, match="changed after freeze"):
        g.governed_evaluate(**options(tmp_path, spec))
    events = ledger(tmp_path).history()["events"]
    assert [e["kind"] for e in events] == ["function_io_register", "function_io_freeze"]


def test_host_import_and_inventory_do_not_load_ml(tmp_path, spec):
    import subprocess
    script = ("import sys; from asea.validation.governed import candidate_binding; "
              "candidate_binding(sys.argv[1]); assert not ({'torch','transformers'} & set(sys.modules))")
    completed = subprocess.run([sys.executable, "-c", script, str(spec)], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr


def test_cli_arguments_preserve_failure_status(tmp_path, spec, monkeypatch, capsys):
    monkeypatch.setattr(g, "_public_cli", fake_error)
    code = main(["governed-evaluate", "--workspace", str(tmp_path / "validation"), "--compose-workspace", str(tmp_path / "compose"),
                 "--spec", str(spec), "--suite", str(PILOT / "development.json"), "--purpose", "development",
                 "--resource-profile", "process_as", "--memory-mib", "256", "--case-timeout", "1", "--max-seconds", "5"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["outcome"]["status"] == "blocked"


@pytest.mark.skipif(sys.platform != 'linux', reason='real Linux process lifecycle')
@pytest.mark.parametrize('signum', [15, 2])
def test_real_governed_cancellation_persists_consumed_and_partial_output(tmp_path, spec, signum):
    import os
    import signal
    import subprocess
    import time
    opts = options(tmp_path, spec, 'development')
    opts['max_seconds'] = 20
    opts = {k: str(v) if isinstance(v, Path) else v for k, v in opts.items()}
    # Test-only Popen seam substitutes a fixed finite trusted fixture, not model
    # execution. The public argv is still validated and production exposes no
    # arbitrary command/fixture parameter.
    script = r"""
import json, os, sys
from asea.validation import governed as g
from asea.execution import _worker
g.candidate_binding(json.loads(sys.argv[1])['spec'])  # Populate native platform cache before launch seam.
original = g.subprocess.Popen
def fixture(argv, **kwargs):
    assert argv[:3] == [sys.executable, '-m', 'asea.compose']
    from pathlib import Path
    partial = Path(argv[argv.index('--workspace') + 1]) / 'runs'
    partial.mkdir(parents=True)
    (partial / 'partial.json').write_bytes(b'{"unit_partial_output":')
    config = {'control_fd': 1, 'expected_parent_pid': os.getpid(),
              'memory_bytes': 96*1024*1024, 'cpu_seconds': 10, 'test_case': 'descendant'}
    return original([sys.executable, '-I', '-S', _worker.__file__, json.dumps(config)], **kwargs)
g.subprocess.Popen = fixture
result = g.governed_evaluate(**json.loads(sys.argv[1]))
print(json.dumps(result), flush=True)
"""
    parent = subprocess.Popen([sys.executable, '-c', script, json.dumps(opts)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              start_new_session=True)
    pidfds = []
    try:
        end = time.monotonic() + 8
        rows = []
        while time.monotonic() < end and parent.poll() is None:
            outputs = list((tmp_path / 'validation').glob('observed-*/stdout.txt'))
            if outputs:
                lines = outputs[0].read_text().splitlines()
                try:
                    rows = [json.loads(line) for line in lines]
                except ValueError:
                    rows = []
                if any('descendant_pid' in row for row in rows):
                    break
            time.sleep(0.01)
        assert any('descendant_pid' in row for row in rows), parent.communicate(timeout=1)
        leader = next(row['parent_death']['worker_pid'] for row in rows if row.get('event') == 'limits')
        descendant = next(row['descendant_pid'] for row in rows if 'descendant_pid' in row)
        if hasattr(os, 'pidfd_open'):
            pidfds = [os.pidfd_open(pid) for pid in (leader, descendant)]
        partial = outputs[0].read_bytes()
        unrelated = tmp_path / 'validation' / 'unrelated-record.txt'
        unrelated.write_bytes(b'never overwrite another record')
        parent.send_signal(signum)
        stdout, stderr = parent.communicate(timeout=8)
        assert parent.returncode == 0, stderr
        result = json.loads(stdout)
        assert result['state'] == 'consumed'
        assert result['receipt']['termination_cause'] == 'interrupted'
        assert result['receipt']['interruption_signal'] == signum
        assert result['outcome']['status'] == 'interrupted'
        assert not result['outcome']['quality_admission'] and not result['certificate_issued']
        assert outputs[0].read_bytes().startswith(partial)
        assert result['receipt']['preserved_outputs']
        assert unrelated.read_bytes() == b'never overwrite another record'
        assert (tmp_path / 'compose' / 'runs' / 'partial.json').read_bytes() == b'{"unit_partial_output":'
        records = result['outcome']['unverified_record_files']
        assert len(records) == 1 and not records[0]['verified'] and records[0]['file_hash']
        if pidfds:
            import select
            for fd in pidfds:
                assert select.select([fd], [], [], 2)[0], 'known group member survived'
        events = ledger(tmp_path).history()['events']
        assert [event['kind'] for event in events] == ['function_io_register', 'function_io_freeze',
                                                     'function_io_begin', 'function_io_finish']
        finish = events[-1]['data']
        assert finish['state'] == 'consumed' and finish['status'] == 'interrupted'
        assert finish['termination_cause'] == 'interrupted'
        observation = load_json(finish['observation'])
        assert observation['receipt']['interruption_signal'] == signum
        with pytest.raises(Blocked, match='consumed'):
            ledger(tmp_path).begin_function_io(events[1]['data'], events[1]['data']['binding'])
    finally:
        if parent.poll() is None:
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=8)
        for fd in pidfds:
            os.close(fd)


def test_managed_signal_handlers_are_restored(tmp_path):
    import signal
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(Blocked):
        g._public_cli(['not-allowed'], tmp_path, 1)
    assert all(signal.getsignal(sig) == old for sig, old in before.items())



def test_lifecycle_implementation_files_are_bound(tmp_path, spec):
    from asea.artifacts import file_hash
    binding = g.candidate_binding(spec)
    implementation = binding['implementation']
    root = Path(g.__file__).resolve().parents[1]
    for name in ('execution/controls.py', 'execution/_worker.py'):
        assert implementation['implementation_hashes'][name] == file_hash(root / name)
    for name in ('governed.py', '__main__.py'):
        assert implementation['governance_hashes'][name] == file_hash(Path(g.__file__).parent / name)


def test_cli_interruption_exit_status(tmp_path, spec, monkeypatch, capsys):
    def interrupted(*args):
        receipt = fake_error(*args)
        receipt.update(termination_cause='interrupted', interruption_signal=15, exit_code=-15)
        return receipt
    monkeypatch.setattr(g, '_public_cli', interrupted)
    code = main(['governed-evaluate', '--workspace', str(tmp_path / 'validation'),
                 '--compose-workspace', str(tmp_path / 'compose'), '--spec', str(spec),
                 '--suite', str(PILOT / 'development.json'), '--purpose', 'development',
                 '--resource-profile', 'process_as', '--memory-mib', '256', '--case-timeout', '1'])
    assert code == 143
    assert not json.loads(capsys.readouterr().out)['outcome']['quality_admission']
