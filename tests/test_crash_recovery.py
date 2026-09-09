"""Offline crash/cancellation invariants. Tiny fixtures, no model or timeout certification."""
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from asea.artifacts import Blocked, Workspace, atomic_json
import asea.certification as cert
import asea.compose.runtime as runtime
from asea.compose.schema import CompositionSpec

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux/POSIX crash contract")


@pytest.fixture
def setup_run(tmp_path):
    model = tmp_path / "fixture"
    model.mkdir()
    (model / "fixture.json").write_text('{"prefix":"echo:"}')
    spec = CompositionSpec.model_validate({"name": "CRASH UNIT FIXTURE ONLY", "input_type": "text",
        "nodes": [{"id": "echo", "input": "$input", "component": {
            "kind": "fixture_text", "task": "fixture", "model_path": str(model), "risk": "low",
            "provenance": [{"source": "unit fixture, not ML", "risk": "low", "license": "test-only"}]}}],
        "output_node": "echo"})
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    return Workspace(tmp_path / "workspace"), spec, spec_path


@contextlib.contextmanager
def child_at_barrier(tmp_path, script, *args):
    """Signal only after the child confirms the exact persistence boundary."""
    ready = tmp_path / "ready"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    prefix = "import sys, signal, time, json\nfrom pathlib import Path\nfrom asea.artifacts import Workspace\n"
    process = subprocess.Popen([sys.executable, "-B", "-c", prefix + textwrap.dedent(script),
                                str(ready), *map(str, args)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=env)
    try:
        deadline = time.monotonic() + 15
        while not ready.exists():
            if process.poll() is not None:
                out, err = process.communicate()
                pytest.fail("child exited before persistence barrier: " + out + err)
            if time.monotonic() > deadline:
                pytest.fail("child never reached persistence barrier")
            time.sleep(0.01)
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def deployments(workspace):
    old, new = "deployment-" + "a" * 32, "deployment-" + "b" * 32
    with workspace.writer():
        for identifier in (old, new):
            # Deliberately not admitted deployment evidence; only pointer mechanics.
            workspace.write_record("deployments", identifier, {"id": identifier, "unit_fixture": True})
        atomic_json(workspace.root / "active.json", {"schema_version": 1, "deployment_id": old})
    return old, new


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("reader", ["active", "list", "run"])
def test_dead_pointer_writer_recovers_before_exposure(tmp_path, setup_run, signum, reader):
    ws, spec, _ = setup_run
    old, new = deployments(ws)
    pointer = ws.root / "active.json"
    before = pointer.read_bytes()
    script = """
        import asea.certification as cert
        ws = Workspace(sys.argv[2])
        original = ws.audit
        def audit(event, **fields):
            if event == 'activate':
                Path(sys.argv[1]).write_text('new pointer durable; final audit not written')
                signal.pause()
            return original(event, **fields)
        ws.audit = audit
        with ws.writer():
            cert._publish_pointer(ws, 'activate', sys.argv[3])
    """
    with child_at_barrier(tmp_path, script, ws.root, new) as process:
        assert json.loads(pointer.read_text())["deployment_id"] == new
        assert (ws.root / "active-transaction.json").exists()
        with pytest.raises(Blocked, match="locked"):
            ws.list_records()
        process.send_signal(signum)
        assert process.wait(timeout=10) == -signum
    restarted = Workspace(ws.root)
    if reader == "active":
        assert restarted.active_state()["deployment_id"] == old
    elif reader == "list":
        assert restarted.list_records()["active"]["deployment_id"] == old
    else:
        assert runtime.run_spec(restarted, spec, "hello", allow_fixtures=True)["status"] == "succeeded"
    assert pointer.read_bytes() == before
    assert not (ws.root / "active-transaction.json").exists()
    assert '"event":"pointer_recovered"' in (ws.root / "audit.jsonl").read_text()


def test_live_exclusion_dead_release_and_stable_inode(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    script = """
        with Workspace(sys.argv[2]).writer():
            Path(sys.argv[1]).write_text('lock held')
            signal.pause()
    """
    with child_at_barrier(tmp_path, script, ws.root) as process:
        lock = ws.root / ".writer.lock"
        inode = lock.stat().st_ino
        # Diagnostic PID text cannot steal a living writer's kernel lock.
        lock.write_text("99999999")
        with pytest.raises(Blocked, match="locked"):
            with ws.writer():
                pass
        process.kill()
        process.wait(timeout=10)
    for _ in range(3):
        with ws.writer():
            assert lock.stat().st_ino == inode
    assert lock.exists() and lock.stat().st_ino == inode


def test_legacy_exclusive_create_lock_is_acquirable(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    lock = ws.root / ".writer.lock"
    lock.write_text("123456789")
    before = lock.stat().st_ino
    with ws.writer():
        assert lock.read_text() == str(os.getpid())
    assert lock.stat().st_ino == before


def test_same_process_nested_writer_excluded(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    with ws.writer():
        with pytest.raises(Blocked, match="locked"):
            with Workspace(ws.root).writer():
                pass
        assert ws.list_records()["active"] is None


def test_no_flock_fails_closed(tmp_path, monkeypatch):
    import asea.artifacts as artifacts
    ws = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(artifacts, "fcntl", None)
    with pytest.raises(Blocked, match="POSIX"):
        with ws.writer():
            pass


@pytest.mark.parametrize("damage", ["json", "missing", "traversal", "version", "workspace", "previous", "duplicate", "unknown_record", "extra_path"])
def test_corrupt_journal_fail_closed_preserves_old_pointer(tmp_path, damage):
    ws = Workspace(tmp_path / "workspace")
    old, new = deployments(ws)
    pointer = ws.root / "active.json"
    before = pointer.read_bytes()
    journal = ws.root / "active-transaction.json"
    transaction = {"schema_version": 1, "workspace": str(ws.root),
                   "previous": json.loads(before), "deployment_id": new}
    if damage == "missing":
        transaction.pop("previous")
    elif damage == "traversal":
        transaction["deployment_id"] = "../../outside"
    elif damage == "version":
        transaction["schema_version"] = True
    elif damage == "workspace":
        transaction["workspace"] = str(tmp_path / "other")
    elif damage == "previous":
        transaction["previous"] = {"schema_version": 1, "deployment_id": "../../outside"}
    elif damage == "unknown_record":
        transaction["deployment_id"] = "deployment-" + "c" * 32
    elif damage == "extra_path":
        transaction["pointer_path"] = str(tmp_path / "outside")
    journal.write_text("{" if damage == "json" else json.dumps(transaction))
    if damage == "duplicate":
        journal.write_text('{"previous":null,"previous":null,"deployment_id":"' + new + '"}')
    raw = journal.read_bytes()
    for action in (ws.active_state, ws.list_records):
        with pytest.raises(Blocked, match="recovery journal"):
            action()
        assert pointer.read_bytes() == before
        assert journal.read_bytes() == raw


def test_journal_symlink_cannot_touch_other_workspace(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    deployments(ws)
    before = (ws.root / "active.json").read_bytes()
    external = tmp_path / "other-journal"
    external.write_text('{"previous":null}')
    (ws.root / "active-transaction.json").symlink_to(external)
    with pytest.raises(Blocked, match="symlink"):
        ws.list_records()
    assert (ws.root / "active.json").read_bytes() == before
    assert external.read_text() == '{"previous":null}'


def test_legacy_journal_validated_and_recovered(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    old, new = deployments(ws)
    previous = ws.active_state()
    atomic_json(ws.root / "active-transaction.json", {"previous": previous, "deployment_id": new})
    atomic_json(ws.root / "active.json", {"schema_version": 1, "deployment_id": new})
    assert ws.active_state()["deployment_id"] == old


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_baseexception_persists_terminal_run_and_restores_handlers(setup_run, monkeypatch, exception):
    ws, spec, _ = setup_run
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def fail(*args, **kwargs):
        raise exception()
    monkeypatch.setattr(runtime, "_adapter", fail)
    with pytest.raises(exception):
        runtime.run_spec(ws, spec, "hello", allow_fixtures=True)
    runs = list((ws.root / "runs").glob("*.json"))
    assert len(runs) == 1
    record = ws.read_record("runs", runs[0].stem)
    assert record["status"] == ("failed" if exception is GeneratorExit else "cancelled")
    assert "output" not in record
    assert not list((ws.root / "outputs").glob("*/pending.json"))
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())
    assert '"status":"running"' not in (ws.root / "audit.jsonl").read_text()


RUN_SCRIPT = """
    import asea.compose.runtime as runtime
    from asea.compose.schema import CompositionSpec
    ws = Workspace(sys.argv[2])
    spec = CompositionSpec.model_validate_json(Path(sys.argv[3]).read_text())
    def adapter(*args, **kwargs):
        Path(sys.argv[1]).write_text('adapter executing after pending marker')
        signal.pause()
    runtime._adapter = adapter
    runtime.run_spec(ws, spec, 'hello', allow_fixtures=True)
"""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_managed_signal_persists_cancelled(tmp_path, setup_run, signum):
    ws, _, spec_path = setup_run
    with child_at_barrier(tmp_path, RUN_SCRIPT, ws.root, spec_path) as process:
        assert len(list((ws.root / "outputs").glob("*/pending.json"))) == 1
        process.send_signal(signum)
        assert process.wait(timeout=10) != 0
    record = ws.list_records()["runs"][0]
    assert record["status"] == "cancelled"
    assert not ws.list_records()["recoveries"]
    assert not list((ws.root / "outputs").glob("*/pending.json"))


def test_sigkill_run_recovery_is_separate_immutable_event_and_outputs_isolated(tmp_path, setup_run):
    ws, spec, spec_path = setup_run
    with child_at_barrier(tmp_path, RUN_SCRIPT, ws.root, spec_path) as process:
        pending = next((ws.root / "outputs").glob("*/pending.json"))
        run_id = pending.parent.name
        partial = pending.parent / "echo.wav"
        partial.write_bytes(b"partial output, never usable as evidence")
        process.kill()
        assert process.wait(timeout=10) == -signal.SIGKILL
    assert not list((ws.root / "runs").glob("*.json"))
    restarted = Workspace(ws.root)
    interrupted = restarted.run_status(run_id)
    assert interrupted["status"] == "interrupted"
    assert not list((ws.root / "runs").glob("*.json"))  # No fictional measured run.
    event_path = ws.root / "recoveries" / (interrupted["recovery_id"] + ".json")
    evidence = event_path.read_bytes()
    assert partial.read_bytes() == b"partial output, never usable as evidence"
    completed = runtime.run_spec(restarted, spec, "again", allow_fixtures=True)
    assert completed["status"] == "succeeded" and completed["id"] != run_id
    assert restarted.run_status(run_id)["status"] == "interrupted"
    assert event_path.read_bytes() == evidence
    assert completed["output"]["text"] == "echo:again"
    assert '"event":"run_recovered"' in (ws.root / "audit.jsonl").read_text()


def test_legacy_signed_running_record_never_rewritten(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    run_id = "run-" + "d" * 32
    record = {"id": run_id, "status": "running", "schema_version": 1}
    with ws.writer():
        ws.write_record("runs", run_id, record)
    path = ws.root / "runs" / (run_id + ".json")
    before = path.read_bytes()
    assert ws.run_status(run_id)["status"] == "interrupted"
    assert ws.read_record("runs", run_id) == record
    assert path.read_bytes() == before
    audit = (ws.root / "audit.jsonl").read_bytes()
    assert ws.list_records()["runs"][0]["status"] == "interrupted"
    assert (ws.root / "audit.jsonl").read_bytes() == audit


def test_final_record_before_audit_crash_keeps_measured_history(setup_run, monkeypatch):
    ws, spec, _ = setup_run
    original_audit = ws.audit
    def fail(event, **fields):
        if event == "run":
            raise OSError("injected final audit failure")
        return original_audit(event, **fields)
    monkeypatch.setattr(ws, "audit", fail)
    with pytest.raises(OSError):
        runtime.run_spec(ws, spec, "hello", allow_fixtures=True)
    path = next((ws.root / "runs").glob("*.json"))
    before = path.read_bytes()
    monkeypatch.setattr(ws, "audit", original_audit)
    assert ws.run_status(path.stem)["status"] == "succeeded"
    assert path.read_bytes() == before
    assert not list((ws.root / "outputs").glob("*/pending.json"))


def test_cancelled_status_survives_missing_final_rss(setup_run, monkeypatch):
    ws, spec, _ = setup_run
    def cancel(*args, **kwargs):
        def unavailable():
            raise Blocked("measurement lost")
        monkeypatch.setattr(runtime, "_rss_mb", unavailable)
        raise KeyboardInterrupt()
    monkeypatch.setattr(runtime, "_adapter", cancel)
    with pytest.raises(KeyboardInterrupt):
        runtime.run_spec(ws, spec, "hello", allow_fixtures=True)
    record = ws.list_records()["runs"][0]
    assert record["status"] == "cancelled"
    assert record["resources"]["measurement_status"] == "unavailable"
