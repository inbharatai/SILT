"""Finite real-kernel tests; no models, root writes, or simulated OS support."""
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import pytest

from asea.execution import probe, run
from asea.execution.controls import _run_test_child

NATIVE_LINUX = platform.system() == "Linux" and hasattr(os, "wait4")
linux = pytest.mark.skipif(not NATIVE_LINUX, reason="Requires real native Linux RLIMIT/wait4")


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    child = subprocess.run([sys.executable, "-m", "asea.execution", *args],
                           env=env, capture_output=True, text=True, timeout=15)
    return child.returncode, json.loads(child.stdout)


def test_probe_is_observation_not_enforcement():
    result = probe()
    assert result["platform"] == platform.system()
    assert result["cgroup_v2"]["delegated"] is False
    assert result["cgroup_v2"]["enforcement_tested"] is False
    assert result["cgroup_v2"]["effective_write_validation"] == "not_attempted"
    assert result["process_as"]["enforcement_tested"] is False
    assert result["process_as"]["hard_rss_enforced"] is False
    code, external = cli("probe")
    assert code == 0
    assert external["platform"] == platform.system()


@pytest.mark.parametrize("root", [None, "/sys/fs/cgroup", "/a/nonexistent/delegation"])
def test_cgroup_never_launches_or_silently_falls_back(root, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No cgroup backend is authorized to launch")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = run("compose", ["--help"], profile="cgroup_v2", delegated_root=root)
    assert result["status"] == "BLOCKED"
    assert not result["launched"]
    assert result["profile"] == "cgroup_v2"
    assert result["returncode"] is None


def test_writable_directory_is_not_delegation(tmp_path):
    result = probe(delegated_root=tmp_path)
    assert not result["cgroup_v2"]["delegated"]
    assert run("compose", ["--help"], profile="cgroup_v2", delegated_root=tmp_path)["status"] == "BLOCKED"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["sh", "python", "asea.compose", "allocate", "-c", "__test__"])
def test_no_public_arbitrary_command_or_test_fixture(operation):
    result = run(operation, ["print('must not run')"])
    assert result["status"] == "BLOCKED"
    assert not result["launched"]


def test_no_command_string_or_nonfinite_budget():
    assert run("compose", "--help")["status"] == "BLOCKED"
    for budget in (0, -1, float("nan"), float("inf"), True):
        assert not run("compose", ["--help"], timeout=budget)["launched"]
    assert not run("compose", ["--help"], memory_mib=True)["launched"]
    with pytest.raises(ValueError):
        _run_test_child("arbitrary source")


@linux
def test_kernel_as_denies_reservation_not_rss_oom():
    result = _run_test_child("allocate")
    assert result["status"] == "RESOURCE_LIMIT", result
    assert result["returncode"] == 70
    assert result["enforcement_established"] and result["enforcement_tested"]
    assert result["effective_limits"]["as"] == [96 * 1024 * 1024] * 2
    assert "MemoryError" in result["resource_cause"]
    assert "not_RSS_OOM" in result["resource_cause"]
    assert not result["hard_rss_enforced"] and not result["aggregate_memory_enforced"]
    assert result["observed_peak_rss_bytes"] > 0
    assert result["elapsed_seconds"] >= result["setup_elapsed_seconds"] > 0


@linux
def test_monotonic_deadline_and_actual_returncode():
    result = _run_test_child("sleep", timeout=0.3)
    assert result["status"] == "TIMEOUT", result
    assert result["returncode"] == -signal.SIGKILL
    assert result["resource_cause"] == "host_monotonic_wall_deadline"
    assert 0.3 <= result["elapsed_seconds"] < 3
    assert result["group_kill_succeeded"] and result["pipes_drained"]
    with pytest.raises(ChildProcessError):
        os.waitpid(result["worker_pid"], os.WNOHANG)


@linux
def test_kernel_cpu_limit_not_bandwidth_claim():
    result = _run_test_child("cpu", timeout=4, cpu_seconds=1)
    assert result["status"] == "RESOURCE_LIMIT", result
    assert result["returncode"] == -signal.SIGXCPU
    assert result["resource_cause"] == "kernel_SIGXCPU"
    assert result["cpu_consumed_seconds"] >= 0.8
    assert not result["cpu_bandwidth_enforced"]


@linux
def test_deadline_kills_same_group_descendant():
    result = _run_test_child("descendant", timeout=0.4)
    assert result["status"] == "TIMEOUT", result
    child = json.loads(result["stdout"])["descendant_pid"]
    # Grandchild reaping belongs to its adopting init, not this supervisor.
    # A zombie is no longer executing and consumes no address space.
    status = Path("/proc") / str(child) / "status"
    end = time.monotonic() + 1
    while status.exists() and time.monotonic() < end:
        text = status.read_text()
        if "State:\tZ" in text:
            break
        time.sleep(0.01)
    assert not status.exists() or "State:\tZ" in status.read_text()
    assert result["group_kill_succeeded"]


@linux
def test_fresh_child_rss_not_process_lifetime_high_water():
    parent_only = bytearray(48 * 1024 * 1024)
    first = _run_test_child("rss")
    second = _run_test_child("environment")
    assert len(parent_only) == 48 * 1024 * 1024
    measured = json.loads(first["stdout"])
    assert first["status"] == second["status"] == "OK"
    # /proc RSS counters are kernel estimates with asynchronous accounting.
    assert first["observed_peak_rss_bytes"] >= measured["self_peak_rss_bytes"] - 1024 * 1024
    assert first["observed_peak_rss_bytes"] < measured["self_peak_rss_bytes"] + 8 * 1024 * 1024
    assert second["observed_peak_rss_bytes"] < first["observed_peak_rss_bytes"]
    assert first["worker_pid"] != second["worker_pid"]
    assert first["peak_rss_source"] == "linux_proc_worker_VmHWM"
    assert first["peak_rss_complete"] and second["peak_rss_complete"]
    assert second["observed_peak_rss_bytes"] < 32 * 1024 * 1024
    assert first["wait4_peak_rss_bytes_including_preexec"] > first["observed_peak_rss_bytes"]


@linux
def test_sanitized_environment_and_thread_hints(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "not-for-child")
    monkeypatch.setenv("LD_PRELOAD", "not-for-child.so")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("OMP_NUM_THREADS", "128")
    result = _run_test_child("environment")
    env = json.loads(result["stdout"])
    assert not any(k in env for k in ("SECRET_TOKEN", "LD_PRELOAD", "PYTHONPATH"))
    assert env["HF_HOME"] == str(tmp_path)
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert not result["hard_thread_count_enforced"]
    assert not Path(env["HOME"]).exists()  # Private home cleaned on return.


@linux
def test_parent_descriptor_not_inherited(tmp_path):
    import fcntl
    with (tmp_path / "not-inherited").open("w") as stream:
        fd = fcntl.fcntl(stream.fileno(), fcntl.F_DUPFD, 200)
        try:
            os.set_inheritable(fd, True)
            result = _run_test_child("fd")
            assert fd not in json.loads(result["stdout"])["fds"]
        finally:
            os.close(fd)


@linux
def test_fixed_compose_public_help_and_list(tmp_path):
    help_result = run("compose", ["--help"], memory_mib=256, timeout=5)
    assert help_result["status"] == "OK", help_result
    assert json.loads(help_result["stdout"])["ok"]
    result = run("compose", ["list"], workspace=tmp_path / "workspace", memory_mib=256, timeout=5)
    assert result["status"] == "OK", result
    assert json.loads(result["stdout"])["command"] == "list"
    assert result["effective_limits"]["as"] == [256 * 1024 * 1024] * 2


@linux
def test_public_cli_fixed_operation(tmp_path):
    code, result = cli("run", "--profile", "process_as", "--memory-mib", "256",
                       "--timeout", "5", "--operation", "compose", "--workspace", str(tmp_path), "--", "list")
    assert code == 0 and result["returncode"] == 0, result
    code, result = cli("run", "--profile", "process_as", "--memory-mib", "256",
                       "--timeout", "5", "--operation", "python", "--", "-c", "print(1)")
    assert code == 2 and not result["launched"]


@linux
def test_fixed_compiler_help():
    result = run("compiler", ["--help"], memory_mib=256, timeout=5)
    assert result["status"] == "OK", result
    assert "inspect" in result["stdout"]
    assert result["returncode"] == 0


@linux
def test_output_is_bounded_and_repeated_runs_close_fds():
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(3):
        result = _run_test_child("output")
        assert result["status"] == "OUTPUT_LIMIT", result
        assert len(result["stdout"].encode()) + len(result["stderr"].encode()) <= 1024 * 1024
        assert result["group_kill_succeeded"]
    assert len(os.listdir("/proc/self/fd")) == before


def test_native_platform_fail_closed_contract():
    # This checks the actual OS, not monkeypatched platform support flags.
    result = probe()
    if platform.system() != "Linux":
        blocked = run("compose", ["--help"])
        assert blocked["status"] == "BLOCKED" and not blocked["launched"]
    if platform.system() == "Windows":
        assert result["windows"]["native"]
        assert not result["windows"]["execution_supported"]
        assert not result["windows"]["enforcement_tested"]


@pytest.mark.skip(reason="Pending implementation AND operator-delegated target Linux runner; never write system roots in CI")
def test_delegated_cgroup_target_enforcement_template():
    """Future target acceptance: authorized leaf/write readback; gated PID attach;
    finite memory reservation, pids/thread denial, CPU bandwidth telemetry;
    no pre-attachment child; memory.events attribution; cgroup.kill + empty
    population verification on timeout/cancel; cleanup of own leaf only.
    A writable ordinary directory or a mocked controller is not passing proof.
    """


@pytest.mark.skip(reason="Pending Job Object + separate host security implementation and real Windows runner")
def test_windows_job_target_enforcement_template():
    """Future target acceptance: suspended launch, noninherited job handle,
    assignment verification before resume, commit-memory/process/CPU limits,
    no breakaway, kill-on-close, and independently tested filesystem/network
    security. Native API discovery alone never turns execution support on.
    """


@linux
@pytest.mark.parametrize("death_signal", [signal.SIGTERM, signal.SIGINT, signal.SIGKILL])
def test_supervisor_death_terminates_real_fixed_worker(tmp_path, death_signal):
    marker = tmp_path / "armed-worker.json"
    # The marker is emitted only when sampling is armed by a VALID bootstrap
    # receipt. No sleep-based guess about whether prctl has been installed.
    script = r"""
import json, sys
from pathlib import Path
from asea.execution import controls as c
original = c._read
marker = Path(sys.argv[1])
def read(path):
    path = Path(path)
    result = original(path)
    if path.name == 'status' and path.parent.name.isdigit() and not marker.exists():
        pid = int(path.parent.name)
        marker.write_text(json.dumps({'pid': pid, 'stat': Path('/proc', str(pid), 'stat').read_text()}))
    return result
c._read = read
print(json.dumps(c._run_test_child('sleep', timeout=20)), flush=True)
"""
    supervisor = subprocess.Popen([sys.executable, "-c", script, str(marker)], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, start_new_session=True)
    pidfd = None
    try:
        end = time.monotonic() + 8
        while not marker.exists() and supervisor.poll() is None and time.monotonic() < end:
            time.sleep(0.01)
        assert marker.exists(), supervisor.communicate(timeout=1)
        worker = json.loads(marker.read_text())["pid"]
        pidfd = os.pidfd_open(worker) if hasattr(os, "pidfd_open") else None
        supervisor.send_signal(death_signal)
        stdout, stderr = supervisor.communicate(timeout=4)
        assert not stderr, stderr
        if death_signal == signal.SIGKILL:
            assert supervisor.returncode == -signal.SIGKILL
        else:
            result = json.loads(stdout)
            assert result['status'] == 'INTERRUPTED', result
            assert result['interruption_signal'] == death_signal
            assert result['parent_death_established']
            assert result['returncode'] == -signal.SIGKILL
        if pidfd is not None:
            import select
            assert select.select([pidfd], [], [], 2)[0], "worker outlived supervisor"
        else:
            status = Path('/proc', str(worker), 'status')
            end = time.monotonic() + 2
            while status.exists() and 'State:\tZ' not in status.read_text() and time.monotonic() < end:
                time.sleep(0.01)
            assert not status.exists() or 'State:\tZ' in status.read_text()
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=3)
        if pidfd is not None:
            os.close(pidfd)


@linux
def test_bootstrap_parent_race_check_prevents_fixture_execution():
    from asea.execution import _worker
    config = {"control_fd": 1, "expected_parent_pid": os.getpid() + 10000000,
              "memory_bytes": 96 * 1024 * 1024, "cpu_seconds": 2, "test_case": "environment"}
    child = subprocess.run([sys.executable, '-I', '-S', _worker.__file__, json.dumps(config)],
                           capture_output=True, text=True, timeout=3)
    assert child.returncode == -signal.SIGKILL
    assert not child.stdout  # No limits receipt, fixture, site or model import.


@linux
def test_bootstrap_missing_parent_contract_fails_closed():
    from asea.execution import _worker
    config = {"control_fd": 1, "memory_bytes": 96 * 1024 * 1024,
              "cpu_seconds": 2, "test_case": "environment"}
    child = subprocess.run([sys.executable, '-I', '-S', _worker.__file__, json.dumps(config)],
                           capture_output=True, text=True, timeout=3)
    assert child.returncode == 125
    events = [json.loads(line) for line in child.stdout.splitlines()]
    assert [e['event'] for e in events] == ['setup_failed']
    assert 'unsupported' in events[0]['reason']


@linux
def test_exited_leader_group_cleanup_precedes_reaping(monkeypatch):
    from asea.execution import controls as c
    original = c._kill_group
    checks = []
    def checked(pid):
        # The supervisor must still own a waitable leader at every group kill,
        # including when the leader exited successfully leaving descendants.
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        checks.append(pid)
        return original(pid)
    monkeypatch.setattr(c, '_kill_group', checked)
    result = _run_test_child('leave_descendant', timeout=3)
    assert result['status'] == 'OK', result
    assert checks and result['group_kill_succeeded']
    child = json.loads(result['stdout'])['descendant_pid']
    status = Path('/proc', str(child), 'status')
    end = time.monotonic() + 2
    while status.exists() and 'State:\tZ' not in status.read_text() and time.monotonic() < end:
        time.sleep(0.01)
    assert not status.exists() or 'State:\tZ' in status.read_text()


def test_parent_death_setup_error_is_not_silently_ignored(monkeypatch):
    import ctypes
    from asea.execution import _worker
    class Unsupported:
        def prctl(self, *args):
            return -1
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **kw: Unsupported())
    if sys.platform == 'linux':
        with pytest.raises(OSError, match='PR_SET_PDEATHSIG unsupported'):
            _worker._parent_death(os.getpid())


@linux
def test_supervisor_rejects_mismatched_parent_death_receipt(monkeypatch):
    from asea.execution import controls as c
    original = c.json.loads
    def changed(value, *args, **kwargs):
        event = original(value, *args, **kwargs)
        if isinstance(event, dict) and event.get('event') == 'limits':
            event['parent_death']['expected_parent_pid'] += 1
        return event
    monkeypatch.setattr(c.json, 'loads', changed)
    result = _run_test_child('environment')
    assert result['returncode'] == 0  # Zero exit is not a valid setup receipt.
    assert result['status'] == 'FAILED'
    assert not result['supported'] and not result['parent_death_established']
    assert not result['enforcement_established']
    assert 'No valid parent-death' in result['reason']
