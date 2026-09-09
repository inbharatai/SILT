"""POSIX fresh-worker supervisor; capabilities are not enforcement evidence."""
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import site
import subprocess
import sys
import tempfile
import time

SCHEMA = "silt.resource-controls.v1"
_OPERATIONS = {"compose", "compiler", "specialist"}
_TEST_CASES = {"allocate", "sleep", "cpu", "descendant", "environment", "rss", "fd", "output", "leave_descendant"}
_OUTPUT_LIMIT = 1024 * 1024
_CONTROL_LIMIT = 16384
_CACHE_KEYS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
               "TRANSFORMERS_CACHE", "XDG_CACHE_HOME")


def _read(path):
    try:
        return Path(path).read_text()[:8192].strip()
    except OSError:
        return None


def probe(delegated_root=None):
    """Read-only native observation. Never creates jobs/cgroups or tests limits.

    An operator's proposed path is recorded but never treated as delegation.
    This release deliberately has no cgroup/Windows execution backend.
    """
    native = platform.system()
    root = Path("/sys/fs/cgroup")
    controls = ("memory.max", "pids.max", "cpu.max", "cgroup.subtree_control")
    cgroup = {
        "available": native == "Linux" and (root / "cgroup.controllers").is_file(),
        "controllers": _read(root / "cgroup.controllers") if native == "Linux" else None,
        "root_observation": {name: {"value": _read(root / name),
                                    "access_w_ok": os.access(root / name, os.W_OK)}
                             for name in controls} if native == "Linux" else {},
        "requested_delegated_root": str(delegated_root) if delegated_root is not None else None,
        "delegated": False,
        "delegation_verified": False,
        "effective_write_validation": "not_attempted",
        "enforcement_tested": False,
        "execution_supported": False,
        "reason": "Verified delegated backend not implemented; no system cgroup writes are performed.",
    }
    as_available = cpu_available = False
    if os.name == "posix":
        try:
            import resource
            as_available = hasattr(resource, "RLIMIT_AS")
            cpu_available = hasattr(resource, "RLIMIT_CPU")
        except ImportError:
            pass
    windows = {"native": native == "Windows", "job_api_available": False,
               "execution_supported": False, "enforcement_tested": False,
               "reason": "No validated Job Object plus host security backend."}
    if native == "Windows":
        try:
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            windows["job_api_available"] = all(hasattr(kernel, name) for name in
                ("CreateJobObjectW", "AssignProcessToJobObject", "QueryInformationJobObject"))
        except (AttributeError, OSError):
            pass
    return {"schema": SCHEMA, "platform": native, "os_name": os.name,
            "cgroup_v2": cgroup, "windows": windows,
            "process_as": {"rlimit_as_available": as_available,
                "rlimit_cpu_available": cpu_available,
                "execution_supported": native == "Linux" and as_available and cpu_available and hasattr(os, "wait4") and hasattr(os, "waitid") and hasattr(os, "WNOWAIT"),
                "enforcement_tested": False, "memory_metric": "virtual_address_space_per_process",
                "aggregate_memory_enforced": False, "hard_rss_enforced": False,
                "hard_process_count_enforced": False, "hard_thread_count_enforced": False,
                "reason": "Linux implementation; actual setrlimit/readback is required on each fresh worker."}}


def _environment(home):
    # Never copy PYTHON*, LD_*, credentials, proxy configuration, or user PATH.
    result = {"PATH": os.defpath, "HOME": home, "TMPDIR": home,
              "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
              "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
              "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "",
              "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1"}
    for name in _CACHE_KEYS:
        value = os.environ.get(name)
        if value and Path(value).is_absolute() and Path(value).is_dir():
            result[name] = str(Path(value).resolve())
    # Preserve the conventional local model cache location without inheriting HOME.
    if not any(name in result for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")):
        cache = Path.home() / ".cache" / "huggingface"
        if cache.is_dir():
            result["HF_HOME"] = str(cache.resolve())
    return result


def run(operation, arguments=(), *, profile="process_as", memory_mib=512,
        timeout=60.0, workspace=None, cpu_seconds=None, delegated_root=None):
    """Run only a fixed public SILT module in a fresh, limited interpreter.

    ``arguments`` are module CLI arguments, never shell/program/Python source.
    Result contains actual returncode, bounded stdout/stderr and host evidence.
    The worker uses isolated Python startup and sets limits BEFORE importing SILT
    or site packages. Do not substitute this API for the candidate OS sandbox.
    """
    return _run(operation, arguments, profile=profile, memory_mib=memory_mib,
                timeout=timeout, workspace=workspace, cpu_seconds=cpu_seconds,
                delegated_root=delegated_root)


def _run_test_child(case, *, memory_mib=96, timeout=3.0, cpu_seconds=None):
    """Private finite kernel fixtures. No arbitrary code/command parameter."""
    if case not in _TEST_CASES:
        raise ValueError("Unknown fixed test case")
    return _run(None, (), profile="process_as", memory_mib=memory_mib,
                timeout=timeout, workspace=None, cpu_seconds=cpu_seconds,
                delegated_root=None, test_case=case)


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False


def _run(operation, arguments, *, profile, memory_mib, timeout, workspace,
         cpu_seconds, delegated_root, test_case=None):
    started = time.monotonic()
    result = {"schema": SCHEMA, "operation": operation, "profile": profile,
              "status": "BLOCKED", "launched": False, "supported": False,
              "returncode": None, "resource_cause": None, "stdout": "", "stderr": "",
              "enforcement_established": False, "enforcement_tested": False,
              "parent_death_required": True, "parent_death_established": False,
              "aggregate_memory_enforced": False, "hard_rss_enforced": False,
              "hard_process_count_enforced": False, "hard_thread_count_enforced": False,
              "cpu_bandwidth_enforced": False, "memory_metric": "virtual_address_space_per_process",
              "observed_peak_rss_bytes": None, "peak_rss_source": "linux_proc_worker_VmHWM",
              "peak_rss_complete": False, "wait4_peak_rss_bytes_including_preexec": None,
              "cpu_consumed_seconds": None, "setup_elapsed_seconds": None,
              "group_kill_attempted": False, "group_kill_succeeded": None,
              "descendant_closure": "not_started"}
    def finish(reason=None):
        result["elapsed_seconds"] = time.monotonic() - started
        if reason:
            result["reason"] = reason
        return result

    try:
        if operation not in _OPERATIONS and test_case not in _TEST_CASES:
            return finish("Only the compose, compiler and specialist public modules are allowed.")
        if isinstance(arguments, (str, bytes)):
            return finish("arguments must be a sequence of strings, not a command string")
        argv = list(arguments)
        if len(argv) > 256 or any(not isinstance(a, str) or "\x00" in a or len(a) > 65536 for a in argv):
            return finish("Invalid or oversized argument vector")
        if sum(len(a) for a in argv) > 131072:
            return finish("Argument vector exceeds 128 KiB")
        if isinstance(memory_mib, bool) or not isinstance(memory_mib, int) or not 32 <= memory_mib <= 1048576:
            return finish("memory_mib must be an integer from 32 to 1048576")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
            return finish("timeout must be finite and in (0, 86400]")
        cpu = math.ceil(timeout) if cpu_seconds is None else cpu_seconds
        if isinstance(cpu, bool) or not isinstance(cpu, int) or not 1 <= cpu <= 86400:
            return finish("cpu_seconds must be an integer from 1 to 86400")
        result["requested_limits"] = {"address_space_bytes": memory_mib * 1024 * 1024,
                                      "cpu_soft_seconds": cpu, "cpu_hard_seconds": cpu + 1,
                                      "wall_seconds": timeout, "combined_output_bytes": _OUTPUT_LIMIT}
        if profile == "cgroup_v2":
            result["memory_metric"] = "cgroup_accounted_memory_requested_not_enforced"
            if delegated_root is None:
                return finish("cgroup_v2 requires an operator-authorized delegated root; none supplied. No fallback.")
            return finish("cgroup_v2 backend pending: authorized root and effective write/attachment validation are required; no launch or fallback.")
        if profile != "process_as":
            return finish("Unknown profile; no fallback")
        if not probe()["process_as"]["execution_supported"]:
            return finish("process_as execution is implemented only on native Linux with RLIMIT_AS, RLIMIT_CPU and wait4; no fallback")
        if workspace is not None:
            if operation != "compose":
                return finish("--workspace belongs only to compose; compiler/specialist use their own command model/output/workspace paths")
            argv = ["--workspace", str(Path(workspace).resolve()), *argv]
        config = {"operation": operation, "arguments": argv, "memory_bytes": memory_mib * 1024 * 1024,
                  "cpu_seconds": cpu, "test_case": test_case,
                  "expected_parent_pid": os.getpid(),
                  "runtime_site_dirs": [str(Path(p).resolve()) for p in site.getsitepackages() if Path(p).is_dir()]}
        return _supervise(config, result, started, timeout, finish)
    except (ValueError, TypeError, OSError) as exc:
        if result["launched"]:
            result["status"] = "FAILED"
            return finish("Resource supervisor failed after launch; execution/measurement may be incomplete: " + str(exc))
        return finish("Resource setup failed before launch: " + str(exc))


def _supervise(config, result, started, timeout, finish):
    proc = None
    read_fd = write_fd = None
    selector = selectors.DefaultSelector()
    chunks = {"stdout": bytearray(), "stderr": bytearray(), "control": bytearray()}
    output_count = 0
    usage = None
    terminal = None
    reaped = False
    deadline = started + timeout
    cleanup_deadline = None
    sampling_armed = False
    previous_handlers = {}
    cancelled = None

    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = cancelled or signum

    def valid_death(setup):
        death = setup.get("parent_death", {})
        return (death.get("mechanism") == "linux_prctl_PDEATHSIG"
                and death.get("signal") == signal.SIGKILL
                and death.get("expected_parent_pid") == config["expected_parent_pid"]
                and death.get("observed_parent_pid") == config["expected_parent_pid"]
                and death.get("worker_pid") == proc.pid
                and death.get("established_before_site") is True)

    def sample_rss():
        # Do not sample before the bootstrap receipt: the process may still
        # carry the requesting parent's pre-exec mm and huge RSS watermark.
        if sampling_armed and not reaped:
            text = _read(Path("/proc") / str(proc.pid) / "status")
            if text:
                for line in text.splitlines():
                    if line.startswith("VmHWM:"):
                        peak = int(line.split()[1]) * 1024
                        result["observed_peak_rss_bytes"] = max(peak, result["observed_peak_rss_bytes"] or 0)
                        break

    try:
        # Python permits signal handlers only on the main thread. Other-thread
        # callers still have the hard kernel worker parent-death contract.
        import threading
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, cancel)
        with tempfile.TemporaryDirectory(prefix="silt-worker-") as home:
            read_fd, write_fd = os.pipe()
            config["control_fd"] = write_fd
            if time.monotonic() >= deadline:
                result["status"] = "TIMEOUT"
                return finish("Total budget exhausted before launch")
            proc = subprocess.Popen([sys.executable, "-I", "-S", str(Path(__file__).with_name("_worker.py")),
                                     json.dumps(config, separators=(",", ":"))],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    env=_environment(home), close_fds=True, pass_fds=(write_fd,),
                                    start_new_session=True)
            result["launched"] = True
            result["worker_pid"] = proc.pid
            os.close(write_fd)
            write_fd = None
            for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr"), (read_fd, "control")):
                fd = stream if isinstance(stream, int) else stream.fileno()
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ, name)
            while not reaped or selector.get_map():
                now = time.monotonic()
                sample_rss()
                if terminal is None and cancelled is not None:
                    terminal = "INTERRUPTED"
                    result["interruption_signal"] = cancelled
                if terminal is None and now >= deadline:
                    terminal = "TIMEOUT"
                if terminal is not None and cleanup_deadline is None:
                    result["group_kill_attempted"] = True
                    result["group_kill_succeeded"] = _kill_group(proc.pid)
                    cleanup_deadline = now + 0.5
                if not reaped:
                    # Keep the leader waitable until its group is killed: a
                    # reaped PID/PGID could otherwise be reused by another job.
                    exited = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    if exited is not None:
                        result["group_kill_attempted"] = True
                        result["group_kill_succeeded"] = _kill_group(proc.pid)
                        _, status, usage = os.wait4(proc.pid, 0)
                        reaped = True
                        proc.returncode = os.waitstatus_to_exitcode(status)
                        result["returncode"] = proc.returncode
                        cleanup_deadline = now + 0.5
                if cleanup_deadline is not None and now >= cleanup_deadline and reaped:
                    break
                for key, _ in selector.select(0.02):
                    try:
                        block = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not block:
                        selector.unregister(key.fd)
                        continue
                    name = key.data
                    if name == "control":
                        if len(chunks[name]) + len(block) > _CONTROL_LIMIT:
                            terminal = terminal or "FAILED"
                        chunks[name].extend(block[:max(0, _CONTROL_LIMIT - len(chunks[name]))])
                        if not sampling_armed and b"\n" in chunks[name]:
                            try:
                                first = json.loads(chunks[name].splitlines()[0])
                                sampling_armed = (first.get("event") == "limits" and first.get("as") == [config["memory_bytes"]] * 2 and valid_death(first))
                            except (ValueError, AttributeError):
                                pass
                            sample_rss()
                    else:
                        available = max(0, _OUTPUT_LIMIT - output_count)
                        chunks[name].extend(block[:available])
                        output_count += len(block)
                        if output_count > _OUTPUT_LIMIT:
                            terminal = terminal or "OUTPUT_LIMIT"
            result["descendant_closure"] = "process_group_killed_not_escape_proof" if result["group_kill_succeeded"] else "kill_failed"
            result["pipes_drained"] = not bool(selector.get_map())
            result["stdout"] = chunks["stdout"].decode("utf-8", "replace")
            result["stderr"] = chunks["stderr"].decode("utf-8", "replace")
            if usage is not None:
                # Keep the raw kernel accounting separately: Linux ru_maxrss
                # can include the parent's inherited pre-exec fork watermark.
                result["wait4_peak_rss_bytes_including_preexec"] = int(usage.ru_maxrss * 1024)
                result["cpu_consumed_seconds"] = usage.ru_utime + usage.ru_stime
            events = []
            try:
                events = [json.loads(line) for line in chunks["control"].splitlines()]
            except (ValueError, UnicodeError):
                terminal = terminal or "FAILED"
            result["worker_events"] = events
            setup = next((e for e in events if isinstance(e, dict) and e.get("event") == "limits"), None)
            result["parent_death_established"] = bool(setup and valid_death(setup))
            if setup and valid_death(setup) and setup.get("as") == [config["memory_bytes"]] * 2 and setup.get("cpu") == [config["cpu_seconds"], config["cpu_seconds"] + 1]:
                result["enforcement_established"] = True
                result["supported"] = True
                result["effective_limits"] = setup
                result["setup_elapsed_seconds"] = setup["monotonic"] - started
            final_usage = next((e for e in events if isinstance(e, dict) and e.get("event") == "usage"), None)
            if final_usage and type(final_usage.get("peak_rss_bytes")) is int and final_usage["peak_rss_bytes"] > 0:
                result["observed_peak_rss_bytes"] = max(final_usage["peak_rss_bytes"], result["observed_peak_rss_bytes"] or 0)
                result["peak_rss_complete"] = True
            memory_error = any(isinstance(e, dict) and e.get("event") == "memory_error" for e in events)
            result["status"] = terminal or ("OK" if proc.returncode == 0 else "FAILED")
            if terminal == "TIMEOUT":
                result["resource_cause"] = "host_monotonic_wall_deadline"
            elif terminal is None and result["supported"] and memory_error:
                result["status"] = "RESOURCE_LIMIT"
                result["resource_cause"] = "worker_MemoryError_under_kernel_RLIMIT_AS_not_RSS_OOM"
                result["enforcement_tested"] = config["test_case"] == "allocate"
            elif terminal is None and result["supported"] and proc.returncode == -signal.SIGXCPU:
                result["status"] = "RESOURCE_LIMIT"
                result["resource_cause"] = "kernel_SIGXCPU"
            if not result["supported"] and terminal is None:
                setup_failed = any(isinstance(e, dict) and e.get("event") == "setup_failed" for e in events)
                result["status"] = "BLOCKED" if setup_failed else "FAILED"
                result["reason"] = ("Required worker parent-death/resource setup unsupported before public-module import." if setup_failed
                                    else "No valid parent-death/kernel-limit receipt; module execution and measurements are unverified.")
            return finish()
    finally:
        # Includes KeyboardInterrupt/cancellation and launch/selector exceptions.
        if proc is not None:
            if not reaped:
                _kill_group(proc.pid)
                proc.wait()
            for stream in (proc.stdout, proc.stderr):
                if stream:
                    stream.close()
        selector.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)
