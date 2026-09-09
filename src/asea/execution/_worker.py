"""Fixed trusted bootstrap. Invoked by absolute path with Python -I -S.

Do not import asea, site packages, or ML before installing the kernel limits.
There is deliberately no command/source/module-name execution argument.
"""
import json
import os
import sys
import time


def _parent_death(expected_parent_pid):
    """Unprivileged Linux contract, before site/model imports; fork clears it.

    Arm first, then check the supervisor PID captured BEFORE Popen. If it died
    before prctl, getppid differs; if it dies after, the kernel sends SIGKILL.
    This protects this worker, not arbitrary descendants or hostile code.
    """
    if sys.platform != "linux" or type(expected_parent_pid) is not int or expected_parent_pid <= 1:
        raise RuntimeError("Linux parent-death contract unsupported")
    import ctypes
    import signal
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG unsupported")
    actual = ctypes.c_int()
    if libc.prctl(2, ctypes.byref(actual), 0, 0, 0) != 0 or actual.value != signal.SIGKILL:
        raise RuntimeError("Parent-death signal readback mismatch")
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
        os._exit(125)
    return {"mechanism": "linux_prctl_PDEATHSIG", "signal": signal.SIGKILL,
            "expected_parent_pid": expected_parent_pid, "observed_parent_pid": os.getppid(),
            "worker_pid": os.getpid(), "established_before_site": True}


def _peak_rss_bytes():
    # VmHWM belongs to this exec's mm. getrusage/wait4 ru_maxrss can retain
    # the parent's pre-exec fork watermark and is unsuitable for this metric.
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return None


def _fixture(case):
    """Finite, fixed test-only cases. Not reachable through the public CLI/API."""
    if case == "allocate":
        # One finite allocation larger than the entire AS ceiling. No RSS stress:
        # the kernel rejects address-space reservation before pages are touched.
        import resource
        limit = resource.getrlimit(resource.RLIMIT_AS)[0]
        bytearray(limit + 1024 * 1024)
        return 9
    if case == "sleep":
        time.sleep(10)
    elif case == "cpu":
        # Finite fallback deadline, plus the separately installed RLIMIT_CPU.
        end = time.monotonic() + 5
        while time.monotonic() < end:
            pass
    elif case in {"descendant", "leave_descendant"}:
        child = os.fork()
        if child == 0:
            time.sleep(10)
            os._exit(0)
        print(json.dumps({"descendant_pid": child}), flush=True)
        if case == "leave_descendant":
            return 0
        time.sleep(10)
    elif case == "environment":
        print(json.dumps(dict(os.environ)), flush=True)
    elif case == "rss":
        import resource
        payload = bytearray(16 * 1024 * 1024)
        print(json.dumps({"self_peak_rss_bytes": _peak_rss_bytes(),
                          "allocated_bytes": len(payload)}), flush=True)
    elif case == "fd":
        print(json.dumps({"fds": sorted(int(fd) for fd in os.listdir("/proc/self/fd"))}), flush=True)
    elif case == "output":
        for _ in range(32):
            os.write(1, b"x" * 65536)
    else:
        raise ValueError("Unknown fixed test fixture")
    return 0


def main():
    config = json.loads(sys.argv[1])
    control = config["control_fd"]
    os.set_inheritable(control, False)

    def event(payload):
        os.write(control, json.dumps(payload, separators=(",", ":")).encode() + b"\n")

    # Setting limits is an irreversible precondition, not a best-effort hint.
    try:
        death = _parent_death(config["expected_parent_pid"])
        import resource
        memory = config["memory_bytes"]
        cpu = config["cpu_seconds"]
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        actual_as = resource.getrlimit(resource.RLIMIT_AS)
        actual_cpu = resource.getrlimit(resource.RLIMIT_CPU)
        if actual_as != (memory, memory) or actual_cpu != (cpu, cpu + 1):
            raise RuntimeError("Kernel resource-limit readback mismatch")
        event({"event": "limits", "as": actual_as, "cpu": actual_cpu,
               "parent_death": death,
               "core": resource.getrlimit(resource.RLIMIT_CORE), "monotonic": time.monotonic()})
    except BaseException as exc:
        event({"event": "setup_failed", "type": type(exc).__name__,
               "reason": "Required parent-death or resource contract unsupported: " + str(exc)[:512]})
        os.close(control)
        return 125
    code = 1
    try:
        if config["test_case"] is not None:
            code = _fixture(config["test_case"])
        else:
            # -S suppresses .pth/sitecustomize execution before limits. Installed
            # site packages are trusted operator runtime, not workspace code.
            import site
            site.main()
            # Use only supervisor-derived installation site directories, never
            # its sys.path, cwd, user site, or a caller-provided module directory.
            for directory in config["runtime_site_dirs"]:
                site.addsitedir(directory)
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            if config["operation"] == "compose":
                from asea.compose.__main__ import main as public_main
            elif config["operation"] == "compiler":
                from asea.compiler.__main__ import main as public_main
            elif config["operation"] == "specialist":
                from asea.specialist.__main__ import main as public_main
            else:
                raise ValueError("Operation is not allowlisted")
            code = public_main(config["arguments"])
    except MemoryError:
        event({"event": "memory_error"})
        code = 70
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:
        event({"event": "worker_error", "type": type(exc).__name__, "message": str(exc)[:1024]})
        code = 1
    finally:
        try:
            event({"event": "usage", "peak_rss_bytes": _peak_rss_bytes()})
        except (OSError, MemoryError):
            pass
        os.close(control)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
