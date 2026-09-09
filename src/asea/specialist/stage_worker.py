"""Trusted fixed specialist bootstrap: Python -I -S, never a caller module.

Only stdlib is imported before the Linux parent-death contract is established.
This protects the stage leader, not arbitrary descendants that escape its group.
"""
import os
import sys


def _parent_death(expected_parent_pid):
    if sys.platform != "linux" or expected_parent_pid <= 1:
        raise RuntimeError("Linux parent-death contract required")
    import ctypes
    import signal
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    actual = ctypes.c_int()
    if libc.prctl(2, ctypes.byref(actual), 0, 0, 0) != 0 or actual.value != signal.SIGKILL:
        raise RuntimeError("parent-death signal readback mismatch")
    # Parent may have died between Popen and prctl. Never import site in that race.
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
        os._exit(125)
    # Popen creates the group/session before exec, so parent knows both IDs.
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        raise RuntimeError("stage must lead its own process group/session")


def _public_cli(argv):
    # -S on older Python omits venv prefix discovery. Derive it from the fixed
    # interpreter path, not PYTHONPATH or caller arguments. Do not execute .pth,
    # sitecustomize or usercustomize; operator-installed packages are read only.
    import site
    from pathlib import Path
    prefix = str(Path(sys.executable).absolute().parent.parent)
    paths = site.getsitepackages([prefix, sys.base_prefix])
    sys.path[:0] = [str(Path(__file__).resolve().parents[2])]
    sys.path.extend(path for path in paths if path not in sys.path and os.path.isdir(path))
    from asea.specialist.__main__ import main
    return main(argv)


def main():
    # Fixed private protocol: parent PID then the public CLI operation/options.
    # No executable, environment, site path, module or source override exists.
    _parent_death(int(sys.argv[1]))
    import signal
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    sys.dont_write_bytecode = True
    argv = sys.argv[2:]
    if not argv or argv[0] not in ("reconstruct", "recover", "evaluate"):
        raise ValueError("unsupported stage operation")
    return _public_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
