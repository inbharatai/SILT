"""Trusted outer-controller bootstrap, launched by fixed Python -I -S path.

Only stdlib runs before Linux PDEATHSIG is installed and the parent race checked.
This internal protocol wraps the controller's existing argv; it is not a new
public executable/module-selection interface. No shell, credential changes or
preexec_fn callbacks. PDEATHSIG survives unprivileged exec of CLI/native probes.
Known-group cleanup is the supervisor's job; escaped descendants are not claimed.
"""
import os
import sys


def _parent_death(expected_parent_pid):
    if sys.platform != "linux" or expected_parent_pid <= 1:
        raise RuntimeError("Linux parent-death contract required; other platforms unsupported")
    import ctypes
    import signal
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    actual = ctypes.c_int()
    if libc.prctl(2, ctypes.byref(actual), 0, 0, 0) != 0 or actual.value != signal.SIGKILL:
        raise RuntimeError("parent-death signal readback mismatch")
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
        os._exit(125)
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        raise RuntimeError("controller child must lead its process group/session")


def main():
    _parent_death(int(sys.argv[1]))
    import signal
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    argv = sys.argv[2:]
    if not argv:
        raise ValueError("missing internal argv")
    # All executable selection is already present in controller run(argv).
    os.execvpe(argv[0], argv, os.environ)


if __name__ == "__main__":
    main()
