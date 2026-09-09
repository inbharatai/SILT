"""Test-only Linux process-death observation; never signal or reap a process."""
import errno
from pathlib import Path
from time import monotonic, sleep

import pytest


def assert_process_dead(pid, timeout=2):
    """Wait within the caller's budget for disappearance or terminal Z/X.

    Linux proc_pid_status(5) documents Z (zombie) and X (dead). Both are
    terminal, even when the adopting init has not reaped an orphan yet.
    Lowercase x is deliberately not accepted: this reads status, not a
    historical proc_pid_stat(5) state table. No live/unknown state is proof.
    """
    status = Path(f"/proc/{pid}/status")
    deadline = monotonic() + timeout
    while True:
        try:
            text = status.read_text()
        except (FileNotFoundError, ProcessLookupError) as error:
            # The process may vanish before open (ENOENT) or during read
            # (ESRCH). Permission/I/O errors are not evidence of death.
            if error.errno not in (errno.ENOENT, errno.ESRCH):
                raise
            return
        state = next((line.split()[1:] for line in text.splitlines()
                      if line.startswith("State:")), [])
        if state and state[0] in {"Z", "X"}:
            return
        remaining = deadline - monotonic()
        if remaining <= 0:
            pytest.fail(f"process {pid} remains live or unconfirmed: {text!r}")
        sleep(min(0.01, remaining))
