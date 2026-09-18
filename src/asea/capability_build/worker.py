"""Host-side client for the isolated GLM worker (brief §F).

The core SILT process talks to ``workers/glm53`` through versioned JSONL
frames (:mod:`asea.capability_build.worker_protocol`) and NEVER imports the
Transformers 5.x runtime itself. The worker can be started either as a
Docker container (image built from ``workers/glm53/Dockerfile``) or as a
bare interpreter (the worker's own env, on the GPU box).

Failure honesty (binding):

  * Docker/binary unavailable, image not built, checkpoint not mounted ->
    :class:`WorkerUnavailable` with the exact remedy.
  * Worker dies mid-request -> :class:`WorkerCrashed` carrying every
    COMPLETE validated frame emitted so far; the lost portion is never
    fabricated.
  * Worker exceeds the request deadline -> :class:`WorkerTimeout` (a
    :class:`WorkerCrashed` subclass): a watchdog kills the worker process
    group at the deadline, and the lost portion is reported as lost. The
    ``timeout`` argument is therefore ENFORCED -- it is never accepted
    and silently ignored, and neither the request write nor the read is
    allowed to block forever: the watchdog is armed BEFORE the request is
    transmitted, so a worker that stops reading its stdin is killed at
    the deadline too.
  * A ``blocked`` frame -> :class:`BlockedResource` with the worker's
    exact requirement + remedy (a small host gets an honest
    ``BLOCKED_RESOURCE``, never a local weaker path).

This module is stdlib-only (subprocess + threading + json): it must work
without any ML dependency, because its whole point is to keep the GLM
runtime away from this process.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import BlockedResource, WorkerCrashed, WorkerTimeout, WorkerUnavailable
from . import worker_protocol as proto

_WORKER_REL = Path("workers") / "glm53" / "worker.py"
_STDERR_TAIL_CHARS = 4000


class GlmWorkerClient:
    """One persistent worker process per client instance."""

    def __init__(
        self,
        *,
        repo_root: Path,
        checkpoint: str,
        use_docker: Optional[bool] = None,
        docker_image: str = "silt-glm53-worker",
        python: str = "python3",
        timeout: float = 3600.0,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.checkpoint = checkpoint
        self.timeout = timeout
        if use_docker is None:
            use_docker = shutil.which("docker") is not None
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.python = python
        self._process: Optional[subprocess.Popen] = None
        self._frames: List[Dict[str, Any]] = []
        self._stderr_file: Optional[Any] = None
        self._deadline_fired = False

    # -- lifecycle ----------------------------------------------------------

    def _command(self) -> List[str]:
        if self.use_docker:
            if shutil.which("docker") is None:
                raise WorkerUnavailable(
                    "docker is not on PATH; cannot start the isolated GLM "
                    "worker. Remedy: install Docker with a Linux engine "
                    "(WSL2 backend on Windows) or run the worker's "
                    "interpreter path on the GPU box."
                )
            return [
                "docker", "run", "--rm", "-i",
                "-v", "%s:/checkpoint:ro" % self.checkpoint,
                "-e", "GLM_CHECKPOINT=/checkpoint",
                self.docker_image,
            ]
        script = self.repo_root / _WORKER_REL
        if not script.is_file():
            raise WorkerUnavailable(
                "worker script not found at %s. Remedy: run from a SILT "
                "checkout." % script
            )
        return [self.python, str(script)]

    def start(self) -> None:
        if self._process is not None:
            # The watchdog's kill (or an unnoticed crash) leaves a DEAD
            # process handle here. Treating it as live wedges the client:
            # the next request would write to a corpse and surface a
            # generic WorkerCrashed instead of the honest restart-or-refuse
            # path. Reap the dead handle and respawn fresh below; a respawned
            # worker has no mask state, and its restore/verify ops refuse
            # honestly rather than silently reporting a clean teacher
            # (audit 2026-09-18).
            if self._process.poll() is None:
                return
            try:
                self._process.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._process = None
            self._close_stderr()
        try:
            # stderr goes to a TEMP FILE, not a PIPE: an un-drained PIPE
            # deadlocks a chatty worker once the OS buffer fills, and a
            # never-drained PIPE means the diagnostic for the crash was
            # thrown away. The file is read (tailed) on failure.
            self._stderr_file = tempfile.TemporaryFile(
                mode="w+", encoding="utf-8", errors="replace"
            )
            self._deadline_fired = False
            self._process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                env=None,
                # New session/process group so a deadline kill takes down
                # the whole worker tree, not just the shim interpreter.
                # POSIX-only argument: Windows Popen rejects it outright.
                **({"start_new_session": True} if os.name == "posix" else {}),
            )
        except (OSError, ValueError) as exc:
            self._close_stderr()
            raise WorkerUnavailable("worker did not start: %s" % exc) from exc

    def _kill_worker(self) -> None:
        """Watchdog callback: kill the worker tree at the deadline."""
        self._deadline_fired = True
        process = self._process
        if process is None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (AttributeError, OSError, PermissionError):
            # POSIX-only or the group is already gone: kill what remains.
            try:
                process.kill()
            except (OSError, ValueError):
                pass

    def _stderr_tail(self) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            self._stderr_file.seek(max(0, size - _STDERR_TAIL_CHARS))
            return self._stderr_file.read() or ""
        except (OSError, ValueError):
            return ""

    def _close_stderr(self) -> None:
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except OSError:
                pass
            self._stderr_file = None

    def stop(self) -> None:
        if self._process is None:
            self._close_stderr()
            return
        try:
            self.request("shutdown", {}, timeout=30.0)
        except Exception:
            pass
        try:
            self._process.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._kill_worker()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self._process = None
        self._close_stderr()

    # -- protocol -----------------------------------------------------------

    def request(
        self, op: str, payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send one request, read one response. The response id MUST echo
        the request id; a mismatch is a protocol fault, not data.

        ``timeout`` (seconds) is ENFORCED with a watchdog that kills the
        worker tree at the deadline; the read can never block unboundedly.
        The watchdog is armed BEFORE the request is written: a worker that
        stops reading stdin cannot block the write past the deadline
        either -- transmission is inside the protected window, not before
        it. The default is the client-level ``timeout``. Whatever the
        worker had not produced by the deadline is reported as lost, never
        fabricated."""
        self.start()
        assert self._process is not None
        request = proto.make_request(op, payload)
        line = proto.encode(request)
        deadline = float(timeout) if timeout is not None else float(self.timeout)
        watchdog = threading.Timer(deadline, self._kill_worker)
        watchdog.daemon = True
        self._deadline_fired = False
        watchdog.start()
        try:
            try:
                self._process.stdin.write(line)  # type: ignore[union-attr]
                self._process.stdin.flush()  # type: ignore[union-attr]
            except (BrokenPipeError, OSError) as exc:
                # The write itself is inside the deadline window: if the
                # watchdog already killed a worker that stopped reading
                # stdin, the deadline (not a generic pipe failure) is the
                # honest diagnosis.
                if self._deadline_fired:
                    raise WorkerTimeout(
                        "worker exceeded the %.1fs deadline during %r "
                        "(it stopped reading its stdin and was killed); the "
                        "request was never accepted.%s"
                        % (deadline, op, self._stderr_suffix())
                    ) from exc
                raise WorkerCrashed(
                    "worker died before accepting %r (partial frames: %d). "
                    "The lost request was never executed.%s"
                    % (op, len(self._frames), self._stderr_suffix())
                ) from exc
            try:
                response_line = self._process.stdout.readline()  # type: ignore[union-attr]
            finally:
                watchdog.cancel()
        finally:
            watchdog.cancel()
        if self._deadline_fired:
            raise WorkerTimeout(
                "worker exceeded the %.1fs deadline during %r and was "
                "killed; %d complete frames were preserved and the lost "
                "portion was never observed.%s"
                % (deadline, op, len(self._frames), self._stderr_suffix())
            )
        if not response_line:
            raise WorkerCrashed(
                "worker closed its stdout during %r after %d complete "
                "frames; the lost portion was never observed and is not "
                "fabricated%s"
                % (op, len(self._frames), self._stderr_suffix())
            )
        try:
            response = proto.decode(response_line.strip())
        except ValueError as exc:
            raise WorkerCrashed(
                "worker emitted an invalid frame during %r: %s (%d "
                "complete frames preserved)%s"
                % (op, exc, len(self._frames), self._stderr_suffix())
            ) from exc
        if response.get("id") != request["id"]:
            raise WorkerCrashed(
                "worker response id %r does not echo request id %r; "
                "responses may be interleaved from another consumer"
                % (response.get("id"), request["id"])
            )
        self._frames.append(response)
        if not response.get("ok"):
            error = response.get("error") or {}
            if error.get("kind") == "blocked":
                raise BlockedResource(
                    requirement=str(error.get("requirement")),
                    remedy=str(error.get("remedy")),
                )
            raise WorkerCrashed(
                "worker refused %s (%s): %s"
                % (op, error.get("kind"), error.get("message"))
            )
        return response.get("result") or {}

    def _stderr_suffix(self) -> str:
        tail = self._stderr_tail()
        if not tail:
            return " (no worker stderr captured)"
        return " Worker stderr tail: %r" % tail[-_STDERR_TAIL_CHARS:]

    # -- high-level operations ----------------------------------------------

    def hello(self) -> Dict[str, Any]:
        """Handshake + hardware preflight. On a small host this raises
        :class:`BlockedResource` with the exact requirement + remedy."""
        return self.request("hello", {})