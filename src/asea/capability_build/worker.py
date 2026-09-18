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
  * A ``blocked`` frame -> :class:`BlockedResource` with the worker's
    exact requirement + remedy (a small host gets an honest
    ``BLOCKED_RESOURCE``, never a local weaker path).

This module is stdlib-only (subprocess + json): it must work without any
ML dependency, because its whole point is to keep the GLM runtime away
from this process.
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import BlockedResource, WorkerCrashed, WorkerUnavailable
from . import worker_protocol as proto

_WORKER_REL = Path("workers") / "glm53" / "worker.py"


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
            return
        try:
            self._process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=None,
            )
        except (OSError, ValueError) as exc:
            raise WorkerUnavailable("worker did not start: %s" % exc) from exc

    def stop(self) -> None:
        if self._process is None:
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
            self._process.kill()
        self._process = None

    # -- protocol -----------------------------------------------------------

    def request(
        self, op: str, payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send one request, read one response. The response id MUST echo
        the request id; a mismatch is a protocol fault, not data."""
        self.start()
        assert self._process is not None
        request = proto.make_request(op, payload)
        line = proto.encode(request)
        try:
            self._process.stdin.write(line)  # type: ignore[union-attr]
            self._process.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError) as exc:
            raise WorkerCrashed(
                "worker died before accepting %r (partial frames: %d). "
                "The lost request was never executed."
                % (op, len(self._frames))
            ) from exc
        response_line = self._process.stdout.readline()  # type: ignore[union-attr]
        if not response_line:
            raise WorkerCrashed(
                "worker closed its stdout during %r after %d complete "
                "frames; the lost portion was never observed and is not "
                "fabricated" % (op, len(self._frames))
            )
        try:
            response = proto.decode(response_line.strip())
        except ValueError as exc:
            raise WorkerCrashed(
                "worker emitted an invalid frame during %r: %s (%d "
                "complete frames preserved)" % (op, exc, len(self._frames))
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

    # -- high-level operations ----------------------------------------------

    def hello(self) -> Dict[str, Any]:
        """Handshake + hardware preflight. On a small host this raises
        :class:`BlockedResource` with the exact requirement + remedy."""
        return self.request("hello", {})