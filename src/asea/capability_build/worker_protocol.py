"""Versioned IPC protocol between SILT core and the isolated GLM worker.

The main SILT process NEVER imports Transformers 5.x or torch's GLM
runtime: the worker (``workers/glm53``) owns them, in its own image with
its own pinned lock file. Both sides import THIS module -- it is stdlib
only, so the worker image needs no asea ML extras and the core environment
needs nothing new.

Wire format: one JSON object per line (JSONL) on stdin/stdout, length-
bounded. Every request carries a protocol version, a unique ``id`` and an
``op``; every response echoes the ``id``. Frames are strictly bounded --
a worker that emits an oversized line is treated as crashed, never
truncated-into-success.

Failure honesty (binding):

  * A worker that dies mid-request yields
    :class:`asea.capability_build.errors.WorkerCrashed` on the client with
    every COMPLETE validated frame it managed to emit; a trace is never
    fabricated for the lost portion.
  * A worker that cannot honestly run (not enough memory, quantized-only
    checkpoint, missing runtime) answers ``{"ok": false, "blocked":
    {"requirement": ..., "remedy": ...}}`` -- the client surfaces
    ``BLOCKED_RESOURCE``, never a weaker path.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, Optional

#: Protocol identity. Both sides must agree on this string exactly; any
#: mismatch is a hard handshake failure, never a best-effort continue.
PROTOCOL = "silt.capability_worker.v1"
PROTOCOL_VERSION = 1

#: Hard bounds: a request or response line longer than this is a crash, not
#: a truncatable frame. 64 MiB accommodates dense per-layer routing tables
#: for 288 experts x 42 sparse layers with plenty of headroom.
MAX_FRAME_BYTES = 64 * 1024 * 1024

#: Operations the worker understands. Unknown op -> typed refusal.
OPS = (
    "hello",        # handshake: versions, arch detection, hardware preflight
    "inspect",      # read-only architecture/parameter inventory
    "routing",      # router usage telemetry over a batch of prompts
    "mask",         # temporary_mask one component (intervention protocol)
    "restore",      # restore_mask one component
    "verify",       # verify_unchanged (parameter hashes)
    "shutdown",     # clean exit
)

#: Error kinds the worker may report. ``blocked`` carries the exact
#: requirement+remedy pair for a BLOCKED_RESOURCE receipt.
ERROR_KINDS = ("invalid_op", "invalid_frame", "blocked", "arch_mismatch",
               "worker_error")


#: Process-lifetime monotonic request ids. ``id(payload)`` was a memory
#: address: two requests sharing one payload object would silently reuse an
#: id (defeating the echo check between interleaved consumers), and ids
#: carried no ordering information. A counter is collision-free per process
#: and monotonic -- a response with an out-of-order id is immediately
#: visible (audit 2026-09-18).
_REQUEST_COUNTER = itertools.count()


def make_request(op: str, payload: Optional[Dict[str, Any]] = None,
                 request_id: Optional[str] = None) -> Dict[str, Any]:
    """Build one request envelope."""
    if op not in OPS:
        raise ValueError("unknown worker op: %r (known: %s)" % (op, ", ".join(OPS)))
    return {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "op": op,
        "id": request_id or ("%s-%d" % (op, next(_REQUEST_COUNTER))),
        "payload": payload or {},
    }


def make_response(request: Dict[str, Any], *, ok: bool,
                  result: Optional[Dict[str, Any]] = None,
                  error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build one response envelope echoing the request id."""
    response: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "op": request.get("op"),
        "id": request.get("id"),
        "ok": bool(ok),
    }
    if ok:
        response["result"] = result or {}
    else:
        if not isinstance(error, dict) or "kind" not in error:
            raise ValueError("error responses need a kind")
        if error.get("kind") not in ERROR_KINDS:
            raise ValueError("unknown error kind: %r" % error.get("kind"))
        response["error"] = error
    return response


def encode(frame: Dict[str, Any]) -> bytes:
    """Encode one frame to a length-checked line (no trailing content)."""
    line = json.dumps(frame, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(line) + 1 > MAX_FRAME_BYTES:
        raise ValueError("frame exceeds %d bytes" % MAX_FRAME_BYTES)
    return line + b"\n"


def decode(line: bytes) -> Dict[str, Any]:
    """Decode and structurally validate one frame line. Raises ValueError
    on anything malformed -- the caller turns that into WorkerCrashed, not
    a silent skip."""
    if len(line) > MAX_FRAME_BYTES:
        raise ValueError("frame exceeds %d bytes" % MAX_FRAME_BYTES)
    try:
        frame = json.loads(line.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("frame is not valid JSON: %s" % exc) from exc
    if not isinstance(frame, dict):
        raise ValueError("frame must be a JSON object")
    if frame.get("protocol") != PROTOCOL:
        raise ValueError("frame protocol mismatch: %r" % frame.get("protocol"))
    if frame.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("frame protocol_version mismatch: %r"
                         % frame.get("protocol_version"))
    if "id" not in frame or not isinstance(frame["id"], str):
        raise ValueError("frame missing string id")
    if "op" not in frame:
        raise ValueError("frame missing op")
    return frame


def blocked_response(request: Dict[str, Any], requirement: str,
                     remedy: str) -> Dict[str, Any]:
    """The honest refusal: exact requirement + exact remedy."""
    return make_response(
        request,
        ok=False,
        error={"kind": "blocked", "requirement": requirement, "remedy": remedy},
    )