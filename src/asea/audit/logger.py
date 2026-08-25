"""Append-only, hash-chained audit log.

Each entry stores the SHA-256 of (previous entry hash + this entry's payload).
Altering or deleting any historical line breaks the chain, and
:meth:`AuditLog.verify` reports the first index where it breaks.

This is tamper-*evidence*, not tamper-proofing: anyone who can write the file
can recompute the whole chain. Real deployments should ship entries to
append-only storage. Saying otherwise would overstate what a local JSONL file
can guarantee.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.errors import AuditIntegrityError

GENESIS = "0" * 64


def _hash(previous: str, payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256((previous + blob).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        # Serialises append()'s read-then-write critical section: append reads
        # last_hash() + count() (the whole file) and then writes a new chained
        # entry. Without a lock, two concurrent appends (FastAPI threadpool +
        # per-job worker threads in the Studio) can both observe the same
        # prev_hash/index and both write -- colliding indices and a broken hash
        # chain, defeating the tamper-evidence this module exists to provide.
        # Single-process lock; matches the default single-worker deployment.
        self._lock = threading.Lock()

    # -- write ------------------------------------------------------------

    def append(
        self,
        event: str,
        actor: str,
        detail: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        packet_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            previous = self.last_hash()
            payload = {
                "index": self.count(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "actor": actor,
                "session_id": session_id,
                "packet_id": packet_id,
                "detail": detail or {},
            }
            entry = dict(payload)
            entry["prev_hash"] = previous
            entry["hash"] = _hash(previous, payload)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return entry

    # -- read -------------------------------------------------------------

    def entries(self) -> List[Dict[str, Any]]:
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def count(self) -> int:
        return len(self.entries())

    def last_hash(self) -> str:
        entries = self.entries()
        return entries[-1]["hash"] if entries else GENESIS

    def for_packet(self, packet_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries() if e.get("packet_id") == packet_id]

    def for_session(self, session_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries() if e.get("session_id") == session_id]

    # -- integrity --------------------------------------------------------

    def verify(self) -> Dict[str, Any]:
        previous = GENESIS
        tracked = ("index", "timestamp", "event", "actor", "session_id",
                   "packet_id", "detail")
        for i, entry in enumerate(self.entries()):
            # A tamper that deletes a tracked field, or a torn write that
            # truncates one, must surface as a structured {ok: False, broken_at,
            # reason} -- the contract in this module's docstring -- not a raw
            # KeyError that crashes assert_intact() and its callers (adversarial
            # audit 2026-08-13: the hard subscript entry[k] defeated that
            # contract). prev_hash/hash already use .get() below; the payload
            # fields now do too.
            for k in tracked:
                if k not in entry:
                    return {"ok": False, "broken_at": i,
                            "reason": "missing field: " + k}
            payload = {k: entry[k] for k in tracked}
            expected = _hash(previous, payload)
            if entry.get("prev_hash") != previous:
                return {"ok": False, "broken_at": i, "reason": "prev_hash mismatch"}
            if entry.get("hash") != expected:
                return {"ok": False, "broken_at": i, "reason": "payload hash mismatch"}
            previous = entry["hash"]
        return {"ok": True, "entries": self.count()}

    def assert_intact(self) -> None:
        result = self.verify()
        if not result["ok"]:
            raise AuditIntegrityError(
                "audit chain broken at entry {}: {}".format(
                    result["broken_at"], result["reason"]
                )
            )
