"""Durable failed-attempt ledger (extraction-brief C7).

Every failed intervention, trace, extraction, standalone load or build
writes an IMMUTABLE attempt record here before the failure surfaces to
the operator. "Immutable" means: the ledger is append-only and
hash-chained -- each record embeds the sha256 of the record before it,
so any edit or deletion of history is detectable by
:func:`verify_chain`. A failed experiment stays visible forever; the
receipt's ``failure_history`` is built from this ledger.

Required record fields (binding, all nine): source fingerprint,
request, component ids, case hashes, error classification, timeout
information, restoration status, stderr tail, generated artifacts with
their artifact hashes.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from asea.artifacts import digest, safe_path

LEDGER_SCHEMA = "silt.extraction.failed_attempt.v1"

#: Restoration vocabulary: a failed attempt leaves the source model in
#: exactly one of these states, and UNKNOWN is never silently upgraded.
RESTORATION_STATES = (
    "RESTORED_AND_VERIFIED",
    "RESTORED_UNVERIFIED",
    "NO_MUTATION_OCCURRED",
    "RESTORATION_FAILED",
    "NOT_APPLICABLE",
    "UNKNOWN",
)

#: Error classification vocabulary used by the ledger.
ERROR_CLASSES = (
    "worker_crashed",
    "timeout",
    "out_of_memory",
    "invalid_frame",
    "arch_mismatch",
    "io_error",
    "schema_rejected",
    "extraction_failure",
    "build_failure",
    "unclassified",
)


class FailedAttemptLedger:
    """Append-only hash-chained record of failed attempts (C7)."""

    def __init__(self, path):
        self.path = safe_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------

    def record(
        self,
        *,
        stage: str,
        source_fingerprint: str,
        request: Dict[str, Any],
        component_ids: Optional[List[str]] = None,
        case_hashes: Optional[List[str]] = None,
        error_class: str = "unclassified",
        timeout_info: Optional[Dict[str, Any]] = None,
        restoration_status: str = "NOT_APPLICABLE",
        stderr_tail: str = "",
        generated_artifacts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Append one immutable failed-attempt record and return it.

        ``generated_artifacts`` maps artifact name -> sha256 (the
        artifact hash); the ledger never stores artifact bytes.
        """
        if error_class not in ERROR_CLASSES:
            raise ValueError(
                "error_class %r is not in the ledger vocabulary %r"
                % (error_class, ERROR_CLASSES))
        if restoration_status not in RESTORATION_STATES:
            raise ValueError(
                "restoration_status %r is not in the ledger vocabulary %r"
                % (restoration_status, RESTORATION_STATES))
        previous = self.tail()
        entry = {
            "schema": LEDGER_SCHEMA,
            "attempt_id": "%d" % ((previous or {}).get("sequence", -1) + 1),
            "recorded_unix": time.time(),
            "stage": stage,
            "source_fingerprint": source_fingerprint,
            "request": request,
            "component_ids": list(component_ids or []),
            "case_hashes": list(case_hashes or []),
            "error_class": error_class,
            "timeout_info": timeout_info or {"timeout": None,
                                             "deadline_hit": False},
            "restoration_status": restoration_status,
            "stderr_tail": stderr_tail[-4000:],
            "generated_artifacts": dict(generated_artifacts or {}),
            "previous_sha256": (previous or {}).get("record_sha256",
                                                    "0" * 64),
        }
        entry["record_sha256"] = digest(entry)
        with open(str(self.path), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        try:
            fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        return entry

    # -- reading ---------------------------------------------------------

    def entries(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        if not self.path.exists():
            return result
        with open(str(self.path), "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result

    def tail(self) -> Optional[Dict[str, Any]]:
        entries = self.entries()
        return entries[-1] if entries else None

    def failure_history(self) -> List[str]:
        """Receipt-ready one-line summaries; every failed attempt stays
        visible."""
        return [
            "%s: stage=%s error=%s restoration=%s"
            % (entry["attempt_id"], entry["stage"], entry["error_class"],
               entry["restoration_status"])
            for entry in self.entries()
        ]

    def verify_chain(self) -> bool:
        """True when every record's sha256 recomputes and the chain
        links; False the moment history was edited."""
        previous = "0" * 64
        for entry in self.entries():
            expected = entry.get("record_sha256")
            body = {key: value for key, value in entry.items()
                    if key != "record_sha256"}
            if digest(body) != expected:
                return False
            if entry.get("previous_sha256") != previous:
                return False
            previous = expected or previous
        return True