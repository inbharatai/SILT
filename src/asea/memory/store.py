"""Memory store.

Candidate and approved learning data live in **physically separate
directories**. This is not tidiness, it is the core containment property: a
receiver reads only from ``approved/``, so an un-evaluated or rejected packet
cannot influence behaviour even through a bug in the pipeline, because it is
simply not in the directory the receiver reads.

Layout::

    <root>/
      candidate/   packets extracted or distilled, not yet promoted
      approved/    promoted packets; the ONLY source the receiver reads
      rejected/    packets refused, kept for audit and pattern analysis
      snapshots/   rollback snapshots of the approved set
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.errors import RollbackError
from ..core.protocol import PromotionStatus, SkillPacket

CANDIDATE = "candidate"
APPROVED = "approved"
REJECTED = "rejected"
SNAPSHOTS = "snapshots"


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for sub in (CANDIDATE, APPROVED, REJECTED, SNAPSHOTS):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        # Serialises the approve() read-check-write critical section so two
        # concurrent approvals of identical-content packets cannot both pass the
        # duplicate-content guard (adversarial audit: TOCTOU race on the A2
        # guard). Single-process lock; matches the default single-worker Studio
        # deployment.
        self._lock = threading.Lock()

    # -- paths ------------------------------------------------------------

    def _dir(self, bucket: str) -> Path:
        return self.root / bucket

    def _path(self, bucket: str, packet_id: str) -> Path:
        return self._dir(bucket) / "{}.json".format(packet_id)

    # -- write ------------------------------------------------------------

    def put(self, packet: SkillPacket, bucket: str) -> Path:
        path = self._path(bucket, packet.packet_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                json.loads(packet.model_dump_json()), fh, ensure_ascii=False, indent=2
            )
        return path

    def put_candidate(self, packet: SkillPacket) -> Path:
        return self.put(packet, CANDIDATE)

    def put_rejected(self, packet: SkillPacket) -> Path:
        return self.put(packet, REJECTED)

    def reject(self, packet: SkillPacket) -> Path:
        """Record a rejection AND remove the packet from the candidate set.

        Symmetric to :meth:`approve`: ``approve`` unlinks the candidate file
        before writing approved/, but the plain ``put_rejected`` path does not
        -- so a packet parked in PENDING_HUMAN (written to candidate/ with
        ``promotion_status=pending_human_approval``) and then rejected by a
        human would leave its OLD candidate file in place. ``cmd_report``'s
        ``pending_human`` listing reads candidate/ and filters on that status,
        so the rejected packet would keep reappearing as "pending human" forever
        -- stale and misleading. This removes the candidate entry so a
        rejected packet exists only in rejected/, as intended.
        """
        self._path(CANDIDATE, packet.packet_id).unlink(missing_ok=True)
        return self.put(packet, REJECTED)

    def approve(self, packet: SkillPacket) -> Path:
        """Move a packet into the approved set.

        Refuses anything not already marked PROMOTED: the gate decides, the
        store only records. Keeping the decision out of here means there is one
        place to audit promotion logic.
        """
        with self._lock:
            if packet.promotion_status != PromotionStatus.PROMOTED:
                raise RollbackError(
                    "refusing to write packet {} to approved/ with status '{}'".format(
                        packet.packet_id, packet.promotion_status.value
                    )
                )
            # Duplicate-content guard (adversarial audit, finding A2): two packets
            # with different ids but identical distilled content for the same
            # receiver must not both enter the approved set -- duplicates inflate
            # retrieval and double-count in L4/L5 exports. Same content for a
            # DIFFERENT receiver is legitimate.
            incoming_hash = packet.content_hash()
            for existing in self.list(APPROVED):
                if (
                    existing.target_module == packet.target_module
                    and existing.content_hash() == incoming_hash
                ):
                    raise RollbackError(
                        "refusing to approve packet {}: identical content already "
                        "approved for '{}' as packet {}".format(
                            packet.packet_id, packet.target_module, existing.packet_id
                        )
                    )
            self._path(CANDIDATE, packet.packet_id).unlink(missing_ok=True)
            return self.put(packet, APPROVED)

    # -- read -------------------------------------------------------------

    def get(self, bucket: str, packet_id: str) -> SkillPacket:
        path = self._path(bucket, packet_id)
        if not path.exists():
            raise RollbackError(
                "packet {} not found in {}/".format(packet_id, bucket)
            )
        with open(path, "r", encoding="utf-8") as fh:
            return SkillPacket.model_validate(json.load(fh))

    def list(self, bucket: str) -> List[SkillPacket]:
        out = []
        for path in sorted(self._dir(bucket).glob("*.json")):
            with open(path, "r", encoding="utf-8") as fh:
                out.append(SkillPacket.model_validate(json.load(fh)))
        return out

    def count(self, bucket: str) -> int:
        return len(list(self._dir(bucket).glob("*.json")))

    def approved_skills(
        self, target_module: str, capability: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Redacted payloads a receiver may consume. The only read path for a module."""
        out = []
        for packet in self.list(APPROVED):
            if packet.target_module != target_module:
                continue
            if capability and packet.sender_capability.as_str() != capability:
                continue
            out.append(packet.redacted_for_receiver())
        return out

    # -- statistics -------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {b: self.count(b) for b in (CANDIDATE, APPROVED, REJECTED)}


class RollbackLayer:
    """Snapshot/restore for the approved set.

    Every promotion takes a snapshot first and stamps the packet with the
    resulting token, so any promotion can be undone without reasoning about
    what it changed. Cheap because the approved set is small JSON.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def snapshot(self, label: str = "") -> str:
        token = "{}-{}".format(
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
        )
        target = self.store._dir(SNAPSHOTS) / token
        target.mkdir(parents=True, exist_ok=False)
        approved_dir = self.store._dir(APPROVED)
        for path in approved_dir.glob("*.json"):
            shutil.copy2(path, target / path.name)
        meta = {
            "token": token,
            "label": label,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "packet_count": len(list(approved_dir.glob("*.json"))),
        }
        with open(target / "_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return token

    def list_snapshots(self) -> List[Dict[str, Any]]:
        out = []
        for path in sorted(self.store._dir(SNAPSHOTS).iterdir()):
            meta_path = path / "_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as fh:
                    out.append(json.load(fh))
        return out

    def snapshot_packets(self, token: str) -> List[SkillPacket]:
        """Load the approved packets captured by a snapshot (B1a Capability
        Diff, audit 2026-08-17).

        A snapshot is a point-in-time copy of the approved set (see
        :meth:`snapshot`). The diff reads two snapshots and compares the
        receiver's capability under each approved set, so it needs the packets
        the snapshot holds -- NOT the live approved/ directory (which may have
        moved on since the snapshot was taken). The same path-escape guard as
        :meth:`rollback` applies: a token containing ``..`` that resolves
        outside ``snapshots/`` is rejected, not silently followed. An unknown
        token raises :class:`~asea.core.errors.SnapshotNotFoundError`; an EMPTY
        snapshot returns ``[]`` (a legitimate, honest delta of zero -- not an
        error, and not silently padded).
        """
        from ..core.errors import SnapshotNotFoundError

        snapshots_dir = self.store._dir(SNAPSHOTS).resolve()
        source = (snapshots_dir / token).resolve()
        if not token or source == snapshots_dir or not source.is_relative_to(snapshots_dir):
            raise SnapshotNotFoundError(token)
        if not source.exists():
            raise SnapshotNotFoundError(token)
        # A token that resolves to a FILE (not a directory) inside snapshots/ is
        # neither the typed "missing" nor the honest "empty" case -- globbing a
        # file silently yields no packets (fabricating a "zero delta") or raises
        # an untyped NotADirectoryError. Treat it as a missing snapshot so the
        # failure stays typed (adversarial audit 2026-08-17, F1).
        if not source.is_dir():
            raise SnapshotNotFoundError(token)
        out: List[SkillPacket] = []
        for path in sorted(source.glob("*.json")):
            if path.name == "_meta.json":
                continue
            with open(path, "r", encoding="utf-8") as fh:
                out.append(SkillPacket.model_validate(json.load(fh)))
        return out

    def rollback(self, token: str) -> Dict[str, Any]:
        # Confine the token inside snapshots/ (adversarial audit, CRITICAL):
        # a token containing '..' resolves outside snapshots/ -- e.g. '..' or
        # '../candidate' -- and the old `if not source.exists()` guard did NOT
        # fire (those paths exist), so rollback() deleted every file in
        # approved/ and copied un-gated candidate/rejected packets (or anything
        # else it found) straight into the only directory the receiver reads,
        # bypassing the gate entirely. Reject any token that escapes.
        snapshots_dir = self.store._dir(SNAPSHOTS).resolve()
        source = (snapshots_dir / token).resolve()
        if not token or source == snapshots_dir or not source.is_relative_to(snapshots_dir):
            raise RollbackError("rollback token escapes snapshots directory: '{}'".format(token))
        if not source.exists():
            raise RollbackError("unknown snapshot token '{}'".format(token))

        approved_dir = self.store._dir(APPROVED)
        removed = 0
        for path in list(approved_dir.glob("*.json")):
            path.unlink()
            removed += 1
        restored = 0
        for path in source.glob("*.json"):
            if path.name == "_meta.json":
                continue
            shutil.copy2(path, approved_dir / path.name)
            restored += 1
        return {"token": token, "removed": removed, "restored": restored}
