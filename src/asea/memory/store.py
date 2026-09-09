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
import os
import shutil
import tempfile
import threading
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional

from ..core.errors import RollbackError
from ..core.protocol import PromotionStatus, SkillPacket

CANDIDATE = "candidate"
APPROVED = "approved"
REJECTED = "rejected"
SNAPSHOTS = "snapshots"


_BUCKETS = (CANDIDATE, APPROVED, REJECTED, SNAPSHOTS)
_BACKUP = ".approved-rollback-backup"
_ROOT_LOCKS = weakref.WeakValueDictionary()
_ROOT_LOCKS_GUARD = threading.Lock()


def _root_lock(root: Path):
    """One reentrant lock per canonical root, shared by all local instances.

    Cooperative threads in ONE process only. Multiple writer processes (or
    inherited stores after fork) are unsupported: deploy a single writer process.
    This is not a filesystem lock or protection from external filesystem edits.
    """
    key = os.path.normcase(str(Path(root).resolve()))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock


def validate_storage_id(value: str) -> str:
    """Reject unsafe path/ZIP components without normalizing legacy labels.

    Spaces, Unicode, dots and punctuation such as +, @ and parentheses remain
    supported. Windows drive/ADS/device names are unsafe even on POSIX exports.
    """
    if (not isinstance(value, str) or not value or value in (".", "..")
            or ".." in value or any(c in value for c in '/\\:<>"|?*')
            or any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
            or len(value.encode("utf-8")) > 250  # leave room for .json (NAME_MAX=255)
            or value.endswith((" ", ".")) or PureWindowsPath(value).drive
            or PureWindowsPath(value).is_reserved()):
        raise ValueError("unsafe storage ID {!r}".format(value))
    return value


def _no_symlinks(path: Path) -> None:
    # Resolving first would erase evidence of symlinks, including ancestors.
    path = path.absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise RollbackError("symlink storage path refused: {}".format(part))


def _regular(path: Path) -> None:
    _no_symlinks(path)
    if not path.is_file():
        raise RollbackError("not a regular storage file: {}".format(path))


def _packet_bytes(path: Path, approved: bool = False):
    _regular(path)
    try:
        raw = path.read_bytes()
        packet = SkillPacket.model_validate_json(raw)
        validate_storage_id(packet.packet_id)
        if path.name != "{}.json".format(packet.packet_id):
            raise ValueError("packet ID does not match filename")
        if approved and packet.promotion_status != PromotionStatus.PROMOTED:
            raise ValueError("snapshot/approved packet is not promoted")
        return raw, packet
    except (ValueError, OSError) as exc:
        raise RollbackError("invalid packet {}: {}".format(path, exc)) from exc


def _atomic_write(path: Path, raw: bytes) -> None:
    """Failure must not truncate an existing packet."""
    _no_symlinks(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".packet-", delete=False) as fh:
            temporary = Path(fh.name)
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        _no_symlinks(path)
        os.replace(temporary, path)
    except OSError as exc:
        raise RollbackError("storage write failed for {}: {}".format(path, exc)) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MemoryStore:
    """Filesystem store for cooperating threads in a single writer process.

    Instances sharing a canonical root share a reentrant lock. Independent
    writer processes / inherited post-fork instances are NOT synchronized;
    run only one writer process per root. External hostile writers and power
    loss are outside this guarantee. Decision-only candidate bookkeeping is
    not a revision token; stale semantic/identity changes are refused.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        _no_symlinks(self.root)
        self._lock = _root_lock(self.root)
        # Constructors must not observe/recreate directories mid-rollback in
        # another instance. Keep the public (possibly relative) root unchanged.
        with self._lock:
            self._check_recovery()
            # Validate all existing boundaries before creating anything.
            for sub in _BUCKETS:
                path = self.root / sub
                _no_symlinks(path)
                if path.exists() and not path.is_dir():
                    raise RollbackError("not a storage directory: {}".format(path))
            for sub in _BUCKETS:
                (self.root / sub).mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------

    def _check_recovery(self) -> None:
        backup = self.root / _BACKUP
        if backup.exists() or backup.is_symlink():
            raise RollbackError(
                "interrupted rollback; original approved data retained at {}. "
                "Operator recovery required before opening this store.".format(backup)
            )

    def _dir(self, bucket: str) -> Path:
        if bucket not in _BUCKETS:
            raise RollbackError("invalid storage bucket {!r}".format(bucket))
        self._check_recovery()
        path = self.root / bucket
        _no_symlinks(path)
        if not path.is_dir():
            raise RollbackError("not a storage directory: {}".format(path))
        return path

    def _path(self, bucket: str, packet_id: str) -> Path:
        try:
            validate_storage_id(packet_id)
        except ValueError as exc:
            raise RollbackError(str(exc)) from exc
        path = self._dir(bucket) / "{}.json".format(packet_id)
        _no_symlinks(path)
        if path.exists() and not path.is_file():
            raise RollbackError("not a regular storage file: {}".format(path))
        return path

    def _files(self, bucket: str) -> List[Path]:
        paths = sorted(self._dir(bucket).glob("*.json"))
        for path in paths:
            self._path(bucket, path.stem)
            _regular(path)
        return paths

    # -- write ------------------------------------------------------------

    def put(self, packet: SkillPacket, bucket: str) -> Path:
        with self._lock:
            if bucket == SNAPSHOTS:
                raise RollbackError("packets require a packet bucket, not snapshots")
            path = self._path(bucket, packet.packet_id)
            try:
                raw = packet.model_dump_json(indent=2).encode("utf-8")
                checked = SkillPacket.model_validate_json(raw)
            except ValueError as exc:
                raise RollbackError("invalid packet: {}".format(exc)) from exc
            if bucket == APPROVED and checked.promotion_status != PromotionStatus.PROMOTED:
                raise RollbackError("refusing to write unpromoted packet to approved/")
            _atomic_write(path, raw)
            return path

    def put_candidate(self, packet: SkillPacket) -> Path:
        return self.put(packet, CANDIDATE)

    def put_rejected(self, packet: SkillPacket) -> Path:
        return self.put(packet, REJECTED)

    def _candidate_for_transition(self, packet: SkillPacket) -> Path:
        """Check the candidate under the root lock before writing/unlinking.

        A caller can hold an older packet across an acknowledged put_candidate
        by another store. Comparing IDs alone would delete that newer content.
        Gate/evaluation bookkeeping may legitimately change after candidate
        storage; compare all other fields, not just the narrow content_hash.
        No on-disk schema or optimistic-version field is added.
        """
        path = self._path(CANDIDATE, packet.packet_id)
        if path.exists():
            current = _packet_bytes(path)[1]
            decision_fields = {"promotion_status", "rejection_reason", "human_approved_by",
                               "rollback_token", "evaluator_score", "safety_score", "scores"}
            if current.model_dump(exclude=decision_fields) != packet.model_dump(exclude=decision_fields):
                raise RollbackError("stale candidate transition refused for {}".format(packet.packet_id))
        return path

    def reject(self, packet: SkillPacket) -> Path:
        """Record a rejection AND remove the packet from the candidate set.

        Symmetric to :meth:`approve`: the destination is written atomically
        before unlinking the candidate, but the plain ``put_rejected`` path does not
        -- so a packet parked in PENDING_HUMAN (written to candidate/ with
        ``promotion_status=pending_human_approval``) and then rejected by a
        human would leave its OLD candidate file in place. ``cmd_report``'s
        ``pending_human`` listing reads candidate/ and filters on that status,
        so the rejected packet would keep reappearing as "pending human" forever
        -- stale and misleading. This removes the candidate entry so a
        rejected packet exists only in rejected/, as intended.
        """
        with self._lock:
            candidate = self._candidate_for_transition(packet)
            self._path(REJECTED, packet.packet_id)
            result = self.put(packet, REJECTED)
            candidate.unlink(missing_ok=True)
            return result

    def approve(self, packet: SkillPacket) -> Path:
        """Move a packet into the approved set.

        Refuses anything not already marked PROMOTED: the gate decides, the
        store only records. Keeping the decision out of here means there is one
        place to audit promotion logic.
        """
        with self._lock:
            candidate = self._candidate_for_transition(packet)
            self._path(APPROVED, packet.packet_id)
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
            result = self.put(packet, APPROVED)
            candidate.unlink(missing_ok=True)
            return result

    # -- read -------------------------------------------------------------

    def get(self, bucket: str, packet_id: str) -> SkillPacket:
        with self._lock:
            path = self._path(bucket, packet_id)
            if not path.exists():
                raise RollbackError("packet {} not found in {}/".format(packet_id, bucket))
            return _packet_bytes(path, approved=bucket == APPROVED)[1]

    def list(self, bucket: str) -> List[SkillPacket]:
        with self._lock:
            return [_packet_bytes(p, approved=bucket == APPROVED)[1]
                    for p in self._files(bucket)]

    def count(self, bucket: str) -> int:
        with self._lock:
            return len(self._files(bucket))

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
        with self._lock:
            return {b: self.count(b) for b in (CANDIDATE, APPROVED, REJECTED)}


class RollbackLayer:
    """Validated snapshot/restore with an intact backup across directory swaps.

    Caught interruptions restore the old set. A hard process death between
    renames leaves the backup: subsequent opens/reads fail closed, never silently
    initialize an empty approved set. Recovery is explicit and operator-driven.
    The lock is process-local, not a sandbox against concurrent hostile OS
    writers or a guarantee against power loss. Keep the workspace service-owned.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def _source(self, token: str) -> Path:
        try:
            validate_storage_id(token)
        except ValueError as exc:
            raise RollbackError("rollback token escapes snapshots directory: {!r}".format(token)) from exc
        source = self.store._dir(SNAPSHOTS) / token
        _no_symlinks(source)
        if not source.is_dir():
            raise RollbackError("unknown snapshot token '{}' (not a directory)".format(token))
        return source

    def _load(self, source: Path, require_meta: bool):
        records = []
        for path in sorted(source.iterdir()):
            _regular(path)
            if path.name == "_meta.json":
                continue
            if path.suffix != ".json":
                raise RollbackError("unexpected snapshot entry: {}".format(path))
            raw, packet = _packet_bytes(path, approved=True)
            records.append((path.name, raw, packet))
        meta_path = source / "_meta.json"
        meta = None
        # Read-only diff fixtures historically omit metadata; restore never may.
        if require_meta or meta_path.exists():
            _regular(meta_path)
            try:
                meta = json.loads(meta_path.read_bytes())
                if (not isinstance(meta, dict) or meta.get("token") != source.name
                        or not isinstance(meta.get("label"), str)
                        or not isinstance(meta.get("created_at"), str)
                        or type(meta.get("packet_count")) is not int
                        or meta["packet_count"] != len(records)):
                    raise ValueError("snapshot metadata/count mismatch")
                datetime.fromisoformat(meta["created_at"])
            except (ValueError, OSError) as exc:
                raise RollbackError("invalid snapshot metadata: {}".format(meta_path)) from exc
        return meta, records

    def snapshot(self, label: str = "") -> str:
        with self.store._lock:
            if not isinstance(label, str):
                raise RollbackError("snapshot label must be a string")
            records = []
            for path in self.store._files(APPROVED):
                if path.name == "_meta.json":
                    raise RollbackError("packet ID _meta conflicts with snapshot metadata")
                records.append((path.name, _packet_bytes(path, approved=True)[0]))
            token = "{}-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
                                   uuid.uuid4().hex[:8])
            parent = self.store._dir(SNAPSHOTS)
            target = parent / token
            staging = Path(tempfile.mkdtemp(dir=self.store.root, prefix=".snapshot-stage-"))
            try:
                for name, raw in records:
                    _atomic_write(staging / name, raw)
                meta = {"token": token, "label": label,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "packet_count": len(records)}
                _atomic_write(staging / "_meta.json", json.dumps(meta, indent=2).encode("utf-8"))
                _no_symlinks(target)
                os.replace(staging, target)
            except OSError as exc:
                raise RollbackError("snapshot write failed: {}".format(exc)) from exc
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            return token

    def list_snapshots(self) -> List[Dict[str, Any]]:
        with self.store._lock:
            out = []
            for path in sorted(self.store._dir(SNAPSHOTS).iterdir()):
                _no_symlinks(path)
                if path.is_dir() and (path / "_meta.json").exists():
                    out.append(self._load(self._source(path.name), require_meta=True)[0])
            return out

    def snapshot_packets(self, token: str) -> List[SkillPacket]:
        """Read legacy metadata-free diff fixtures; never use them for restore."""
        from ..core.errors import SnapshotNotFoundError

        with self.store._lock:
            try:
                source = self._source(token)
            except RollbackError as exc:
                raise SnapshotNotFoundError(token) from exc
            return [record[2] for record in self._load(source, require_meta=False)[1]]

    def rollback(self, token: str) -> Dict[str, Any]:
        with self.store._lock:
            source = self._source(token)
            _, records = self._load(source, require_meta=True)
            approved = self.store._dir(APPROVED)
            removed = len(self.store._files(APPROVED))
            # Validate ALL live entries before staging; retain unrelated regular
            # files (the original restore only replaced packet JSON).
            extras = []
            for path in approved.iterdir():
                _regular(path)
                if path.suffix != ".json":
                    extras.append((path.name, path.read_bytes()))
            backup = self.store.root / _BACKUP
            staging = Path(tempfile.mkdtemp(dir=self.store.root, prefix=".rollback-stage-"))
            try:
                for name, raw, _ in records:
                    _atomic_write(staging / name, raw)
                for name, raw in extras:
                    _atomic_write(staging / name, raw)
                # Same-filesystem renames. Never unlink live packets.
                os.replace(approved, backup)
                os.replace(staging, approved)
            except BaseException as exc:
                # Includes KeyboardInterrupt/SystemExit. Failed recovery retains
                # the backup and _check_recovery then blocks all further access.
                if backup.exists():
                    try:
                        if approved.exists():
                            os.rename(approved, staging)
                        os.rename(backup, approved)
                    except OSError as recovery_exc:
                        raise RollbackError("rollback recovery required; original data at {}".format(
                            backup)) from recovery_exc
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise RollbackError("rollback failed; approved data unchanged: {}".format(exc)) from exc
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                raise RollbackError("rollback installed; backup cleanup required: {}".format(backup)) from exc
            return {"token": token, "removed": removed, "restored": len(records)}
