"""Adapter store -- separated candidate/approved/rejected, with rollback.

Mirrors the discipline of :class:`asea.memory.store.MemoryStore`:

* Candidate, approved and rejected adapters live in **physically separate
  directories**. A receiver reads only from ``approved_adapters/``, so an
  un-gated or rejected adapter cannot influence behaviour even through a bug --
  it is simply not in the directory the receiver reads.
* ``approve()`` refuses anything not ``PROMOTED``: Gate 2 decides, the store
  only records.
* Duplicate-content guard: two adapters with identical ``content_hash`` for the
  same base model must not both enter the approved set.
* Snapshot before every admission; rollback restores the prior approved-adapter
  set, so the receiver returns to its exact pre-admission behaviour. Adapters
  are removable by construction -- v1 never merges into base weights.

The rollback token path-escape guard is the same as
``RollbackLayer.rollback`` (adversarial audit a11): a token containing ``..``
resolves outside ``snapshots/`` and must be rejected, not followed.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.errors import RollbackError
from ..core.protocol import PromotionStatus
from .adapter_packet import AdapterPacket
from .errors import AdapterNotPromoted

CANDIDATE = "candidate_adapters"
APPROVED = "approved_adapters"
REJECTED = "rejected_adapters"
SNAPSHOTS = "snapshots"


class AdapterStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for sub in (CANDIDATE, APPROVED, REJECTED, SNAPSHOTS):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        # Serialises approve()'s read-check-write so two concurrent admissions
        # of identical-content adapters cannot both pass the duplicate guard.
        self._lock = threading.Lock()

    # -- paths ------------------------------------------------------------

    def _dir(self, bucket: str) -> Path:
        return self.root / bucket

    def _path(self, bucket: str, adapter_id: str) -> Path:
        return self._dir(bucket) / "{}.json".format(adapter_id)

    # -- write ------------------------------------------------------------

    def put(self, adapter: AdapterPacket, bucket: str) -> Path:
        path = self._path(bucket, adapter.adapter_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                json.loads(adapter.model_dump_json()), fh, ensure_ascii=False, indent=2
            )
        return path

    def put_candidate(self, adapter: AdapterPacket) -> Path:
        return self.put(adapter, CANDIDATE)

    def put_rejected(self, adapter: AdapterPacket) -> Path:
        return self.put(adapter, REJECTED)

    def approve(self, adapter: AdapterPacket) -> Path:
        """Move an adapter into the approved set.

        Refuses anything not PROMOTED: Gate 2 decides, the store only records.
        """
        with self._lock:
            if adapter.promotion_status != PromotionStatus.PROMOTED:
                raise AdapterNotPromoted(
                    "refusing to write adapter {} to approved_adapters/ with status '{}'".format(
                        adapter.adapter_id, adapter.promotion_status.value
                    )
                )
            incoming_hash = adapter.content_hash()
            for existing in self.list(APPROVED):
                if (
                    existing.base_model == adapter.base_model
                    and existing.content_hash() == incoming_hash
                ):
                    raise RollbackError(
                        "refusing to approve adapter {}: identical content already "
                        "approved for '{}' as adapter {}".format(
                            adapter.adapter_id, adapter.base_model, existing.adapter_id
                        )
                    )
            self._path(CANDIDATE, adapter.adapter_id).unlink(missing_ok=True)
            return self.put(adapter, APPROVED)

    # -- read -------------------------------------------------------------

    def get(self, bucket: str, adapter_id: str) -> AdapterPacket:
        path = self._path(bucket, adapter_id)
        if not path.exists():
            raise RollbackError(
                "adapter {} not found in {}/".format(adapter_id, bucket)
            )
        with open(path, "r", encoding="utf-8") as fh:
            return AdapterPacket.model_validate(json.load(fh))

    def list(self, bucket: str) -> List[AdapterPacket]:
        out: List[AdapterPacket] = []
        for path in sorted(self._dir(bucket).glob("*.json")):
            with open(path, "r", encoding="utf-8") as fh:
                out.append(AdapterPacket.model_validate(json.load(fh)))
        return out

    def count(self, bucket: str) -> int:
        return len(list(self._dir(bucket).glob("*.json")))

    def stats(self) -> Dict[str, int]:
        return {b: self.count(b) for b in (CANDIDATE, APPROVED, REJECTED)}


class AdapterRollbackLayer:
    """Snapshot/restore for the approved-adapter set.

    Every admission takes a snapshot first and stamps the adapter with the
    resulting token, so any admission can be undone. Rolling back detaches the
    adapter (restores the prior approved-adapter set) -- the receiver returns to
    its exact pre-admission behaviour because adapters are removable by
    construction and are never merged into base weights in v1.
    """

    def __init__(self, store: AdapterStore) -> None:
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
            "adapter_count": len(list(approved_dir.glob("*.json"))),
        }
        with open(target / "_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return token

    def list_snapshots(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for path in sorted(self.store._dir(SNAPSHOTS).iterdir()):
            meta_path = path / "_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as fh:
                    out.append(json.load(fh))
        return out

    def rollback(self, token: str) -> Dict[str, Any]:
        # Path-escape guard (adversarial audit a11, identical to RollbackLayer):
        # a token containing '..' resolves outside snapshots/ and must be
        # rejected, not followed.
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