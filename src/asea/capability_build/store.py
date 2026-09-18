"""Capability-build workspace store.

A directory layout under an operator-chosen workspace, following the
containment discipline of :class:`asea.memory.store.MemoryStore` and
:class:`asea.deepapply.store.AdapterStore`: one immutable JSON document per
artifact, physically separated directories, path-escape guards (no ``..``
tokens, no symlinks) and content-addressed duplicate refusal. The store only
RECORDS what a command produced; it never admits quality and never
activates anything.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List

from asea.artifacts import digest, safe_path

from .errors import CapabilityBuildError

#: Artifact names become real files (``<name>.json``) in the workspace, so
#: the full Windows-illegal set matters -- a ``:`` in a name would write fine
#: on Linux and then make the workspace unreadable on Windows, exactly the
#: class of latent cross-platform corruption the data/ .gitattributes fix
#: closed. DOS reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
#: are refused with or without an extension, and trailing dots/spaces are
#: stripped silently by Win32 (a write that "succeeds" under a name the
#: directory listing then shows differently).
_BAD_NAME_CHARS = re.compile(r'[:*?"<>|\x00-\x1f]')
_RESERVED_NAMES = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE)


def _require_portable_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise CapabilityBuildError("artifact name must be a non-empty string")
    if _BAD_NAME_CHARS.search(name) or name != name.strip() or name.endswith("."):
        raise CapabilityBuildError(
            "artifact name %r contains characters that are illegal or "
            "unstable in filenames on some supported platform" % name
        )
    if _RESERVED_NAMES.match(name.split(".")[0]):
        raise CapabilityBuildError(
            "artifact name %r collides with a reserved device name on Windows" % name
        )

LAYOUT = {
    "specs": "specs",
    "traces": "traces",
    "footprints": "footprints",
    "receipts": "receipts",
    "baselines": "baselines",
    "interventions": "interventions",
    "candidates": "candidates",
}


class CapabilityStore:
    """Immutable artifact store for one capability-build workspace."""

    def __init__(self, workspace: Path) -> None:
        root = Path(workspace)
        if root.exists() and not root.is_dir():
            raise CapabilityBuildError("workspace exists and is not a directory")
        root.mkdir(parents=True, exist_ok=True)
        for name in LAYOUT.values():
            (root / name).mkdir(parents=True, exist_ok=True)
        self.root = root

    # -- paths -------------------------------------------------------------

    def _safe(self, kind: str, name: str, must_exist: bool) -> Path:
        if kind not in LAYOUT:
            raise CapabilityBuildError("unknown artifact kind: %s" % kind)
        if "/" in name or "\\" in name or ".." in name or name in (".", ""):
            raise CapabilityBuildError("artifact name must be a single path segment")
        _require_portable_name(name)
        path = safe_path(self.root / LAYOUT[kind] / ("%s.json" % name))
        if must_exist and not path.is_file():
            raise CapabilityBuildError("artifact does not exist: %s" % path)
        return path

    # -- write -------------------------------------------------------------

    def put(self, kind: str, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Write one immutable JSON artifact. Refuses to overwrite (append-only
        discipline: a rewrite is a NEW artifact with a new name, e.g. suffixed
        by revision or timestamp). Refuses duplicate content under a different
        name only within the same kind when content-addressed naming is used.
        """
        path = self._safe(kind, name, must_exist=False)
        if path.exists():
            raise CapabilityBuildError("artifact already exists: %s" % path)
        record = dict(payload)
        record.setdefault("artifact_sha256", digest(record))
        text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)
        return {"path": str(path), "artifact_sha256": record["artifact_sha256"]}

    # -- read --------------------------------------------------------------

    def get(self, kind: str, name: str) -> Dict[str, Any]:
        path = self._safe(kind, name, must_exist=True)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise CapabilityBuildError("artifact is not valid JSON: %s" % exc) from exc
        if not isinstance(raw, dict):
            raise CapabilityBuildError("artifact must be a JSON object")
        stored = raw.pop("artifact_sha256", None)
        if stored is None:
            # Fail closed: every artifact written by :meth:`put` carries this
            # marker, so its absence means the file was not written by the
            # store (or the marker was stripped after write) -- an integrity
            # failure, not a legacy artifact to trust silently.
            raise CapabilityBuildError(
                "artifact has no artifact_sha256 integrity marker (foreign or "
                "tampered): %s" % path
            )
        if digest(raw) != stored:
            raise CapabilityBuildError(
                "artifact content hash mismatch (edited after write): %s" % path
            )
        return raw

    def list(self, kind: str) -> List[str]:
        directory = self.root / LAYOUT[kind]
        return sorted(p.stem for p in directory.glob("*.json"))