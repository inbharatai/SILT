"""Distribution-owned audit support; source checkouts are anchored to this file.

Builds copy exact repository bytes, never hand-maintained duplicates. Installed
resources are required and integrity checked; CWD and external repos are ignored.
Frozen records continue to verify their original absolute paths, without remapping.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

AUDIT_ORIGINS = (
    "README.md",
    "docs/SPECIALIST_WORKFLOW.md",
    "scripts/run_specialist_quality_experiment.py",
    "tests/test_specialist_workflow.py",
    "tests/test_specialist_quality_experiment_runner.py",
    "tests/test_hardware_dispatch.py",
    "tests/test_controller_integrity_review.py",
)
PACKAGE_ROOT = Path(__file__).resolve().parent


def source_root():
    """Only the src/asea tree containing this executed module is a checkout."""
    root = PACKAGE_ROOT.parent.parent
    if PACKAGE_ROOT.parent.name == "src" and (root / "pyproject.toml").is_file():
        return root
    return None


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or nonregular package audit resource: " + str(path))
    return path


def resource_manifest():
    path = PACKAGE_ROOT / "_audit_support" / "manifest.json"
    if path.parent.resolve() != path.parent.absolute():
        raise ValueError("redirected package audit resource manifest: " + str(path))
    _regular(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema_version") != 1 or
            set(value.get("files", {})) != set(AUDIT_ORIGINS)):
        raise ValueError("invalid package audit resource manifest: " + str(path))
    return path, value


def resource_path(origin):
    """Return a persistent absolute file path, fail closed on absent/tampered bytes."""
    if origin not in AUDIT_ORIGINS:
        raise ValueError("unknown package audit origin: " + str(origin))
    root = source_root()
    if root is not None:
        return _regular(root / origin)
    _, manifest = resource_manifest()
    base = PACKAGE_ROOT / "_audit_support"
    path = base / origin
    # Refuse directory symlinks too: a package must own its evidence, not redirect it.
    if path.resolve().parent != (base / Path(origin).parent).absolute():
        raise ValueError("redirected package audit resource: " + str(path))
    raw = _regular(path).read_bytes()
    expected = manifest["files"][origin]
    if expected != {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}:
        raise ValueError("package audit resource digest mismatch: " + str(path))
    return path


def audit_paths():
    """All support files plus the installed provenance manifest, when present."""
    paths = [resource_path(origin) for origin in AUDIT_ORIGINS]
    if source_root() is None:
        paths.append(resource_manifest()[0])
    return paths
