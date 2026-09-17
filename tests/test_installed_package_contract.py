"""Source and installed-wheel contract tests. No ML or network dependency."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from asea import _package_resources as resources
from asea.artifacts import file_hash
from asea.specialist.workflow import _check_unchanged, implementation_manifest


def test_runtime_manifest_hashes_actual_files():
    manifest = implementation_manifest()
    paths = manifest["source_files"]
    assert len(paths) >= 29
    for name, fingerprint in paths.items():
        assert Path(name).is_absolute()
        assert file_hash(name) == fingerprint
    for path in resources.audit_paths():
        assert str(path) in paths
    assert str(Path(resources.__file__).resolve()) in paths
    assert manifest["command_exit_is_certificate"] is False
    assert manifest["device_contract"] == "reconstruction_cpu; other_stages_frozen_cpu_or_cuda_index"
    _check_unchanged(paths)


def test_known_root_not_cwd(tmp_path, monkeypatch):
    expected = resources.resource_path("README.md")
    (tmp_path / "README.md").write_text("untrusted working directory")
    monkeypatch.chdir(tmp_path)
    assert resources.resource_path("README.md") == expected


@pytest.fixture
def installed(tmp_path, monkeypatch):
    target = tmp_path / "site-packages/asea"
    base = target / "_audit_support"
    files = {}
    for origin in resources.AUDIT_ORIGINS:
        raw = resources.resource_path(origin).read_bytes()
        dest = base / origin
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        files[origin] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    (base / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}))
    monkeypatch.setattr(resources, "PACKAGE_ROOT", target)
    return base


@pytest.mark.parametrize("origin", resources.AUDIT_ORIGINS)
def test_installed_missing_fails_closed(installed, origin):
    (installed / origin).unlink()
    with pytest.raises(ValueError, match="missing or nonregular"):
        resources.audit_paths()


@pytest.mark.parametrize("origin", resources.AUDIT_ORIGINS)
def test_installed_tamper_fails_closed(installed, origin):
    path = installed / origin
    path.write_bytes(path.read_bytes() + b"\ntampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        resources.audit_paths()


def test_missing_installed_manifest_never_falls_back(installed, tmp_path, monkeypatch):
    (installed / "manifest.json").unlink()
    (tmp_path / "README.md").write_text("not distribution owned")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="missing or nonregular"):
        resources.resource_path("README.md")


def test_manifest_cannot_drop_audit_inputs(installed):
    path = installed / "manifest.json"
    value = json.loads(path.read_text())
    value["files"].pop("tests/test_hardware_dispatch.py")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid package audit"):
        resources.audit_paths()


def test_no_symlinked_support_directory(installed, tmp_path):
    external = tmp_path / "external-docs"
    shutil.move(str(installed / "docs"), external)
    (installed / "docs").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="redirected"):
        resources.audit_paths()


def test_old_frozen_records_are_not_remapped(installed, tmp_path):
    original = tmp_path / "old-checkout/docs/SPECIALIST_WORKFLOW.md"
    original.parent.mkdir(parents=True)
    original.write_bytes((installed / "docs/SPECIALIST_WORKFLOW.md").read_bytes())
    frozen = {str(original): file_hash(original)}
    _check_unchanged(frozen)
    original.unlink()
    # An identical installed replacement does NOT rescue the original lock.
    with pytest.raises(Exception):
        _check_unchanged(frozen)


def test_source_missing_resource_never_uses_packaged_copy(installed, monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(resources, "source_root", lambda: root)
    with pytest.raises(ValueError, match="missing or nonregular"):
        resources.resource_path("README.md")


def test_no_unknown_origin_or_traversal():
    with pytest.raises(ValueError, match="unknown package audit origin"):
        resources.resource_path("../../README.md")


def test_static_assets_are_readable():
    import asea.studio
    static = Path(asea.studio.__file__).resolve().parent / "static"
    for name in ("index.html", "experimental.html", "logo.svg", "favicon.svg"):
        assert (static / name).is_file()
        assert (static / name).read_bytes()
        assert (static / name).stat().st_mode & 0o444
