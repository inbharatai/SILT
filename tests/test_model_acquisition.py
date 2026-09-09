"""Offline acquisition/CLI audits: tiny inert fixtures, never real model loading.

Every HTTP request is intercepted, including metadata. Hashes use the public HF
metadata shape (complete size plus LFS SHA-256 or Git blob SHA-1). These tests
establish transport integrity only, not model capability or license approval.
"""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from asea.artifacts import Blocked
from asea.artifacts import __main__ as acquisition
from asea.artifacts import bundle


REPO = "public-fixtures/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
FILES = {
    "config.json": b'{"model_type":"fixture-only"}',
    "README.md": b"# Offline transport fixture\nNo model capability claims.\n",
    "tokenizer.json": b'{"version":"1.0"}',
    "model.safetensors": b"inert-weight-fixture-not-an-executable-model",
}
_MISSING = object()


def git_digest(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def metadata_for(files):
    siblings = []
    for name, data in files.items():
        entry = {"rfilename": name, "size": len(data)}
        if name.endswith(".safetensors"):
            entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            # For LFS this identifies a pointer, not the downloaded large asset.
            entry["blobId"] = "1" * 40
        else:
            entry["blobId"] = git_digest(data)
        siblings.append(entry)
    return {"sha": REVISION, "siblings": siblings, "cardData": {"license": "apache-2.0"}}


class ChunkedResponse(io.BytesIO):
    """Force multiple bounded reads even for tiny fixture files."""
    def read(self, size=-1):
        assert 0 < size <= 1024 * 1024
        return super().read(min(size, 7))


@pytest.fixture
def remote(monkeypatch):
    fixture = SimpleNamespace(files=dict(FILES), metadata=metadata_for(FILES),
                              calls=[], responses=[], before_asset=None, fail_asset=None)

    def urlopen(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        fixture.calls.append(url)
        if url == "https://huggingface.co/api/models/" + REPO + "/revision/" + REVISION + "?blobs=true":
            assert timeout == 60
            response = io.BytesIO(json.dumps(fixture.metadata).encode())
        else:
            prefix = "https://huggingface.co/" + REPO + "/resolve/" + REVISION + "/"
            assert url.startswith(prefix), "unexpected network endpoint: " + url
            assert timeout == 90
            assert request.get_header("Authorization") is None
            name = unquote(url[len(prefix):])
            assert name in fixture.files, "unexpected asset fetched: " + name
            if fixture.before_asset:
                fixture.before_asset(name)
            if fixture.fail_asset == name:
                raise OSError("fixture stream unavailable")
            response = ChunkedResponse(fixture.files[name])
        fixture.responses.append(response)
        return response

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    return fixture


def entry_for(remote, name):
    return next(item for item in remote.metadata["siblings"] if item["rfilename"] == name)


def assert_clean(tmp_path, remote):
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".silt-model-*"))
    assert all(response.closed for response in remote.responses)


def fetch(tmp_path, max_bytes=4096):
    return acquisition.download(REPO, REVISION, tmp_path / "output", max_bytes)


def test_known_metadata_receipt_readme_identity_and_every_digest(tmp_path, remote, capsys):
    result = fetch(tmp_path)
    output = tmp_path / "output"
    receipt = json.loads((output / "acquisition.json").read_text())
    assert result == {"output": str(output), "bytes": sum(map(len, FILES.values())), **receipt}
    assert receipt["repo"] == REPO and receipt["revision"] == REVISION
    assert receipt["license"] == "apache-2.0"
    assert receipt["schema_version"] == 1
    assert receipt["scope"] == "downloaded source identity/integrity only; no inference or capability certification"
    assert set(receipt["files"]) == set(FILES)
    assert {path.name for path in output.iterdir()} == set(FILES) | {"acquisition.json"}
    for name, data in FILES.items():
        assert (output / name).read_bytes() == data
        asset = receipt["files"][name]
        assert asset["sha256"] == hashlib.sha256(data).hexdigest()
        assert asset["bytes"] == len(data)
        assert asset["remote_lfs_verified"] is name.endswith(".safetensors")
        assert asset["remote_git_blob_verified"] is (not name.endswith(".safetensors"))
    assert receipt["files"]["README.md"]["sha256"] == hashlib.sha256(FILES["README.md"]).hexdigest()
    assert len(remote.calls) == 1 + len(FILES)
    assert all(response.closed for response in remote.responses)
    assert not list(tmp_path.glob(".silt-model-*"))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("card", [_MISSING, None, {}, {"license": None}])
def test_unknown_optional_license_has_no_commercial_permission(tmp_path, remote, card):
    if card is _MISSING:
        remote.metadata.pop("cardData")
    else:
        remote.metadata["cardData"] = card
    result = fetch(tmp_path)
    assert result["license"] is None
    receipt = json.loads((tmp_path / "output" / "acquisition.json").read_text())
    assert receipt["license"] is None
    assert "commercial" not in json.dumps(receipt).lower()
    assert "no inference or capability certification" in receipt["scope"]


def test_non_lfs_weights_with_matching_git_blob_are_permitted(tmp_path, remote):
    entry = entry_for(remote, "model.safetensors")
    entry.pop("lfs")
    entry["blobId"] = git_digest(FILES["model.safetensors"])
    result = fetch(tmp_path)
    assert result["files"]["model.safetensors"]["remote_git_blob_verified"] is True


@pytest.mark.parametrize("name", ["config.json", "README.md", "model.safetensors"])
def test_mismatched_git_blob_id_is_blocked_and_cleaned(tmp_path, remote, name):
    entry = entry_for(remote, name)
    entry.pop("lfs", None)
    entry["blobId"] = "0" * 40
    with pytest.raises(Blocked, match="Git-blob digest"):
        fetch(tmp_path)
    assert_clean(tmp_path, remote)


@pytest.mark.parametrize("blob_id", [_MISSING, None, "", "x" * 40, "0" * 39, 123])
def test_absent_or_invalid_all_remote_hashes_are_blocked(tmp_path, remote, blob_id):
    entry = entry_for(remote, "model.safetensors")
    entry.pop("lfs")
    entry.pop("blobId")
    if blob_id is not _MISSING:
        entry["blobId"] = blob_id
    with pytest.raises(Blocked, match="Git-blob digest"):
        fetch(tmp_path)
    assert_clean(tmp_path, remote)


def test_bad_lfs_digest_cannot_fall_back_to_matching_git_digest(tmp_path, remote):
    entry = entry_for(remote, "model.safetensors")
    entry["lfs"]["sha256"] = "0" * 64
    entry["blobId"] = git_digest(FILES["model.safetensors"])
    with pytest.raises(Blocked, match="LFS digest mismatch"):
        fetch(tmp_path)
    assert_clean(tmp_path, remote)


@pytest.mark.parametrize("alteration, message", [(b"", "truncated"), (b"extra", "exceeds declared size")])
def test_truncated_or_oversized_stream_is_blocked(tmp_path, remote, alteration, message):
    remote.files["model.safetensors"] = (FILES["model.safetensors"] + alteration
                                         if alteration else FILES["model.safetensors"][:-1])
    with pytest.raises(Blocked, match=message):
        fetch(tmp_path)
    assert_clean(tmp_path, remote)


@pytest.mark.parametrize("size", [_MISSING, None, -1, True, "42", 42.0])
def test_complete_integer_size_required_before_any_asset_fetch(tmp_path, remote, size):
    entry = entry_for(remote, "README.md")
    entry.pop("size")
    if size is not _MISSING:
        entry["size"] = size
    with pytest.raises(Blocked, match="missing declared size"):
        fetch(tmp_path)
    assert len(remote.calls) == 1
    assert_clean(tmp_path, remote)


def test_metadata_revision_mismatch_is_blocked(tmp_path, remote):
    remote.metadata["sha"] = "f" * 40
    with pytest.raises(Blocked, match="resolved model revision mismatch"):
        fetch(tmp_path)
    assert len(remote.calls) == 1
    assert_clean(tmp_path, remote)


@pytest.mark.parametrize("repo", ["model", "a/b/c", "../model", "a/..", "/a/b", "a/b/",
    "https://huggingface.co/a/b", "a:b/model", "a/b:main", "a/b?token=secret", "a/b#tag",
    "a/b%2fother", "a\\b/model", "a/b\n", "a/ b", "a@evil/b", "a/b\0"])
def test_invalid_repo_path_tokens_rejected_before_network(tmp_path, remote, repo):
    with pytest.raises(Blocked, match="namespace/model"):
        acquisition.download(repo, REVISION, tmp_path / "output", 4096)
    assert remote.calls == []
    assert_clean(tmp_path, remote)


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 39, "a" * 41, "A" * 40,
    "g" * 40, "a" * 40 + "\n", "../main", "a:b", ""])
def test_invalid_revision_rejected_before_network(tmp_path, remote, revision):
    with pytest.raises(Blocked, match="exact 40-character"):
        acquisition.download(REPO, revision, tmp_path / "output", 4096)
    assert remote.calls == []
    assert_clean(tmp_path, remote)


@pytest.mark.parametrize("max_bytes", [0, -1, 12 * 1024**3 + 1])
def test_invalid_byte_ceiling_rejected_before_network(tmp_path, remote, max_bytes):
    with pytest.raises(Blocked, match="byte ceiling"):
        fetch(tmp_path, max_bytes)
    assert remote.calls == []
    assert_clean(tmp_path, remote)


def test_declared_total_over_budget_rejected_before_asset_fetch(tmp_path, remote):
    with pytest.raises(Blocked, match="byte/disk budget"):
        fetch(tmp_path, sum(map(len, FILES.values())) - 1)
    assert len(remote.calls) == 1
    assert_clean(tmp_path, remote)


def test_disk_reserve_is_enforced_before_asset_fetch(tmp_path, remote, monkeypatch):
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3))
    with pytest.raises(Blocked, match="byte/disk budget"):
        fetch(tmp_path)
    assert len(remote.calls) == 1
    assert_clean(tmp_path, remote)


@pytest.mark.parametrize("kind", ["empty-directory", "user-directory", "file"])
def test_existing_output_never_overwritten_or_networked(tmp_path, remote, kind):
    output = tmp_path / "output"
    if kind == "file":
        output.write_bytes(b"user-owned-data")
    else:
        output.mkdir()
        if kind == "user-directory":
            (output / "keep.txt").write_bytes(b"user-owned-data")
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(Blocked, match="output must not exist"):
        fetch(tmp_path)
    assert remote.calls == []
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert output.exists()
    assert not list(tmp_path.glob(".silt-model-*"))


def test_output_appearing_while_streaming_preserves_user_directory(tmp_path, remote):
    output = tmp_path / "output"

    def create_user_directory(name):
        if name == "model.safetensors":
            output.mkdir()
            (output / "keep.txt").write_text("do not overwrite")

    remote.before_asset = create_user_directory
    with pytest.raises(Blocked, match="output appeared"):
        fetch(tmp_path)
    assert {p.name for p in output.iterdir()} == {"keep.txt"}
    assert (output / "keep.txt").read_text() == "do not overwrite"
    assert not list(tmp_path.glob(".silt-model-*"))
    assert all(response.closed for response in remote.responses)


@pytest.mark.parametrize("kind", ["parent", "destination", "dangling-destination"])
def test_symlink_output_or_parent_rejected_without_touching_target(tmp_path, remote, kind):
    target = tmp_path / "user-target"
    target.mkdir()
    (target / "keep.txt").write_text("untouched")
    if kind == "parent":
        link = tmp_path / "linked-parent"
        link.symlink_to(target, target_is_directory=True)
        output = link / "output"
    else:
        output = tmp_path / "output"
        output.symlink_to(target if kind == "destination" else tmp_path / "missing", target_is_directory=True)
    with pytest.raises(Blocked, match="symlinks are forbidden"):
        acquisition.download(REPO, REVISION, output, 4096)
    assert remote.calls == []
    assert {p.name for p in target.iterdir()} == {"keep.txt"}
    assert (target / "keep.txt").read_text() == "untouched"
    assert not list(tmp_path.glob(".silt-model-*"))


def test_publish_check_race_does_not_replace_new_empty_user_directory(tmp_path, remote, monkeypatch):
    """Reproduce an empty directory appearing after the final exists() check."""
    output = tmp_path / "output"
    original_exists = Path.exists
    observations = []

    def racing_exists(path):
        exists = original_exists(path)
        if path == output and not exists:
            observations.append(False)
            if len(observations) == 2:
                output.mkdir()
        return exists

    monkeypatch.setattr(Path, "exists", racing_exists)
    with pytest.raises((Blocked, OSError)):
        fetch(tmp_path)
    assert len(observations) == 2
    assert output.is_dir() and list(output.iterdir()) == []
    assert not list(tmp_path.glob(".silt-model-*"))
    assert all(response.closed for response in remote.responses)


def test_missing_parent_not_created(tmp_path, remote):
    with pytest.raises(Blocked, match="parent must exist"):
        acquisition.download(REPO, REVISION, tmp_path / "missing" / "output", 4096)
    assert not (tmp_path / "missing").exists()
    assert remote.calls == []


def test_network_failure_cleans_partial_assets(tmp_path, remote):
    remote.fail_asset = "model.safetensors"
    with pytest.raises(OSError, match="fixture stream unavailable"):
        fetch(tmp_path)
    assert len(remote.calls) == len(FILES) + 1
    assert_clean(tmp_path, remote)


def test_code_pickle_and_paths_never_requested_or_written(tmp_path, remote):
    unsafe = ["model.py", "model.pyc", "pytorch_model.bin", "model.pt", "model.pth", "x.pkl",
              "x.pickle", "model.ckpt", "adapter_config.json", "../config.json", "/config.json",
              "nested/model.safetensors", "model.safetensors.py", "acquisition.json"]
    for name in unsafe:
        # Missing size/hashes deliberately: excluded assets are not processed.
        remote.metadata["siblings"].append({"rfilename": name})
    result = fetch(tmp_path)
    assert set(result["files"]) == set(FILES)
    assert len(remote.calls) == len(FILES) + 1
    assert {p.name for p in (tmp_path / "output").iterdir()} == set(FILES) | {"acquisition.json"}


def test_pickle_only_revision_does_not_fall_back(tmp_path, remote):
    remote.metadata["siblings"] = [{"rfilename": "pytorch_model.bin", "size": 5}]
    with pytest.raises(Blocked, match="pickle fallback is forbidden"):
        fetch(tmp_path)
    assert len(remote.calls) == 1
    assert_clean(tmp_path, remote)


@pytest.mark.parametrize("max_bytes", [None, 1234])
def test_import_cli_dispatches_safe_module_function_only(monkeypatch, capsys, max_bytes):
    calls = []

    def import_bundle(archive, output, spec_output, ceiling):
        calls.append((archive, output, spec_output, ceiling))
        return {"status": "imported-not-admitted", "admitted": False}

    def forbidden_download(*args):
        pytest.fail("import CLI must not dispatch downloader")

    monkeypatch.setattr(bundle, "import_bundle", import_bundle)
    monkeypatch.setattr(acquisition, "download", forbidden_download)
    args = ["import-bundle", "--archive", "safe.zip", "--output", "new-output", "--spec-output", "new-spec.json"]
    if max_bytes is not None:
        args += ["--max-bytes", str(max_bytes)]
    assert acquisition.main(args) == 0
    assert calls == [("safe.zip", "new-output", "new-spec.json", max_bytes if max_bytes is not None else 4 * 1024**3)]
    assert json.loads(capsys.readouterr().out) == {"ok": True, "result": {"status": "imported-not-admitted", "admitted": False}}


def test_import_cli_safe_failure_json(monkeypatch, capsys):
    def refuse(*args):
        raise Blocked("fixture archive denied")

    monkeypatch.setattr(bundle, "import_bundle", refuse)
    assert acquisition.main(["import-bundle", "--archive", "safe.zip", "--output", "out", "--spec-output", "spec.json"]) == 2
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": {"type": "Blocked", "message": "fixture archive denied"}}


def test_download_cli_uses_mock_streams_and_prints_receipt(tmp_path, remote, capsys):
    assert acquisition.main(["download", "--repo", REPO, "--revision", REVISION,
                             "--output", str(tmp_path / "output"), "--max-bytes", "4096"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"]["revision"] == REVISION
    assert len(remote.calls) == 1 + len(FILES)


def test_download_cli_validation_failure_never_networks(tmp_path, remote, capsys):
    assert acquisition.main(["download", "--repo", "a/b:main", "--revision", REVISION,
                             "--output", str(tmp_path / "output")]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["error"]["type"] == "Blocked"
    assert remote.calls == []
