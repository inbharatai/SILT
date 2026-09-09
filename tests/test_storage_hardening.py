"""Phase0 source-level negative tests; these intentionally exercise internals.

Not user-facing demonstrations. No model downloads or optional connectors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import pytest

from asea.core.errors import RollbackError, SnapshotNotFoundError
from asea.core.protocol import PacketType, PromotionStatus, SkillPacket
from asea.distill import export as exporting
from asea.distill.export import export_artifact_bundle, export_dataset
from asea.memory import store as storage
from asea.memory.store import APPROVED, CANDIDATE, REJECTED, MemoryStore, RollbackLayer


@pytest.fixture
def packet(capability, clean_provenance, packet_factory):
    return packet_factory(
        capability, clean_provenance, packet_id="legacy safe + (1)@v1",
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "ভাত", "target": "rice"}]},
        rollback_token="before", promotion_status=PromotionStatus.PROMOTED,
    )


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory")


def tree(path):
    return {str(p.relative_to(path)): p.read_bytes()
            for p in path.rglob("*") if p.is_file() and not p.is_symlink()}


UNSAFE = ["", ".", "..", "../approved/escape", "x/../../escape", "/tmp/escape",
          "x/y", "x\\y", "C:\\escape", "C:escape", "\\\\host\\share", "a..b",
          "bad\x00id", "bad\nid", "bad.", "bad ", "CON", "a:stream", "a*", "a?",
          "x" * 251, "bad\ud800id"]


@pytest.mark.parametrize("packet_id", UNSAFE)
def test_packet_ids_rejected_at_all_store_and_export_boundaries(store, packet, tmp_path, packet_id):
    packet.packet_id = packet_id
    before = tree(store.root)
    actions = [lambda: store.put_candidate(packet), lambda: store.put_rejected(packet),
               lambda: store.approve(packet), lambda: store.reject(packet),
               lambda: store.get(CANDIDATE, packet_id),
               lambda: store._path(APPROVED, packet_id)]
    for action in actions:
        with pytest.raises(RollbackError):
            action()
        assert tree(store.root) == before
    for writer in (export_artifact_bundle, export_dataset):
        out = tmp_path / "not-created"
        with pytest.raises(ValueError):
            writer([packet], out, "bundle")
        assert not out.exists()
    assert not (tmp_path / "escape.json").exists()


@pytest.mark.parametrize("bucket", ["../outside", "/tmp", "candidate/../approved", "", "other"])
def test_bucket_boundaries(store, packet, bucket):
    before = tree(store.root)
    for action in (lambda: store.put(packet, bucket), lambda: store.get(bucket, packet.packet_id),
                   lambda: store.list(bucket), lambda: store.count(bucket), lambda: store._dir(bucket)):
        with pytest.raises(RollbackError):
            action()
    assert tree(store.root) == before


@pytest.mark.parametrize("packet_id", ["legacy.safe-v1_2", "Skill 2001 + (draft)@site", "ভাত-日本語", ".hidden"])
def test_legacy_safe_ids_roundtrip_and_export(store, packet, tmp_path, packet_id):
    packet.packet_id = packet_id
    store.put_candidate(packet)
    store.approve(packet)
    layer = RollbackLayer(store)
    token = layer.snapshot("Legacy label: punctuation is not a path!")
    original = tree(store.root / APPROVED)
    assert layer.rollback(token)["restored"] == 1
    assert tree(store.root / APPROVED) == original
    assert store.get(APPROVED, packet_id).packet_id == packet_id
    with zipfile.ZipFile(export_artifact_bundle([packet], tmp_path / "export", "old.v1")) as zf:
        assert "approved/{}.json".format(packet_id) in zf.namelist()
        assert json.loads(zf.read("manifest.json"))["version"] == 1
        assert json.loads(zf.read("old.v1.jsonl"))["input"] == "ভাত"


def test_relative_path_return_contract_is_preserved(packet, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = MemoryStore(Path("memory"))
    assert local.root == Path("memory")
    assert local.put_candidate(packet) == Path("memory/candidate") / (packet.packet_id + ".json")
    assert export_artifact_bundle([packet], Path("out"), "bundle") == Path("out/bundle.zip")
    assert export_dataset([packet], Path("out"), "dataset")["dataset_path"] == "out/dataset.jsonl"


def test_file_token_cannot_delete_approved(store, packet):
    store.approve(packet)
    (store.root / "snapshots" / "file-token").write_text("{}")
    before = tree(store.root)
    layer = RollbackLayer(store)
    with pytest.raises(RollbackError, match="not a directory"):
        layer.rollback("file-token")
    with pytest.raises(SnapshotNotFoundError):
        layer.snapshot_packets("file-token")
    assert tree(store.root) == before


@pytest.mark.parametrize("damage", ["missing_meta", "bad_meta", "count", "token", "date", "label",
                                   "partial", "schema", "status", "mismatch", "extra_dir"])
def test_corrupt_partial_snapshot_prevalidation_never_deletes(store, packet, damage):
    store.approve(packet)
    layer = RollbackLayer(store)
    token = layer.snapshot()
    source = store.root / "snapshots" / token
    path = source / (packet.packet_id + ".json")
    meta_path = source / "_meta.json"
    if damage == "missing_meta":
        meta_path.unlink()
    elif damage == "bad_meta":
        meta_path.write_text("{")
    elif damage in ("count", "token", "date", "label"):
        meta = json.loads(meta_path.read_text())
        field, value = {"count": ("packet_count", 2), "token": ("token", "wrong"),
                        "date": ("created_at", "not-a-date"), "label": ("label", 3)}[damage]
        meta[field] = value
        meta_path.write_text(json.dumps(meta))
    elif damage == "partial":
        # One valid packet plus a torn second packet: not just a wholly bad set.
        (source / "zzz.json").write_text('{"packet_id":')
    elif damage == "schema":
        path.write_text('{"packet_id": "bad", "unknown_field": true}')
    elif damage in ("status", "mismatch"):
        data = json.loads(path.read_text())
        data["promotion_status" if damage == "status" else "packet_id"] = (
            "distilled" if damage == "status" else "different")
        path.write_text(json.dumps(data))
    else:
        (source / "extra").mkdir()
    before = tree(store.root)
    with pytest.raises(RollbackError):
        layer.rollback(token)
    assert tree(store.root) == before
    assert not list(store.root.glob(".rollback-stage-*"))


@pytest.mark.parametrize("where", ["root", "bucket", "packet", "snapshot", "snapshot_packet", "metadata"])
def test_symlink_boundaries_fail_closed(store, packet, tmp_path, where):
    store.approve(packet)
    layer = RollbackLayer(store)
    token = layer.snapshot()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.json"
    sentinel.write_text(packet.model_dump_json())
    if where == "root":
        link = tmp_path / "linked"
        link.symlink_to(store.root, target_is_directory=True)
        actions = [lambda: MemoryStore(link)]
    elif where == "bucket":
        (store.root / CANDIDATE).rmdir()
        (store.root / CANDIDATE).symlink_to(outside, target_is_directory=True)
        actions = [lambda: store.put_candidate(packet), lambda: store.list(CANDIDATE),
                   lambda: store.count(CANDIDATE), lambda: MemoryStore(store.root)]
    elif where == "packet":
        path = store.root / APPROVED / (packet.packet_id + ".json")
        path.unlink()
        path.symlink_to(sentinel)
        actions = [lambda: store.get(APPROVED, packet.packet_id), lambda: store.list(APPROVED),
                   lambda: store.count(APPROVED), lambda: store.put(packet, APPROVED),
                   lambda: layer.snapshot(), lambda: layer.rollback(token)]
    elif where == "snapshot":
        (store.root / "snapshots" / "link").symlink_to(outside, target_is_directory=True)
        actions = [lambda: layer.rollback("link"), lambda: layer.list_snapshots()]
    else:
        source = store.root / "snapshots" / token
        path = source / ("_meta.json" if where == "metadata" else packet.packet_id + ".json")
        path.unlink()
        path.symlink_to(sentinel)
        actions = [lambda: layer.rollback(token), lambda: layer.snapshot_packets(token)]
    original = tree(outside)
    for action in actions:
        with pytest.raises(RollbackError):
            action()
    assert tree(outside) == original


@pytest.mark.parametrize("destination", ["bundle.zip", "bundle.jsonl", "manifest.json", "README.txt"])
def test_export_symlinks_fail_before_any_write(packet, tmp_path, destination):
    out = tmp_path / "exports"
    out.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    (out / destination).symlink_to(outside)
    before = tree(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        export_artifact_bundle([packet], out, "bundle")
    assert tree(tmp_path) == before


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_packet_write_failure_preserves_existing_and_candidate(store, packet, monkeypatch, failure):
    store.put_candidate(packet)
    store.put(packet, APPROVED)
    before = tree(store.root)
    packet.distilled_skill = {"entries": [{"source": "new", "target": "new"}]}
    def fail(*args, **kwargs):
        raise OSError("injected write failure")
    monkeypatch.setattr(storage.os, failure, fail)
    with pytest.raises(RollbackError):
        store.approve(packet)
    assert tree(store.root) == before
    assert not list((store.root / APPROVED).glob(".packet-*"))


@pytest.mark.parametrize("failure", ["write", "first_replace", "second_replace", "after_replace", "interrupt"])
def test_rollback_failure_restores_original_bytes(store, packet, monkeypatch, failure):
    store.approve(packet)
    layer = RollbackLayer(store)
    token = layer.snapshot()
    packet.distilled_skill = {"entries": [{"source": "new", "target": "new"}]}
    store.put(packet, APPROVED)
    before = tree(store.root)
    original_replace = storage.os.replace
    def replace(src, dst):
        if failure == "after_replace" and Path(src).name.startswith(".rollback-stage-"):
            original_replace(src, dst)
            raise OSError("injected failure after directory installation")
        if ((failure == "first_replace" and Path(src).name == APPROVED)
                or (failure in ("second_replace", "interrupt")
                    and Path(src).name.startswith(".rollback-stage-"))):
            if failure == "interrupt":
                raise KeyboardInterrupt()
            raise OSError("injected directory replace failure")
        return original_replace(src, dst)
    def fail_write(*args):
        raise OSError("injected write failure")
    monkeypatch.setattr(storage.os, "replace", replace)
    if failure == "write":
        monkeypatch.setattr(storage.os, "fsync", fail_write)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RollbackError):
        layer.rollback(token)
    assert tree(store.root) == before
    assert store.count(APPROVED) == 1
    assert not (store.root / storage._BACKUP).exists()


def test_process_death_retains_backup_and_reopen_fails_closed(store, packet):
    store.approve(packet)
    token = RollbackLayer(store).snapshot()
    before = tree(store.root / APPROVED)
    script = '''
import os, sys
from pathlib import Path
from asea.memory.store import MemoryStore, RollbackLayer
store = MemoryStore(Path(sys.argv[1]))
original = os.replace
def replace(src, dst):
    if Path(src).name.startswith('.rollback-stage-'):
        os._exit(23)
    return original(src, dst)
os.replace = replace
RollbackLayer(store).rollback(sys.argv[2])
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(store.root), token],
                            cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
    assert result.returncode == 23
    assert tree(store.root / storage._BACKUP) == before
    with pytest.raises(RollbackError, match="Operator recovery required"):
        MemoryStore(store.root)
    with pytest.raises(RollbackError, match="interrupted rollback"):
        store.list(APPROVED)
    assert not (store.root / APPROVED).exists()  # constructor did NOT create empty approved/


def test_v1_sparse_packet_and_original_snapshot_bytes_preserved(store, packet, tmp_path):
    # A historical v1 fixture omitting defaulted optional fields, not a new
    # model_dump roundtrip masquerading as an old on-disk schema.
    raw = json.dumps({
        "packet_id": "old safe + v1", "version": 1, "task_type": "translate",
        "source_module": "trusted-source", "target_module": "learner",
        "sender_capability": {"task_type": "translate", "modality": "text",
                              "domain": "translation", "language": "as->en"},
        "modality": "text", "provenance": {"origin_kind": "curated_corpus",
                                            "chain": ["trusted-source"]},
        "promotion_status": "promoted", "rollback_token": "legacy",
        "packet_type": "glossary", "distilled_skill": {
            "entries": [{"source": "ভাত", "target": "rice"}]},
    }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path = store.root / APPROVED / "old safe + v1.json"
    path.write_bytes(raw)
    loaded = store.get(APPROVED, "old safe + v1")
    token = RollbackLayer(store).snapshot()
    path.unlink()
    RollbackLayer(store).rollback(token)
    assert path.read_bytes() == raw  # no silent rewriting or new fields
    with zipfile.ZipFile(export_artifact_bundle([loaded], tmp_path / "old-export", "legacy")) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["kind"] == "silt_skill_bundle" and manifest["version"] == 1
        exported = SkillPacket.model_validate_json(zf.read("approved/old safe + v1.json"))
        assert exported == loaded
        assert json.loads(zf.read("legacy.manifest.json"))["row_count"] == 1
        assert "legacy.job.json" not in zf.namelist()


def test_readonly_metadata_free_legacy_snapshot_supported_but_not_restored(store, packet):
    source = store.root / "snapshots" / "legacy label +1"
    source.mkdir()
    (source / (packet.packet_id + ".json")).write_text(packet.model_dump_json())
    layer = RollbackLayer(store)
    assert layer.snapshot_packets(source.name)[0] == packet
    with pytest.raises(RollbackError):
        layer.rollback(source.name)


def test_invalid_skipped_id_and_duplicate_zip_ids_fail_before_output(packet, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="duplicate"):
        export_artifact_bundle([packet, packet], out)
    assert not out.exists()
    packet.promotion_status = PromotionStatus.DISTILLED
    packet.packet_id = "../skipped"
    with pytest.raises(ValueError):
        export_artifact_bundle([packet], out)
    assert not out.exists()


def test_failed_recovery_retains_backup_for_explicit_operator_action(store, packet, monkeypatch):
    store.approve(packet)
    layer = RollbackLayer(store)
    token = layer.snapshot()
    before = tree(store.root / APPROVED)
    original_replace = storage.os.replace
    def replace(src, dst):
        if Path(src).name.startswith(".rollback-stage-"):
            raise OSError("install failed")
        return original_replace(src, dst)
    def rename(*args):
        raise OSError("recovery failed")
    monkeypatch.setattr(storage.os, "replace", replace)
    monkeypatch.setattr(storage.os, "rename", rename)
    with pytest.raises(RollbackError, match="recovery required"):
        layer.rollback(token)
    assert tree(store.root / storage._BACKUP) == before
    with pytest.raises(RollbackError, match="Operator recovery"):
        MemoryStore(store.root)


def test_actual_staged_write_failure_leaves_no_snapshot_or_lost_packets(store, packet, monkeypatch):
    store.approve(packet)
    before = tree(store.root)
    original = storage.tempfile.NamedTemporaryFile
    class BrokenWriter:
        def __init__(self, *args, **kwargs):
            self.fh = original(*args, **kwargs)
            self.name = self.fh.name
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.fh.close()
        def write(self, data):
            self.fh.write(data[:5])  # real partial write before failure
            self.fh.flush()
            raise OSError("torn staged write")
    monkeypatch.setattr(storage.tempfile, "NamedTemporaryFile", BrokenWriter)
    with pytest.raises(RollbackError):
        RollbackLayer(store).snapshot()
    assert tree(store.root) == before
    assert not list(store.root.glob(".snapshot-stage-*"))
    with pytest.raises(RollbackError):
        store.put(packet, APPROVED)
    assert tree(store.root) == before


def test_rejection_validates_destination_before_removing_candidate(store, packet, tmp_path):
    store.put_candidate(packet)
    outside = tmp_path / "outside.json"
    outside.write_text("keep")
    (store.root / REJECTED / (packet.packet_id + ".json")).symlink_to(outside)
    before = tree(store.root)
    with pytest.raises(RollbackError):
        store.reject(packet)
    assert tree(store.root) == before
    assert outside.read_text() == "keep"


def test_snapshot_reserved_metadata_id_fails_before_writes(store, packet):
    packet.packet_id = "_meta"
    store.approve(packet)  # safe legacy packet filename remains readable
    before = tree(store.root)
    with pytest.raises(RollbackError, match="conflicts"):
        RollbackLayer(store).snapshot()
    assert tree(store.root) == before


def test_unpromoted_generic_put_cannot_bypass_approved_boundary(store, packet):
    packet.promotion_status = PromotionStatus.DISTILLED
    with pytest.raises(RollbackError):
        store.put(packet, APPROVED)
    assert store.count(APPROVED) == 0


_EXPORT_NAMES = ["bundle.jsonl", "bundle.manifest.json", "manifest.json", "README.txt",
                 "bundle.job.json", "bundle.zip"]


def _export(writer, packet, out):
    if writer == "bundle":
        return export_artifact_bundle([packet], out, "bundle", base_model="local-model")
    return export_dataset([packet], out, "bundle")


def _previous_export(writer, packet, out, previous):
    if previous != "missing":
        _export(writer, packet, out)
        if previous == "empty":
            for path in out.iterdir():
                path.write_bytes(b"")  # existence matters even with zero old bytes
    out.mkdir(exist_ok=True)
    (out / "unrelated.txt").write_text("keep")
    return tree(out)


@pytest.mark.parametrize("writer,failure", [
    (writer, failure) for writer in ("dataset", "bundle")
    for failure in ("row_first", "row_second", "manifest", "fsync")
] + [("bundle", "zip_first"), ("bundle", "zip_after")])
@pytest.mark.parametrize("previous", ["valid", "empty", "missing"])
def test_export_serialization_failures_preserve_all_prior_outputs(
        packet, tmp_path, monkeypatch, writer, failure, previous):
    out = tmp_path / "out"
    before = _previous_export(writer, packet, out, previous)
    packet.distilled_skill = {"entries": [{"source": "new1", "target": "one"},
                                         {"source": "new2", "target": "two"}]}
    dumps, dump = exporting.json.dumps, exporting.json.dump
    writestr = zipfile.ZipFile.writestr
    rows = []
    def fail():
        raise OSError("injected export failure")
    def serialize(value, *args, **kwargs):
        if isinstance(value, dict) and "input" in value:
            rows.append(value)
            if failure == "row_first" or (failure == "row_second" and len(rows) == 2):
                fail()
        return dumps(value, *args, **kwargs)
    def manifest(value, fh, *args, **kwargs):
        if failure == "manifest":
            fh.write('{"torn":')  # partial staged companion must not reach live files
            fail()
        return dump(value, fh, *args, **kwargs)
    def member(zf, *args, **kwargs):
        if failure == "zip_first":
            fail()
        result = writestr(zf, *args, **kwargs)
        if failure == "zip_after":
            fail()
        return result
    monkeypatch.setattr(exporting.json, "dumps", serialize)
    monkeypatch.setattr(exporting.json, "dump", manifest)
    monkeypatch.setattr(zipfile.ZipFile, "writestr", member)
    if failure == "fsync":
        monkeypatch.setattr(exporting.os, "fsync", lambda *args: fail())
    with pytest.raises(OSError, match="injected"):
        _export(writer, packet, out)
    assert tree(out) == before
    assert not list(out.glob(".export-*"))


@pytest.mark.parametrize("writer,destination", [
    ("dataset", name) for name in _EXPORT_NAMES[:2]
] + [("bundle", name) for name in _EXPORT_NAMES])
@pytest.mark.parametrize("previous", ["valid", "empty", "missing"])
@pytest.mark.parametrize("when", ["before", "after", "interrupt"])
def test_export_install_failure_rolls_back_every_companion(
        packet, tmp_path, monkeypatch, writer, destination, previous, when):
    out = tmp_path / "out"
    before = _previous_export(writer, packet, out, previous)
    packet.distilled_skill = {"entries": [{"source": "new", "target": "new"}]}
    original = exporting.os.replace
    def replace(src, dst):
        if Path(dst) == out / destination:
            if when == "after":
                original(src, dst)
            if when == "interrupt":
                raise KeyboardInterrupt()
            raise OSError("injected install failure")
        return original(src, dst)
    monkeypatch.setattr(exporting.os, "replace", replace)
    with pytest.raises(KeyboardInterrupt if when == "interrupt" else OSError):
        _export(writer, packet, out)
    assert tree(out) == before
    assert not list(out.glob(".export-*"))


@pytest.mark.parametrize("writer", ["dataset", "bundle"])
@pytest.mark.parametrize("failure", ["write", "backup", "marker_before", "marker_after"])
def test_export_staging_io_and_marker_failures_preserve_previous(
        packet, tmp_path, monkeypatch, writer, failure):
    out = tmp_path / "out"
    before = _previous_export(writer, packet, out, "valid")
    original_replace, original_copy = exporting.os.replace, exporting.shutil.copyfile
    original_open = open
    def replace(src, dst):
        if Path(dst) == out / exporting._EXPORT_RECOVERY:
            if failure == "marker_after":
                original_replace(src, dst)
            if failure.startswith("marker_"):
                raise OSError("injected marker failure")
        return original_replace(src, dst)
    def copy(src, dst, *args, **kwargs):
        if failure == "backup":
            Path(dst).write_bytes(b"torn backup")
            raise OSError("injected backup failure")
        return original_copy(src, dst, *args, **kwargs)
    class TornWriter:
        def __init__(self, fh):
            self.fh = fh
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.fh.close()
        def write(self, value):
            self.fh.write(value[:3])
            self.fh.flush()
            raise OSError("injected partial write")
    def opening(path, mode="r", *args, **kwargs):
        fh = original_open(path, mode, *args, **kwargs)
        if failure == "write" and mode == "w" and Path(path).suffix == ".jsonl":
            return TornWriter(fh)
        return fh
    monkeypatch.setattr(exporting.os, "replace", replace)
    monkeypatch.setattr(exporting.shutil, "copyfile", copy)
    monkeypatch.setattr(exporting, "open", opening, raising=False)
    with pytest.raises(OSError, match="injected"):
        _export(writer, packet, out)
    assert tree(out) == before
    assert not list(out.glob(".export-*"))


@pytest.mark.parametrize("writer", ["dataset", "bundle"])
def test_repeat_export_succeeds_with_legacy_paths_and_matching_companions(packet, tmp_path, writer):
    out = tmp_path / "out"
    _export(writer, packet, out)
    packet.distilled_skill = {"entries": [{"source": "new", "target": "new"}]}
    result = _export(writer, packet, out)
    manifest = json.loads((out / "bundle.manifest.json").read_text())
    assert manifest["dataset_path"] == str(out / "bundle.jsonl")
    assert json.loads((out / "bundle.jsonl").read_text())["input"] == "new"
    assert manifest["dataset_sha256"] == exporting.hashlib.sha256((out / "bundle.jsonl").read_bytes()).hexdigest()
    if writer == "bundle":
        assert result == out / "bundle.zip"
        with zipfile.ZipFile(result) as zf:
            assert zf.testzip() is None
            for name in _EXPORT_NAMES[:-1]:
                assert zf.read(name) == (out / name).read_bytes()
    else:
        assert result == manifest
    assert not list(out.glob(".export-*"))


def test_repeat_bundle_removes_obsolete_job_but_restores_it_on_failure(packet, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _export("bundle", packet, out)
    before = tree(out)
    original = exporting.os.replace
    def replace(src, dst):
        if Path(dst) == out / "bundle.zip":
            raise OSError("last install failed")
        return original(src, dst)
    with monkeypatch.context() as patch:
        patch.setattr(exporting.os, "replace", replace)
        with pytest.raises(OSError):
            export_artifact_bundle([packet], out, "bundle")
    assert tree(out) == before
    export_artifact_bundle([packet], out, "bundle")
    assert not (out / "bundle.job.json").exists()
    with zipfile.ZipFile(out / "bundle.zip") as zf:
        assert "bundle.job.json" not in zf.namelist()


def test_export_failed_recovery_retains_backup_and_refuses_reuse(packet, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _export("dataset", packet, out)
    before = tree(out)
    original = exporting.os.replace
    def replace(src, dst):
        if Path(dst) == out / "bundle.manifest.json":
            raise OSError("install failed")
        return original(src, dst)
    def rename(*args):
        raise OSError("recovery failed")
    with monkeypatch.context() as patch:
        patch.setattr(exporting.os, "replace", replace)
        patch.setattr(exporting.os, "rename", rename)
        with pytest.raises(RuntimeError, match="recovery required"):
            _export("dataset", packet, out)
    assert tree(out / exporting._EXPORT_RECOVERY / "old") == before
    frozen = tree(out)
    with pytest.raises(ValueError, match="operator recovery"):
        _export("bundle", packet, out)
    assert tree(out) == frozen


def test_export_process_death_retains_old_generation_and_refuses_reuse(packet, tmp_path):
    out = tmp_path / "out"
    _export("bundle", packet, out)
    before = tree(out)
    script = '''
import os, sys
from pathlib import Path
from asea.core.protocol import SkillPacket
from asea.distill.export import export_artifact_bundle
out = Path(sys.argv[1])
packet = SkillPacket.model_validate_json(sys.argv[2])
original = os.replace
def replace(src, dst):
    if Path(dst) == out / 'bundle.manifest.json':
        os._exit(24)
    return original(src, dst)
os.replace = replace
export_artifact_bundle([packet], out, 'bundle', base_model='local-model')
'''
    packet.distilled_skill = {"entries": [{"source": "new", "target": "new"}]}
    result = subprocess.run([sys.executable, "-B", "-c", script, str(out), packet.model_dump_json()],
                            cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
    assert result.returncode == 24
    recovery = out / exporting._EXPORT_RECOVERY
    assert tree(recovery / "old") == before
    journal = json.loads((recovery / "journal.json").read_text())
    assert set(journal["previous"]) == set(before)
    with pytest.raises(ValueError, match="operator recovery"):
        _export("bundle", packet, out)


def test_root_locks_are_shared_across_instances_and_path_spellings(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = MemoryStore(Path("memory"))
    alias = MemoryStore(store.root / ".." / "memory")
    assert store._lock is relative._lock is alias._lock
    assert MemoryStore(tmp_path / "different")._lock is not store._lock


@pytest.mark.parametrize("transition", ["approve", "reject"])
@pytest.mark.parametrize("changed", ["payload", "provenance", "notes", "target"])
def test_stale_transition_cannot_remove_acknowledged_new_candidate(store, packet, transition, changed):
    store.put_candidate(packet)
    stale = store.get(CANDIDATE, packet.packet_id)
    second = MemoryStore(store.root)
    update = packet.model_copy(deep=True)
    if changed == "payload":
        update.distilled_skill = {"entries": [{"source": "new", "target": "new"}]}
    elif changed == "provenance":
        update.provenance.chain.append("new-review")
    elif changed == "notes":
        update.notes["review"] = "newer candidate"
    else:
        update.target_module = "new-receiver"
    second.put_candidate(update)  # acknowledged before the stale operation starts
    before = tree(store.root)
    with pytest.raises(RollbackError, match="stale candidate"):
        getattr(store, transition)(stale)
    assert tree(store.root) == before


@pytest.mark.parametrize("transition", ["approve", "reject"])
def test_two_instance_candidate_update_waits_for_destructive_transition(
        store, packet, monkeypatch, transition):
    second = MemoryStore(store.root)
    store.put_candidate(packet)
    candidate = store.root / CANDIDATE / (packet.packet_id + ".json")
    reached, resume, started, acknowledged = (threading.Event() for _ in range(4))
    errors = []
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == candidate and threading.current_thread().name == "transition":
            reached.set()
            if not resume.wait(5):
                raise TimeoutError("transition not resumed")
        return original(path, *args, **kwargs)
    def moving():
        try:
            getattr(store, transition)(packet)
        except BaseException as exc:
            errors.append(exc)
    update = packet.model_copy(deep=True)
    update.distilled_skill = {"entries": [{"source": "new", "target": "acknowledged"}]}
    def writing():
        started.set()
        try:
            second.put_candidate(update)
            acknowledged.set()
        except BaseException as exc:
            errors.append(exc)
    monkeypatch.setattr(Path, "unlink", unlink)
    first = threading.Thread(target=moving, name="transition")
    other = threading.Thread(target=writing)
    first.start()
    try:
        assert reached.wait(5)
        other.start()
        assert started.wait(5)
        assert not acknowledged.wait(0.1)  # writer cannot bypass root lock
    finally:
        resume.set()
        first.join(5)
        if other.ident is not None:
            other.join(5)
    assert not first.is_alive() and not other.is_alive()
    assert not errors and acknowledged.is_set()
    assert store.get(CANDIDATE, packet.packet_id) == update


def test_two_instance_duplicate_check_is_one_critical_section(store, packet, monkeypatch):
    second = MemoryStore(store.root)
    reached, resume, started, finished = (threading.Event() for _ in range(4))
    original = store.list
    errors = []
    def listing(bucket):
        result = original(bucket)
        if bucket == APPROVED:
            reached.set()
            if not resume.wait(5):
                raise TimeoutError("first approval not resumed")
        return result
    monkeypatch.setattr(store, "list", listing)
    duplicate = packet.model_copy(deep=True)
    duplicate.packet_id = "duplicate"
    def approve(instance, item, other=False):
        if other:
            started.set()
        try:
            instance.approve(item)
        except BaseException as exc:
            errors.append(exc)
        finally:
            if other:
                finished.set()
    first = threading.Thread(target=approve, args=(store, packet))
    other = threading.Thread(target=approve, args=(second, duplicate, True))
    first.start()
    try:
        assert reached.wait(5)
        other.start()
        assert started.wait(5)
        assert not finished.wait(0.1)
    finally:
        resume.set()
        first.join(5)
        if other.ident is not None:
            other.join(5)
    assert not first.is_alive() and not other.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RollbackError)
    assert "identical content" in str(errors[0])
    assert second.count(APPROVED) == 1


def test_constructor_waits_for_another_instances_rollback(store, packet, monkeypatch):
    store.approve(packet)
    layer = RollbackLayer(store)
    token = layer.snapshot()
    reached, resume, started, opened = (threading.Event() for _ in range(4))
    errors = []
    original = storage.os.replace
    def replace(src, dst):
        if Path(src).name.startswith(".rollback-stage-"):
            reached.set()
            if not resume.wait(5):
                raise TimeoutError("rollback not resumed")
        return original(src, dst)
    def rollback():
        try:
            layer.rollback(token)
        except BaseException as exc:
            errors.append(exc)
    def opening():
        started.set()
        try:
            MemoryStore(store.root)
            opened.set()
        except BaseException as exc:
            errors.append(exc)
    monkeypatch.setattr(storage.os, "replace", replace)
    first = threading.Thread(target=rollback)
    other = threading.Thread(target=opening)
    first.start()
    try:
        assert reached.wait(5)
        other.start()
        assert started.wait(5)
        assert not opened.wait(0.1)
    finally:
        resume.set()
        first.join(5)
        if other.ident is not None:
            other.join(5)
    assert not first.is_alive() and not other.is_alive()
    assert not errors and opened.is_set()
    assert store.count(APPROVED) == 1
