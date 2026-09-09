"""Offline transport/security tests. Tiny weights are not executed or admitted."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import warnings
import zipfile

import pytest

from asea.artifacts import Blocked, Workspace, canonical, digest, model_inventory
from asea.artifacts import bundle
from asea.artifacts.bundle import MANIFEST, STATUS, import_bundle
from asea.compose.schema import CompositionSpec, load_json


# A real safetensors-shaped file containing one F32 value, not a runnable model.
_HEADER = canonical({"unit": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
_HEADER += b" " * (-len(_HEADER) % 8)
WEIGHTS = struct.pack("<Q", len(_HEADER)) + _HEADER + struct.pack("<f", 0.0)


def make_manifest(files=None, kind="hf_text", two_nodes=False):
    files = files or {"model.safetensors": WEIGHTS, "config.json": b"{}"}
    component = {"schema_version": 1, "kind": kind, "task": "fixture" if kind == "fixture_text" else "causal",
                 "model_path": "/nonexistent/original/workspace/model", "risk": "low",
                 "provenance": [{"source": "UNIT TRANSPORT FIXTURE ONLY", "risk": "medium", "license": "test-only"}]}
    spec = CompositionSpec.model_validate({"name": "UNIT TRANSPORT ONLY", "input_type": "text",
        "nodes": [{"id": "first", "input": "$input", "component": component}], "output_node": "first"}).model_dump(mode="json")
    if two_nodes:
        spec["nodes"].append({**copy.deepcopy(spec["nodes"][0]), "id": "second", "input": "first"})
        spec["output_node"] = "second"
    artifact = {"schema_version": 1, "manifest": component,
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)} for name, data in files.items()}}
    aid = "artifact-" + digest(artifact)
    bindings = {node["id"]: aid for node in spec["nodes"]}
    runtime = "composition-cpu-v2-chat-smolvlm"
    graph = digest({"spec": spec, "artifacts": bindings, "runtime_version": runtime})
    candidate = {"schema_version": 1, "id": "candidate-" + graph, "status": "candidate",
                 "graph_hash": graph, "spec": spec, "artifacts": bindings, "runtime_version": runtime,
                 "config_hash": digest({"spec": spec, "runtime_version": runtime}), "risk": "medium",
                 "fixture_only": kind == "fixture_text"}
    manifest = {"kind": "asea-composition-deployment", "format_version": 1, "dependencies_included": True,
                "candidate": candidate, "artifacts": {aid: artifact},
                "deployment": {"id": "UNTRUSTED-OLD-DEPLOYMENT"},
                "evaluation": {"status": "admitted", "id": "UNTRUSTED-OLD-EVALUATION"},
                "runs": [{"status": "succeeded", "id": "UNTRUSTED-OLD-RUN"}],
                "note": "Synthetic format-only fixture; never production admission evidence."}
    entries = {"dependencies/" + aid + "/" + name: data for name, data in files.items()}
    return manifest, entries


def write_zip(path, manifest, entries, compression=zipfile.ZIP_STORED, raw=None):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr(MANIFEST, canonical(manifest) if raw is None else raw)
        for name, value in entries.items():
            archive.writestr(name, value)
    return path


@pytest.fixture
def exported(tmp_path):
    manifest, entries = make_manifest()
    return write_zip(tmp_path / "export.zip", manifest, entries), manifest, entries


def assert_clean(tmp_path):
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "spec.json").exists()
    assert not list(tmp_path.glob(".composition-*-*"))


@pytest.mark.parametrize("internal", [False, True])
def test_round_trip_paths_only_no_trust(tmp_path, internal):
    manifest, entries = make_manifest(two_nodes=True)
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    output = tmp_path / "output"
    spec_file = output / "nested" / "spec.json" if internal else tmp_path / "spec.json"
    result = import_bundle(archive, output, spec_file)
    assert result["status"] == STATUS
    assert result["admitted"] is result["activated"] is result["evidence_valid"] is False
    assert type(result["dependency_files"]) is type(result["dependency_bytes"]) is int
    raw = json.loads(spec_file.read_text())
    original = copy.deepcopy(manifest["candidate"]["spec"])
    assert result["model_paths"]["first"] == result["model_paths"]["second"]
    for node, old in zip(raw["nodes"], original["nodes"]):
        aid = manifest["candidate"]["artifacts"][node["id"]]
        assert node["component"]["model_path"] == str(output / "dependencies" / aid)
        assert model_inventory(node["component"]["model_path"]) == manifest["artifacts"][aid]["files"]
        node["component"]["model_path"] = old["component"]["model_path"]
    assert raw == original
    assert load_json(spec_file, CompositionSpec).output_node == "second"
    assert json.loads((output / "import.json").read_text()) == result
    assert json.loads(Path(result["reference_manifest"]).read_text())["evaluation"]["status"] == "admitted"
    assert not list(output.rglob(".integrity-key"))
    assert not (output / "active.json").exists()
    assert not (output / "evaluations").exists()


def test_real_cli_inspect_rebinds_hashes_without_loading_weights(tmp_path, exported):
    archive, manifest, _ = exported
    output, spec_file = tmp_path / "output", tmp_path / "spec.json"
    import_bundle(archive, output, spec_file)
    command = [sys.executable, "-m", "asea.compose", "--workspace", str(tmp_path / "fresh"),
               "inspect", "--spec", str(spec_file)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    payload = json.loads(process.stdout)["result"]
    assert payload["status"] == "candidate"
    assert payload["graph_hash"] != manifest["candidate"]["graph_hash"]
    assert payload["artifacts"] != manifest["candidate"]["artifacts"]
    assert not (tmp_path / "fresh" / "active.json").exists()
    assert not list((tmp_path / "fresh" / "evaluations").iterdir())


@pytest.mark.parametrize("command", ["run", "evaluate"])
def test_real_cli_accepts_spec_but_fixture_adapter_remains_blocked(tmp_path, command):
    # Public CLI intentionally refuses fixture execution. This tests actual JSON
    # dispatch/load, without pretending test weights are an executable ML model.
    manifest, entries = make_manifest(kind="fixture_text")
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    args = [sys.executable, "-m", "asea.compose", "--workspace", str(tmp_path / "fresh"),
            command, "--spec", str(tmp_path / "spec.json")]
    if command == "run":
        args += ["--input", "hello"]
    else:
        suite = {"name": "UNIT ONLY", "reference_source": "unit literals", "cases": [
            {"id": group, "group": group, "input": "hello", "reference": "hello",
             "metric": "text_exact", "threshold": 1.0} for group in ("target", "control")]}
        (tmp_path / "suite.json").write_bytes(canonical(suite))
        args += ["--suite", str(tmp_path / "suite.json")]
    process = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert process.returncode == 2
    assert "unit-test-only" in process.stdout
    assert json.loads(process.stdout)["ok"] is False
    assert not (tmp_path / "fresh" / "active.json").exists()


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(dependencies_included=False),
    lambda m: m.update(dependencies_included=1),
    lambda m: m.update(kind="legacy-bundle"),
    lambda m: m.update(format_version=True),
    lambda m: m.update(format_version=2),
    lambda m: m.update(unknown=True),
    lambda m: m["candidate"].update(status="admitted"),
    lambda m: m["candidate"].update(schema_version=True),
    lambda m: m["candidate"].update(graph_hash="0" * 64),
    lambda m: m["candidate"]["artifacts"].update(unknown="artifact-" + "0" * 64),
    lambda m: m["candidate"]["spec"].update(admitted=True),
    lambda m: m["candidate"]["spec"]["nodes"][0].update(max_new_tokens="128"),
    lambda m: next(iter(m["artifacts"].values())).update(schema_version=True),
    lambda m: next(iter(m["artifacts"].values()))["files"]["config.json"].update(size=True),
    lambda m: next(iter(m["artifacts"].values()))["files"]["config.json"].update(sha256="0" * 64),
])
def test_bad_schemas_and_bindings_are_refused(tmp_path, mutation):
    manifest, entries = make_manifest()
    mutation(manifest)
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("raw", [
    b'{"kind":1,"kind":2}',
    b'{"nested":{"x":1,"x":2}}',
    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'[]', b'null', b'\xff',
])
def test_strict_json(tmp_path, raw):
    manifest, entries = make_manifest()
    archive = write_zip(tmp_path / "export.zip", manifest, entries, raw=raw)
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\escape", "a//b", "a/./b",
                                  "a/../b", "a:stream", "a/", "NUL", "a/COM1.txt", "a/trailing.",
                                  "a/trailing ", "unknown.json", ".integrity-key", "active.json"])
def test_unsafe_or_unknown_zip_names(tmp_path, name):
    manifest, entries = make_manifest()
    entries[name] = b"unsafe"
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\b", "a//b", "a/./b", "NUL"])
def test_unsafe_inventory_names(tmp_path, name):
    manifest, entries = make_manifest()
    artifact = next(iter(manifest["artifacts"].values()))
    artifact["files"][name] = artifact["files"].pop("config.json")
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("duplicate", [MANIFEST, "dependency"])
def test_duplicate_zip_entries(tmp_path, exported, duplicate):
    archive, _, entries = exported
    name = MANIFEST if duplicate == MANIFEST else next(iter(entries))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a") as stream:
            stream.writestr(name, b"duplicate")
    with pytest.raises(Blocked, match="duplicate"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600, stat.S_IFDIR | 0o755])
def test_nonregular_zip_entry(tmp_path, mode):
    manifest, entries = make_manifest()
    name = next(iter(entries))
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = mode << 16
    data = entries.pop(name)
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr(info, data)
    with pytest.raises(Blocked, match="regular"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("files", [
    {"model.safetensors": WEIGHTS, "unsafe.py": b"raise RuntimeError('must not execute')"},
    {"model.safetensors": WEIGHTS, "unsafe.pkl": b"not a pickle"},
    {"model.safetensors": WEIGHTS, "adapter_config.json": b"{}"},
    {"model.safetensors": WEIGHTS, "tokenizer_config.json": b'{"vocab_file":"../outside"}'},
    {"model.safetensors": WEIGHTS, "tokenizer_config.json": b'{"x":1,"x":2}'},
    {"model.safetensors": WEIGHTS, "tokenizer_config.json": b'[]'},
    {"model.safetensors": WEIGHTS, "model.safetensors.index.json": b'{"weight_map":{"x":"missing.safetensors"}}'},
    {"config.json": b"{}"},
])
def test_model_inventory_safety_is_reused(tmp_path, files):
    manifest, entries = make_manifest(files)
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("mutation", ["hash", "size", "missing", "crc", "truncated"])
def test_dependency_corruption(tmp_path, mutation):
    manifest, entries = make_manifest()
    name = next(iter(entries))
    if mutation == "hash":
        entries[name] = b"x" * len(entries[name])
    elif mutation == "size":
        entries[name] += b"x"
    elif mutation == "missing":
        del entries[name]
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    if mutation == "crc":
        data = bytearray(archive.read_bytes())
        offset = data.index(WEIGHTS)
        data[offset] ^= 1
        archive.write_bytes(data)
    elif mutation == "truncated":
        archive.write_bytes(archive.read_bytes()[:-30])
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("limit", [0, -1, True, 10.0, "100", 10])
def test_byte_budget(tmp_path, exported, limit):
    archive, _, _ = exported
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json", max_bytes=limit)
    assert_clean(tmp_path)


def test_deflate_absolute_budget_not_ratio(tmp_path):
    manifest, entries = make_manifest({"model.safetensors": WEIGHTS, "data.txt": b"0" * (2 * 1024 * 1024)})
    archive = write_zip(tmp_path / "export.zip", manifest, entries, zipfile.ZIP_DEFLATED)
    total = len(canonical(manifest)) + sum(map(len, entries.values()))
    with pytest.raises(Blocked, match="max_bytes"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json", total - 1)
    assert_clean(tmp_path)
    result = import_bundle(archive, tmp_path / "output", tmp_path / "spec.json", total)
    assert result["dependency_bytes"] == sum(map(len, entries.values()))


def test_entry_count_preflight(tmp_path):
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for number in range(10001):
            stream.writestr(str(number), b"")
    with pytest.raises(Blocked, match="10000"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


def test_disk_headroom_checked_before_staging(tmp_path, exported, monkeypatch):
    archive, _, _ = exported
    monkeypatch.setattr(bundle.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(100, 100, 0))
    with pytest.raises(Blocked, match="headroom"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("existing", ["output", "spec.json"])
def test_existing_resources_never_overwritten(tmp_path, exported, existing):
    archive, _, _ = exported
    resource = tmp_path / existing
    if existing == "output":
        resource.mkdir()  # Even an empty directory must not be replaced.
    else:
        resource.write_text("untouched")
    with pytest.raises(Blocked, match="already exist"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert resource.exists()
    if existing == "spec.json":
        assert resource.read_text() == "untouched"
    assert not list(tmp_path.glob(".composition-*-*"))


@pytest.mark.parametrize("which", ["archive", "source-parent", "output", "spec"])
def test_symlink_paths(tmp_path, exported, which):
    archive, _, _ = exported
    output, spec = tmp_path / "output", tmp_path / "spec.json"
    if which == "archive":
        alias = tmp_path / "alias.zip"
        alias.symlink_to(archive)
        archive = alias
    elif which == "source-parent":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        archive = alias / archive.name
    elif which == "output":
        output.symlink_to(tmp_path / "nonexistent")
    else:
        spec.symlink_to(tmp_path / "nonexistent")
    with pytest.raises(Blocked, match="symlink"):
        import_bundle(archive, output, spec)
    assert archive.exists()
    assert not list(tmp_path.glob(".composition-*-*"))


def test_source_inside_output_rejected(tmp_path, exported):
    archive, _, _ = exported
    output = tmp_path / "output"
    output.mkdir()
    moved = output / "source.zip"
    archive.rename(moved)
    with pytest.raises(Blocked, match="outside output"):
        import_bundle(moved, output, tmp_path / "spec.json")
    assert moved.exists()


@pytest.mark.parametrize("spec_name", ["dependencies/spec.json", "reference/spec.json", "import.json", "import.json/spec.json"])
def test_internal_spec_reserved_paths(tmp_path, exported, spec_name):
    archive, _, _ = exported
    with pytest.raises(Blocked, match="reserved"):
        import_bundle(archive, tmp_path / "output", tmp_path / "output" / spec_name)
    assert_clean(tmp_path)


def test_missing_parent_not_created(tmp_path, exported):
    archive, _, _ = exported
    with pytest.raises(Blocked, match="parent"):
        import_bundle(archive, tmp_path / "missing" / "output", tmp_path / "spec.json")
    assert not (tmp_path / "missing").exists()
    with pytest.raises(Blocked, match="parent"):
        import_bundle(archive, tmp_path / "output", tmp_path / "missing" / "spec.json")
    assert_clean(tmp_path)


def test_directory_publication_failure_rolls_back_external_spec(tmp_path, exported, monkeypatch):
    archive, _, _ = exported
    def fail(source, destination):
        assert (tmp_path / "spec.json").exists()
        raise OSError("injected directory publication failure")
    monkeypatch.setattr(bundle, "_publish_directory", fail)
    with pytest.raises(Blocked, match="injected"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


def test_racing_empty_directory_is_not_replaced_or_deleted(tmp_path, exported, monkeypatch):
    archive, _, _ = exported
    publish = bundle._publish_directory
    def race(source, destination):
        destination.mkdir()
        publish(source, destination)
    monkeypatch.setattr(bundle, "_publish_directory", race)
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert (tmp_path / "output").is_dir()
    assert not list((tmp_path / "output").iterdir())
    assert not (tmp_path / "spec.json").exists()
    assert not list(tmp_path.glob(".composition-*-*"))


def test_racing_spec_is_not_replaced_or_deleted(tmp_path, exported, monkeypatch):
    archive, _, _ = exported
    link = os.link
    def race(source, destination):
        Path(destination).write_text("racing resource")
        link(source, destination)
    monkeypatch.setattr(bundle.os, "link", race)
    with pytest.raises(Blocked):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert (tmp_path / "spec.json").read_text() == "racing resource"
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".composition-*-*"))


def test_post_publication_fsync_failure_rolls_back_both(tmp_path, exported, monkeypatch):
    archive, _, _ = exported
    sync = bundle._sync_directory
    def fail(path):
        if path == tmp_path and (tmp_path / "output").exists():
            raise OSError("injected publication fsync failure")
        sync(path)
    monkeypatch.setattr(bundle, "_sync_directory", fail)
    with pytest.raises(Blocked, match="injected"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


def test_existing_exporter_format_compatibility(tmp_path, monkeypatch):
    # Exercise the real serializer/ZIP64 writer without fabricating production
    # admission. Only its prevalidated deployment lookup is replaced locally.
    import asea.certification as certification
    manifest, entries = make_manifest()
    old_aid = next(iter(manifest["artifacts"]))
    artifact = manifest["artifacts"][old_aid]
    root = tmp_path / "model"
    root.mkdir()
    for name, data in entries.items():
        (root / name.split("/", 2)[2]).write_bytes(data)
    artifact["manifest"]["model_path"] = str(root)
    aid = "artifact-" + digest(artifact)
    candidate = manifest["candidate"]
    candidate["spec"]["nodes"][0]["component"]["model_path"] = str(root)
    candidate["artifacts"] = {"first": aid}
    candidate["graph_hash"] = digest({"spec": candidate["spec"], "artifacts": candidate["artifacts"],
                                      "runtime_version": candidate["runtime_version"]})
    candidate["id"] = "candidate-" + candidate["graph_hash"]
    candidate["config_hash"] = digest({"spec": candidate["spec"], "runtime_version": candidate["runtime_version"]})
    evaluation = {"case_evidence": []}
    monkeypatch.setattr(certification, "_deployment", lambda ws, identifier: ({"id": identifier}, evaluation, candidate))
    workspace = Workspace(tmp_path / "exporter-workspace")
    with workspace.writer():
        workspace.write_record("artifacts", aid, artifact)
    archive = tmp_path / "real-export.zip"
    certification.export_deployment(workspace, "unit-format-only", archive, include_dependencies=True)
    result = import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert result["status"] == STATUS
    assert load_json(tmp_path / "spec.json", CompositionSpec).nodes[0].component.kind == "hf_text"


@pytest.mark.parametrize("operation", ["rename", "link"])
def test_interruption_after_publication_syscall_rolls_back_owned_inodes(tmp_path, exported, monkeypatch, operation):
    archive, _, _ = exported
    if operation == "rename":
        publish = bundle._publish_directory
        def interrupt(source, destination):
            publish(source, destination)
            raise KeyboardInterrupt("injected after rename")
        monkeypatch.setattr(bundle, "_publish_directory", interrupt)
    else:
        link = os.link
        def interrupt(source, destination):
            link(source, destination)
            raise KeyboardInterrupt("injected after link")
        monkeypatch.setattr(bundle.os, "link", interrupt)
    with pytest.raises(KeyboardInterrupt):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


def test_temporary_spec_cleanup_error_is_before_commit(tmp_path, exported, monkeypatch):
    archive, _, _ = exported
    unlink = Path.unlink
    failed = False
    def fail_once(path, *args, **kwargs):
        nonlocal failed
        if path.name.startswith(".composition-spec-") and not failed:
            failed = True
            raise OSError("injected temporary cleanup error")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(Blocked, match="injected"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)


@pytest.mark.parametrize("extra", ["MODEL.safetensors", "model.safetensors/child"])
def test_case_alias_and_file_parent_collision(tmp_path, extra):
    manifest, entries = make_manifest({"model.safetensors": WEIGHTS, extra: b"conflict"})
    archive = write_zip(tmp_path / "export.zip", manifest, entries)
    with pytest.raises(Blocked, match="aliased|collision"):
        import_bundle(archive, tmp_path / "output", tmp_path / "spec.json")
    assert_clean(tmp_path)
