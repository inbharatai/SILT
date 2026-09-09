"""Portable dependency transport, deliberately NOT a portable admission credential.

Only the Compose deployment export is accepted; legacy bundle kinds are untouched.
No workspace, signing key, adapter, tensor loader, or network client is constructed.
"""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import zipfile
import zlib

from . import Blocked, canonical, digest, model_inventory, safe_file, safe_path
from asea.compose.schema import ComponentManifest, CompositionSpec


MANIFEST = "composition-export-v1.json"
STATUS = "UNADMITTED_REQUIRES_REEVALUATION"
_MAX_ENTRIES = 10000
_MAX_JSON = 16 * 1024 * 1024
_CHUNK = 1024 * 1024
_HEADROOM = 16 * 1024 * 1024
_REASON = ("Imported evidence is human reference only. Model paths and graph bindings "
           "change across workspaces; inspect, run and evaluate locally before admission.")


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Blocked("duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise Blocked("nonfinite JSON number: " + value)

    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    # Also catches overflowing finite-looking exponents (e.g. 1e999).
    canonical(value)
    return value


def _fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise Blocked("invalid " + label + " fields")


def _version(value, label):
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise Blocked("invalid " + label + " schema_version")


def _relative(name):
    # POSIX ZIP paths only, with Windows aliases rejected even on POSIX hosts.
    if (not isinstance(name, str) or not name or len(name) > 4096
            or "\\" in name or ":" in name or any(ord(c) < 32 for c in name)):
        raise Blocked("unsafe archive/inventory path")
    for part in name.split("/"):
        if (part in {"", ".", ".."} or part.endswith((".", " "))
                or re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)):
            raise Blocked("unsafe archive/inventory path: " + name)
    return name


def _entries(archive, max_bytes):
    entries, aliases, total = {}, set(), 0
    infos = archive.infolist()
    if len(infos) > _MAX_ENTRIES:
        raise Blocked("archive exceeds 10000 entries")
    for info in infos:
        name = _relative(info.filename)
        if info.orig_filename != name or name in entries or name.casefold() in aliases:
            raise Blocked("duplicate or aliased archive name")
        mode = info.external_attr >> 16
        if (info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or info.external_attr & 0x10):
            raise Blocked("archive entries must be regular files, not symlinks/directories")
        if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise Blocked("encrypted or unsupported archive compression")
        if info.file_size < 0 or info.compress_size < 0:
            raise Blocked("negative ZIP entry size")
        total += info.file_size
        if total > max_bytes:
            raise Blocked("archive uncompressed bytes exceed max_bytes")
        entries[name] = info
        aliases.add(name.casefold())
    # A regular file may not also be an ancestor of another regular file.
    for name in entries:
        parts = name.casefold().split("/")
        if any("/".join(parts[:i]) in aliases for i in range(1, len(parts))):
            raise Blocked("archive file/directory path collision")
    if MANIFEST not in entries or entries[MANIFEST].file_size > _MAX_JSON:
        raise Blocked("missing or oversized composition export manifest")
    return entries, total


def _stream(archive, info, target=None, expected=None):
    count, hasher, chunks = 0, hashlib.sha256(), []
    with archive.open(info, "r") as source:
        while True:
            chunk = source.read(min(_CHUNK, info.file_size - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > info.file_size:
                raise Blocked("entry exceeds declared uncompressed size")
            hasher.update(chunk)
            if target is None:
                chunks.append(chunk)
            else:
                target.write(chunk)
    if count != info.file_size:
        raise Blocked("entry size mismatch")
    actual = {"sha256": hasher.hexdigest(), "size": count}
    if expected is not None and actual != expected:
        raise Blocked("dependency SHA-256/size mismatch: " + info.filename)
    return b"".join(chunks) if target is None else actual


def _validate(manifest, entries):
    _fields(manifest, {"kind", "format_version", "dependencies_included", "deployment",
                       "evaluation", "candidate", "artifacts", "runs", "note"}, "export manifest")
    if (manifest["kind"] != "asea-composition-deployment"
            or type(manifest["format_version"]) is not int or manifest["format_version"] != 1):
        raise Blocked("unsupported bundle kind/version")
    if manifest["dependencies_included"] is not True:
        raise Blocked("portable import requires dependencies_included=true")
    if (not isinstance(manifest["deployment"], dict) or not isinstance(manifest["evaluation"], dict)
            or not isinstance(manifest["runs"], list)
            or not all(isinstance(run, dict) for run in manifest["runs"])
            or not isinstance(manifest["note"], str)):
        raise Blocked("invalid reference metadata")
    candidate, artifacts = manifest["candidate"], manifest["artifacts"]
    _fields(candidate, {"schema_version", "id", "status", "graph_hash", "spec", "artifacts",
                        "runtime_version", "config_hash", "risk", "fixture_only"}, "candidate")
    _version(candidate, "candidate")
    spec = CompositionSpec.model_validate(candidate["spec"])
    if (candidate["status"] != "candidate" or not isinstance(candidate["runtime_version"], str)
            or not candidate["runtime_version"] or candidate["risk"] != spec.effective_risk
            or type(candidate["fixture_only"]) is not bool
            or candidate["fixture_only"] != any(n.component.kind == "fixture_text" for n in spec.nodes)):
        raise Blocked("invalid candidate metadata")
    bindings = candidate["artifacts"]
    if (not isinstance(bindings, dict) or set(bindings) != {node.id for node in spec.nodes}
            or not all(isinstance(aid, str) for aid in bindings.values())
            or not isinstance(artifacts, dict) or set(artifacts) != set(bindings.values())):
        raise Blocked("candidate node/artifact inventory binding mismatch")
    expected = {}
    for aid, artifact in artifacts.items():
        if not re.fullmatch(r"artifact-[a-f0-9]{64}", aid):
            raise Blocked("invalid artifact identifier")
        _fields(artifact, {"schema_version", "manifest", "files"}, "artifact")
        _version(artifact, "artifact")
        ComponentManifest.model_validate(artifact["manifest"])
        files = artifact["files"]
        if not isinstance(files, dict) or not files:
            raise Blocked("empty or invalid artifact inventory")
        for name, fingerprint in files.items():
            name = _relative(name)
            _fields(fingerprint, {"sha256", "size"}, "file fingerprint")
            if (type(fingerprint["size"]) is not int or fingerprint["size"] < 0
                    or not isinstance(fingerprint["sha256"], str)
                    or not re.fullmatch(r"[a-f0-9]{64}", fingerprint["sha256"])):
                raise Blocked("invalid inventory SHA-256/size")
            archive_name = "dependencies/" + aid + "/" + name
            expected[archive_name] = fingerprint
            if archive_name not in entries or entries[archive_name].file_size != fingerprint["size"]:
                raise Blocked("missing dependency or inventory size mismatch")
        if aid != "artifact-" + digest(artifact):
            raise Blocked("artifact manifest hash mismatch")
    if set(entries) != {MANIFEST} | set(expected):
        raise Blocked("extra unknown archive entries")
    for node in spec.nodes:
        if node.component.model_dump(mode="json") != artifacts[bindings[node.id]]["manifest"]:
            raise Blocked("spec component/artifact manifest mismatch")
    graph = digest({"spec": candidate["spec"], "artifacts": bindings,
                    "runtime_version": candidate["runtime_version"]})
    config = digest({"spec": candidate["spec"], "runtime_version": candidate["runtime_version"]})
    if (candidate["graph_hash"] != graph or candidate["id"] != "candidate-" + graph
            or candidate["config_hash"] != config):
        raise Blocked("candidate graph/config hash mismatch")
    return expected


def _publish_directory(source, destination):
    """Atomic exclusive rename; ordinary os.rename can overwrite an empty directory."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    else:
        raise Blocked("atomic exclusive directory publication requires Linux/macOS")
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _identity(path):
    value = path.lstat()
    return value.st_dev, value.st_ino


def _owned(path, identity):
    try:
        return _identity(path) == identity
    except FileNotFoundError:
        return False


def _sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def import_bundle(archive, output, spec_output, max_bytes=4 * 1024**3) -> dict:
    """Verify/extract an export and write a path-rebound, UNADMITTED Compose spec.

    ``output`` must not exist. ``spec_output`` is a new JSON file, either inside
    output (single atomic publication) or in an existing external directory.
    External publication uses an exclusive hard link and rolls back our own
    publications on exceptions. No unrelated paths are replaced or removed.
    ``max_bytes`` caps ALL uncompressed ZIP entries including the manifest.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise Blocked("max_bytes must be a positive integer")
    source, destination, spec_path = safe_file(archive), safe_path(output), safe_path(spec_output)
    if source == destination or destination in source.parents:
        raise Blocked("source archive must be outside output")
    if destination.exists() or spec_path.exists():
        raise Blocked("output and spec_output must not already exist")
    if not destination.parent.is_dir():
        raise Blocked("output parent must be an existing directory")
    internal = destination in spec_path.parents
    if spec_path == destination or spec_path in destination.parents:
        raise Blocked("spec_output must be a file, not an output ancestor")
    relative_spec = spec_path.relative_to(destination) if internal else None
    if internal:
        _relative(relative_spec.as_posix())
        if relative_spec.parts[0] in {"dependencies", "reference", "import.json"}:
            raise Blocked("spec_output overlaps reserved import paths")
    elif not spec_path.parent.is_dir():
        raise Blocked("external spec_output parent must already exist")
    stage = spec_temp = None
    stage_identity = spec_identity = None
    try:
        source_fd = os.open(str(source), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                            | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(source_fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise Blocked("archive must be a regular file")
            # Bound central-directory allocation before ZipFile builds ZipInfo objects.
            # _EndRecData also handles ZIP64 (used by the existing exporter).
            end = zipfile._EndRecData(stream)
            if (end is None or end[zipfile._ECD_ENTRIES_TOTAL] > _MAX_ENTRIES
                    or end[zipfile._ECD_SIZE] > 64 * 1024 * 1024):
                raise Blocked("invalid or oversized ZIP directory (maximum 10000 entries)")
            stream.seek(0)
            with zipfile.ZipFile(stream) as bundle:
                entries, total = _entries(bundle, max_bytes)
                manifest_bytes = _stream(bundle, entries[MANIFEST])
                manifest = _json(manifest_bytes)
                expected = _validate(manifest, entries)
                spec = copy.deepcopy(manifest["candidate"]["spec"])
                bindings = manifest["candidate"]["artifacts"]
                model_paths = {node["id"]: str(destination / "dependencies" / bindings[node["id"]])
                               for node in spec["nodes"]}
                for node in spec["nodes"]:
                    node["component"]["model_path"] = model_paths[node["id"]]
                CompositionSpec.model_validate(spec)
                spec_bytes = canonical(spec)
                if len(spec_bytes) > 2 * 1024 * 1024:
                    raise Blocked("rebuilt spec exceeds Compose JSON limit")
                result = {"status": STATUS, "admitted": False, "activated": False,
                          "evidence_valid": False, "output": str(destination), "spec_output": str(spec_path),
                          "spec_hash": digest(spec), "model_paths": model_paths,
                          "dependencies_included": True, "dependency_files": len(expected),
                          "dependency_bytes": sum(item["size"] for item in expected.values()),
                          "reference_manifest": str(destination / "reference" / MANIFEST), "reason": _REASON}
                result_bytes = canonical(result)
                needed = total + len(spec_bytes) + len(result_bytes) + _HEADROOM
                if shutil.disk_usage(destination.parent).free < needed:
                    raise Blocked("insufficient disk headroom for import")
                if not internal and shutil.disk_usage(spec_path.parent).free < len(spec_bytes) + _HEADROOM:
                    raise Blocked("insufficient disk headroom for spec_output")
                stage = Path(tempfile.mkdtemp(prefix=".composition-import-", dir=str(destination.parent)))
                stage_identity = _identity(stage)
                for name, fingerprint in sorted(expected.items()):
                    target = stage / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as target_stream:
                        _stream(bundle, entries[name], target_stream, fingerprint)
                        target_stream.flush()
                        os.fsync(target_stream.fileno())
                for aid, artifact in manifest["artifacts"].items():
                    root = stage / "dependencies" / aid
                    # These are the configuration files interpreted by model_inventory.
                    for name, fingerprint in artifact["files"].items():
                        if name == "tokenizer_config.json" or name.endswith(".safetensors.index.json"):
                            if fingerprint["size"] > 2 * 1024 * 1024:
                                raise Blocked("dependency configuration exceeds 2 MiB")
                            if not isinstance(_json((root / name).read_bytes()), dict):
                                raise Blocked("dependency configuration must be a JSON object")
                    if model_inventory(root) != artifact["files"]:
                        raise Blocked("extracted model inventory mismatch")
                    if (artifact["manifest"]["kind"] != "fixture_text"
                            and not any(name.endswith(".safetensors") for name in artifact["files"])):
                        raise Blocked("local safetensors weights required")
                _write(stage / "reference" / MANIFEST, manifest_bytes)
                _write(stage / "import.json", result_bytes)
                if internal:
                    _write(stage / relative_spec, spec_bytes)
                else:
                    fd, temporary = tempfile.mkstemp(prefix=".composition-spec-", dir=str(spec_path.parent))
                    spec_temp = Path(temporary)
                    spec_identity = _identity(spec_temp)
                    with os.fdopen(fd, "wb") as spec_stream:
                        spec_stream.write(spec_bytes)
                        spec_stream.flush()
                        os.fsync(spec_stream.fileno())
                for directory, _, _ in os.walk(stage, topdown=False):
                    _sync_directory(Path(directory))
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise Blocked("source archive changed during import")
        # Recheck symlink parents before publication. Parents must be trusted local
        # directories; no filesystem transaction can defend an adversarial owner.
        safe_path(destination)
        safe_path(spec_path)
        if spec_temp is not None:
            os.link(str(spec_temp), str(spec_path))  # atomic, exclusive, no replacement
            # Remove the temporary hard link before the transaction commits, so
            # a cleanup error cannot turn a successful publication into failure.
            spec_temp.unlink()
            spec_temp = None
            _sync_directory(spec_path.parent)
        _publish_directory(stage, destination)
        _sync_directory(destination.parent)
        return result
    except BaseException as exc:
        # Inode ownership also covers interruption immediately after a successful
        # syscall, before Python could record a publication-completed flag.
        if stage_identity is not None and _owned(destination, stage_identity):
            shutil.rmtree(destination)
        if spec_identity is not None and _owned(spec_path, spec_identity):
            spec_path.unlink()
        if isinstance(exc, (ValueError, OSError, KeyError, TypeError, RecursionError,
                            zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError)) and not isinstance(exc, Blocked):
            raise Blocked("bundle import refused: " + str(exc)) from exc
        raise
    finally:
        if stage is not None and _owned(stage, stage_identity):
            shutil.rmtree(stage)
        if spec_temp is not None and _owned(spec_temp, spec_identity):
            spec_temp.unlink()
