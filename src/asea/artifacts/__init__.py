"""Local content-addressed registrations and integrity-checked records (no ML imports)."""
import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import threading
import uuid

try:
    import fcntl
except ImportError:  # POSIX advisory locks are required; never fall back to O_EXCL.
    fcntl = None


class Blocked(ValueError):
    """A deliberate safety or admission refusal."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_path(value):
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise Blocked("parent traversal is forbidden")
    path = Path(os.path.abspath(str(path)))
    for parent in reversed([path] + list(path.parents)):
        if parent.is_symlink():
            raise Blocked("symlinks are forbidden: " + str(parent))
    return path


def safe_file(value):
    path = safe_path(value)
    if not path.is_file():
        raise Blocked("not a regular local file: " + str(path))
    return path


def file_hash(value):
    path = safe_file(value)
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise Blocked("not a regular file")
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise Blocked("file changed during hashing")
    return {"sha256": result.hexdigest(), "size": after.st_size}


def model_inventory(value):
    root = safe_path(value)
    if not root.is_dir():
        raise Blocked("model_path must be an existing local directory")
    result = {}
    for directory, directories, files in os.walk(str(root), followlinks=False):
        for name in directories + files:
            candidate = safe_path(Path(directory) / name)
            if candidate.is_symlink():
                raise Blocked("symlink in model directory")
        for name in sorted(files):
            path = safe_file(Path(directory) / name)
            if path.suffix.lower() in {".py", ".pyc", ".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"}:
                raise Blocked("unsafe/unused executable or pickle file in model directory: " + name)
            result[path.relative_to(root).as_posix()] = file_hash(path)
            if len(result) > 10000:
                raise Blocked("model directory exceeds 10000 files")
    if not result:
        raise Blocked("empty model directory")
    if "adapter_config.json" in result:
        raise Blocked("PEFT/base-model indirection is not a supported standalone component")
    # HF shard indexes and tokenizer configuration must not escape the hashed root.
    def local_reference(reference):
        if not isinstance(reference, str):
            raise Blocked("invalid local dependency reference")
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts or "\\" in reference or relative.as_posix() not in result:
            raise Blocked("dependency reference outside registered inventory: " + reference)
    for name in result:
        if name.endswith(".safetensors.index.json") or name == "tokenizer_config.json":
            if result[name]["size"] > 2 * 1024 * 1024:
                raise Blocked("dependency configuration exceeds 2 MiB")
            config = json.loads(safe_file(root / name).read_text())
            if name.endswith(".safetensors.index.json"):
                if not isinstance(config.get("weight_map"), dict) or not config["weight_map"]:
                    raise Blocked("invalid safetensors shard index")
                for reference in config["weight_map"].values():
                    local_reference(reference)
                    if not reference.endswith(".safetensors"):
                        raise Blocked("only safetensors shards are supported")
            else:
                for key, reference in config.items():
                    if key.endswith("_file") and reference is not None:
                        local_reference(reference)
    return result


def atomic_json(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Workspace:
    """Separate single-writer store. HMAC is tamper-evidence, not protection from its owner."""
    collections = {"artifacts", "candidates", "runs", "evaluations", "deployments", "recoveries"}

    def __init__(self, path):
        self.root = safe_path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / "composition-workspace-v1.json"
        if not marker.exists():
            if any(self.root.iterdir()):
                raise Blocked("workspace must be empty or an existing composition workspace")
            atomic_json(marker, {"kind": "composition-workspace", "schema_version": 1})
        elif json.loads(safe_file(marker).read_text()) != {"kind": "composition-workspace", "schema_version": 1}:
            raise Blocked("invalid workspace marker")
        for collection in self.collections | {"outputs", "imports"}:
            safe_path(self.root / collection).mkdir(exist_ok=True)
        key_path = safe_path(self.root / ".integrity-key")
        if not key_path.exists():
            try:
                fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(os.urandom(32))
            except FileExistsError:
                pass
        self._key = safe_file(key_path).read_bytes()
        if len(self._key) != 32:
            raise Blocked("invalid workspace key")

    @contextlib.contextmanager
    def writer(self):
        """Crash-released POSIX lock on a stable inode (including legacy lock files).

        Never unlink this file: a waiter on the old inode and a new file's owner
        could otherwise both become writers. PID text is diagnostic, not authority.
        """
        if fcntl is None:
            raise Blocked("workspace writing requires POSIX fcntl.flock")
        path = safe_path(self.root / ".writer.lock")
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
                raise Blocked("workspace writer lock must be a single regular inode")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Blocked("workspace writer locked by a living writer") from exc
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (path.stat().st_dev, path.stat().st_ino):
                raise Blocked("workspace writer lock inode changed")
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
            self._writer_owner = (os.getpid(), threading.get_ident())
            try:
                # Lazy import prevents artifacts <-> certification import cycles.
                if safe_path(self.root / "active-transaction.json").exists():
                    from asea.certification import recover_active
                    recover_active(self)
                self._recover_runs()
                yield
            finally:
                self._writer_owner = None
        finally:
            # Closing releases flock even after exceptions; process death does too.
            os.close(fd)

    def _owns_writer(self):
        return getattr(self, "_writer_owner", None) == (os.getpid(), threading.get_ident())

    def _recovery_id(self, run_id):
        self._path("runs", run_id)
        return "recovery-" + digest({"run_id": run_id, "event": "run_recovered"})

    def _recover_runs(self):
        """Reconcile abandoned outputs/legacy running records without rewriting history."""
        pending = {}
        for directory in sorted((self.root / "outputs").iterdir()):
            directory = safe_path(directory)
            self._path("runs", directory.name)  # Never accept arbitrary output paths.
            if not directory.is_dir():
                raise Blocked("invalid run output directory")
            if (safe_path(directory / "pending.json").exists()
                    or (not self._path("runs", directory.name).exists()
                        and not self._path("recoveries", self._recovery_id(directory.name)).exists())):
                pending[directory.name] = directory
        for path in sorted((self.root / "runs").glob("*.json")):
            if (self.read_record("runs", path.stem).get("status") == "running"
                    and not self._path("recoveries", self._recovery_id(path.stem)).exists()):
                pending.setdefault(path.stem, None)
        for run_id, directory in pending.items():
            path = self._path("runs", run_id)
            record = self.read_record("runs", run_id) if path.exists() else None
            status = record.get("status") if record else None
            status = "interrupted" if status in (None, "running") else status
            event = {"schema_version": 1, "id": self._recovery_id(run_id), "event": "run_recovered",
                     "run_id": run_id, "status": status, "record_present": record is not None,
                     "reason": "writer ended before run finalization; isolated outputs are not admitted evidence"}
            self.write_record("recoveries", event["id"], event)
            self.audit("run_recovered", run_id=run_id, status=status, recovery_id=event["id"])
            if directory is not None:
                marker = safe_path(directory / "pending.json")
                if marker.exists():
                    marker.unlink()
                    fd = os.open(str(directory), os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)

    def run_status(self, run_id):
        """Derived status view; signed run records remain immutable and independently readable."""
        if not self._owns_writer():
            with self.writer():
                return self.run_status(run_id)
        path = self._path("runs", run_id)
        record = self.read_record("runs", run_id) if path.exists() else {"id": run_id}
        recovery_id = self._recovery_id(run_id)
        if self._path("recoveries", recovery_id).exists():
            event = self.read_record("recoveries", recovery_id)
            return {**record, "status": event["status"], "recovery_id": recovery_id}
        if not path.exists():
            raise Blocked("unknown run")
        return record

    def active_state(self):
        """Only expose an active pointer after locked recovery, never a pending publication."""
        if not self._owns_writer():
            with self.writer():
                return self.active_state()
        pointer = safe_path(self.root / "active.json")
        return json.loads(safe_file(pointer).read_text()) if pointer.exists() else None

    def new_id(self, prefix):
        return prefix + "-" + uuid.uuid4().hex

    def _path(self, collection, identifier):
        if collection not in self.collections or not re.fullmatch(r"[a-z]+-[a-f0-9]{32,64}", identifier):
            raise Blocked("invalid record identifier or collection")
        return safe_path(self.root / collection / (identifier + ".json"))

    def write_record(self, collection, identifier, value):
        path = self._path(collection, identifier)
        if path.exists():
            if self.read_record(collection, identifier) != value:
                raise Blocked("immutable record already exists")
            return
        signature = hmac.new(self._key, canonical(value), hashlib.sha256).hexdigest()
        atomic_json(path, {"payload": value, "hmac_sha256": signature})

    def read_record(self, collection, identifier):
        path = safe_file(self._path(collection, identifier))
        record = json.loads(path.read_text())
        value = record["payload"]
        expected = hmac.new(self._key, canonical(value), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, record.get("hmac_sha256", "")):
            raise Blocked("record integrity check failed")
        return value

    def register(self, manifest):
        files = model_inventory(manifest.model_path)
        if manifest.kind != "fixture_text" and not any(name.endswith(".safetensors") for name in files):
            raise Blocked("local safetensors weights required")
        value = {"schema_version": 1, "manifest": manifest.model_dump(mode="json"), "files": files}
        identifier = "artifact-" + digest(value)
        self.write_record("artifacts", identifier, value)
        return identifier

    def verify_artifact(self, identifier):
        record = self.read_record("artifacts", identifier)
        if model_inventory(record["manifest"]["model_path"]) != record["files"]:
            raise Blocked("model files changed since registration: " + identifier)
        return record

    def import_directory(self, source):
        """Optional immutable copy; never extracts archives or follows symlinks."""
        inventory = model_inventory(source)
        identifier = digest(inventory)
        destination = safe_path(self.root / "imports" / identifier)
        if destination.exists():
            if model_inventory(destination) != inventory:
                raise Blocked("existing import corrupted")
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=str(self.root / "imports")))
        try:
            root = safe_path(source)
            for relative in inventory:
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(safe_file(root / relative)), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            if model_inventory(temporary) != inventory:
                raise Blocked("source changed during import")
            os.replace(str(temporary), str(destination))
        finally:
            if temporary.exists():
                shutil.rmtree(str(temporary))
        return destination

    def audit(self, event, **fields):
        path = safe_path(self.root / "audit.jsonl")
        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "ab") as stream:
            stream.write(canonical({"time": time.time(), "event": event, **fields}) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def list_records(self):
        if not self._owns_writer():
            with self.writer():
                return self.list_records()
        result = {collection: [self.read_record(collection, p.stem) for p in sorted((self.root / collection).glob("*.json"))]
                  for collection in ("candidates", "evaluations", "deployments", "recoveries")}
        run_ids = {p.stem for p in (self.root / "runs").glob("*.json")}
        run_ids.update(event["run_id"] for event in result["recoveries"])
        result["runs"] = [self.run_status(run_id) for run_id in sorted(run_ids)]
        result["active"] = self.active_state()
        return result
