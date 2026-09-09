"""Local validation workflow ledger. Not a security boundary or certificate issuer."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
import uuid

from asea.artifacts import Blocked, canonical, digest, safe_path, safe_file
from .schema import LicenseCard, Policy, ResultInput, Suite

try:
    import fcntl
except ImportError:
    fcntl = None

BOUNDARY = "local_owner_workflow_only; not independent custodian authorization"
# Writers and readers share a byte budget, including the complete event envelope.
MAX_JSON_BYTES = 8 * 1024 * 1024


def read_bytes(path, limit=32 * 1024 * 1024):
    path = safe_file(path)
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise Blocked("input is not a bounded regular file")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > limit or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise Blocked("input oversized or changed while reading")
    return raw


def load_json(path):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise Blocked("duplicate JSON key")
            obj[key] = value
        return obj
    return json.loads(read_bytes(path, MAX_JSON_BYTES), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(Blocked("nonfinite JSON")))


def exclusive_json(path, value):
    """Publish bounded, fsynced JSON without replacing an existing name.

    Validate the actual serialized bytes before creating even a pending file.
    On a live publication error, attempt to roll back our link. A process/power
    interruption or failed rollback requires reopening to inspect committed state.
    """
    raw = canonical(value)
    if len(raw) > MAX_JSON_BYTES:
        raise Blocked("serialized JSON exceeds reader byte budget")
    path = safe_path(path)
    if not path.parent.is_dir():
        raise Blocked("output parent must exist")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=str(path.parent))
    directory, linked = None, False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(str(path.parent), os.O_RDONLY)
        os.link(temporary, str(path))
        linked = True
        os.fsync(directory)
    except BaseException:
        if linked:
            os.unlink(str(path))
            os.fsync(directory)
        raise
    finally:
        if directory is not None:
            os.close(directory)
        os.unlink(temporary)


def license_status(cards):
    """Conservative terms classification, not legal clearance or an SPDX parser."""
    cards = [c if isinstance(c, LicenseCard) else LicenseCard.model_validate(c) for c in cards]
    known = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "CC-BY-4.0"}
    nc = any("NC" in c.spdx.upper() for c in cards)
    unknown = any(c.spdx not in known | {"CC-BY-NC-4.0"} for c in cards)
    privacy = any(c.consent_privacy == "unknown" for c in cards)
    restrictions = [r for c in cards for r in c.usage_restrictions]
    return {"licenses": [c.model_dump() for c in cards],
            "commercial_product_eligible": False if nc or unknown or privacy or restrictions else None,
            "commercial_terms_status": "blocked" if nc or unknown or privacy or restrictions else "requires_rights_review",
            "use_scope": "noncommercial_experimental" if nc else "subject_to_recorded_terms",
            "unknown_terms": unknown, "privacy_unresolved": privacy,
            "legal_authorization_granted": False}


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise Blocked("expected exact lowercase SHA-256")
    return value


class ValidationRegistry:
    """Append-only single-writer event store outside the candidate workspace.

    Owner can rewrite all records. Hash chaining detects accidental alteration only.
    No signing key is created, inspected, or read. POSIX flock required.
    """
    def __init__(self, workspace, registry=None):
        self.workspace = safe_path(workspace)
        self.root = safe_path(registry or self.workspace.with_name(self.workspace.name + ".validation-registry"))
        if self.root == self.workspace or self.workspace in self.root.parents or self.root in self.workspace.parents:
            raise Blocked("registry and candidate workspace must be disjoint")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock():
            marker = self.root / "validation-registry-v1.json"
            expected = {"schema_version": 1, "workspace": str(self.workspace), "boundary": BOUNDARY}
            if not marker.exists():
                entries = [p for p in self.root.iterdir() if p.name != ".writer.lock"]
                if any(not p.name.startswith(".pending-") for p in entries):
                    raise Blocked("registry must be empty or already initialized")
                # Interrupted first marker publication has no committed state.
                for pending in entries:
                    safe_file(pending).unlink()
                exclusive_json(marker, expected)
            elif load_json(marker) != expected:
                raise Blocked("registry workspace/format mismatch")
            safe_path(self.root / "events").mkdir(exist_ok=True)
            self._events()

    @contextlib.contextmanager
    def _lock(self):
        if fcntl is None:
            raise Blocked("POSIX single-writer locking required")
        path = safe_path(self.root / ".writer.lock")
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise Blocked("invalid writer lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Blocked("registry writer busy; retry explicitly") from exc
            yield
        finally:
            os.close(fd)

    def _events(self):
        entries, previous = [], None
        directory = safe_path(self.root / "events")
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".pending-"):
                continue  # never considered committed; recover removes these under lock
            if not re.fullmatch(r"[0-9]{8}\.json", path.name):
                raise Blocked("unexpected registry entry")
            try:
                event = load_json(path)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise Blocked("malformed registry event; no automatic repair") from exc
            if not isinstance(event, dict) or set(event) != {"sequence", "previous", "time_ns", "kind", "data", "sha256"}:
                raise Blocked("malformed registry event")
            expected = dict(event)
            checksum = expected.pop("sha256")
            if (event["sequence"] != len(entries) + 1 or path.stem != "%08d" % event["sequence"]
                    or event["previous"] != previous or digest(expected) != checksum):
                raise Blocked("registry chain corrupted; no automatic repair")
            entries.append(event)
            previous = checksum
        return entries

    def _append(self, kind, data, events):
        event = {"sequence": len(events) + 1, "previous": events[-1]["sha256"] if events else None,
                 "time_ns": time.time_ns(), "kind": kind, "data": data}
        event["sha256"] = digest(event)
        exclusive_json(self.root / "events" / ("%08d.json" % event["sequence"]), event)
        return data

    def register(self, suite):
        suite = Suite.model_validate(suite).model_dump()
        with self._lock():
            events = self._events()
            prior = [e["data"]["suite"] for e in events if e["kind"] == "register"]
            if any(s["suite_id"] == suite["suite_id"] for s in prior):
                raise Blocked("suite ID immutable; register a new ID/version")
            if suite["parent_suite_id"] and not any(s["suite_id"] == suite["parent_suite_id"] for s in prior):
                raise Blocked("unknown parent suite")
            # Global family identifiers deliberately span datasets/translations.
            existing = [m for s in prior for m in s["members"]]
            governed = [m for e in events if e["kind"] == "function_io_register" for m in e["data"]["members"]]
            if governed:
                from .governed import _overlap
                if any(_overlap(m, old) and m["split"] != old["split"] for m in suite["members"] for old in governed):
                    raise Blocked("member/content/family crosses governed splits")
            for member in suite["members"]:
                for old in existing:
                    same_member = (member["dataset_id"], member["member_id"]) == (old["dataset_id"], old["member_id"])
                    if (same_member or member["family_id"] == old["family_id"] or member["content_sha256"] == old["content_sha256"] or member["input"] == old["input"]) and member["split"] != old["split"]:
                        raise Blocked("member/content/family crosses registered splits")
            splits = {split: digest([m for m in suite["members"] if m["split"] == split]) for split in ("development", "final")}
            return self._append("register", {"suite": suite, "suite_sha256": digest(suite), "split_hashes": splits,
                                             "state": "development", "boundary": BOUNDARY}, events)

    def freeze(self, suite_id, candidate_hash, policy):
        _hash(candidate_hash)
        policy = Policy.model_validate(policy).model_dump()
        with self._lock():
            events = self._events()
            registrations = [e["data"] for e in events if e["kind"] == "register" and e["data"]["suite"]["suite_id"] == suite_id]
            if not registrations:
                raise Blocked("unknown suite")
            if any(e["kind"] == "freeze" and e["data"]["suite_id"] == suite_id for e in events):
                raise Blocked("suite already reserved or consumed; no reuse")
            registered = registrations[0]
            suite = registered["suite"]
            final = [m for m in suite["members"] if m["split"] == "final"]
            if not final or len(final) > policy["max_cases"] or {m["role"] for m in final} != {"target", "control"}:
                raise Blocked("final split must fit budget and contain target and control")
            rights = license_status(list(suite["datasets"].values()))
            if rights["unknown_terms"] or rights["privacy_unresolved"]:
                raise Blocked("unknown license/consent terms block final freeze")
            if policy["purpose"] == "commercial" and rights["commercial_terms_status"] == "blocked":
                raise Blocked("license restrictions block commercial use")
            # A different suite name must not reset a consumed/frozen holdout.
            frozen_ids = {e["data"]["suite_id"] for e in events if e["kind"] == "freeze"}
            old_final = [m for e in events if e["kind"] == "register" and e["data"]["suite"]["suite_id"] in frozen_ids
                         for m in e["data"]["suite"]["members"] if m["split"] == "final"]
            governed_final = [m for e in events if e["kind"] == "function_io_freeze" and e["data"]["purpose"] == "final" for m in e["data"]["members"]]
            if governed_final:
                from .governed import _overlap
                if any(_overlap(m, old) for m in final for old in governed_final):
                    raise Blocked("final member/family already reserved by governed evaluation")
            for m in final:
                if any(m["family_id"] == old["family_id"] or m["content_sha256"] == old["content_sha256"] or m["input"] == old["input"]
                       or (m["dataset_id"], m["member_id"]) == (old["dataset_id"], old["member_id"]) for old in old_final):
                    raise Blocked("final member/family already reserved; use genuinely new final data")
            return self._append("freeze", {"reservation_id": uuid.uuid4().hex, "suite_id": suite_id,
                "suite_sha256": registered["suite_sha256"], "split_hashes": registered["split_hashes"],
                "candidate_hash": candidate_hash, "policy": policy, "policy_sha256": digest(policy),
                "state": "frozen_final", "boundary": BOUNDARY}, events)

    def begin(self, reservation_id, candidate_hash, policy):
        _hash(candidate_hash)
        policy = Policy.model_validate(policy).model_dump()
        with self._lock():
            events = self._events()
            reservations = [e["data"] for e in events if e["kind"] == "freeze" and e["data"]["reservation_id"] == reservation_id]
            if not reservations:
                raise Blocked("unknown reservation")
            reservation = reservations[0]
            if any(e["kind"] == "begin" and e["data"]["reservation_id"] == reservation_id for e in events):
                raise Blocked("reservation consumed, including failed/interrupted attempts")
            if reservation["candidate_hash"] != candidate_hash or reservation["policy_sha256"] != digest(policy):
                raise Blocked("candidate/policy differs from exact reservation binding")
            return self._append("begin", {**reservation, "run_id": uuid.uuid4().hex, "state": "consumed",
                                         "result_status": "pending", "quality_admission": False}, events)

    def finish(self, run_id, result):
        result = ResultInput.model_validate(result).model_dump()
        with self._lock():
            events = self._events()
            runs = [e["data"] for e in events if e["kind"] == "begin" and e["data"]["run_id"] == run_id]
            if not runs:
                raise Blocked("unknown run")
            if any(e["kind"] == "finish" and e["data"]["run_id"] == run_id for e in events):
                raise Blocked("run already finished")
            if result["evaluator_revision"] != runs[0]["policy"]["evaluator_revision"]:
                raise Blocked("evaluator revision differs from frozen policy")
            return self._append("finish", {"run_id": run_id, "result_input": result, "result_sha256": digest(result),
                "state": "consumed", "trust": "unverified_result_input_not_certificate", "quality_admission": False,
                "boundary": BOUNDARY}, events)

    def history(self):
        with self._lock():
            events = self._events()
            self._append("access", {"operation": "history", "boundary": BOUNDARY}, events)
            return {"events": events, "boundary": BOUNDARY}

    def final_inputs(self, run_id):
        """Evaluator integration hook; logs access, but cannot isolate owner/candidate.

        Call only after begin, pass inputs to a separately implemented oracle. This
        does not supply observed outcomes or authorize a trusted evaluator.
        """
        with self._lock():
            events = self._events()
            runs = [e["data"] for e in events if e["kind"] == "begin" and e["data"]["run_id"] == run_id]
            if not runs or any(e["kind"] == "finish" and e["data"]["run_id"] == run_id for e in events):
                raise Blocked("run absent or already finished")
            suite = next(e["data"]["suite"] for e in events if e["kind"] == "register" and e["data"]["suite"]["suite_id"] == runs[0]["suite_id"])
            self._append("access", {"operation": "final_inputs", "run_id": run_id, "boundary": BOUNDARY}, events)
            return [m for m in suite["members"] if m["split"] == "final"]

    def recover(self):
        """Remove uncommitted temp files, never reset consumed reservations."""
        with self._lock():
            events = self._events()
            removed = []
            for directory in (self.root, self.root / "events"):
                for path in directory.iterdir():
                    if path.name.startswith(".pending-"):
                        safe_file(path).unlink()
                        removed.append(path.name)
            return self._append("recovery", {"removed_uncommitted_files": removed,
                "consumed_reservations_unchanged": True}, events)


# Public convenience exports. Importing these never imports an ML runtime.
from .voice import prepare_voice_evidence, prepare_listening_batch, validate_listening_review  # noqa: E402,F401
from .statistics import proportion_summary  # noqa: E402,F401
