"""function_io_pilot_v1: observed public CLI workflow, never a new quality gate.

No reference program is executed here. Owner-readable data and same-owner HMACs
are workflow evidence, not a custodian/security/contamination boundary.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from asea.artifacts import (Blocked, Workspace, digest, file_hash, model_inventory,
                            safe_file, safe_path)
from asea.compose.schema import CompositionSpec, EvaluationSuite
from . import BOUNDARY, ValidationRegistry, exclusive_json, load_json, read_bytes

ADAPTER = "function_io_pilot_v1"


def _sha(path):
    return file_hash(path)["sha256"]


def _checked(root, relative, expected):
    path = safe_file(root / relative)
    if root not in path.parents or _sha(path) != expected:
        raise Blocked("manifest artifact hash/path mismatch: " + relative)
    return path


def adapt_pilot(suite_path, purpose, manifest_path=None):
    """Read only the selected split; lock metadata contains IDs, not final answers.

    Pilot-v2's actual manifest has sources/source_pins/artifact_sha256, not
    invented LicenseCard fields. Keep its rights evidence intact and do not
    infer consent/privacy/commercial authorization from an SPDX label.
    """
    if purpose not in {"development", "final"}:
        raise Blocked("unknown purpose")
    suite_path = safe_file(suite_path)
    manifest_path = safe_file(manifest_path or suite_path.parent / "manifest.json")
    manifest = load_json(manifest_path)
    root = manifest_path.parent
    name = purpose + ".json"
    # Check path and manifest binding BEFORE reading any suite contents.
    if suite_path != safe_path(root / name):
        raise Blocked("purpose must match the manifest's exact split path")
    if (manifest.get("version") != 2 or manifest.get("license_unresolved") != []
            or manifest.get("license_status") != "verified_from_preserved_primary_repository_and_publisher_dataset_card"):
        raise Blocked("verified pilot-v2 rights/provenance manifest required; supply --dataset-manifest")
    hashes = manifest["artifact_sha256"]
    lock_path = _checked(root, "selection-lock.json", hashes["selection-lock.json"])
    lock = load_json(lock_path)
    if _sha(lock_path) != manifest["selection_lock_sha256"] or digest(lock["selection"]) != manifest["selection_sha256"]:
        raise Blocked("selection binding mismatch")
    selection = lock["selection"]
    ids, families, sources = set(), {}, {}
    for row in selection:
        if row["id"] in ids or row["split"] not in {"development", "final"}:
            raise Blocked("duplicate source ID or invalid split")
        ids.add(row["id"])
        if row["family"] in families:
            raise Blocked("pilot members must have distinct families")
        families[row["family"]] = row["split"]
        source_id = (row["source"], row["source_task_id"])
        if source_id in sources:
            raise Blocked("source task reuse")
        sources[source_id] = row["source_record_sha256"]
    evidence_names = {"sources/HumanEval-LICENSE.txt", "sources/mbpp-LICENSE-evidence.md", "sources/CC-BY-4.0.txt"}
    evidence = [row for row in manifest["sources"] if row["file"] in evidence_names]
    if {row["file"] for row in evidence} != evidence_names:
        raise Blocked("missing preserved license evidence")
    for row in evidence:
        _checked(root, row["file"], row["sha256"])
    attribution_path = _checked(root, "ATTRIBUTION.md", hashes["ATTRIBUTION.md"])
    _checked(root, name, hashes[name])
    raw_suite = load_json(suite_path)
    suite = EvaluationSuite.model_validate(raw_suite)
    if suite.claims != ["coding"] or any(c.metric != "function_io" or c.input_file is not None for c in suite.cases):
        raise Blocked("adapter accepts text-input function_io coding suites only")
    selected = {row["id"]: row for row in selection if row["split"] == purpose}
    if {c.id for c in suite.cases} != set(selected):
        raise Blocked("suite members differ from frozen selection")
    members = []
    for case in suite.cases:
        row = selected[case.id]
        if case.group != row["group"] or row["source"] not in {"HumanEval", "MBPP"}:
            raise Blocked("source/group binding mismatch")
        members.append({"member_id": case.id, "dataset_id": row["source"],
                        "source_task_id": row["source_task_id"], "source_record_sha256": row["source_record_sha256"],
                        "family_id": row["family"], "split": purpose, "role": case.group,
                        "input_sha256": digest(case.input), "content_sha256": digest(case.model_dump(mode="json")),
                        "license": "MIT" if row["source"] == "HumanEval" else "CC-BY-4.0"})
    registration = {"adapter": ADAPTER, "suite_id": suite.name, "version": lock["version"],
                    "purpose": purpose, "suite_artifact_sha256": hashes[name],
                    "suite_sha256": digest(suite.model_dump(mode="json")),
                    "manifest_sha256": _sha(manifest_path), "selection_lock_sha256": _sha(lock_path),
                    "split_family_sha256": digest(selection), "source_mapping": selection,
                    "members": members, "source_pins": manifest["source_pins"],
                    "rights_evidence": evidence, "attribution": read_bytes(attribution_path).decode(),
                    "exposure": manifest["pretraining_exposure"], "scope": manifest["scope"],
                    "legal_authorization_granted": False, "boundary": BOUNDARY}
    return suite, registration


def candidate_binding(spec_path):
    """Same content graph as compose._prepare, without registration or model loads."""
    from asea.compose.runtime import RUNTIME_VERSION
    from asea.certification import implementation_fingerprint
    spec = CompositionSpec.model_validate(load_json(spec_path))
    if spec.input_type != "text" or any(n.component.kind != "hf_text" for n in spec.nodes):
        raise Blocked("governed coding requires real local hf_text components; fixtures are unit-test-only")
    inventories, artifacts = {}, {}
    for node in spec.nodes:
        files = model_inventory(node.component.model_path)
        if not any(name.endswith(".safetensors") for name in files):
            raise Blocked("local safetensors weights required")
        value = {"schema_version": 1, "manifest": node.component.model_dump(mode="json"), "files": files}
        inventories[node.id] = value
        artifacts[node.id] = "artifact-" + digest(value)
    normalized = spec.model_dump(mode="json")
    graph_hash = digest({"spec": normalized, "artifacts": artifacts, "runtime_version": RUNTIME_VERSION})
    implementation = implementation_fingerprint()
    wrapper_root = Path(__file__).parent
    implementation["governance_hashes"] = {p.name: file_hash(p) for p in sorted(wrapper_root.glob("*.py"))}
    binding = {"spec": normalized, "spec_artifact_sha256": _sha(spec_path), "inventories": inventories,
               "graph_hash": graph_hash, "runtime_version": RUNTIME_VERSION, "implementation": implementation}
    return binding


def _overlap(a, b):
    return (a["family_id"] == b["family_id"] or a["content_sha256"] == b["content_sha256"]
            or a.get("input_sha256", digest(a.get("input"))) == b.get("input_sha256", digest(b.get("input")))
            or (a["dataset_id"], a.get("source_task_id", a["member_id"])) ==
               (b["dataset_id"], b.get("source_task_id", b["member_id"])))


class GovernedRegistry(ValidationRegistry):
    """Versioned events share the old lock/hash chain; old data-only APIs unchanged."""
    def register_function_io(self, registration):
        with self._lock():
            events = self._events()
            for e in events:
                if e["kind"] == "function_io_register":
                    old = e["data"]
                    if old["suite_id"] == registration["suite_id"]:
                        if old != registration:
                            raise Blocked("suite ID immutable; no source or threshold overwrite")
                        return old
                    # An ID cannot be relabeled as another source or split.
                    for a in registration["source_mapping"]:
                        for b in old["source_mapping"]:
                            if (a["id"] == b["id"] or (a["source"], a["source_task_id"]) == (b["source"], b["source_task_id"])) and a != b:
                                raise Blocked("immutable source mapping changed")
                if e["kind"] in {"register", "function_io_register"}:
                    prior = e["data"]["suite"]["members"] if e["kind"] == "register" else e["data"]["members"]
                    if any(_overlap(a, b) and a["split"] != b["split"] for a in registration["members"] for b in prior):
                        raise Blocked("member/content/family crosses registered splits")
            return self._append("function_io_register", registration, events)

    def freeze_function_io(self, registration, binding):
        with self._lock():
            events = self._events()
            if not any(e["kind"] == "function_io_register" and e["data"] == registration for e in events):
                raise Blocked("suite not registered")
            if registration["purpose"] == "final":
                for e in events:
                    if e["kind"] == "function_io_freeze" and e["data"]["purpose"] == "final":
                        old = e["data"]["members"]
                    elif e["kind"] == "freeze":
                        old = next(r["data"]["suite"]["members"] for r in events if r["kind"] == "register" and r["data"]["suite"]["suite_id"] == e["data"]["suite_id"])
                        old = [m for m in old if m["split"] == "final"]
                    else:
                        continue
                    if any(_overlap(a, b) for a in registration["members"] for b in old):
                        raise Blocked("final member/family already reserved across candidates; genuinely new final data required")
            value = {"reservation_id": uuid.uuid4().hex, "suite_id": registration["suite_id"],
                     "purpose": registration["purpose"], "members": registration["members"],
                     "binding": binding, "reservation_hash": digest(binding), "state": "frozen_final" if registration["purpose"] == "final" else "frozen_development",
                     "boundary": BOUNDARY}
            return self._append("function_io_freeze", value, events)

    def begin_function_io(self, reservation, binding):
        with self._lock():
            events = self._events()
            if not any(e["kind"] == "function_io_freeze" and e["data"] == reservation for e in events):
                raise Blocked("unknown reservation")
            if digest(binding) != reservation["reservation_hash"]:
                raise Blocked("immutable reservation dependency changed")
            if any(e["kind"] == "function_io_begin" and e["data"]["reservation_id"] == reservation["reservation_id"] for e in events):
                raise Blocked("reservation consumed, including interrupted attempts")
            return self._append("function_io_begin", {"run_id": uuid.uuid4().hex, "reservation_id": reservation["reservation_id"],
                "reservation_hash": reservation["reservation_hash"], "state": "consumed", "boundary": BOUNDARY}, events)

    def finish_observed(self, run, observation):
        with self._lock():
            events = self._events()
            if not any(e["kind"] == "function_io_begin" and e["data"] == run for e in events):
                raise Blocked("unknown observed run")
            if any(e["kind"] == "function_io_finish" and e["data"]["run_id"] == run["run_id"] for e in events):
                raise Blocked("run already finished")
            return self._append("function_io_finish", {**run, **observation, "trust": "observed_public_cli",
                "same_owner_evidence": True, "certificate_issued": False}, events)


_CANCELLATION = ContextVar("governed_cancellation", default=None)


@contextmanager
def _managed_signals():
    """Defer cancellation through cleanup and append-only terminal persistence.

    Never raise in Popen's child-created/handle-returned window or ledger writes.
    Nested calls share the first signal. Main-thread ownership is required;
    failure happens before consumption. SIGKILL cannot persist a terminal event.
    """
    state = _CANCELLATION.get()
    if state is not None:
        yield state
        return
    state = {"signal": None}
    old = {}
    token = _CANCELLATION.set(state)
    def cancel(signum, frame):
        state["signal"] = state["signal"] or signum
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old[sig] = signal.signal(sig, cancel)
        yield state
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        _CANCELLATION.reset(token)


def _managed_call(function):
    @wraps(function)
    def call(*args, **kwargs):
        with _managed_signals():
            return function(*args, **kwargs)
    return call


def _child_exited(proc):
    # WNOWAIT pins the child PID until final group cleanup, avoiding recycled
    # PID/PGID kills. No process-name/global scans.
    return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _stop_cli(proc):
    """Allow compose to reap isolated workers, then kill same-group leftovers."""
    try:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        end = time.monotonic() + 5
        while not _child_exited(proc) and time.monotonic() < end:
            time.sleep(0.02)
    finally:
        # A normally exited leader may leave children. Keep it unreaped until
        # after the LAST group signal, so its group identity cannot be reused.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


@_managed_call
def _public_cli(argv, directory, seconds):
    """Only fixed public compose, bounded time/output and managed cancellation."""
    if argv[:3] != [sys.executable, "-m", "asea.compose"]:
        raise Blocked("Only the fixed public compose CLI is allowed")
    started = time.monotonic()
    stdout, stderr = directory / "stdout.txt", directory / "stderr.txt"
    state = _CANCELLATION.get()
    cause, proc = None, None
    with stdout.open("xb") as out, stderr.open("xb") as err:
        try:
            if state["signal"] is None:
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                        close_fds=True, start_new_session=True)
                while not _child_exited(proc):
                    if state["signal"] is not None:
                        cause = "interrupted"
                        break
                    if time.monotonic() - started >= seconds:
                        cause = "overall_timeout"
                        break
                    if stdout.stat().st_size + stderr.stat().st_size > 8 * 1024 * 1024:
                        cause = "output_budget_exceeded"
                        break
                    time.sleep(0.05)
        finally:
            if proc is not None:
                _stop_cli(proc)
    if state["signal"] is not None:
        cause = "interrupted"
    return {"argv": argv, "exit_code": proc.returncode if proc else None,
            "child_pid": proc.pid if proc else None, "termination_cause": cause,
            "interruption_signal": state["signal"], "elapsed_seconds": time.monotonic() - started,
            "stdout": str(stdout), "stdout_hash": file_hash(stdout), "stderr": str(stderr), "stderr_hash": file_hash(stderr),
            "descendant_closure": "known_cli_group_killed_compose_worker_cleanup_requested_not_escape_proof"}


def _observe(compose_root, receipt, binding, suite):
    """Verify signed records and preserve actual gate status, not just exit zero."""
    result = {"status": "blocked", "quality_admission": False, "signed_records": [],
              "counts": {"cases": len(suite.cases), "measured": 0, "passed": 0, "failed": 0, "missing": len(suite.cases)}}
    try:
        envelope = json.loads(read_bytes(receipt["stdout"], 8 * 1024 * 1024))
        result["public_cli_outcome"] = envelope
        evaluation = envelope.get("result", {})
        if envelope.get("command") != "evaluate" or not evaluation.get("id"):
            raise Blocked("public CLI produced no signed evaluation")
        if not (compose_root / "composition-workspace-v1.json").is_file():
            raise Blocked("public CLI did not initialize composition workspace")
        workspace = Workspace(compose_root)
        if workspace.read_record("evaluations", evaluation["id"]) != evaluation:
            raise Blocked("stdout differs from signed evaluation")
        candidate = workspace.read_record("candidates", evaluation["candidate_id"])
        if (evaluation["graph_hash"] != binding["candidate"]["graph_hash"] or candidate["spec"] != binding["candidate"]["spec"]
                or evaluation["suite_hash"] != digest(suite.model_dump(mode="json"))
                or evaluation["suite"] != suite.model_dump(mode="json")):
            raise Blocked("signed candidate/suite differs from frozen binding")
        config = evaluation["execution_config"]
        if any(config[k] != binding["resources"][k] for k in ("resource_profile", "memory_mib", "case_timeout")):
            raise Blocked("signed execution limits differ")
        implementation = dict(binding["candidate"]["implementation"])
        implementation.pop("governance_hashes")
        if evaluation["implementation_fingerprint"] != implementation:
            raise Blocked("signed implementation differs")
        records = [("evaluations", evaluation["id"]), ("candidates", evaluation["candidate_id"])]
        for node_id, expected in binding["candidate"]["inventories"].items():
            artifact_id = candidate["artifacts"][node_id]
            if workspace.read_record("artifacts", artifact_id) != expected:
                raise Blocked("signed artifact differs from frozen inventory")
            records.append(("artifacts", artifact_id))
        rows = evaluation["case_evidence"]
        if len({r["case_id"] for r in rows}) != len(rows) or not {r["case_id"] for r in rows} <= {c.id for c in suite.cases}:
            raise Blocked("invalid measured case coverage")
        for row in rows:
            run = workspace.read_record("runs", row["run_id"])
            if run["candidate_id"] != evaluation["candidate_id"] or run["graph_hash"] != evaluation["graph_hash"]:
                raise Blocked("signed run binding mismatch")
            records.append(("runs", row["run_id"]))
        # Failed workers can leave signed runs absent from case_evidence. Keep
        # those too, without counting them as completed grades.
        for path in sorted((compose_root / "runs").glob("*.json")):
            run = workspace.read_record("runs", path.stem)
            if run["candidate_id"] != evaluation["candidate_id"] or run["graph_hash"] != evaluation["graph_hash"]:
                raise Blocked("unrelated signed run in fresh composition root")
            if ("runs", path.stem) not in records:
                records.append(("runs", path.stem))
        result["signed_records"] = [{"collection": c, "id": i, "path": str(compose_root / c / (i + ".json")),
                                     "file_hash": file_hash(compose_root / c / (i + ".json"))} for c, i in records]
        archive = Path(receipt["stdout"]).parent / "signed-records.json"
        exclusive_json(archive, [{"collection": c, "id": i, "envelope": load_json(compose_root / c / (i + ".json"))} for c, i in records])
        result["signed_records_archive"] = {"path": str(archive), "file_hash": file_hash(archive)}
        passed = sum(r.get("passed") is True for r in rows)
        result["counts"] = {"cases": len(suite.cases), "measured": len(rows), "passed": passed,
                            "failed": len(rows) - passed, "missing": len(suite.cases) - len(rows)}
        result["status"] = evaluation["status"]
        result["completion_status"] = "done" if evaluation["status"] in {"admitted", "rejected"} else "backend_blocked"
        result["quality_admission"] = (evaluation["status"] == "admitted" and receipt["exit_code"] == 0
            and not evaluation["fixture_only"] and len(rows) == len(suite.cases) and passed == len(rows)
            and receipt["termination_cause"] is None)
        if receipt["termination_cause"]:
            result.update(status="blocked", completion_status="backend_blocked", quality_admission=False)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result.update(status="blocked", completion_status="backend_blocked", quality_admission=False, observation_error=str(exc))
    return result


@_managed_call
def governed_evaluate(*, workspace, compose_workspace, spec, suite, purpose, resource_profile="process_as",
                      memory_mib=512, case_timeout=60.0, max_seconds=3600.0, dataset_manifest=None, registry=None):
    """Freeze + consume + observe one public compose evaluation. No activation.

    Lifecycle patch is fingerprinted by governance_hashes and execution hashes;
    prior development evidence is stale, never rewritten or upgraded in place.
    """
    if sys.platform != "linux" or not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        raise Blocked("Managed process_as lifecycle requires Linux waitid/WNOWAIT; no launch or fallback")
    if resource_profile != "process_as" or type(memory_mib) is not int or not 32 <= memory_mib <= 1048576:
        raise Blocked("process_as and memory_mib in [32,1048576] required")
    if any(type(n) not in (int, float) or not math.isfinite(n) or not 0 < n <= 86400 for n in (case_timeout, max_seconds)):
        raise Blocked("finite case/overall timeout in (0,86400] required")
    store, compose_root = safe_path(workspace), safe_path(compose_workspace)
    # Legacy registry workspace and root must be disjoint. The governed store is
    # the explicit root, with a logical sibling identity (never a model workspace).
    ledger = GovernedRegistry(store.with_name(store.name + ".identity"), registry or store)
    if any(a == b or a in b.parents or b in a.parents for a, b in ((store, compose_root), (ledger.root, compose_root))):
        raise Blocked("validation and compose workspaces must be disjoint")
    if compose_root.exists() and (not compose_root.is_dir() or any(compose_root.iterdir())):
        raise Blocked("compose workspace must be a new empty root")
    evaluation_suite, registration = adapt_pilot(suite, purpose, dataset_manifest)
    candidate = candidate_binding(spec)
    for case in evaluation_suite.cases:
        case.validate_input_type(CompositionSpec.model_validate(candidate["spec"]))
    resources = {"resource_profile": resource_profile, "memory_mib": memory_mib, "case_timeout": case_timeout,
                 "max_seconds": max_seconds, "termination_grace_seconds": 10,
                 "memory_semantics": "per_process_virtual_address_space_not_RSS", "output_budget_bytes": 8 * 1024 * 1024}
    binding = {"adapter": ADAPTER, "registration": registration, "candidate": candidate, "resources": resources,
               "stopping_rule": "one_attempt_no_retry_all_cases_or_backend_block", "thresholds": {c.id: c.threshold for c in evaluation_suite.cases}}
    ledger.register_function_io(registration)
    reservation = ledger.freeze_function_io(registration, binding)
    directory = ledger.root / ("observed-" + reservation["reservation_id"])
    directory.mkdir()
    exclusive_json(directory / "spec.json", candidate["spec"])
    exclusive_json(directory / "suite.json", evaluation_suite.model_dump(mode="json"))
    # Re-hash dependencies immediately before consumption; never launch changed weights.
    if candidate_binding(spec) != candidate or adapt_pilot(suite, purpose, dataset_manifest)[1] != registration:
        raise Blocked("dependencies changed after freeze; not consumed, reservation remains frozen")
    argv = [sys.executable, "-m", "asea.compose", "--workspace", str(compose_root), "evaluate",
            "--spec", str(directory / "spec.json"), "--suite", str(directory / "suite.json"),
            "--resource-profile", resource_profile, "--memory-mib", str(memory_mib), "--case-timeout", str(case_timeout)]
    if _CANCELLATION.get()["signal"] is not None:
        raise Blocked("Interrupted before consumption; no public child launched")
    run = ledger.begin_function_io(reservation, binding)  # consumes BEFORE launch / first final item
    receipt = {"argv": argv, "exit_code": None, "termination_cause": "not_launched"}
    outcome = None
    try:
        receipt = _public_cli(argv, directory, max_seconds)
        outcome = _observe(compose_root, receipt, binding, evaluation_suite)
        if _CANCELLATION.get()["signal"] is None and candidate_binding(spec) != candidate:
            outcome.update(status="blocked", completion_status="backend_blocked", quality_admission=False, observation_error="candidate changed during execution")
    except (Exception, KeyboardInterrupt) as exc:
        receipt["observation_error"] = str(exc) or type(exc).__name__
        if outcome is None:
            outcome = {"counts": {"cases": len(evaluation_suite.cases), "measured": 0, "passed": 0, "failed": 0, "missing": len(evaluation_suite.cases)}}
        outcome.update(status="blocked", completion_status="backend_blocked", quality_admission=False)
    if _CANCELLATION.get()["signal"] is not None:
        receipt.update(termination_cause="interrupted", interruption_signal=_CANCELLATION.get()["signal"])
        outcome.update(status="interrupted", completion_status="interrupted", quality_admission=False)
    # Preserve outputs and partial compose records without inventing grades.
    receipt["preserved_outputs"] = [{"path": str(path), "file_hash": file_hash(path)}
        for path in (directory / "stdout.txt", directory / "stderr.txt") if path.is_file()]
    outcome["partial_records_root"] = str(compose_root)
    outcome["partial_records_are_quality_evidence"] = False
    outcome["unverified_record_files"] = []
    for collection in ("candidates", "artifacts", "runs", "evaluations"):
        for path in sorted((compose_root / collection).glob("*.json")):
            item = {"path": str(path), "verified": False}
            try:
                item["file_hash"] = file_hash(safe_file(path))
            except (ValueError, OSError) as exc:
                item["observation_error"] = str(exc)
            outcome["unverified_record_files"].append(item)
    exclusive_json(directory / "observation.json", {"receipt": receipt, "outcome": outcome})
    finish = ledger.finish_observed(run, {"status": outcome["status"], "quality_admission": outcome["quality_admission"],
        "termination_cause": receipt["termination_cause"], "interruption_signal": receipt.get("interruption_signal"),
        "counts": outcome["counts"], "observation": str(directory / "observation.json"), "observation_hash": file_hash(directory / "observation.json")})
    return {"adapter": ADAPTER, "reservation_id": reservation["reservation_id"], "reservation_hash": reservation["reservation_hash"],
            "run_id": run["run_id"], "state": "consumed", "receipt": receipt, "outcome": outcome,
            "trust": finish["trust"], "boundary": BOUNDARY, "certificate_issued": False}
