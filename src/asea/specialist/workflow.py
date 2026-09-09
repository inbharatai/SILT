"""Bounded, append-only local specialist study orchestration (no ML imports)."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from asea.artifacts import file_hash, safe_file, safe_path
from .evaluation import json_hash, load_suite, representation_inventory
from .recovery import _bounded_json, _read_samples

CAPTURE_LIMIT = 1024 * 1024
REPORT_LIMIT = 16 * 1024 * 1024
COMPACT_LIMIT = 64 * 1024


def _pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def compact_result(command, result):
    """Explicit projections only: arrays/configs/inventories stay in full evidence."""
    if command == "reconstruct":
        summary = _pick(result, ("schema_version", "artifact_kind", "status", "engineering_complete",
                                 "method", "runtime"))
        for key, fields in {
            "reduction": ("parameters_removed", "parameter_fraction", "tensor_bytes_removed", "safetensors_bytes_removed"),
            "calibration": ("file", "sha256", "size", "samples_available_used", "max_samples", "max_length", "actual_max_length"),
            "verification": ("native_forward_unchanged", "hook_parity", "parity_comparison", "cache_during_observation",
                "samples_used", "reconstructed_forward_finite", "serialized_state_shapes_and_keys_strict",
                "source_files_unchanged", "source_loading_strict", "capability_evaluated", "promotion_performed"),
        }.items():
            if key in result:
                summary[key] = _pick(result[key], fields)
        if "counts" in result:
            summary["counts"] = {key: _pick(value, ("parameters", "tensor_bytes", "safetensors_bytes"))
                                 for key, value in result["counts"].items() if key in ("source", "output")}
        for key, fields in (("source", ("family", "architecture")), ("output", ("architecture", "model_type",
                "standalone", "teacher_required_at_serve", "safe_serialization", "max_shard_size"))):
            if key in result:
                value = result[key]
                summary[key] = _pick(value, fields)
                for field in ("files", "config"):
                    if field in value:
                        summary[key][field + "_sha256"] = json_hash(value[field])
        if "selected_indices" in result:
            selected = result["selected_indices"]
            summary["selection_summary"] = {"groups": len(selected),
                "selected_indices_total": sum(len(v) for v in selected.values()),
                "selected_indices_sha256": json_hash(selected)}
        for field, name in (("weights_provenance", "weights_provenance_entries"),
                            ("removed_source_tensors", "removed_source_tensors_count")):
            if field in result:
                summary[name] = len(result[field])
    elif command == "recover":
        summary = _pick(result, ("schema_version", "status", "engineering_complete", "artifact_admitted",
            "method", "export_mode", "requested_steps", "actual_steps", "seed", "device", "family",
            "teacher_model_type", "teacher_forward_calls", "validation_teacher_forward_calls",
            "trainable_parameters", "total_parameters_with_adapter", "adapter_delta_l2", "source_unchanged",
            "student_store_unchanged", "teacher_frozen_no_grad", "teacher_unloaded_before_student",
            "standalone_native", "standalone_teacher_independent", "factor_preserving", "merged",
            "standalone_tokenizer_verified", "standalone_inventory_and_native_keys_verified", "elapsed_seconds"))
        for key, fields in {
            "standalone_reload_probe": ("passed", "atol", "rtol", "forward_equal", "generation_equal"),
            "merge_parity_probe": ("passed", "atol", "rtol", "max_abs_error"),
            "teacher_bank": ("required_at_serve", "external_cache_accepted", "removed_before_export"),
            "training": ("response_tokens", "steps"),
        }.items():
            if key in result:
                summary[key] = _pick(result[key], fields)
        for key in ("source_hashes_before", "source_hashes_after", "student_store_hashes_before",
                    "student_store_hashes_after", "output_weight_sha256"):
            if key in result:
                summary[key + "_sha256"] = json_hash(result[key])
    else:
        raise ValueError("compact receipts support reconstruction/recovery only")
    if "error" in result:
        summary["error"] = _pick(result["error"], ("type", "message"))
    return summary


def receipt_pointer(prefix, path):
    path = safe_file(path)
    if path.stat().st_size > REPORT_LIMIT:
        raise ValueError("receipt evidence exceeds 16 MiB")
    info = file_hash(path)
    if info["size"] > REPORT_LIMIT:
        raise ValueError("receipt evidence exceeds 16 MiB")
    return {prefix + "_file": str(path), prefix + "_sha256": info["sha256"], prefix + "_bytes": info["size"]}


def compact_receipt(receipt, command, report_path, output_dir):
    """Project only after full receipt publication; hashes are not quality proof."""
    value = _pick(receipt, ("schema_version", "command", "status", "completed", "engineering_complete",
                            "error", "error_type", "operational_failure"))
    value.update(stdout_contract=command + "_compact_v1", result=compact_result(command, receipt.get("result", {})))
    value.update(receipt_pointer("full_receipt", report_path))
    if receipt.get("completed") is True:
        name = "reconstruction_manifest.json" if command == "reconstruct" else "recovery_report.json"
        prefix = "reconstruction_manifest" if command == "reconstruct" else "recovery_manifest"
        value.update(receipt_pointer(prefix, safe_path(output_dir) / name))
    history = receipt.get("result", {}).get("training_history", receipt.get("result", {}).get("recovery_history"))
    if history is not None:
        value["result"].update(receipt_pointer("recovery_history", str(report_path) + ".history.json"))
        value["result"]["recovery_history_rows"] = len(history)
    compact_payload(value)  # No quiet truncation of proof or unexpected growth.
    return value


def compact_payload(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload.encode("utf-8")) + 1 > COMPACT_LIMIT:
        raise ValueError("compact receipt exceeds 64 KiB")
    return payload


def _read_verified_receipt(value, expected):
    """Read/hash the SAME bounded bytes; never follow the child-provided path."""
    import hashlib
    import stat
    from .recovery import _unique_pairs
    expected = safe_file(expected)
    if (type(value.get("full_receipt_file")) is not str or value["full_receipt_file"] != str(expected)
            or type(value.get("full_receipt_bytes")) is not int
            or not 0 < value["full_receipt_bytes"] <= REPORT_LIMIT):
        raise ValueError("compact full receipt path/size mismatch")
    fd = os.open(str(expected), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > REPORT_LIMIT:
            raise ValueError("full receipt exceeds 16 MiB or is not regular")
        raw = handle.read(REPORT_LIMIT + 1)
        after = os.fstat(handle.fileno())
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or len(raw) != value["full_receipt_bytes"] or len(raw) > REPORT_LIMIT
            or hashlib.sha256(raw).hexdigest() != value.get("full_receipt_sha256")):
        raise ValueError("compact full receipt hash/size mismatch")
    def nonfinite(_):
        raise ValueError("nonfinite full receipt")
    # Unlike dataset readers, channel selection arrays may exceed 10,000 entries.
    report = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=nonfinite)
    if type(report) is not dict or type(report.get("result", {})) is not dict:
        raise ValueError("invalid full receipt shape")
    return report


def _resolve_compact_receipt(argv, value):
    """A terminal stage stores only a verified projection, never a full manifest."""
    requested = "--receipt-mode" in argv and argv[argv.index("--receipt-mode") + 1] == "compact"
    if not requested and "stdout_contract" not in value:
        return value  # Legacy full receipts retain their existing contract/cap.
    command = argv[0]
    if (not requested or command not in ("reconstruct", "recover") or "--report" not in argv
            or value.get("stdout_contract") != command + "_compact_v1"):
        raise ValueError("unexpected compact receipt contract")
    compact_payload(value)
    expected = safe_path(argv[argv.index("--report") + 1])
    # Pre-publication/serialization failures must never be promoted to success.
    if value.get("transport_error") is True:
        if value.get("completed") is not False or value.get("engineering_complete") is not False:
            raise ValueError("invalid compact transport failure")
        failure = _pick(value, ("schema_version", "command", "status", "completed", "engineering_complete",
                               "stdout_contract", "transport_error", "error", "error_type", "error_truncated", "operational_failure"))
        if failure.get("status") not in ("BLOCKED", "REJECTED"):
            raise ValueError("invalid compact failure status")
        if "full_receipt_file" in value:
            _read_verified_receipt(value, expected)
            failure.update(receipt_pointer("full_receipt", expected))
        return failure
    full = _read_verified_receipt(value, expected)
    if full.get("command") != command or type(full.get("completed")) is not bool:
        raise ValueError("full receipt command/completed mismatch")
    projected = compact_receipt(full, command, expected, argv[argv.index("--output-dir") + 1])
    if value != projected:
        raise ValueError("compact receipt differs from verified full evidence")
    return projected


class StageBlocked(ValueError):
    """Unavailable execution, not a completed model-quality failure."""


class ChildFailure(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def write_json(path, value):
    """Exclusive, private publication: no successful or rejected report overwritten."""
    path = safe_path(path)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > REPORT_LIMIT:
        raise ValueError("report exceeds 16 MiB")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return str(path)


def _fields(value, allowed, required=()):
    if type(value) is not dict or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError("invalid/unknown recipe fields; allowed: " + ", ".join(sorted(allowed)))


def _number(value, low, high, integer=False):
    import math
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("invalid bounded numeric recipe value")


def recipe_config(path):
    path = safe_file(path)
    value = _bounded_json(path)
    required = {"source_path", "calibration", "training", "validation_data", "validation_suite"}
    allowed = required | {"schema_version", "dtype", "seed", "encoding", "max_length", "max_new_tokens",
        "output_budget_bytes", "timeout_seconds", "reconstruction", "recovery", "data_manifest", "selection_lock", "source_metadata",
        "memory_budget_bytes", "source_quality_floor"}
    _fields(value, allowed, required)
    result = dict(schema_version=1, dtype="bfloat16", seed=17, encoding="native_recovery_v1",
        max_length=256, max_new_tokens=256, output_budget_bytes=4 * 1024**3,
        timeout_seconds=3600, source_metadata={}, memory_budget_bytes=None, source_quality_floor=1.0)
    result.update(value)
    if type(result["schema_version"]) is not int or result["schema_version"] != 1 or result["encoding"] != "native_recovery_v1":
        raise ValueError("unsupported recipe schema/encoding")
    if result["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("unsupported dtype")
    for key, low, high in (("seed", 0, 2**32 - 1), ("max_length", 4, 2048),
            ("max_new_tokens", 1, 384), ("output_budget_bytes", 1, 64 * 1024**3), ("timeout_seconds", 1, 3600)):
        _number(result[key], low, high, True)
    if result["memory_budget_bytes"] is not None:
        _number(result["memory_budget_bytes"], 1, 2**63 - 1, True)
    _number(result["source_quality_floor"], 0, 1)
    for key in required | {"data_manifest", "selection_lock"}:
        if key in result:
            if type(result[key]) is not str or not result[key]:
                raise ValueError(key + " must be a local path string")
            candidate = Path(result[key])
            result[key] = str(safe_path(candidate if candidate.is_absolute() else path.parent / candidate))
    data_root = Path(result["training"]).parent
    result.setdefault("data_manifest", str(data_root / "manifest.json"))
    result.setdefault("selection_lock", str(data_root / "selection-lock.json"))
    _fields(result["source_metadata"], {"canonical_id", "revision", "acquisition_pr", "acquisition_url", "license"})
    if any(type(v) is not str or not v.strip() or len(v) > 4096 for v in result["source_metadata"].values()):
        raise ValueError("source metadata values must be nonblank bounded strings")
    reconstruction = result.get("reconstruction", {})
    _fields(reconstruction, {"family", "method", "retention", "max_samples"})
    result["reconstruction"] = dict(family="auto", method="activation", retention=0.75, max_samples=32, **{})
    result["reconstruction"].update(reconstruction)
    r = result["reconstruction"]
    if r["family"] not in ("auto", "qwen2", "switch_transformers") or r["method"] not in ("activation", "magnitude", "uniform"):
        raise ValueError("unsupported reconstruction method/family")
    _number(r["retention"], 0.000001, 0.999999)
    _number(r["max_samples"], 1, 4096, True)
    recovery = result.get("recovery", {})
    _fields(recovery, {"steps", "rank", "learning_rate", "kd_weight", "method", "max_samples", "teacher_mode", "export_mode"})
    result["recovery"] = dict(steps=64, rank=8, learning_rate=0.0001, kd_weight=0.7, method="lora_kd", max_samples=256, teacher_mode="cached", export_mode="native_merged")
    result["recovery"].update(recovery)
    r = result["recovery"]
    for key, low, high in (("steps", 1, 4096), ("rank", 1, 128), ("max_samples", 1, 4096)):
        _number(r[key], low, high, True)
    if r["export_mode"] not in ("native_merged", "factor_preserving"):
        raise ValueError("export_mode must be native_merged or factor_preserving; no fallback")
    if r["teacher_mode"] not in ("cached", "resident"):
        raise ValueError("teacher_mode must be cached or resident")
    _number(r["learning_rate"], 1e-12, 0.1)
    _number(r["kd_weight"], 0, 1)
    if r["method"] not in ("lora_kd", "supervised_lora") or (r["method"] == "supervised_lora" and r["kd_weight"] != 0):
        raise ValueError("invalid recovery method/KD combination")
    return result


def data_preflight(config, source_files):
    """Read permitted train/dev + answer-free lock only. Never open final artifacts."""
    manifest_path, lock_path = safe_path(config["data_manifest"]), safe_path(config["selection_lock"])
    permitted_paths = [safe_path(config[key]) for key in
                       ("training", "calibration", "validation_data", "validation_suite")]
    if any(any(part.lower().replace("-", "_").startswith(("final", "source_calibration_only"))
               for part in path.parts) for path in permitted_paths):
        raise ValueError("quarantined final/source-calibration-only paths cannot enter training or validation")
    if manifest_path.name != "manifest.json" or lock_path != manifest_path.parent / "selection-lock.json":
        raise ValueError("only designated metadata manifest.json and selection-lock.json may be opened")
    manifest, lock = _bounded_json(safe_file(manifest_path)), _bounded_json(safe_file(lock_path))
    if manifest.get("source_calibration_only") is True or manifest.get("usage") == "source_calibration_only":
        raise ValueError("source-calibration-only data is not eligible for a training build")
    if manifest.get("schema") != "silt.specialist.manifest.v1" or lock.get("schema") != "silt.specialist.selection-lock.v1":
        raise ValueError("versioned specialist data manifest and selection lock required")
    if lock.get("frozen_before_model_generation") is not True or lock.get("model_outputs_consulted") is not False:
        raise ValueError("selection must be frozen without model-output selection")
    if file_hash(lock_path)["sha256"] != manifest.get("selection_lock_sha256"):
        raise ValueError("selection lock hash mismatch")
    hashes = {str(manifest_path): file_hash(manifest_path), str(lock_path): file_hash(lock_path)}
    artifacts = manifest.get("artifact_sha256", {})
    # Explicit allowed split basenames also prevent swapping final for training.
    for key, name in (("training", "train.json"), ("calibration", "calibration.json"),
                      ("validation_data", "validation.json"), ("validation_suite", "validation-suite.json")):
        path = safe_path(config[key])
        if path != manifest_path.parent / name or file_hash(path)["sha256"] != artifacts.get(name):
            raise ValueError("permitted data path/hash mismatch: " + key)
        hashes[str(path)] = file_hash(path)
    rows = lock.get("selection", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty selection lock")
    index = {row["id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate selected IDs")
    consumed = lock.get("prior_consumed_selection", [])
    consumed_ids = {r["id"] for r in consumed}
    consumed_families = {r["family"] for r in consumed}
    families, ids = {}, {}
    for split in ("train", "validation", "final"):
        subset = [r for r in rows if r.get("split") == split]
        if not subset or any(not isinstance(r.get("family"), str) or not r["family"] for r in subset):
            raise ValueError("missing split or family identity")
        ids[split] = {r["id"] for r in subset}
        families[split] = {r["family"] for r in subset}
        if ids[split] & consumed_ids or families[split] & consumed_families:
            raise ValueError("previously consumed task/family selected")
    for a, b in (("train", "validation"), ("train", "final"), ("validation", "final")):
        if ids[a] & ids[b] or families[a] & families[b]:
            raise ValueError("cross-split ID/family overlap")
    train, train_ids, train_pairs = _read_samples(safe_file(config["training"]))
    dev, dev_ids, dev_pairs = _read_samples(safe_file(config["validation_data"]))
    if train_ids & dev_ids or train_pairs & dev_pairs:
        raise ValueError("training/validation content overlap")
    # Inspect the whole files before any selection or weight load.
    for split, samples in (("train", train), ("validation", dev)):
        if {r["id"] for r in samples} != ids[split]:
            raise ValueError("sample IDs do not match frozen selection")
        for row in samples:
            selected = index[row["id"]]
            if row.get("family") != selected["family"]:
                raise ValueError("sample family differs from selection lock")
            for key in ("response_sha256", "canonical_prompt_sha256", "canonical_source_sha256", "source_record_sha256", "alpha_source_sha256"):
                if key in selected and row.get(key) != selected[key]:
                    raise ValueError("sample provenance differs from selection lock")
    train_by_id = {r["id"]: r for r in train}
    calibration = _read_samples(safe_file(config["calibration"]))[0]
    if any(row != train_by_id.get(row["id"]) for row in calibration):
        raise ValueError("calibration must be an exact TRAIN-only subset")
    suite = load_suite(config["validation_suite"])
    if {c.id for c in suite.cases} != ids["validation"]:
        raise ValueError("validation suite IDs differ from validation data")
    dev_by_id = {r["id"]: r for r in dev}
    if any(c.input != dev_by_id[c.id]["prompt"] for c in suite.cases):
        raise ValueError("suite prompt must equal supervised native-chat user prompt")
    tokenizer_pins = manifest.get("environment", {}).get("tokenizer_files_sha256", {})
    # Current manifest uses tokenizer_check; discover only named non-final metadata.
    for section in ("tokenizer", "tokenizer_check", "environment"):
        tokenizer_pins = manifest.get(section, {}).get("tokenizer_files_sha256", tokenizer_pins)
    if tokenizer_pins and any(source_files.get(k, {}).get("sha256") != v for k, v in tokenizer_pins.items()):
        raise ValueError("source tokenizer differs from locked data encoding")
    return {"hashes": hashes, "counts": {s: len(ids[s]) for s in ids},
        "validation_tasks": len(suite.cases), "ids": {s: sorted(ids[s]) for s in ids},
        "families": {s: sorted(families[s]) for s in families},
        "final_opened": False, "prior_consumed_checked": True,
        "manifest": manifest, "selection_lock_sha256": file_hash(lock_path)["sha256"],
        "final_suite_path": str(manifest_path.parent / "final-suite.json"),
        "final_suite_sha256": artifacts.get("final-suite.json")}


class StageInterrupted(KeyboardInterrupt):
    """Managed termination, retained as incomplete evidence (never promotion)."""


@contextmanager
def _managed_signals():
    import threading
    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
        raise StageBlocked("stage supervision requires Linux and the main thread")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    if getattr(previous[signal.SIGTERM], "specialist_managed", False):
        yield
        return
    interrupted = False
    def stop(sig, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise StageInterrupted("interrupted by " + signal.Signals(sig).name)
    stop.specialist_managed = True
    try:
        for sig in previous:
            signal.signal(sig, stop)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@_managed_signals()
def run_child(argv, timeout):
    """Fixed public CLI, pre-site PDEATHSIG, bounded drain and known group."""
    if not argv or argv[0] not in ("reconstruct", "recover", "evaluate"):
        raise ValueError("unsupported child command")
    _number(timeout, 0.000001, 3600)
    worker = Path(__file__).resolve().with_name("stage_worker.py")
    command = [sys.executable, "-I", "-S", str(worker), str(os.getpid())] + [str(v) for v in argv]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    started = time.monotonic()
    deadline = started + timeout
    # Reserve cleanup inside the supplied remaining budget, not a fresh timeout.
    run_deadline = deadline - min(0.2, timeout / 10)
    failure, value, interrupted, process = None, None, False, None
    selector = selectors.DefaultSelector()
    # A signal between fork and assignment must not orphan an untracked worker.
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, start_new_session=True, shell=False)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        for name in buffers:
            pipe = getattr(process, name)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while True:
            # Peek, NEVER poll/wait here: retain the zombie leader to pin its PID
            # and process-group identity until all known group members are killed.
            exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exited is not None and not selector.get_map():
                break
            if time.monotonic() >= run_deadline:
                failure = "stage deadline exceeded"
                break
            for key, _ in selector.select(min(0.05, max(0, run_deadline - time.monotonic()))):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                remaining_capture = CAPTURE_LIMIT - sum(map(len, buffers.values()))
                buffers[key.data].extend(chunk[:remaining_capture])
                if len(chunk) > remaining_capture:
                    failure = "child output exceeds 1 MiB"
                    break
            if failure:
                break
    except BaseException as exc:
        interrupted = not isinstance(exc, Exception)
        failure = str(exc) or type(exc).__name__
    finally:
        # Block managed signals until group kill and leader reap have completed.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            selector.close()
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    failure = failure or "stage cleanup deadline exceeded (SIGKILL sent)"
                process.stdout.close()
                process.stderr.close()
        finally:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            except BaseException as exc:
                interrupted = True
                failure = str(exc) or type(exc).__name__
    try:
        value = json.loads(buffers["stdout"].decode("utf-8"))
        if not isinstance(value, dict):
            value = None
            raise ValueError("child stdout must contain exactly one JSON receipt")
    except (ValueError, UnicodeError) as exc:
        failure = failure or "invalid child receipt: " + str(exc)
    if not failure:
        try:
            value = _resolve_compact_receipt(argv, value)
        except Exception as exc:
            failure = "invalid compact child receipt: " + str(exc)[:2048]
            value = {"status": "BLOCKED", "completed": False, "engineering_complete": False,
                     "transport_validation_failed": True}
    evidence = {"returncode": process.returncode if process is not None else None, "receipt": value,
        "interrupted": interrupted,
        "worker_pid": process.pid if process is not None else None,
        "process_group": process.pid if process is not None else None,
        "launcher": "fixed_stage_worker_python_isolated_no_site",
        "stdout": buffers["stdout"].decode("utf-8", "replace"),
        "stderr": buffers["stderr"].decode("utf-8", "replace"),
        "stdout_bytes": len(buffers["stdout"]), "stderr_bytes": len(buffers["stderr"]),
        "stdout_sha256": __import__("hashlib").sha256(buffers["stdout"]).hexdigest(),
        "stderr_sha256": __import__("hashlib").sha256(buffers["stderr"]).hexdigest(),
        "resources": {"wall_seconds": time.monotonic() - started, "capture_limit_bytes": CAPTURE_LIMIT,
                      "measurement": "fresh_child_wall_clock; see receipt for process RSS"}}
    if failure:
        raise ChildFailure(failure, dict(evidence, error=failure))
    return evidence


def _options(values):
    args = []
    for key, value in values.items():
        if key == "receipt_mode" and (type(value) is not str or value not in ("full", "compact")):
            raise ValueError("receipt_mode must be full or compact")
        if value is None:
            continue
        if key == "memory_budget_bytes":
            from decimal import Decimal, localcontext
            with localcontext() as context:
                context.prec = 64
                key, value = "memory_budget_mib", Decimal(value) / Decimal(1024**2)
        args += ["--" + key.replace("_", "-"), str(value)]
    return args


def _check_unchanged(hashes):
    if any(file_hash(path) != before for path, before in hashes.items()):
        raise ValueError("permitted data/config changed during study")


@_managed_signals()
def _stage(root, name, argv, deadline):
    config = {"argv": [sys.executable, "-m", "asea.specialist"] + argv,
              "argv_sha256": json_hash(argv), "state": "RUNNING", "stage": name,
              "launcher": str(Path(__file__).resolve().with_name("stage_worker.py")),
              "launcher_flags": ["-I", "-S"]}
    write_json(root / (name + ".started.json"), config)
    result = {"stage": name, "state": "REJECTED", "completed": False, "returncode": None}
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StageBlocked("whole study deadline exceeded before child start")
        child = run_child(argv, remaining)
        result.update(child)
        receipt = child["receipt"]
        expected_status = "RECONSTRUCTED_UNVALIDATED" if argv[0] == "reconstruct" else "completed"
        summary = receipt.get("result") or {}
        if receipt.get("status", "").upper() == "BLOCKED" or summary.get("engineering_complete") is False:
            raise StageBlocked("child execution unavailable: " + str(receipt.get("status")))
        if child["returncode"] != 0 or receipt.get("completed") is not True or receipt.get("status") != expected_status:
            raise ValueError("child did not complete: " + str(receipt.get("status")))
        if argv[0] == "evaluate":
            output = Path(argv[argv.index("--output") + 1])
            evaluation = _bounded_json(output)
            declared = load_suite(argv[argv.index("--suite") + 1])
            total = evaluation.get("tasks_total")
            passed, failed = evaluation.get("tasks_passed"), evaluation.get("tasks_failed")
            if (evaluation.get("status") != "completed" or evaluation.get("completed") is not True
                    or evaluation.get("engineering_complete") is not True
                    or evaluation.get("operational_failures") != 0 or evaluation.get("tasks_blocked") != 0
                    or type(total) is not int or total <= 0
                    or type(passed) is not int or type(failed) is not int
                    or passed < 0 or failed < 0 or passed + failed != total
                    or evaluation.get("tasks_graded") != total or total != len(declared.cases)
                    or evaluation.get("pass_rate") != passed / total
                    or evaluation.get("quality_pass") is not (passed == total)):
                raise StageBlocked("evaluation lacks complete operational grades")
        result.update(state="COMPLETED", completed=True)
    except BaseException as exc:
        result["completed"] = False
        if not isinstance(exc, Exception):
            result.update(state="BLOCKED", interrupted=True)
        if isinstance(exc, ChildFailure):
            result.update(exc.evidence)
        if isinstance(exc, (ChildFailure, StageBlocked)) or isinstance(exc, (OSError, subprocess.SubprocessError)):
            result["state"] = "BLOCKED"
        result["error"] = str(exc)
    logs = {key: result.pop(key) for key in ("stdout", "stderr") if key in result}
    if logs:
        result["logs"] = write_json(root / (name + ".logs.json"), logs)
    write_json(root / (name + ".result.json"), result)
    if not result["completed"]:
        error = StageBlocked if result["state"] == "BLOCKED" else ValueError
        exc = error(name + " " + result["state"].lower() + "; inspect preserved stage report")
        exc.interrupted = result.get("interrupted", False)
        raise exc
    return result["receipt"]


def comparison(source, pruned, recovered):
    baseline = source["pass_rate"]
    return {"source": source["pass_rate"], "reconstructed": pruned["pass_rate"],
        "recovered": recovered["pass_rate"], "recovery_delta": recovered["pass_rate"] - pruned["pass_rate"],
        "observed_retention_ratio": recovered["pass_rate"] / baseline if baseline else None,
        "source_zero_denominator": baseline == 0, "quality_pass": recovered["quality_pass"],
        "noninferiority": "insufficient_evidence_no_power98_gate", "certificate": False,
        "scope": "fixed exercised short-function cohort only; no automatic promotion"}


def recovery_complete(report, export_mode):
    """Representation-specific engineering gate; parity is never quality evidence."""
    common = (report.get("status") == "completed" and report.get("artifact_admitted") is True
        and report.get("standalone_teacher_independent") is True
        and report.get("standalone_reload_probe", {}).get("passed") is True)
    if export_mode == "factor_preserving":
        return common and report.get("factor_preserving") is True and report.get("export_mode") == export_mode
    return common and export_mode == "native_merged" and report.get("standalone_native") is True


def implementation_manifest():
    """Freeze executable source/API identity, not just the model inventory."""
    import inspect
    from .reconstruction import reconstruct
    from .recovery import recover
    from .evaluation import infer, evaluate, validate
    from .standalone import inspect_bundle, load_standalone
    package = Path(__file__).resolve().parent
    files = [package / name for name in ("__init__.py", "__main__.py", "workflow.py", "stage_worker.py", "evaluation.py", "reconstruction.py", "recovery.py", "standalone.py", "controls.py")]
    files += [package.parent / "artifacts/__init__.py", package.parent / "certification/function_oracle.py",
              package.parent / "certification/sandbox.py", package.parent / "certification/__init__.py"]
    # This revision also freezes the transport contract and its regression tests.
    files += [package.parents[2] / "docs/SPECIALIST_WORKFLOW.md",
              package.parents[2] / "tests/test_specialist_workflow.py"]
    return {"source_files": {str(path): file_hash(path) for path in files},
            "api_signatures": {f.__name__: str(inspect.signature(f)) for f in (reconstruct, recover, infer, evaluate, validate, inspect_bundle, load_standalone)},
            "device_contract": "cpu_only; no GPU capability claim", "command_exit_is_certificate": False}


@_managed_signals()
def build(recipe, workspace):
    root = safe_path(workspace)
    if root.exists() or not root.parent.is_dir():
        raise ValueError("workspace must be a new immutable directory with existing parent")
    config = recipe_config(recipe)
    source = safe_path(config["source_path"])
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("workspace and source must be disjoint")
    root.mkdir(mode=0o700)
    started = time.monotonic()
    deadline = started + config["timeout_seconds"]
    report = {"schema_version": 1, "status": "REJECTED", "completed": False,
        "engineering_complete": False, "quality_pass": False, "certificate": False,
        "workspace": str(root), "config": config, "config_sha256": json_hash(config),
        "source_metadata": config["source_metadata"], "stages": [], "attempted_stages": [],
        "qualified_source": False, "source_qualification": "not_measured",
        "study_role": "RESEARCH", "device_contract": "cpu_only"}
    write_json(root / "recipe.json", config)
    try:
        implementation = implementation_manifest()
        report["implementation"] = implementation
        report["implementation_sha256"] = json_hash(implementation)
        write_json(root / "implementation-lock.json", implementation)
        source_config = _bounded_json(safe_file(source / "config.json"))
        if source_config.get("model_type") not in ("qwen2", "switch_transformers"):
            raise ValueError("build source must be a reconstructable native Qwen2 or Switch model")
        source_files = representation_inventory(source)
        report["source"] = {"path": str(source), "files": source_files,
                            "config": source_config, "metadata": config["source_metadata"]}
        data = data_preflight(config, source_files)
        report["data"] = data
        write_json(root / "preflight.json", {"status": "completed", "source": report["source"], "data": data})
        hashes = dict(data["hashes"], **{str(safe_file(recipe)): file_hash(recipe)})
        hashes.update(implementation["source_files"])
        common_eval = dict(suite=config["validation_suite"], dtype=config["dtype"],
                           max_new_tokens=config["max_new_tokens"], trace_policy="digest", memory_budget_bytes=config["memory_budget_bytes"])
        models = {"source": source, "reconstructed": root / "reconstructed", "recovered": root / "recovered"}
        evaluations = {}
        def stage(name, argv):
            _check_unchanged(hashes)
            if representation_inventory(source) != source_files:
                raise ValueError("source weights/metadata changed")
            report["attempted_stages"].append(name)
            value = _stage(root, name, argv, deadline)
            _check_unchanged(hashes)
            if representation_inventory(source) != source_files:
                raise ValueError("source teacher changed after stage")
            report["stages"].append(name)
            return value
        def eval_model(label):
            output = root / (label + "-validation.json")
            stage(label + "-validation", ["evaluate"] + _options(dict(common_eval, model=str(models[label]), output=str(output))))
            evaluations[label] = _bounded_json(output)
        eval_model("source")
        source_rate = evaluations["source"]["pass_rate"]
        report.update(qualified_source=source_rate is not None and source_rate >= config["source_quality_floor"],
            source_qualification="observed_validation_floor_only_not_teacher_certificate",
            source_baseline={"role": "source_validation_prerequisite", "pass_rate": source_rate,
                "required_floor": config["source_quality_floor"], "training_policy": "continue_research_if_operationally_complete"})
        rargs = dict(source_dir=str(source), output_dir=str(models["reconstructed"]), calibration_path=config["calibration"],
                     dtype=config["dtype"], seed=config["seed"], max_length=config["max_length"],
                     memory_budget_bytes=config["memory_budget_bytes"], report=str(root / "reconstruct.receipt.json"),
                     receipt_mode="compact", **config["reconstruction"])
        stage("reconstruct", ["reconstruct"] + _options(rargs))
        reconstructed_files = representation_inventory(models["reconstructed"])
        eval_model("reconstructed")
        rargs = dict(teacher_dir=str(source), student_dir=str(models["reconstructed"]), output_dir=str(models["recovered"]),
            training_path=config["training"], validation_path=config["validation_data"], dtype=config["dtype"],
            seed=config["seed"], max_length=config["max_length"], memory_budget_bytes=config["memory_budget_bytes"],
            report=str(root / "recover.receipt.json"), receipt_mode="compact", **config["recovery"])
        stage("recover", ["recover"] + _options(rargs))
        recovery = _bounded_json(models["recovered"] / "recovery_report.json")
        if not recovery_complete(recovery, config["recovery"]["export_mode"]) or recovery.get("actual_steps") != config["recovery"]["steps"]:
            raise ValueError("recovery lacks completed real-step evidence")
        if representation_inventory(models["reconstructed"]) != reconstructed_files:
            raise ValueError("reconstruction control changed during recovery")
        recovered_files = representation_inventory(models["recovered"])
        eval_model("recovered")
        frozen = {name: {"path": str(path), "files": representation_inventory(path)} for name, path in models.items()}
        if frozen["reconstructed"]["files"] != reconstructed_files or frozen["recovered"]["files"] != recovered_files:
            raise ValueError("control or candidate changed before freeze")
        recovered_total_bytes = sum(v["size"] for v in frozen["recovered"]["files"].values())
        size = sum(v["size"] for k, v in frozen["recovered"]["files"].items() if k.endswith(".safetensors"))
        source_size = sum(v["size"] for k, v in source_files.items() if k.endswith(".safetensors"))
        if not 0 < size < source_size or recovered_total_bytes > config["output_budget_bytes"]:
            raise ValueError("standalone recovered artifact not smaller or exceeds output budget")
        report.update(status="BUILT_UNCERTIFIED", completed=True, engineering_complete=True,
            quality_pass=evaluations["recovered"]["quality_pass"], comparison=comparison(**evaluations_to_comparison(evaluations)),
            candidate_frozen=True, frozen_models=frozen, frozen_models_sha256=json_hash(frozen),
            source_unchanged=True, recovered_weight_bytes=size, source_weight_bytes=source_size,
            isolated_model_stages=True, recovered_total_bundle_bytes=recovered_total_bytes,
            representation=config["recovery"]["export_mode"],
            factor_preserving=recovery.get("factor_preserving", False),
            standalone_native=recovery.get("standalone_native", False),
            standalone_teacher_independent=recovery.get("standalone_teacher_independent", False),
            standalone_reload_probe=recovery["standalone_reload_probe"],
            outputs={"source": {"role": "immutable_source_control", "path": str(source)},
                "reconstructed": {"role": "unrecovered_control", "path": str(models["reconstructed"])},
                "candidate": {"role": "frozen_recovered_candidate_not_certified", "path": str(models["recovered"])},
                "validation": {"role": "development_only_not_final",
                    "files": {name: str(root / (name + "-validation.json")) for name in evaluations}}})
    except BaseException as exc:
        report["error"] = str(exc) or type(exc).__name__
        report.update(completed=False, engineering_complete=False, quality_pass=False)
        report.pop("candidate_frozen", None)
        if not isinstance(exc, Exception) or getattr(exc, "interrupted", False):
            report["interrupted"] = True
        if isinstance(exc, StageBlocked) or not isinstance(exc, Exception):
            report["status"] = "BLOCKED"
    report["wall_seconds"] = time.monotonic() - started
    write_json(root / "manifest.json", report)
    return report


def evaluations_to_comparison(evaluations):
    return {"source": evaluations["source"], "pruned": evaluations["reconstructed"], "recovered": evaluations["recovered"]}


@_managed_signals()
def finalize(study, suite, output):
    root, output = safe_path(study), safe_path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("final output must be new with existing parent")
    report = _bounded_json(root / "manifest.json")
    if report.get("status") != "BUILT_UNCERTIFIED" or report.get("candidate_frozen") is not True:
        raise ValueError("only a completed frozen candidate study may consume final")
    implementation = report.get("implementation")
    if not implementation or json_hash(implementation) != report.get("implementation_sha256"):
        raise ValueError("implementation lock missing or changed")
    _check_unchanged(implementation["source_files"])
    frozen = report["frozen_models"]
    if json_hash(frozen) != report["frozen_models_sha256"]:
        raise ValueError("frozen control metadata changed")
    for store in frozen.values():
        if representation_inventory(store["path"]) != store["files"]:
            raise ValueError("frozen candidate/source/pruned control changed")
    data = report["data"]
    # Do not open/hash suite until exclusive consumed marker is durable.
    if safe_path(suite) != safe_path(data["final_suite_path"]):
        raise ValueError("final suite must be the preregistered quarantined suite")
    write_json(root / "final-consumed.json", {"schema_version": 1, "consumed": True,
        "suite_path": str(safe_path(suite)), "expected_sha256": data["final_suite_sha256"],
        "frozen_models_sha256": report["frozen_models_sha256"], "output": str(output)})
    result = {"schema_version": 1, "status": "REJECTED", "completed": False,
        "engineering_complete": False, "quality_pass": False, "certificate": False,
        "final_consumed": True, "training_on_final": False,
        "qualified_source": report.get("qualified_source", False)}
    try:
        if file_hash(suite)["sha256"] != data["final_suite_sha256"]:
            raise ValueError("final suite hash mismatch (consumed; no retry)")
        loaded = load_suite(suite)
        if {c.id for c in loaded.cases} != set(data["ids"]["final"]):
            raise ValueError("final suite IDs differ from frozen selection")
        deadline = time.monotonic() + min(3600, report["config"]["timeout_seconds"])
        evaluations = {}
        for name in ("source", "reconstructed", "recovered"):
            _check_unchanged(implementation["source_files"])
            for store in frozen.values():
                if representation_inventory(store["path"]) != store["files"]:
                    raise ValueError("frozen control changed before final stage")
            path = root / (name + "-final.json")
            _stage(root, name + "-final", ["evaluate"] + _options(dict(model=frozen[name]["path"], suite=str(suite),
                output=str(path), dtype=report["config"]["dtype"], max_new_tokens=report["config"]["max_new_tokens"], trace_policy="digest",
                memory_budget_bytes=report["config"].get("memory_budget_bytes"))), deadline)
            _check_unchanged(implementation["source_files"])
            if file_hash(suite)["sha256"] != data["final_suite_sha256"]:
                raise ValueError("final suite changed during evaluation")
            evaluations[name] = _bounded_json(path)
            for store in frozen.values():
                if representation_inventory(store["path"]) != store["files"]:
                    raise ValueError("frozen model changed during final")
        result.update(status="BUILT_UNCERTIFIED", completed=True, engineering_complete=True,
            quality_pass=evaluations["recovered"]["quality_pass"],
            comparison=comparison(**evaluations_to_comparison(evaluations)),
            evaluation_files={n: str(root / (n + "-final.json")) for n in evaluations})
    except BaseException as exc:
        result["error"] = str(exc) or type(exc).__name__
        result.update(completed=False, engineering_complete=False, quality_pass=False)
        if not isinstance(exc, Exception) or getattr(exc, "interrupted", False):
            result["interrupted"] = True
        if isinstance(exc, StageBlocked) or not isinstance(exc, Exception):
            result["status"] = "BLOCKED"
    write_json(output, result)
    return result
