"""Measured, reference-bound admission. A successful run is never a certification."""
from difflib import SequenceMatcher
import ast
import contextlib
import importlib.metadata
import platform
import shutil
import sys
import hashlib
import json
import re
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile

from asea.artifacts import Blocked, atomic_json, canonical, digest, file_hash, safe_file, safe_path
from asea.compose.schema import CompositionSpec, EvaluationSuite, SelectionBudgets, PORTS


ADMISSION_POLICY = {"version": "composition-admission-v2", "text_exact": 1.0,
                    "functional_code": 1.0, "similarity_min": 0.8, "wer_max": 0.5,
                    "functional_scope": "OS-contained tests; same-interpreter harness is not adversarial correctness proof"}


def _check_policy(suite):
    # Validate even model_construct/model_copy inputs; callers cannot bypass floors.
    suite = EvaluationSuite.model_validate(suite.model_dump(mode="python"))
    if "safety" in suite.claims:
        raise Blocked("safety claims BLOCKED: no safety certification policy implemented")
    if "coding" in suite.claims and {c.group for c in suite.cases if c.metric in {"functional_code", "function_io"}} != {"target", "control"}:
        raise Blocked("coding claims require functional_code or function_io targets and controls; text/proxy metrics are not coding evidence")
    for case in suite.cases:
        if case.metric == "functional_code":
            _validate_trusted_tests(case.reference)
    return suite


def _validate_trusted_tests(reference):
    # Shape check only, not proof of test adequacy or harness tamper resistance.
    try:
        tree = ast.parse(reference)
    except SyntaxError as exc:
        raise Blocked("functional-code sandbox requires valid trusted unittest source") from exc
    if not any(isinstance(node, ast.ClassDef)
               and any((isinstance(base, ast.Attribute) and base.attr == "TestCase")
                       or (isinstance(base, ast.Name) and base.id == "TestCase") for base in node.bases)
               and any(isinstance(method, ast.FunctionDef) and method.name.startswith("test") for method in node.body)
               for node in tree.body):
        raise Blocked("functional-code sandbox requires nonempty trusted unittest TestCase methods")


def preprocess_code(actual):
    """Raw Python or exactly one python/py fenced block; never join blocks."""
    if "```" not in actual and "~~~" not in actual:
        return actual
    if actual.count("```") != 2 or "~~~" in actual:
        raise Blocked("functional-code preprocessing rejects multiple or malformed fenced blocks")
    match = re.search(r"```(?:python|py)\s*\n(.*?)\n```", actual, flags=re.DOTALL | re.IGNORECASE)
    if not match or not match.group(1).strip():
        raise Blocked("functional-code preprocessing requires a single fenced Python block")
    return match.group(1)


def extract_source(actual):
    """Raw source or one Python fence; no concatenation or host execution."""
    return preprocess_code(actual)


def _function_result(actual, cases, trace_policy=None):
    from .function_oracle import evaluate_functions
    try:
        source = extract_source(actual)
        if not source.strip():
            raise Blocked("empty generated source")
    except Blocked as exc:
        # Candidate formatting is a scored failure, not an unavailable backend.
        # No candidate execution or comparison is claimed for this case.
        return {"oracle_mode": "host_data_only_functions_v1", "status": "INVALID_SOURCE",
                "supported": False, "passed": False, "tests_total": len(cases),
                "tests_run": 0, "tests_passed": 0, "cases": [], "returncode": None,
                "resource_flags": [], "resources": {"limits_established": False},
                "reason": str(exc), "failure_stage": "candidate_preprocessing"}
    result = evaluate_functions(source, cases, trace_policy=trace_policy)
    if not result.get("supported") or result.get("status") == "BLOCKED":
        error = Blocked("function_io oracle unavailable: " + result.get("reason", "missing isolation receipt"))
        error.oracle_evidence = result
        raise error
    return result


def _function_passed(result, cases):
    # Candidate text and candidate-reported counters are never authority. Even
    # an inconsistent host result must not become a passing scalar.
    return (result.get("oracle_mode") == "host_data_only_functions_v1"
            and result.get("status") == "PASSED" and result.get("passed") is True
            and result.get("supported") is True and result.get("returncode") == 0
            and not result.get("resource_flags") and not result.get("reason")
            and result.get("resources", {}).get("limits_established") is True
            and all(type(result.get(key)) is int for key in ("tests_total", "tests_run", "tests_passed"))
            and result.get("tests_total") == len(cases)
            and result.get("tests_run") == len(cases)
            and result.get("tests_passed") == len(cases)
            and result.get("cases") == [{"id": c["id"], "matched": True} for c in cases])


def metric_score(metric, actual, reference, function_cases=None):
    if metric == "function_io":
        return float(_function_passed(_function_result(actual, function_cases), function_cases))
    if metric == "functional_code":
        _validate_trusted_tests(reference)
        from .sandbox import evaluate_code
        result = evaluate_code(preprocess_code(actual), reference)
        if not result.supported or result.status == "BLOCKED":
            raise Blocked("functional-code sandbox unavailable: " + result.reason)
        return float(result.status == "PASSED" and result.passed and result.tests_run > 0)
    if metric == "text_exact":
        return float(actual == reference)
    if metric == "text_similarity_proxy":
        # Bound quadratic string similarity and label it as a proxy, not code correctness.
        if max(len(actual), len(reference)) > 8192:
            raise Blocked("text similarity proxy limited to 8192 characters")
        return SequenceMatcher(None, reference, actual, autojunk=False).ratio()
    if metric == "word_error_rate":
        a, b = reference.split(), actual.split()
        if not a or max(len(a), len(b)) > 2048:
            raise Blocked("WER requires nonempty reference and at most 2048 words")
        previous = list(range(len(b) + 1))
        for i, token in enumerate(a, 1):
            current = [i]
            for j, other in enumerate(b, 1):
                current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (token != other)))
            previous = current
        return previous[-1] / len(a)
    raise Blocked("unsupported metric")


def _passes(metric, score, threshold):
    if not math.isfinite(score) or not math.isfinite(threshold):
        return False
    if metric in {"text_exact", "functional_code", "function_io"} and threshold != 1.0:
        return False
    if metric == "text_similarity_proxy" and not 0.8 <= threshold <= 1:
        return False
    if metric == "word_error_rate" and not 0 <= threshold <= 0.5:
        return False
    return score <= threshold if metric == "word_error_rate" else score >= threshold


def _validate_resources(measured, limits):
    wall, rss = measured["wall_seconds"], measured["process_peak_rss_mb"]
    if (type(wall) not in (float, int) or type(rss) not in (float, int)
            or not math.isfinite(wall) or not math.isfinite(rss)
            or not 0 < wall <= limits.max_seconds or not 0 < rss <= limits.max_peak_rss_mb
            or measured["device"] != "cpu" or type(measured["threads"]) is not int
            or not 1 <= measured["threads"] <= limits.threads):
        raise Blocked("missing or over-budget measured resources")


def _case_input(case, spec):
    """Reconstruct the runtime's input digest, including image bytes AND question."""
    case.validate_input_type(spec)
    if spec.input_type == "text":
        return {"type": "text", "text": case.input}
    from asea.compose.runtime import _audio_info, _image_info, DEFAULT_IMAGE_PROMPT
    if spec.input_type == "image":
        prompt = case.input if case.input is not None else DEFAULT_IMAGE_PROMPT
        if len(prompt) > spec.limits.max_input_chars:
            raise Blocked("evaluation image prompt exceeds graph character limit")
        return {"type": "image", **_image_info(case.input_file, spec.limits), "text": prompt}
    return {"type": "audio", **_audio_info(case.input_file, spec.limits)}


EXECUTION_CONTRACT = "composition-execution-v1"
ORACLE_SCOPES = {
    "function_io": {"mode": "host_data_only_functions_v1", "scope": "host compares observable typed outputs on these data cases only; not purity, algorithm identity or universal correctness"},
    "functional_code": {"mode": "legacy_same_interpreter_unittest", "scope": ADMISSION_POLICY["functional_scope"]},
}


OBSERVABILITY_SCHEMA = "obs_v1"


class SENSITIVE_TRACE_EXPORT_REQUIRES_OPT_IN(Blocked):
    """A valid original bundle contains potentially sensitive observation data."""


def _evaluation_trace_policy(value=None):
    from .function_oracle import _trace_policy
    if type(value) is str:
        value = {"schema_version": 1, "return_retention": value}
    return _trace_policy(value)


def _preflight_formats(spec, suite):
    from asea.compose.generation_contract import preflight_node
    declarations = [case.output_format for case in suite.cases if case.output_format is not None]
    # All nodes contribute to the selected output in this graph schema. Check
    # every explicit contract against ALL cases before the first inference.
    for node in spec.nodes:
        relevant = node.id == spec.output_node or node.output_contract is not None
        preflight_node(node, declarations if relevant else ())
        if declarations and node.id == spec.output_node and node.output_contract is None:
            raise Blocked("structured case output_format requires an explicit strict output_contract on the output node; use preview and author a new spec")


def _sensitive_observations(value, in_trace=False):
    """Scan actual payloads, not just flags; never invoke candidate repr/str.

    Case-insensitive conservative detection also protects malformed legacy data.
    This is a privacy gate, not a schema normalizer or admission authority.
    """
    if type(value) is list:
        return any(_sensitive_observations(item, in_trace) for item in value)
    if type(value) is not dict:
        return False
    for key, item in value.items():
        name = key.casefold() if type(key) is str else ""
        if (name in {"return_retention", "trace_policy"} or (in_trace and name == "retention")) and type(item) is str and item.casefold() == "value":
            return True
        if name == "evidence_flags" and type(item) is list and any(
                type(flag) is str and "sensitive" in flag.casefold() for flag in item):
            return True
        if (in_trace and name == "value") or (name in {"generated_token_ids", "returned_sequence_token_ids"} and item):
            return True
        if name in {"stdout", "stderr", "candidate_error", "exception_message", "error_message"} and item:
            return True
        if name in {"diagnostic", "diagnostics"} and item:
            # Even apparently structured candidate diagnostics may carry PII.
            return True
        if _sensitive_observations(item, in_trace or name == "return_trace"):
            return True
    return False


def _generation_observation(run, spec):
    from asea.compose.generation_contract import TRACE_SCHEMA, MAX_TRACE_BYTES, MAX_RUN_TRACE_BYTES, MAX_TOKEN_IDS
    traces = run.get("generation_trace_v1")
    if traces is None:
        return {"status": "original_unknown", "reason": "legacy_generation_trace_unavailable", "sha256": None}
    if (run.get("generation_trace_schema") != TRACE_SCHEMA or type(traces) is not list
            or len(traces) > len(spec.nodes) or len(canonical(traces)) > MAX_RUN_TRACE_BYTES):
        raise Blocked("generation trace schema/bounds mismatch")
    by_id, seen = {node.id: node for node in spec.nodes}, set()
    for trace in traces:
        if (type(trace) is not dict or trace.get("schema") != TRACE_SCHEMA
                or trace.get("node") not in by_id or trace["node"] in seen
                or len(canonical(trace)) > MAX_TRACE_BYTES):
            raise Blocked("generation trace node/schema/bounds mismatch")
        seen.add(trace["node"])
        node = by_id[trace["node"]]
        if "evidence_omitted" in trace:
            if (trace["evidence_omitted"] not in {"node_byte_limit", "run_byte_limit"}
                    or not _sha256(trace.get("sha256"))):
                raise Blocked("invalid generation trace omission marker")
            continue
        expected = {"output_contract": node.output_contract.model_dump(mode="json") if node.output_contract else None,
                    "cache_policy": node.cache_policy, "legacy_use_cache": node.use_cache,
                    "dtype": node.dtype, "max_new_tokens": node.max_new_tokens}
        if trace.get("requested") != expected:
            raise Blocked("generation trace requested settings binding mismatch")
        if "call_metadata_sha256" in trace and "metadata_omitted" not in trace:
            if trace["call_metadata_sha256"] != digest({"forwarded": trace.get("forwarded"), "resolved": trace.get("resolved")}):
                raise Blocked("generation call metadata hash mismatch")
        for name in ("generated_token_ids", "returned_sequence_token_ids", "eos_positions"):
            ids = trace.get(name)
            if ids is not None and (type(ids) is not list or len(ids) > MAX_TOKEN_IDS
                    or any(type(i) is not int or i < 0 for i in ids)):
                raise Blocked("generation token trace bounds mismatch")
        for name in ("input_token_count", "output_token_count", "generated_token_count"):
            count = trace.get(name)
            if count is not None and (type(count) is not int or count < 0):
                raise Blocked("invalid generation token count")
        if "metadata_omitted" in trace:
            omitted = trace["metadata_omitted"]
            if (type(omitted) is not dict or set(omitted) != {"reason", "sha256"}
                    or omitted["reason"] != "byte_limit" or not _sha256(omitted["sha256"])):
                raise Blocked("invalid generation metadata omission")
        for name in ("cap_reached", "token_ids_truncated"):
            if trace.get(name) is not None and type(trace[name]) is not bool:
                raise Blocked("invalid generation cap/truncation observation")
        count, ids = trace.get("generated_token_count"), trace.get("generated_token_ids")
        if ids is not None and count is not None and len(ids) != min(count, MAX_TOKEN_IDS):
            raise Blocked("generation token count/IDs mismatch")
        if count is not None and trace.get("cap_reached") is not None:
            if trace["cap_reached"] != (count >= node.max_new_tokens):
                raise Blocked("generation cap observation mismatch")
        positions = trace.get("eos_positions")
        basis = trace.get(trace.get("eos_positions_basis", "generated_token_ids"))
        if positions is not None and basis is not None and any(i >= len(basis) for i in positions):
            raise Blocked("generation EOS position bounds mismatch")
    missing = [node.id for node in spec.nodes if node.id not in seen]
    return {"status": "captured" if traces else "original_unknown",
            "reason": "generation_trace_captured" if traces else "generation_trace_unavailable",
            "sha256": digest(traces), "run_id": run["id"], "graph_hash": run["graph_hash"],
            "input_digest": run.get("input_digest"), "output_sha256": digest(run["output"]),
            "missing_nodes": missing}


def _sha256(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _oracle_observation(oracle, cases, policy):
    """Validate only captured evidence. Never backfill it from a fresh execution."""
    if oracle is None:
        return {"status": "original_unknown", "reason": "oracle_trace_unavailable", "sha256": None}
    trace = oracle.get("return_trace")
    if trace is None:
        return {"status": "original_unknown", "reason": "legacy_return_trace_unavailable", "sha256": None}
    from .function_oracle import replay_returns, MAX_CASES
    if (type(trace) is not dict or set(trace) != {"schema_version", "policy", "returns"}
            or type(trace["schema_version"]) is not int or trace["schema_version"] != 1
            or _evaluation_trace_policy(trace["policy"]) != policy
            or type(trace["returns"]) is not list or len(trace["returns"]) > min(MAX_CASES, len(cases))):
        raise Blocked("return trace policy/schema/bounds mismatch")
    retention = policy["return_retention"]
    flags = ["sensitive_actual_value_trace"] if retention == "value" else []
    if oracle.get("evidence_flags") != flags:
        raise Blocked("return trace evidence flags inconsistent with retention policy")
    completed = oracle.get("cases", [])
    if len(trace["returns"]) != len(completed):
        raise Blocked("return trace completed-case count mismatch")
    for entry, case, result in zip(trace["returns"], cases, completed):
        keys = {"id", "phase", "retention", "reason"}
        if retention != "none":
            keys |= {"value_type", "sha256"}
        if retention == "value":
            keys.add("value")
        if (type(entry) is not dict or set(entry) != keys or entry.get("id") != case["id"]
                or result.get("id") != case["id"] or entry.get("phase") != "response"
                or entry.get("retention") != retention
                or entry.get("reason") != ("policy_none" if retention == "none" else "retained")):
            raise Blocked("return trace case/policy binding mismatch")
        if retention != "none" and (entry["value_type"] not in {"null", "bool", "int", "string", "list", "dict"}
                                     or not _sha256(entry["sha256"])):
            raise Blocked("return trace type/digest mismatch")
    if retention == "value":
        # Existing oracle codec validates exact builtins, frame/aggregate byte,
        # depth/node/string bounds and hashes. Recomparison can only invalidate,
        # never replace the existing host-counter pass criteria.
        replayed = replay_returns(trace, cases)
        if replayed["cases"] != completed:
            raise Blocked("retained return values disagree with host comparisons")
    elif len(canonical(trace)) > 128 * 1024:
        raise Blocked("return trace metadata bounds exceeded")
    diagnostic = oracle.get("diagnostic")
    if diagnostic is not None and (type(diagnostic) is not dict or type(diagnostic.get("schema_version")) is not int
            or diagnostic["schema_version"] != 1 or len(canonical(diagnostic)) > 8192):
        raise Blocked("diagnostic schema/bounds mismatch")
    termination = oracle.get("termination")
    if termination is not None:
        if (type(termination) is not dict or type(termination.get("schema_version")) is not int
                or termination["schema_version"] != 1 or len(canonical(termination)) > 8192):
            raise Blocked("invalid termination observation")
        for field in ("observed_exit", "final_exit"):
            code = termination.get(field)
            if code is not None and type(code) is not int:
                raise Blocked("invalid observed exit")
            if termination.get(field + "_state") != ("known" if code is not None else "unknown"):
                raise Blocked("termination known/unknown mismatch")
        if termination.get("final_exit") != oracle.get("returncode"):
            raise Blocked("termination final exit binding mismatch")
    return {"status": "captured", "reason": "return_trace_captured", "sha256": digest(trace),
            "diagnostic_status": "captured" if diagnostic is not None else "original_unknown",
            "diagnostic_sha256": digest(diagnostic) if diagnostic is not None else None,
            "termination_status": "captured" if termination is not None else "original_unknown",
            "termination_sha256": digest(termination) if termination is not None else None}


def _evaluation_evidence_flags(workspace, evaluation):
    # Exclude the aggregate flags themselves, then recompute from actual signed
    # observations (including blocked attempts), not an author's privacy label.
    payload = {key: value for key, value in evaluation.items() if key != "evidence_flags"}
    if _sensitive_observations(payload):
        return ["sensitive_observations"]
    for row in evaluation.get("case_evidence", []):
        if _sensitive_observations(workspace.read_record("runs", row["run_id"])):
            return ["sensitive_observations"]
    return []


def _case_observability(run, spec, oracle, case, policy):
    return {"schema": OBSERVABILITY_SCHEMA,
            "generation": _generation_observation(run, spec),
            "oracle": _oracle_observation(oracle, case.function_cases, policy) if case.metric == "function_io"
                      else {"status": "original_unknown", "reason": "metric_has_no_return_trace", "sha256": None}}


def implementation_fingerprint():
    """Hash installed code and record dependency versions without importing ML."""
    root = Path(__file__).resolve().parents[1]
    names = ("certification/__init__.py", "certification/function_oracle.py", "certification/sandbox.py",
             "compose/runtime.py", "compose/schema.py", "compose/generation_contract.py", "compose/__main__.py", "artifacts/__init__.py",
             "execution/__init__.py", "execution/controls.py", "execution/_worker.py")
    versions = {}
    for name in ("pydantic", "torch", "transformers", "tokenizers", "safetensors", "numpy",
                 "Pillow", "scipy", "soundfile", "sentencepiece", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    tools = {}
    for name, path in (("python", sys.executable), ("base_python", getattr(sys, "_base_executable", sys.executable)),
                       ("unshare", shutil.which("unshare")), ("ldd", shutil.which("ldd"))):
        tools[name] = {"path": str(Path(path).resolve()), "file_hash": file_hash(Path(path).resolve())} if path else None
    return {"schema_version": 1, "implementation_hashes": {name: file_hash(root / name) for name in names},
            "environment": {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
                            "platform": platform.platform(), "dependencies": versions, "runtime_tools": tools}}


def _execution_config(profile, memory_mib, timeout):
    if profile not in {"observe_only", "process_as", "cgroup_v2"}:
        raise Blocked("unsupported resource profile; no fallback")
    if type(memory_mib) is not int or not 32 <= memory_mib <= 1048576:
        raise Blocked("memory_mib must be an integer from 32 to 1048576")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise Blocked("case_timeout must be finite and in (0, 86400]")
    return {"version": EXECUTION_CONTRACT, "resource_profile": profile, "memory_mib": memory_mib,
            "case_timeout": timeout,
            "memory_semantics": "virtual_address_space_per_process" if profile == "process_as" else
                                "cgroup_accounted_memory_requested_not_enforced" if profile == "cgroup_v2" else "observed_process_lifetime_rss_no_new_hard_limit",
            "limits_enforced_by_request": False if profile == "observe_only" else None}


def _preflight_execution(config):
    if config["resource_profile"] != "observe_only":
        from asea.execution import probe
        observed = probe()
        if not observed[config["resource_profile"]]["execution_supported"]:
            raise Blocked(config["resource_profile"] + " unavailable before model generation; no fallback")


def _snapshot_spec(workspace, spec):
    directory = safe_path(workspace.root / "evaluation-specs")
    directory.mkdir(exist_ok=True)
    path = safe_path(directory / (digest(spec.model_dump(mode="json")) + ".json"))
    if not path.exists():
        atomic_json(path, spec.model_dump(mode="json"))
        path.chmod(0o444)
    if json.loads(safe_file(path).read_text()) != spec.model_dump(mode="json"):
        raise Blocked("immutable evaluation spec changed")
    return {"path": str(path), "file_hash": file_hash(path)}


def _validate_implementation(evaluation):
    if evaluation.get("execution_config", {}).get("version") != EXECUTION_CONTRACT or "implementation_fingerprint" not in evaluation:
        raise Blocked("stale_evidence_version_not_supported: missing execution contract/implementation fingerprint; reevaluate")
    if evaluation["implementation_fingerprint"] != implementation_fingerprint():
        raise Blocked("stale_implementation_fingerprint: implementation or environment changed; reevaluate")
    if (evaluation.get("evidence_schema") != OBSERVABILITY_SCHEMA
            or _evaluation_trace_policy(evaluation.get("trace_policy")) != evaluation.get("trace_policy")
            or evaluation.get("trace_policy_hash") != digest(evaluation["trace_policy"])):
        raise Blocked("observability schema/trace policy binding mismatch; original unknown, reevaluate")
    config = evaluation["execution_config"]
    if (config != _execution_config(config["resource_profile"], config["memory_mib"], config["case_timeout"])
            or evaluation.get("execution_config_hash") != digest(config)):
        raise Blocked("execution configuration binding mismatch")
    expected_modes = {c["metric"]: ORACLE_SCOPES[c["metric"]] for c in evaluation["suite"]["cases"] if c["metric"] in ORACLE_SCOPES}
    if evaluation.get("oracle_modes") != expected_modes:
        raise Blocked("oracle mode/scope binding mismatch")
    if config["resource_profile"] != "observe_only":
        snapshot = evaluation.get("spec_snapshot", {})
        if file_hash(snapshot["path"]) != snapshot.get("file_hash"):
            raise Blocked("immutable evaluation spec changed")


def _validate_worker(workspace, receipt, receipt_hash, config, run):
    """Receipt is host supervision data; application stdout is never admission."""
    if digest(receipt) != receipt_hash:
        raise Blocked("worker receipt hash mismatch")
    external = receipt["resource_evidence"]
    pid = external.get("worker_pid")
    if (external.get("schema") != "silt.resource-controls.v1" or external.get("operation") != "compose"
            or external.get("profile") != config["resource_profile"] or config["resource_profile"] != "process_as"
            or external.get("status") != "OK" or external.get("returncode") != 0
            or external.get("supported") is not True or external.get("launched") is not True
            or external.get("enforcement_established") is not True
            or type(pid) is not int or pid <= 0 or external.get("group_kill_succeeded") is not True):
        raise Blocked("worker did not complete with requested resource enforcement")
    limits = external.get("requested_limits", {})
    effective = external.get("effective_limits", {})
    memory_bytes = config["memory_mib"] * 1024 * 1024
    cpu = math.ceil(config["case_timeout"])
    if (limits.get("address_space_bytes") != memory_bytes or limits.get("wall_seconds") != config["case_timeout"]
            or limits.get("cpu_soft_seconds") != cpu or limits.get("cpu_hard_seconds") != cpu + 1
            or effective.get("as") != [memory_bytes, memory_bytes]
            or effective.get("cpu") != [cpu, cpu + 1]
            or external.get("memory_metric") != "virtual_address_space_per_process"):
        raise Blocked("worker kernel limit receipt mismatch (AS, not RSS)")
    envelope = json.loads(external["stdout"])
    if (envelope.get("ok") is not True or envelope.get("command") != "run"
            or envelope.get("execution_pid") != pid or envelope.get("result") != run
            or receipt.get("run_id") != run["id"] or receipt.get("run_hash") != digest(run)
            or receipt.get("workspace") != str(workspace.root)
            or receipt.get("output_hash") != digest(run["output"])):
        raise Blocked("worker PID/run/output binding mismatch")
    # Signature is checked by read_record; never copy/sign a supplied run verdict.
    if workspace.read_record("runs", run["id"]) != run:
        raise Blocked("worker signed run mismatch")


def _worker_run(workspace, spec, case, evaluation):
    from asea.execution import run as execute_worker
    snapshot = evaluation["spec_snapshot"]
    if file_hash(snapshot["path"]) != snapshot["file_hash"]:
        raise Blocked("immutable evaluation spec changed")
    arguments = ["run", "--spec", snapshot["path"]]
    if case.input is not None:
        arguments.extend(["--input", case.input])
    if case.input_file is not None:
        arguments.extend(["--input-file", str(safe_file(case.input_file))])
    # Freshness is independent of child stdout. Existing runs cannot be replayed.
    before = {p.stem for p in (workspace.root / "runs").glob("*.json")}
    config = evaluation["execution_config"]
    try:
        external = execute_worker("compose", arguments, profile=config["resource_profile"],
                                  memory_mib=config["memory_mib"], timeout=config["case_timeout"],
                                  workspace=str(workspace.root))
    except BaseException as exc:
        evaluation["worker_attempts"].append({"case_id": case.id, "status": "interrupted",
            "reason": type(exc).__name__, "resource_evidence": None,
            "cleanup": "supervisor finally owns kill/reap; no completed receipt, never admissible"})
        raise
    # Retain failures, including timeout, rather than dropping an unresolved child.
    evaluation["worker_attempts"].append({"case_id": case.id, "resource_evidence": external,
                                         "resource_evidence_hash": digest(external)})
    if external.get("status") != "OK" or external.get("returncode") != 0:
        raise Blocked("resource worker " + external.get("status", "FAILED") + ": " + external.get("reason", "child did not succeed; no fallback"))
    envelope = json.loads(external["stdout"])
    run_id = envelope["result"]["id"]
    if run_id in before:
        raise Blocked("stale child run replay rejected")
    run = workspace.read_record("runs", run_id)
    receipt = {"resource_evidence": external, "workspace": str(workspace.root),
               "arguments": arguments, "run_id": run_id, "run_hash": digest(run),
               "output_hash": digest(run["output"])}
    _validate_worker(workspace, receipt, digest(receipt), config, run)
    return run, receipt


def evaluate(workspace, spec, suite, allow_fixtures=False, *, resource_profile="observe_only", memory_mib=512, case_timeout=60.0, trace_policy=None):
    from asea.compose.runtime import _prepare, _run, _managed_run_signals
    spec = CompositionSpec.model_validate(spec.model_dump(mode="python"))
    config = _execution_config(resource_profile, memory_mib, case_timeout)
    trace_policy = _evaluation_trace_policy(trace_policy)
    # Observe-only retains the old single-process, single-writer execution path.
    # Protected children acquire this same writer themselves; never hold it
    # across child launch. Only parent preparation and finalization hold it.
    with contextlib.ExitStack() as stack:
        # Convert main-thread SIGTERM/SIGINT to unwinding exceptions so the
        # supervisor's kill/reap finally runs instead of orphaning its child.
        stack.enter_context(_managed_run_signals())
        if resource_profile == "observe_only":
            stack.enter_context(workspace.writer())
        with (workspace.writer() if resource_profile != "observe_only" else contextlib.nullcontext()):
            candidate = _prepare(workspace, spec, allow_fixtures)
            identifier = workspace.new_id("evaluation")
            evaluation = {"schema_version": 1, "id": identifier, "candidate_id": candidate["id"],
                          "graph_hash": candidate["graph_hash"], "status": "blocked", "fixture_only": candidate["fixture_only"],
                          "suite": suite.model_dump(mode="json"), "suite_hash": digest(suite.model_dump(mode="json")),
                          "policy": dict(ADMISSION_POLICY), "policy_hash": digest(ADMISSION_POLICY),
                          "claims": list(suite.claims), "case_evidence": [], "worker_attempts": [],
                          "evidence_schema": OBSERVABILITY_SCHEMA, "trace_policy": trace_policy,
                          "trace_policy_hash": digest(trace_policy), "evidence_flags": [],
                          "execution_config": config, "execution_config_hash": digest(config),
                          "implementation_fingerprint": implementation_fingerprint(),
                          "oracle_modes": {c.metric: ORACLE_SCOPES[c.metric] for c in suite.cases if c.metric in ORACLE_SCOPES},
                          "scope": "named reference metrics only; proxy is not coding/safety evidence; legacy functional_code shares an interpreter and is not adversarial correctness proof; function_io host-compares observable outputs only",
                          "created_at": time.time()}
            if resource_profile != "observe_only":
                evaluation["spec_snapshot"] = _snapshot_spec(workspace, spec)
        cancelled = None
        try:
            suite = _check_policy(suite)
            _preflight_formats(spec, suite)
            # Validate the complete suite before generating ANY candidate output.
            expected_inputs = {case.id: _case_input(case, spec) for case in suite.cases}
            _preflight_execution(config)
            if any(case.metric in {"functional_code", "function_io"} for case in suite.cases):
                from .sandbox import probe_code_sandbox
                probe = probe_code_sandbox()
                if not probe.supported or not probe.passed:
                    raise Blocked("functional sandbox unavailable: " + probe.reason)
            output_node = next(n for n in spec.nodes if n.id == spec.output_node)
            if PORTS[output_node.component.kind][1] != "text":
                raise Blocked("audio-output quality evaluation not implemented; evaluate a text-output subgraph separately; this cannot admit the full speech-output graph")
            for case in suite.cases:
                expected_input = expected_inputs[case.id]
                source_hash = file_hash(case.input_file) if case.input_file else None
                receipt = None
                if resource_profile == "observe_only":
                    run = _run(workspace, spec, case.input, case.input_file, allow_fixtures)
                else:
                    run, receipt = _worker_run(workspace, spec, case, evaluation)
                _validate_implementation(evaluation)
                for artifact_id in candidate["artifacts"].values():
                    workspace.verify_artifact(artifact_id)
                if run["candidate_id"] != candidate["id"] or run["graph_hash"] != candidate["graph_hash"]:
                    raise Blocked("candidate changed during evaluation")
                if case.input_file and source_hash != file_hash(case.input_file):
                    raise Blocked("evaluation fixture changed during run")
                if digest(expected_input) != run.get("input_digest") or expected_input != _case_input(case, spec):
                    raise Blocked("measured run does not bind reference fixture input")
                if run["status"] != "succeeded" or run["fixture_only"] != candidate["fixture_only"] or run["output"]["type"] != "text":
                    raise Blocked("invalid measured run status or output")
                _validate_resources(run["resources"], spec.limits)
                oracle = None
                if case.metric == "function_io":
                    try:
                        oracle = _function_result(run["output"]["text"], case.function_cases, trace_policy)
                    except Blocked as exc:
                        if hasattr(exc, "oracle_evidence"):
                            original = exc.oracle_evidence
                            evaluation["blocked_oracle_evidence"] = {"case_id": case.id, "run_id": run["id"],
                                "oracle_evidence": original, "oracle_evidence_hash": digest(original),
                                "generation": _generation_observation(run, spec)}
                        raise
                    score = float(_function_passed(oracle, case.function_cases))
                else:
                    score = metric_score(case.metric, run["output"]["text"], case.reference)
                row = {"case_id": case.id, "run_id": run["id"], "metric": case.metric,
                       "score": score, "threshold": case.threshold, "passed": _passes(case.metric, score, case.threshold),
                       "reference_sha256": hashlib.sha256(case.reference.encode()).hexdigest(), "input_file_hash": source_hash}
                row["obs_v1"] = _case_observability(run, spec, oracle, case, trace_policy)
                row["obs_v1_hash"] = digest(row["obs_v1"])
                if oracle is not None:
                    row.update(oracle_evidence=oracle, oracle_evidence_hash=digest(oracle),
                               function_cases_hash=digest(case.function_cases))
                if receipt is not None:
                    row.update(worker_receipt=receipt, worker_receipt_hash=digest(receipt))
                evaluation["case_evidence"].append(row)
            runs = [workspace.read_record("runs", row["run_id"]) for row in evaluation["case_evidence"]]
            evaluation["measured_resources"] = {"max_wall_seconds": max(r["resources"]["wall_seconds"] for r in runs),
                "max_peak_rss_mb": max(r["resources"]["process_peak_rss_mb"] for r in runs), "measurement_count": len(runs)}
            evaluation["status"] = "admitted" if all(row["passed"] for row in evaluation["case_evidence"]) and not candidate["fixture_only"] else "rejected"
            if candidate["fixture_only"]:
                evaluation["reason"] = "fixture-only adapters cannot be admitted by production gate"
        except BaseException as exc:
            evaluation["status"] = "blocked"
            evaluation["reason"] = str(exc) or type(exc).__name__
            if not isinstance(exc, Exception):
                cancelled = exc
                evaluation["reason"] = "evaluation cancelled: " + evaluation["reason"]
        with (workspace.writer() if resource_profile != "observe_only" else contextlib.nullcontext()):
            # Writer recovery retains interrupted pending runs after child death.
            if evaluation["status"] in {"admitted", "rejected"}:
                try:
                    _validate_implementation(evaluation)
                    for artifact_id in candidate["artifacts"].values():
                        workspace.verify_artifact(artifact_id)
                    for case in suite.cases:
                        if _case_input(case, spec) != expected_inputs[case.id]:
                            raise Blocked("evaluation input changed before finalization")
                    for row in evaluation["case_evidence"]:
                        signed = workspace.read_record("runs", row["run_id"])
                        if resource_profile != "observe_only":
                            _validate_worker(workspace, row["worker_receipt"], row["worker_receipt_hash"], config, signed)
                except Exception as exc:
                    evaluation.update(status="blocked", reason=str(exc))
            evaluation["evidence_flags"] = _evaluation_evidence_flags(workspace, evaluation)
            workspace.write_record("evaluations", identifier, evaluation)
            workspace.audit("evaluation", evaluation_id=identifier, candidate_id=candidate["id"], status=evaluation["status"])
        if cancelled is not None:
            raise cancelled
        return evaluation


def _validate_admitted(workspace, identifier):
    evaluation = workspace.read_record("evaluations", identifier)
    if evaluation.get("status") != "admitted" or evaluation.get("fixture_only"):
        raise Blocked("activation requires a real admitted evaluation, never a run or fixture")
    _validate_implementation(evaluation)
    candidate = workspace.read_record("candidates", evaluation["candidate_id"])
    spec = CompositionSpec.model_validate(candidate["spec"])
    if candidate.get("fixture_only") or any(n.component.kind == "fixture_text" for n in spec.nodes):
        raise Blocked("fixture cannot pass production admission")
    from asea.compose.runtime import RUNTIME_VERSION
    if candidate.get("runtime_version") != RUNTIME_VERSION:
        raise Blocked("runtime changed since evaluation; reevaluate")
    expected_config = digest({"spec": spec.model_dump(mode="json"), "runtime_version": RUNTIME_VERSION})
    if candidate.get("config_hash") != expected_config:
        raise Blocked("runtime/configuration evidence mismatch")
    graph_hash = digest({"spec": spec.model_dump(mode="json"), "artifacts": candidate["artifacts"], "runtime_version": RUNTIME_VERSION})
    if graph_hash != candidate["graph_hash"] or graph_hash != evaluation["graph_hash"] or candidate["id"] != "candidate-" + graph_hash:
        raise Blocked("graph/evidence binding mismatch")
    for node in spec.nodes:
        artifact = workspace.verify_artifact(candidate["artifacts"][node.id])
        if artifact["manifest"] != node.component.model_dump(mode="json"):
            raise Blocked("node manifest does not match registered artifact")
    if evaluation.get("policy") != ADMISSION_POLICY or evaluation.get("policy_hash") != digest(ADMISSION_POLICY):
        raise Blocked("admission policy/version binding mismatch; reevaluate under current policy")
    suite = _check_policy(EvaluationSuite.model_validate(evaluation["suite"]))
    _preflight_formats(spec, suite)
    if evaluation.get("evidence_flags") != _evaluation_evidence_flags(workspace, evaluation):
        raise Blocked("observability evidence flags inconsistent with captured payloads")
    if evaluation.get("claims") != list(suite.claims):
        raise Blocked("evaluation claims binding mismatch")
    if digest(suite.model_dump(mode="json")) != evaluation["suite_hash"]:
        raise Blocked("reference suite fingerprint mismatch")
    evidence = evaluation["case_evidence"]
    if len(evidence) != len(suite.cases) or {r["case_id"] for r in evidence} != {c.id for c in suite.cases}:
        raise Blocked("missing measured target/control evidence")
    rows = {row["case_id"]: row for row in evidence}
    if evaluation["execution_config"]["resource_profile"] != "observe_only":
        if json.loads(safe_file(evaluation["spec_snapshot"]["path"]).read_text()) != spec.model_dump(mode="json"):
            raise Blocked("worker spec/graph binding mismatch")
        attempts = evaluation.get("worker_attempts", [])
        if len(attempts) != len(suite.cases):
            raise Blocked("missing or extra worker attempts")
        pids, run_ids = set(), set()
        for case, attempt in zip(suite.cases, attempts):
            external = rows[case.id].get("worker_receipt", {}).get("resource_evidence")
            if (attempt.get("case_id") != case.id or attempt.get("resource_evidence") != external
                    or attempt.get("resource_evidence_hash") != digest(external)):
                raise Blocked("worker attempt binding mismatch")
            pids.add(external.get("worker_pid"))
            run_ids.add(rows[case.id]["run_id"])
        if len(pids) != len(suite.cases) or len(run_ids) != len(suite.cases):
            raise Blocked("stale worker PID/run reuse")
    elif evaluation.get("worker_attempts"):
        raise Blocked("observe_only cannot claim worker attempts")
    resources = []
    for case in suite.cases:
        row = rows[case.id]
        run = workspace.read_record("runs", row["run_id"])
        if run["status"] != "succeeded" or run["fixture_only"] or run["candidate_id"] != candidate["id"] or run["graph_hash"] != graph_hash:
            raise Blocked("invalid measured run binding")
        if run["output"]["type"] != "text":
            raise Blocked("unsupported measured output type")
        if evaluation["execution_config"]["resource_profile"] != "observe_only":
            _validate_worker(workspace, row["worker_receipt"], row["worker_receipt_hash"], evaluation["execution_config"], run)
            arguments = ["run", "--spec", evaluation["spec_snapshot"]["path"]]
            if case.input is not None:
                arguments.extend(["--input", case.input])
            if case.input_file is not None:
                arguments.extend(["--input-file", str(safe_file(case.input_file))])
            if row["worker_receipt"].get("arguments") != arguments:
                raise Blocked("worker arguments binding mismatch")
        elif "worker_receipt" in row:
            raise Blocked("observe_only cannot claim an enforced worker receipt")
        observed = _case_observability(run, spec, row.get("oracle_evidence"), case, evaluation["trace_policy"])
        if row.get("obs_v1") != observed or row.get("obs_v1_hash") != digest(observed):
            raise Blocked("captured observability trace binding mismatch; original unknown, never backfill")
        if case.metric == "function_io":
            oracle = row.get("oracle_evidence", {})
            if (row.get("oracle_evidence_hash") != digest(oracle)
                    or row.get("function_cases_hash") != digest(case.function_cases)
                    or not _function_passed(oracle, case.function_cases)):
                raise Blocked("function oracle evidence mismatch")
            fresh = _function_result(run["output"]["text"], case.function_cases, evaluation["trace_policy"])
            # Diagnostics, observed timing, and exception text are deliberately
            # NOT compared across executions and never authorize a passing score.
            for field in ("oracle_mode", "source_sha256", "inputs_sha256", "data_sha256", "oracle_sha256", "sandbox_sha256"):
                if oracle.get(field) != fresh.get(field):
                    raise Blocked("function oracle source/data/implementation binding mismatch")
            score = float(_function_passed(fresh, case.function_cases))
        else:
            score = metric_score(case.metric, run["output"]["text"], case.reference)
        if (row["metric"] != case.metric or row["threshold"] != case.threshold or row["score"] != score
                or row.get("passed") is not True
                or row.get("reference_sha256") != hashlib.sha256(case.reference.encode()).hexdigest()
                or not _passes(case.metric, score, case.threshold)):
            raise Blocked("threshold or measured evidence mismatch")
        expected_input = _case_input(case, spec)
        if case.input_file and file_hash(case.input_file) != row["input_file_hash"]:
            raise Blocked("reference input file changed after evaluation")
        if digest(expected_input) != run["input_digest"]:
            raise Blocked("measured run does not bind reference fixture input")
        measured = run["resources"]
        _validate_resources(measured, spec.limits)
        resources.append(measured)
    expected_resources = {"max_wall_seconds": max(r["wall_seconds"] for r in resources),
        "max_peak_rss_mb": max(r["process_peak_rss_mb"] for r in resources), "measurement_count": len(resources)}
    if evaluation.get("measured_resources") != expected_resources:
        raise Blocked("resource summary does not match measured runs")
    _validate_implementation(evaluation)
    return evaluation, candidate


def _unlink_durable(path):
    path = safe_path(path)
    if path.exists():
        path.unlink()
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _restore_pointer(workspace, previous):
    pointer = workspace.root / "active.json"
    if previous is None:
        _unlink_durable(pointer)
    else:
        atomic_json(pointer, previous)


def recover_active(workspace):
    """Caller holds writer(); validate everything before conservatively restoring.

    A journal is not admission evidence. Unknown, cross-workspace or corrupt
    journals block reads/writes and are retained for inspection, without touching
    the pointer. Legacy journals are accepted only with valid local record IDs.
    """
    journal = safe_path(workspace.root / "active-transaction.json")
    if not journal.exists():
        return
    try:
        if safe_file(journal).stat().st_size > 65536:
            raise Blocked("active recovery journal too large")
        def unique_object(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise Blocked("duplicate recovery journal key")
                value[key] = item
            return value
        transaction = json.loads(journal.read_text(), object_pairs_hook=unique_object)
        legacy = {"previous", "deployment_id"}
        if not isinstance(transaction, dict) or set(transaction) not in (legacy, legacy | {"schema_version", "workspace"}):
            raise Blocked("invalid active recovery journal fields")
        if "workspace" in transaction and (type(transaction["schema_version"]) is not int
                or transaction["schema_version"] != 1 or transaction["workspace"] != str(workspace.root)):
            raise Blocked("active recovery journal belongs to a different workspace/version")

        def validate_id(identifier):
            if not isinstance(identifier, str) or not re.fullmatch(r"deployment-[a-f0-9]{32,64}", identifier):
                raise Blocked("invalid recovery deployment ID")
            record = workspace.read_record("deployments", identifier)
            if record.get("id") != identifier:
                raise Blocked("recovery deployment record binding mismatch")

        def validate_pointer(value):
            if value is None:
                return
            if (not isinstance(value, dict) or set(value) != {"schema_version", "deployment_id"}
                    or type(value["schema_version"]) is not int or value["schema_version"] != 1):
                raise Blocked("invalid recovery pointer")
            validate_id(value["deployment_id"])

        validate_id(transaction["deployment_id"])
        previous = transaction["previous"]
        validate_pointer(previous)
        pointer = safe_path(workspace.root / "active.json")
        current = json.loads(safe_file(pointer).read_text(), object_pairs_hook=unique_object) if pointer.exists() else None
        validate_pointer(current)
        target = {"schema_version": 1, "deployment_id": transaction["deployment_id"]}
        if current not in (previous, target):
            raise Blocked("active pointer conflicts with recovery journal")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Blocked("invalid active recovery journal: " + str(exc)) from exc
    _restore_pointer(workspace, previous)
    workspace.audit("pointer_recovered", deployment_id=transaction["deployment_id"])
    _unlink_durable(journal)


def _publish_pointer(workspace, event, deployment_id, **fields):
    # Write-ahead audit + recovery journal precede publication. A failed final
    # audit restores the prior pointer; a failed restore retains recovery state.
    pointer = workspace.root / "active.json"
    previous = json.loads(safe_file(pointer).read_text()) if pointer.exists() else None
    journal = workspace.root / "active-transaction.json"
    atomic_json(journal, {"schema_version": 1, "workspace": str(workspace.root),
                          "previous": previous, "deployment_id": deployment_id})
    try:
        workspace.audit(event + "_intent", deployment_id=deployment_id, previous=previous, **fields)
        atomic_json(pointer, {"schema_version": 1, "deployment_id": deployment_id})
        workspace.audit(event, deployment_id=deployment_id, **fields)
        _unlink_durable(journal)
    except BaseException:
        _restore_pointer(workspace, previous)
        _unlink_durable(journal)
        raise


def activate(workspace, evaluation_id, approve_high_risk=False, approver=None):
    with workspace.writer():
        recover_active(workspace)
        evaluation, candidate = _validate_admitted(workspace, evaluation_id)
        risk = CompositionSpec.model_validate(candidate["spec"]).effective_risk
        if approver is not None and (not isinstance(approver, str) or not approver.strip() or len(approver) > 200):
            raise Blocked("approver must be a nonempty name of at most 200 characters")
        if risk == "high" and not approver:
            raise Blocked("high-risk provenance requires named --approver NAME; boolean --approve-high-risk alone is insufficient")
        identifier = workspace.new_id("deployment")
        record = {"schema_version": 1, "id": identifier, "evaluation_id": evaluation_id, "candidate_id": candidate["id"],
                  "graph_hash": candidate["graph_hash"], "risk": risk, "high_risk_approved": bool(approver),
                  "approver": approver.strip() if approver else None, "policy_hash": evaluation["policy_hash"], "created_at": time.time()}
        workspace.write_record("deployments", identifier, record)
        _publish_pointer(workspace, "activate", identifier, evaluation_id=evaluation_id, approver=record["approver"])
        return record


def _deployment(workspace, identifier):
    record = workspace.read_record("deployments", identifier)
    evaluation, candidate = _validate_admitted(workspace, record["evaluation_id"])
    if record["candidate_id"] != candidate["id"] or record["graph_hash"] != candidate["graph_hash"]:
        raise Blocked("deployment binding mismatch")
    if record.get("policy_hash") != evaluation["policy_hash"]:
        raise Blocked("deployment policy binding mismatch")
    if CompositionSpec.model_validate(candidate["spec"]).effective_risk == "high" and (
            not record.get("high_risk_approved") or not isinstance(record.get("approver"), str) or not record["approver"].strip()):
        raise Blocked("missing named deployment high-risk approval")
    return record, evaluation, candidate


def rollback(workspace, deployment_id):
    with workspace.writer():
        recover_active(workspace)
        record, _, _ = _deployment(workspace, deployment_id)
        _publish_pointer(workspace, "rollback", deployment_id)
        return record


def plan_graph(spec):
    """Schema-only planning; no artifact availability, quality or feasibility claim."""
    spec = CompositionSpec.model_validate(spec.model_dump(mode="json"))
    return {"status": "draft", "admitted": False, "activated": False,
            "spec": spec.model_dump(mode="json"), "spec_hash": digest(spec.model_dump(mode="json")),
            "node_order": [node.id for node in spec.topological()],
            "scope": "schema-valid planning graph only; inspect, run and evaluate separately"}


def select_measured(workspace, evaluation_ids, input_type, output_type, max_wall_seconds, max_peak_rss_mb,
                    suite_hash=None, max_disk_bytes=None, graph_hash=None):
    """Select supplied admitted bundles for an explicit suite; never activate."""
    with workspace.writer():
        return _select_measured(workspace, evaluation_ids, input_type, output_type, max_wall_seconds,
                                max_peak_rss_mb, suite_hash, max_disk_bytes, graph_hash)


def _select_measured(workspace, evaluation_ids, input_type, output_type, max_wall_seconds, max_peak_rss_mb,
                     suite_hash, max_disk_bytes, graph_hash):
    if input_type not in {"text", "audio", "image"} or output_type not in {"text", "audio"}:
        raise Blocked("selection requires text/audio/image input and text/audio output types")
    try:
        SelectionBudgets(max_wall_seconds=max_wall_seconds, max_peak_rss_mb=max_peak_rss_mb, max_disk_bytes=max_disk_bytes)
    except ValueError as exc:
        raise Blocked("selection budgets must be finite positive numbers; disk bytes a positive integer") from exc
    for value in (suite_hash, graph_hash):
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)):
            raise Blocked("suite/graph hash must be a SHA-256 hex digest")
    if (not math.isfinite(max_wall_seconds) or not math.isfinite(max_peak_rss_mb)
            or max_wall_seconds <= 0 or max_peak_rss_mb <= 0):
        raise Blocked("selection budgets must be finite and positive")
    if suite_hash is None:
        return {"status": "no_feasible_bundle", "reason": "suite_hash required: matching I/O is not a capability requirement",
                "scope": "supplied already-admitted bundles only", "activated": False}
    feasible = []
    for identifier in evaluation_ids:
        try:
            evaluation, candidate = _validate_admitted(workspace, identifier)
            spec = CompositionSpec.model_validate(candidate["spec"])
            output = next(n for n in spec.nodes if n.id == spec.output_node)
            measured = evaluation["measured_resources"]
            disk_bytes = sum(item["size"] for aid in set(candidate["artifacts"].values())
                             for item in workspace.verify_artifact(aid)["files"].values())
            if (spec.input_type == input_type and PORTS[output.component.kind][1] == output_type
                    and evaluation["suite_hash"] == suite_hash
                    and (graph_hash is None or candidate["graph_hash"] == graph_hash)
                    and (max_disk_bytes is None or disk_bytes <= max_disk_bytes)
                    and measured["max_wall_seconds"] <= max_wall_seconds and measured["max_peak_rss_mb"] <= max_peak_rss_mb):
                feasible.append((measured["max_peak_rss_mb"], measured["max_wall_seconds"], identifier, evaluation["suite_hash"], disk_bytes, candidate["graph_hash"]))
        except (Blocked, OSError, ValueError, KeyError, TypeError):
            continue
    if not feasible:
        return {"status": "no_feasible_bundle", "scope": "supplied measured bundles only"}
    if len({row[3] for row in feasible}) > 1:
        return {"status": "no_feasible_bundle", "reason": "incomparable reference suites; supply a suite_hash", "scope": "supplied measured bundles only"}
    chosen = min(feasible)
    return {"status": "selected", "evaluation_id": chosen[2], "suite_hash": chosen[3],
            "dependency_disk_bytes": chosen[4], "graph_hash": chosen[5], "activated": False,
            "scope": "minimum measured RSS then latency among supplied suite-matching admitted bundles; not global optimality; disk counts unique registered artifact inventories, not runtime temporary storage"}


def export_deployment(workspace, deployment_id, output, include_dependencies=False, *, include_sensitive_traces=False):
    if type(include_sensitive_traces) is not bool:
        raise Blocked("include_sensitive_traces must be a boolean")
    with workspace.writer():
        deployment, evaluation, candidate = _deployment(workspace, deployment_id)
        runs = [workspace.read_record("runs", row["run_id"]) for row in evaluation["case_evidence"]]
        if not include_sensitive_traces and _sensitive_observations([deployment, evaluation, candidate, runs]):
            # Version-one import checks the exact original bundle. Refuse rather
            # than silently redact HMAC-bound evidence or create invalid bundles.
            raise SENSITIVE_TRACE_EXPORT_REQUIRES_OPT_IN(
                "sensitive actual values, token IDs or untrusted diagnostics require --include-sensitive-traces; originals were not modified")
        artifacts = {identifier: workspace.verify_artifact(identifier) for identifier in set(candidate["artifacts"].values())}
        destination = safe_path(output)
        if destination == workspace.root or workspace.root in destination.parents:
            raise Blocked("export destination must be outside the managed workspace")
        for artifact in artifacts.values():
            model = safe_path(artifact["manifest"]["model_path"])
            if destination == model or model in destination.parents:
                raise Blocked("export destination must be outside model directories")
        if not destination.parent.is_dir():
            raise Blocked("export parent directory does not exist")
        fd, temporary = tempfile.mkstemp(prefix=".composition-export-", dir=str(destination.parent))
        os.close(fd)
        try:
            manifest = {"kind": "asea-composition-deployment", "format_version": 1,
                        "dependencies_included": bool(include_dependencies), "deployment": deployment,
                        "evaluation": evaluation, "candidate": candidate, "artifacts": artifacts,
                        "runs": runs,
                        "note": "Integrity bundle, not a portable trust credential. Import/activation in another workspace is not implemented."}
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr("composition-export-v1.json", canonical(manifest))
                if include_dependencies:
                    for identifier, artifact in sorted(artifacts.items()):
                        root = safe_path(artifact["manifest"]["model_path"])
                        for relative, expected in sorted(artifact["files"].items()):
                            src = safe_file(root / relative)
                            hasher, count = hashlib.sha256(), 0
                            source_fd = os.open(str(src), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                            with os.fdopen(source_fd, "rb") as stream, archive.open("dependencies/" + identifier + "/" + relative, "w", force_zip64=True) as target:
                                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                    hasher.update(chunk)
                                    count += len(chunk)
                                    target.write(chunk)
                            if {"sha256": hasher.hexdigest(), "size": count} != expected:
                                raise Blocked("dependency changed during export")
                for identifier in artifacts:
                    workspace.verify_artifact(identifier)
            os.replace(temporary, str(destination))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        result = {"kind": "asea-composition-deployment", "format_version": 1, "deployment_id": deployment_id,
                  "output": str(destination), "dependencies_included": bool(include_dependencies), **file_hash(destination)}
        workspace.audit("export", **result)
        return result
