"""Opt-in, loopback-only public CLI bridge. No model imports or legacy catalog use.

Mount ``router`` only when SILT_ENABLE_EXPERIMENTAL=1. The router also fails
closed itself. This is a trusted single-user local tool, not a hosted service.
"""
from __future__ import annotations

import hmac
import hashlib
import ipaddress
import json
import logging
import os
from pathlib import Path
import queue
import re
import secrets
import selectors
from itertools import islice
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import List, Literal, Optional
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

LOG = logging.getLogger("asea.studio.experimental")
MAX_OUTPUT = 10 * 1024 * 1024
MAX_JOBS = 20
MAX_QUEUE = 3
MAX_BODY = 128 * 1024
SPECIALIST_REPORT_LIMIT = 16 * 1024 * 1024
SPECIALIST_STAGES = ("source-validation", "reconstruct", "reconstructed-validation", "recover", "recovered-validation")
SPECIALIST_REPORTS = ("manifest.json", "recipe.json", "preflight.json", "implementation-lock.json",
                      "source-validation.json", "reconstructed-validation.json", "recovered-validation.json",
                      "recover.receipt.json", "recover.receipt.json.history.json") + tuple(
                          stage + suffix for stage in SPECIALIST_STAGES
                          for suffix in (".started.json", ".result.json", ".logs.json"))
ACTIONS = {"compose": ["inspect", "preview", "plan", "run", "evaluate", "select", "activate", "rollback", "export", "list"],
           "compiler": ["inspect", "prune", "evaluate", "infer", "certify", "diagnose", "roundtrip"],
           "specialist": ["build", "infer", "evaluate", "finalize"],
           "execution": ["probe"],
           "validation": ["voice-check", "export-listen-batch"]}
TERMINAL = {"succeeded", "failed", "cancelled", "timed_out"}


class StudioError(Exception):
    def __init__(self, kind, message, status=400):
        self.kind, self.message, self.status = kind, message, status
        super().__init__(message)


def failure(kind, message):
    return {"type": kind, "message": message}


class ExperimentalRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request):
            try:
                response = await handler(request)
            except StudioError as exc:
                response = JSONResponse({"ok": False, "error": failure(exc.kind, exc.message)}, status_code=exc.status)
            except (RequestValidationError, ValidationError, ValueError) as exc:
                response = JSONResponse({"ok": False, "error": failure("INVALID_REQUEST", str(exc)[:4096])}, status_code=422)
            except OSError as exc:
                LOG.exception("Experimental filesystem failure")
                response = JSONResponse({"ok": False, "error": failure("IO_ERROR", str(exc)[:1024])}, status_code=500)
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            return response
        return guarded


router = APIRouter(route_class=ExperimentalRoute)


def local_gate(request: Request):
    if os.environ.get("SILT_ENABLE_EXPERIMENTAL") != "1":
        raise StudioError("DISABLED", "Experimental Studio is disabled", 404)
    host = request.url.hostname or ""
    try:
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if not local:
        raise StudioError("LOCAL_ONLY", "Use a loopback hostname; do not expose this service", 403)
    origin = request.headers.get("origin")
    expected = str(request.base_url).rstrip("/")
    if origin and origin != expected:
        raise StudioError("ORIGIN_REJECTED", "Experimental routes require the same origin", 403)
    if request.headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
        raise StudioError("ORIGIN_REJECTED", "Cross-site access is not allowed", 403)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["compose", "compiler", "execution", "validation", "specialist"]
    recipe: Optional[str] = Field(default=None, max_length=4096)
    study_name: Optional[str] = Field(default=None, max_length=80)
    study: Optional[str] = Field(default=None, max_length=4096)
    operation: str = Field(max_length=32)
    operator: str = Field(min_length=1, max_length=120)
    local_files_confirmed: bool = False
    spec: Optional[str] = Field(default=None, max_length=4096)
    suite: Optional[str] = Field(default=None, max_length=4096)
    model: Optional[str] = Field(default=None, max_length=4096)
    reference_model: Optional[str] = Field(default=None, max_length=4096)
    calibration: Optional[str] = Field(default=None, max_length=4096)
    output: Optional[str] = Field(default=None, max_length=180)
    input_text: Optional[str] = Field(default=None, max_length=32768)
    input_file: Optional[str] = Field(default=None, max_length=4096)
    evaluations: Optional[List[str]] = Field(default=None, min_length=1, max_length=64)
    input_type: Optional[Literal["text", "audio", "image"]] = None
    output_type: Optional[Literal["text", "audio"]] = None
    max_wall_seconds: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    max_peak_rss_mb: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    max_disk_bytes: Optional[int] = Field(default=None, gt=0)
    suite_hash: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    graph_hash: Optional[str] = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evaluation: Optional[str] = Field(default=None, max_length=128)
    deployment: Optional[str] = Field(default=None, max_length=128)
    confirmation: str = Field(default="", max_length=32)
    approve_high_risk: bool = False
    include_dependencies: bool = False
    output_format: Optional[Literal["raw_python", "fenced_python"]] = None
    trace_policy: Literal["none", "digest", "value"] = "none"
    value_trace_consent: bool = False
    include_sensitive_traces: bool = False
    sensitive_export_confirmed: bool = False
    keep_experts: int = Field(default=4, ge=1, le=1024)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    resource_profile: Optional[Literal["observe_only", "process_as", "cgroup_v2"]] = None
    memory_mib: Optional[int] = Field(default=None, ge=32, le=1048576)
    case_timeout: Optional[float] = Field(default=None, gt=0, le=86400, allow_inf_nan=False)
    preserve_router_fp32: Optional[bool] = None
    scorer: Optional[Literal["probability_mass", "reap_dispatched"]] = None
    minimum_expert_observations: Optional[int] = Field(default=None, ge=1, le=1000000)
    worker_timeout_seconds: Optional[float] = Field(default=None, ge=0.05, le=900, allow_inf_nan=False)
    audio_file: Optional[str] = Field(default=None, max_length=4096)
    text: Optional[str] = Field(default=None, max_length=32768)
    language: Optional[str] = Field(default=None, max_length=64)
    model_manifest: Optional[str] = Field(default=None, max_length=4096)
    manifest: Optional[str] = Field(default=None, max_length=4096)
    # Operator intent only: the public voice CLI has no --purpose/--use flag.
    use_purpose: Optional[Literal["noncommercial_experimental", "commercial"]] = None
    max_length: int = Field(default=128, ge=1, le=256)
    max_new_tokens: int = Field(default=32, ge=1, le=384)
    memory_budget_mib: int = Field(default=2048, ge=128, le=3072)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)

    @model_validator(mode="after")
    def operation_budgets(self):
        # API-side bounds: hidden inputs, crafted JSON and non-browser clients
        # cannot borrow build's longer budget for old operations or finalization.
        if (self.mode, self.operation) != ("specialist", "build") and self.timeout_seconds > 900:
            raise ValueError("timeout_seconds must be <=900 except specialist build (<=3600)")
        if self.mode != "specialist" and self.max_new_tokens > 128:
            raise ValueError("max_new_tokens must be <=128 outside specialist")
        if self.mode != "specialist" and any(getattr(self, key) is not None for key in ("recipe", "study_name", "study")):
            raise ValueError("recipe/study fields apply only to specialist")
        return self


def safe_local(path: Path, root: Optional[Path] = None):
    """Reject symlinks/special files; resolve containment before any serving/write."""
    path = path.absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise StudioError("UNSAFE_PATH", "Symlink paths are not allowed")
    resolved = path.resolve()
    if root is not None and not resolved.is_relative_to(root.resolve()):
        raise StudioError("UNSAFE_PATH", "Path escapes the experimental workspace")
    if resolved.exists():
        info = resolved.stat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise StudioError("UNSAFE_PATH", "Only regular files/directories are accepted")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise StudioError("NOT_OWNER", "Local inputs must be owned by the Studio operator")
    return resolved


def approved_recipe(path):
    """Validate only the operator-approved recipe, never source weights/final data.

    Normalize relative paths before taking the private queued-job snapshot. The
    public strict recipe parser is the authority for training knobs, not the GUI.
    """
    path = safe_local(Path(path).expanduser())
    if not path.is_file() or path.stat().st_size > MAX_BODY:
        raise StudioError("INVALID_RECIPE", "Recipe must be an existing regular JSON file <=128 KiB")
    from asea.specialist.workflow import recipe_config
    config = recipe_config(path)
    for key in ("source_path", "calibration", "training", "validation_data", "validation_suite", "data_manifest", "selection_lock"):
        candidate = safe_local(Path(config[key]))
        if not (candidate.is_dir() if key == "source_path" else candidate.is_file()):
            raise StudioError("INVALID_RECIPE", key + " must exist with the expected local file/directory type")
    return config


def specialist_argv(request, root, artifacts):
    """Closed initial workflow surface; reconstruct/recover remain public CLI only."""
    op = request.operation
    allowed = {"mode", "operation", "operator", "local_files_confirmed", "timeout_seconds"}
    allowed |= {"build": {"recipe", "study_name"},
                "infer": {"model", "input_text", "dtype", "max_new_tokens", "memory_budget_mib"},
                "evaluate": {"model", "suite", "output", "dtype", "max_new_tokens", "memory_budget_mib"},
                "finalize": {"study", "suite", "output", "confirmation"}}[op]
    # Even explicitly supplied irrelevant defaults are rejected, not silently ignored.
    if request.model_fields_set - allowed:
        raise StudioError("INVALID_ARGUMENT", "Fields do not apply to specialist " + op + ": " +
                          ", ".join(sorted(request.model_fields_set - allowed)))
    argv = [sys.executable, "-m", "asea.specialist", op]
    produced = []

    def add(flag, value, kind=None):
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise StudioError("MISSING_ARGUMENT", flag + " requires a nonblank value without NUL")
        if kind:
            candidate = safe_local(Path(value).expanduser())
            if kind == "file" and not candidate.is_file():
                raise StudioError("INVALID_ARGUMENT", flag + " requires an existing local file")
            if kind == "directory" and candidate.exists() and not candidate.is_dir():
                raise StudioError("INVALID_ARGUMENT", flag + " requires a local directory")
            value = str(candidate)
        argv.append(flag + "=" + value)
        return value

    if op == "build":
        add("--recipe", request.recipe, "file")
        config = approved_recipe(request.recipe)
        if not request.study_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", request.study_name):
            raise StudioError("UNSAFE_OUTPUT", "Study name must be a bounded name, not a path")
        study = safe_local(artifacts.parent / "studies" / request.study_name, artifacts.parent)
        source = Path(config["source_path"])
        if study.exists() or study == source or source in study.parents or study in source.parents:
            raise StudioError("UNSAFE_OUTPUT", "Study must be a new directory disjoint from the source")
        add("--workspace", str(study))
        produced.append((study, "specialist_study"))
    elif op in ("infer", "evaluate"):
        add("--model", request.model, "directory")
        if op == "infer":
            add("--prompt", request.input_text)  # Exact prompt: never prefix or retune.
        dtype = request.dtype if "dtype" in request.model_fields_set else "bfloat16"
        if dtype not in ("bfloat16", "float32"):
            raise StudioError("INVALID_ARGUMENT", "Specialist supports only bfloat16 or float32")
        add("--dtype", dtype)
        add("--max-new-tokens", str(request.max_new_tokens if "max_new_tokens" in request.model_fields_set else 256))
        if "memory_budget_mib" in request.model_fields_set:
            add("--memory-budget-mib", str(request.memory_budget_mib))
        if op == "evaluate":
            # No value-retention UI. Keep exact public default, do not weaken old consent.
            add("--trace-policy", "digest")
    elif op == "finalize":
        if request.confirmation != "FINALIZE":
            raise StudioError("CONFIRM_FINALIZE", "Type FINALIZE: consumes the frozen study's final once, even on failure; no retuning or retry")
        study = Path(add("--study", request.study, "directory"))
        manifest = safe_local(study / "manifest.json")
        if not manifest.is_file():
            raise StudioError("INVALID_ARGUMENT", "Study must contain its local manifest.json")
        # Do not read final answers or touch the consumed marker in Studio. The
        # public CLI verifies frozen source/code/suite paths before one-shot use.
    if op in ("evaluate", "finalize"):
        add("--suite", request.suite, "file")
        name = request.output or ("evaluation.json" if op == "evaluate" else "final-result.json")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,174}\.json", name):
            raise StudioError("UNSAFE_OUTPUT", "Specialist output must be a new JSON filename, not a path")
        path = safe_local(artifacts / name, artifacts.parent)
        if path.exists():
            raise StudioError("OUTPUT_EXISTS", "Output must not already exist")
        add("--output", str(path))
        produced.append((path, "specialist_report"))
    return argv, produced


def build_argv(request: JobRequest, root: Path, artifacts: Path):
    if request.operation not in ACTIONS[request.mode]:
        raise StudioError("INVALID_ACTION", "Operation is not allowlisted for this mode")
    if not request.operator.strip() or not request.local_files_confirmed:
        raise StudioError("CONFIRM_LOCAL_FILES", "Name the operator and confirm permission to read the local inputs")
    if request.mode == "specialist":
        return specialist_argv(request, root, safe_local(artifacts, root))
    op = request.operation
    target = (request.mode, op)
    # These controls are closed, typed operator intents, not arbitrary CLI flags.
    controls = {"output_format": ("compose", "preview"),
                "trace_policy": ("compose", "evaluate"),
                "value_trace_consent": ("compose", "evaluate"),
                "include_sensitive_traces": ("compose", "export"),
                "sensitive_export_confirmed": ("compose", "export")}
    for name, action in controls.items():
        if target != action and getattr(request, name) != JobRequest.model_fields[name].default:
            raise StudioError("INVALID_ARGUMENT", name + " does not apply to this action")
    if request.trace_policy == "value" and not request.value_trace_consent:
        raise StudioError("CONFIRM_VALUE_TRACE", "Value traces contain return values; explicit local storage consent is required")
    if request.value_trace_consent and request.trace_policy != "value":
        raise StudioError("INVALID_ARGUMENT", "Value consent requires trace_policy=value")
    if request.include_sensitive_traces != request.sensitive_export_confirmed:
        raise StudioError("CONFIRM_SENSITIVE_EXPORT", "Sensitive export requires its separate explicit confirmation")
    if target == ("compose", "preview"):
        allowed = {"mode", "operation", "operator", "local_files_confirmed", "timeout_seconds", "spec", "output_format"}
        for name in request.model_fields_set - allowed:
            if getattr(request, name) != JobRequest.model_fields[name].default:
                raise StudioError("INVALID_ARGUMENT", name + " does not apply to preview")
    audits = {( "compiler", name) for name in ("diagnose", "roundtrip")}
    applicable = {
        "resource_profile": {("compose", "evaluate")}, "memory_mib": {("compose", "evaluate")},
        "case_timeout": {("compose", "evaluate")},
        "preserve_router_fp32": audits | {("compiler", "prune")},
        "scorer": {("compiler", "prune")}, "minimum_expert_observations": {("compiler", "prune")},
        "worker_timeout_seconds": audits,
        **{name: {("validation", "voice-check")} for name in ("audio_file", "text", "language", "model_manifest")},
        "manifest": {("validation", "export-listen-batch")},
        "use_purpose": {("validation", "voice-check"), ("validation", "export-listen-batch")},
    }
    for name, targets in applicable.items():
        if getattr(request, name) is not None and target not in targets:
            raise StudioError("INVALID_ARGUMENT", name + " does not apply to this action")
    if request.mode in ("execution", "validation"):
        allowed = {"mode", "operation", "operator", "local_files_confirmed", "timeout_seconds"}
        if request.mode == "validation":
            allowed |= {"output", "use_purpose"} | ({"audio_file", "text", "language", "model_manifest"}
                       if op == "voice-check" else {"manifest"})
        for name in request.model_fields_set - allowed:
            if getattr(request, name) != JobRequest.model_fields[name].default:
                raise StudioError("INVALID_ARGUMENT", name + " does not apply to this action")
    if request.worker_timeout_seconds is not None and request.worker_timeout_seconds > request.timeout_seconds:
        raise StudioError("INVALID_ARGUMENT", "Worker timeout must not exceed the whole-job timeout")
    if target == ("compiler", "certify") and request.dtype == "bfloat16":
        raise StudioError("INVALID_ARGUMENT", "The public certify CLI does not accept bfloat16")
    argv = [sys.executable, "-m", "asea." + request.mode]
    if request.mode in ("compose", "validation") and target != ("compose", "preview"):
        argv += ["--workspace", str(root / request.mode)]
    argv.append(op)
    produced = []

    def add(flag, value, required=True, path=False):
        if value is None or value == "":
            if required:
                raise StudioError("MISSING_ARGUMENT", flag + " is required")
            return
        if "\x00" in str(value):
            raise StudioError("INVALID_ARGUMENT", "NUL characters are forbidden")
        if path:
            value = str(safe_local(Path(value).expanduser()))
        # Equals binds even text beginning with '-' as a value, never a new flag.
        argv.append(flag + "=" + str(value))

    def output(default, role):
        name = request.output or default
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", name):
            raise StudioError("UNSAFE_OUTPUT", "Output must be a filename, not an absolute path or directory traversal")
        target = safe_local(artifacts / name, root)
        if target.exists():
            raise StudioError("OUTPUT_EXISTS", "Output must not already exist")
        produced.append((target, role))
        return str(target)

    if request.mode == "compose":
        if op in ("inspect", "preview", "plan", "run", "evaluate"):
            add("--spec", request.spec, path=True)
        if op == "preview":
            add("--output-format", request.output_format)
        elif op == "run":
            # The CLI validates combinations: image inputs may include a text question.
            add("--input", request.input_text, required=False)
            add("--input-file", request.input_file, required=False, path=True)
        elif op == "evaluate":
            add("--suite", request.suite, path=True)
            # Omitted requests retain the public CLI's none default and legacy argv.
            if "trace_policy" in request.model_fields_set:
                add("--trace-policy", request.trace_policy)
            for flag, value in (("--resource-profile", request.resource_profile),
                                ("--memory-mib", request.memory_mib), ("--case-timeout", request.case_timeout)):
                add(flag, value, required=False)
        elif op == "select":
            if not request.evaluations:
                raise StudioError("MISSING_ARGUMENT", "--evaluations is required")
            if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,127}", value) for value in request.evaluations):
                raise StudioError("INVALID_ARGUMENT", "Evaluation IDs must be bounded identifiers, never CLI flags")
            # nargs requires separate values. Identifiers cannot begin with '-' or contain whitespace.
            argv.extend(["--evaluations", *request.evaluations])
            for flag, value in (("--input-type", request.input_type), ("--output-type", request.output_type),
                                ("--max-wall-seconds", request.max_wall_seconds), ("--max-peak-rss-mb", request.max_peak_rss_mb)):
                add(flag, value)
            for flag, value in (("--suite-hash", request.suite_hash), ("--graph-hash", request.graph_hash),
                                ("--max-disk-bytes", request.max_disk_bytes)):
                add(flag, value, required=False)
        elif op == "activate":
            if request.confirmation != "ACTIVATE":
                raise StudioError("CONFIRM_ACTIVATION", "Type ACTIVATE and name the operator; nothing is activated automatically")
            add("--evaluation", request.evaluation)
            if request.approve_high_risk:
                argv.append("--approve-high-risk")
                add("--approver", request.operator)
        elif op == "rollback":
            if request.confirmation != "ROLLBACK":
                raise StudioError("CONFIRM_ROLLBACK", "Type ROLLBACK and name the operator")
            add("--deployment", request.deployment)
        elif op == "export":
            add("--deployment", request.deployment)
            add("--output", output("deployment.zip", "deployment_export"))
            if request.include_dependencies:
                argv.append("--include-dependencies")
            if request.include_sensitive_traces:
                argv.append("--include-sensitive-traces")
    elif request.mode == "validation":
        if op == "voice-check":
            add("--audio", request.audio_file, path=True)
            add("--text", request.text)
            add("--language", request.language)
            add("--model-manifest", request.model_manifest, path=True)
        else:
            add("--manifest", request.manifest, path=True)
        add("--output", output("voice-evidence.json" if op == "voice-check" else "listening-packet.json",
                               "validation_evidence"))
    elif request.mode == "compiler" and target in audits:
        dtype = request.dtype if "dtype" in request.model_fields_set else "float16"
        length = request.max_length if "max_length" in request.model_fields_set else 64
        tokens = request.max_new_tokens if "max_new_tokens" in request.model_fields_set else 16
        if dtype not in ("float16", "bfloat16") or length > 64 or tokens > 32:
            raise StudioError("INVALID_ARGUMENT", "Audits require float16/bfloat16, max_length <=64 and max_new_tokens <=32")
        add("--model", request.model, path=True)
        add("--suite", request.suite, required=op == "diagnose", path=True)
        if op == "diagnose":
            add("--reference-model", request.reference_model, required=False, path=True)
        elif request.reference_model is not None:
            raise StudioError("INVALID_ARGUMENT", "Roundtrip has no --reference-model option")
        add("--output", output("diagnostics.json" if op == "diagnose" else "roundtrip", "compiler_audit"))
        for flag, value in (("--dtype", dtype), ("--max-length", length), ("--max-new-tokens", tokens),
                            ("--memory-budget-mib", request.memory_budget_mib),
                            ("--worker-timeout-seconds", request.worker_timeout_seconds
                             if request.worker_timeout_seconds is not None else min(120, request.timeout_seconds))):
            add(flag, value)
        if request.preserve_router_fp32:
            argv.append("--preserve-router-fp32")
    elif request.mode == "compiler":
        add("--model", request.model, path=True)
        if op != "inspect":
            for flag, value in (("--dtype", request.dtype), ("--max-length", request.max_length),
                                ("--memory-budget-mib", request.memory_budget_mib)):
                add(flag, value)
        if op == "prune":
            add("--output", output("candidate", "compiler_candidate"))
            add("--keep-experts", request.keep_experts)
            add("--calibration", request.calibration, path=True)
            add("--scorer", request.scorer, required=False)
            add("--minimum-expert-observations", request.minimum_expert_observations, required=False)
            if request.preserve_router_fp32:
                argv.append("--preserve-router-fp32")
        elif op == "evaluate":
            add("--suite", request.suite, path=True)
            add("--reference-model", request.reference_model, required=False, path=True)
            add("--evidence-output", output("evaluation.json", "compiler_evidence"))
        elif op == "certify":
            add("--suite", request.suite, path=True)
            add("--reference-model", request.reference_model, path=True)
            add("--certificate-output", output("compiler-certificate.json", "compiler_certificate"))
        elif op == "infer":
            add("--prompt", request.input_text)
        if op in ("evaluate", "infer", "certify"):
            add("--max-new-tokens", request.max_new_tokens)
    if request.approve_high_risk and (request.mode, op) != ("compose", "activate"):
        raise StudioError("INVALID_ARGUMENT", "High-risk approval only applies to explicit Compose activation")
    return argv, produced


def decode_cli(job, stdout, stderr, code):
    """Preserve each public envelope; never manufacture a boolean quality pass."""
    source = stderr if job["mode"] == "validation" and code != 0 and not stdout.strip() else stdout
    payload = json.loads(source, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(payload, dict):
        raise ValueError("Expected one public CLI JSON object")
    if job["mode"] == "specialist":
        if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or payload.get("command") != job["operation"]
                or type(payload.get("completed")) is not bool
                or not isinstance(payload.get("status"), str)):
            raise ValueError("Expected the specialist public command/completed receipt")
        expected = "BUILT_UNCERTIFIED" if job["operation"] in ("build", "finalize") else "completed"
        if payload["completed"] and payload["status"] != expected:
            raise ValueError("Specialist completion/status fields disagree")
        return payload, code == 0 and payload["completed"]
    if job["mode"] == "execution":
        if (payload.get("schema") != "silt.resource-controls.v1" or
                not all(isinstance(payload.get(key), dict) for key in ("process_as", "cgroup_v2", "windows"))):
            raise ValueError("Expected the read-only public execution probe envelope")
        return payload, code == 0
    if job["mode"] == "validation":
        if code != 0 and payload.get("status") == "blocked":
            return payload, False
        expected = "voice_evidence_pending" if job["operation"] == "voice-check" else "proposed_listening_packet"
        if (payload.get("kind") != expected or payload.get("status") != "pending_human"
                or payload.get("quality_admission") is not False):
            raise ValueError("Expected pending_human evidence, never listening approval")
        return payload, code == 0
    if not isinstance(payload.get("ok"), bool):
        raise ValueError("Expected one JSON object with boolean ok")
    if (job["mode"], job["operation"]) == ("compose", "preview") and payload["ok"] and code == 0:
        preview = payload.get("result")
        if not isinstance(preview, dict) or preview.get("read_only") is not True or preview.get("applied") is not False:
            raise ValueError("Expected read-only preview with applied=false")
    if (job["mode"] == "compiler" and job["operation"] in ("diagnose", "roundtrip")
            and payload["ok"] and code == 0):
        if (payload.get("status") != "AUDIT_ONLY" or payload.get("admission") != "UNADMITTED"
                or payload.get("quality_certification") is not False
                or payload.get("stdout_contract") != "bounded_receipt_v1"):
            raise ValueError("Expected bounded AUDIT_ONLY / UNADMITTED receipt; no legacy evidence upgrade")
    return payload, code == 0 and payload["ok"]


def completion_summary(job):
    result = job.get("result") or {}
    summary = {"process_status": job["status"], "exit_code": job.get("exit_code"),
               "claim": "Process completion is not admission; inspect the original CLI evidence."}
    if job["mode"] == "specialist":
        evidence = result.get("result") if isinstance(result.get("result"), dict) else {}
        summary.update(claim="Completion, engineering, observed suite quality, and certificate are independent. BUILT_UNCERTIFIED is never a quality PASS or certification.",
                       cli_status=result.get("status", UNAVAILABLE), completed=result.get("completed", UNAVAILABLE),
                       engineering_complete=evidence.get("engineering_complete", result.get("engineering_complete", UNAVAILABLE)),
                       quality_pass=evidence.get("quality_pass", UNAVAILABLE),
                       certificate=evidence.get("certificate", UNAVAILABLE))
        if job["status"] != "succeeded":
            summary["diagnosis"] = "infrastructure_blocked" if result.get("status") == "BLOCKED" or (job.get("error") or {}).get("type") in {"TIMEOUT", "SPAWN_ERROR", "OUTPUT_LIMIT", "WORKER_ERROR"} else "rejected_or_incomplete"
        elif evidence.get("quality_pass") is False:
            summary["diagnosis"] = "completed_quality_not_passed"
        elif evidence.get("quality_pass") is True:
            summary["diagnosis"] = "observed_suite_pass_only_not_certificate"
        else:
            summary["diagnosis"] = "completed_quality_not_evaluated"
        for key in ("manifest", "workspace", "output", "output_sha256"):
            if key in result:
                summary[key] = result[key]
    elif (job["mode"], job["operation"]) == ("compose", "preview"):
        summary["claim"] = "Read-only proposal; applied=false. No spec rewrite or target workspace writes; outer Studio job log only."
    elif job["mode"] == "execution":
        summary["claim"] = "Read-only capability observation; no enforcement test or guarantee."
    elif job["mode"] == "validation":
        summary.update(claim="Waveform facts / proposed packet only. Quality remains pending_human.",
                       quality_admission=False, use_purpose=job.get("use_purpose"),
                       purpose_scope="operator metadata only; not CLI policy enforcement or authorization")
    elif job["mode"] == "compiler" and job["operation"] in ("diagnose", "roundtrip"):
        summary.update(claim="Research audit only; never admission or preserved-quality certification.",
                       admission="UNADMITTED", quality_certification=False)
        for key in ("status", "publication", "report", "report_sha256", "report_bytes", "output",
                    "manifest", "manifest_sha256", "comparisons", "dtype", "preserve_router_fp32"):
            if key in result:
                summary[key] = result[key]
    return summary


REDACTED = "[redacted: explicit viewer reveal required]"
UNAVAILABLE = "unavailable / original_unknown (not captured; no backfill)"


def viewer_reveal(request: Request):
    value = request.headers.get("X-SILT-Reveal-Sensitive", "false")
    if value not in ("true", "false"):
        raise StudioError("INVALID_ARGUMENT", "X-SILT-Reveal-Sensitive must be true or false", 422)
    return value == "true"


def project_job(original, reveal=False):
    """Display copy only; never mutate/re-sign originals or claim HMAC validation.

    Retention consent is not viewer/export consent. Historic jobs have no inferred
    storage authorization. A value trace also needs its oracle's evidence flag.
    Raw logs are withheld by default, including failed/truncated CLI output.
    """
    denied_values = False

    def scrub(value, value_authorized=False, in_trace=False):
        nonlocal denied_values
        if isinstance(value, list):
            return [scrub(item, value_authorized, in_trace) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        flags = value.get("evidence_flags", [])
        flagged = isinstance(flags, list) and "sensitive_actual_value_trace" in flags
        for key, item in value.items():
            if key == "return_trace":
                # Only the containing host-oracle envelope can flag its trace.
                policy = item.get("policy") if isinstance(item, dict) else None
                value_policy = isinstance(policy, dict) and policy.get("return_retention") == "value"
                result[key] = scrub(item, reveal and flagged and value_policy, True)
            elif key == "value" and (in_trace or "retention" in value or "value_type" in value):
                if value.get("retention") == "value" and value_authorized:
                    result[key] = item  # Exact bounded JSON; never interpret its contents.
                else:
                    result[key] = REDACTED
                    denied_values = True
            elif not reveal and (key in ("stdout", "stderr", "generated_token_ids", "returned_sequence_token_ids")
                    or (original.get("mode") == "specialist" and key in ("input_token_ids", "native_sequence_token_ids"))):
                result[key] = REDACTED if item else item
            else:
                result[key] = scrub(item, value_authorized, in_trace)
        return result

    projected = scrub(original)
    raw_allowed = reveal and not denied_values
    if not raw_allowed:
        def hide_streams(value):
            if isinstance(value, list):
                return [hide_streams(item) for item in value]
            if isinstance(value, dict):
                return {key: (REDACTED if item else item) if key in ("stdout", "stderr")
                        else item if key == "value" else hide_streams(item) for key, item in value.items()}
            return value
        projected = hide_streams(projected)
        # Never keep an unredacted alternate copy in raw JSON, logs or error text.
        projected["stdout"] = REDACTED if original.get("stdout") else ""
        projected["stderr"] = REDACTED if original.get("stderr") else ""
        if projected.get("error"):
            projected["error"] = {"type": projected["error"].get("type", "UNKNOWN"),
                                  "message": "See the privacy-filtered result; raw error text withheld"}
    projected["privacy"] = {"projection_only": True, "original_integrity_validated": False,
        "viewer_reveal": reveal, "raw_available": raw_allowed,
        "values_without_evidence_flag_withheld": denied_values and reveal,
        "storage_authorization": original.get("observability_authorization", UNAVAILABLE)}
    projected["observability"] = observability_panels(projected.get("result"))
    return projected


def observability_panels(payload):
    """Bounded existing-envelope traversal; no workspace/dataset/model reads."""
    generation, oracle, bindings = [], [], []

    def visit(value, path="result", depth=0):
        if depth > 32 or len(generation) + len(oracle) + len(bindings) >= 256:
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + "/" + str(index), depth + 1)
        elif isinstance(value, dict):
            if "generation_trace_v1" in value:
                traces = value["generation_trace_v1"]
                keys = ("node", "status", "stage", "requested", "forwarded", "resolved", "input_token_count",
                        "output_token_count", "generated_token_count", "eos_token_ids", "eos_positions",
                        "eos_positions_basis", "cap_reached", "token_ids_truncated", "stop_reason", "media")
                nodes = [{**{key: node.get(key, UNAVAILABLE) for key in keys}, "original_trace_projection": node}
                         if isinstance(node, dict) else node for node in traces] if isinstance(traces, list) else traces
                generation.append({"path": path, "nodes": nodes})
            if any(key in value for key in ("return_trace", "diagnostic", "termination")):
                oracle.append({"path": path, "trace_policy": (value.get("return_trace") or {}).get("policy", UNAVAILABLE),
                    "retained_or_redacted_returns": value.get("return_trace", UNAVAILABLE),
                    "HOST_diagnostic_with_UNTRUSTED_candidate_claims": value.get("diagnostic", UNAVAILABLE),
                    "HOST_original_exit_and_separate_cleanup_termination": value.get("termination", UNAVAILABLE),
                    "evidence_flags": value.get("evidence_flags", UNAVAILABLE)})
            if "obs_v1" in value:
                bindings.append({"path": path, "binding": value["obs_v1"]})
            for key, child in value.items():
                # Candidate return values/strings are data, never telemetry schemas.
                if key not in ("value", "generation_trace_v1", "return_trace", "diagnostic", "termination", "obs_v1"):
                    visit(child, path + "/" + key, depth + 1)
    visit(payload)
    return {"generation_requested_forwarded_resolved_tokens_media": generation or UNAVAILABLE,
            "oracle": oracle or UNAVAILABLE, "case_bindings": bindings or UNAVAILABLE,
            "scope": "Existing CLI envelope only; linked run details may be unavailable here. No inference, backfill, or integrity certification."}


class JobManager:
    """One daemon worker, bounded pending queue and logs; POSIX process groups."""
    def __init__(self, root):
        if os.name != "posix":
            raise StudioError("UNSUPPORTED_PLATFORM", "Process-group isolation currently requires POSIX", 503)
        self.root = safe_local(Path(root).expanduser())
        if ".studio" in self.root.parts:
            raise StudioError("UNSAFE_WORKSPACE", "Use a workspace outside legacy .studio")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        # Never allow two server workers to load models concurrently in this root.
        import fcntl
        self.lockfile = safe_local(self.root / ".worker.lock", self.root).open("a")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lockfile.close()
            raise StudioError("WORKSPACE_BUSY", "Use one Studio server worker per experimental root", 503)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.pending = queue.Queue(maxsize=MAX_QUEUE)
        self.jobs = {}
        self.events = {}
        self.stopping = threading.Event()
        self.process = None
        jobs_root = safe_local(self.root / "jobs", self.root)
        jobs_root.mkdir(exist_ok=True, mode=0o700)
        # Persisted terminal records are useful after restart; never replay a command.
        for file in sorted(jobs_root.glob("*/evidence.json"), key=lambda p: p.stat().st_mtime)[-MAX_JOBS:]:
            try:
                safe_local(file, self.root)
                if file.stat().st_size > MAX_OUTPUT * 8:
                    continue
                job = json.loads(file.read_text())
                if not re.fullmatch(r"[a-f0-9]{32}", job["id"]):
                    continue
                if job["status"] not in TERMINAL:
                    job.update(status="failed", error=failure("SERVER_RESTARTED", "Job interrupted; not replayed"))
                self.jobs[job["id"]] = job
                self.events[job["id"]] = threading.Event()
                self._persist(job)
            except (OSError, ValueError, KeyError, StudioError):
                LOG.warning("Ignoring invalid experimental job record %s", file)
        self.worker = threading.Thread(target=self._worker, name="silt-experimental-cli", daemon=True)
        self.worker.start()

    def _persist(self, job):
        directory = safe_local(self.root / "jobs" / job["id"], self.root)
        directory.mkdir(exist_ok=True, mode=0o700)
        path = safe_local(directory / "evidence.json", self.root)
        temp = safe_local(directory / "evidence.tmp", self.root)
        temp.write_text(json.dumps(job, ensure_ascii=True, allow_nan=False))
        os.chmod(temp, 0o600)
        temp.replace(path)

    def submit(self, request):
        with self.lock:
            if self.stopping.is_set():
                raise StudioError("SHUTTING_DOWN", "Worker is shutting down", 503)
            if self.pending.full():
                raise StudioError("QUEUE_FULL", "At most three jobs may wait behind the active job", 429)
            job_id = uuid.uuid4().hex
            directory = self.root / "jobs" / job_id
            artifacts = directory / "artifacts"
            argv, produced = build_argv(request, self.root, artifacts)
            artifacts.mkdir(parents=True, mode=0o700)
            specialist_authorization = None
            if request.mode == "specialist":
                specialist_authorization = {"operator": request.operator.strip(),
                    "local_files_confirmed": request.local_files_confirmed,
                    "confirmation": request.confirmation,
                    "approved_at": time.time(), "request": request.model_dump(exclude_unset=True),
                    "scope": "Trusted local input/path authorization, not a signature or quality approval"}
                if request.operation == "build":
                    # Freeze normalized controls (including source/dataset annotations)
                    # before queueing. Later edits to the original recipe cannot retune this job.
                    config = approved_recipe(request.recipe)
                    snapshot = safe_local(directory / "recipe.approved.json", directory)
                    raw = json.dumps(config, ensure_ascii=True, sort_keys=True, allow_nan=False).encode()
                    with snapshot.open("xb") as stream:
                        stream.write(raw)
                    os.chmod(snapshot, 0o600)
                    specialist_authorization.update(recipe_snapshot=str(snapshot),
                        recipe_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                        source_path=config["source_path"], source_metadata=config["source_metadata"],
                        dataset_paths={key: config[key] for key in ("calibration", "training", "validation_data", "validation_suite", "data_manifest", "selection_lock")})
                    argv = ["--recipe=" + str(snapshot) if arg.startswith("--recipe=") else arg for arg in argv]
                    safe_local(directory / "studies", directory).mkdir(mode=0o700)
            job = {"id": job_id, "mode": request.mode, "operation": request.operation,
                   "operator": request.operator.strip(), "status": "queued", "created_at": time.time(),
                   "use_purpose": request.use_purpose,
                   "observability_authorization": {"schema_version": 1, "source": "studio_operator",
                       "operator": request.operator.strip(), "trace_policy": request.trace_policy,
                       "value_storage_consent": request.value_trace_consent,
                       "include_sensitive_traces": request.include_sensitive_traces,
                       "sensitive_export_confirmed": request.sensitive_export_confirmed},
                   "argv": argv, "timeout_seconds": request.timeout_seconds,
                   "high_risk_approved": request.approve_high_risk,
                   "confirmation": request.confirmation, "stdout": "", "stderr": "", "result": None,
                   "error": None, "exit_code": None, "artifacts": []}
            if specialist_authorization is not None:
                job["specialist_authorization"] = specialist_authorization
                job["specialist_produced"] = [{"path": str(path.relative_to(self.root)), "role": role} for path, role in produced]
            self.jobs[job_id] = job
            self.events[job_id] = threading.Event()
            self._persist(job)
            self.pending.put_nowait((job_id, produced))
            self._trim()
            return self.get(job_id)

    def _trim(self):
        for job_id in list(self.jobs):
            if len(self.jobs) <= MAX_JOBS:
                break
            if self.jobs[job_id]["status"] in TERMINAL:
                for name in ("evidence.json", "stderr.log"):
                    safe_local(self.root / "jobs" / job_id / name, self.root).unlink(missing_ok=True)
                self.jobs.pop(job_id)
                self.events.pop(job_id, None)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise StudioError("JOB_NOT_FOUND", "Unknown or expired job", 404)
            result = json.loads(json.dumps(self.jobs[job_id]))
            if result["mode"] == "specialist" and result["operation"] == "build":
                workspace = safe_local(self.root / "jobs" / job_id, self.root)
                for item in result.get("specialist_produced", []):
                    if item["role"] == "specialist_study":
                        study = safe_local(self.root / item["path"], workspace)
                        result["study_status"] = {"path": str(study),
                            "claim": "Read-only file presence, not stage success; a started record alone is not completion.",
                            "stages": [{"stage": stage,
                                "started_record_exists": safe_local(study / (stage + ".started.json"), workspace).is_file(),
                                "result_record_exists": safe_local(study / (stage + ".result.json"), workspace).is_file()}
                                for stage in SPECIALIST_STAGES],
                            "manifest_exists": safe_local(study / "manifest.json", workspace).is_file()}
            return result

    def list(self):
        with self.lock:
            return [{k: v for k, v in job.items() if k not in ("stdout", "stderr", "result", "argv")}
                    for job in reversed(list(self.jobs.values()))]

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] not in TERMINAL:
                self.events[job_id].set()
                if job["status"] == "queued":
                    self.jobs[job_id].update(status="cancelled", error=failure("CANCELLED", "Cancelled before starting"))
                    self._persist(self.jobs[job_id])
            return self.get(job_id)

    @staticmethod
    def _kill(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def _worker(self):
        while not self.stopping.is_set():
            try:
                job_id, produced = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                with self.lock:
                    if job_id not in self.jobs or self.events[job_id].is_set():
                        continue
                    job = self.jobs[job_id]
                    job.update(status="running", started_at=time.time())
                    self._persist(job)
                self._run(job, produced)
            except Exception as exc:
                LOG.exception("Experimental CLI worker failed")
                with self.lock:
                    if job_id in self.jobs:
                        self.jobs[job_id].update(status="failed", error=failure("WORKER_ERROR", str(exc)[:2048]))
                        self._persist(self.jobs[job_id])
            finally:
                self.pending.task_done()

    def _run(self, job, produced):
        env = dict(os.environ)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        # CLI test-fixture switches are never enabled by Studio.
        env.pop("ASEA_COMPOSE_ALLOW_FIXTURES", None)
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        reason = None
        process = None
        started = time.monotonic()
        total = 0
        selector = selectors.DefaultSelector()
        try:
            process = subprocess.Popen(job["argv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=env,
                                       cwd=str(Path.cwd()), shell=False)
            self.process = process
            for name in buffers:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            stopping_at = None
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if reason is None:
                    if self.events[job["id"]].is_set() or self.stopping.is_set():
                        reason = failure("CANCELLED", "Operator cancelled the job")
                    elif now - started > job["timeout_seconds"]:
                        reason = failure("TIMEOUT", "CLI exceeded its bounded wall-time budget")
                if reason and stopping_at is None:
                    self._kill(process, signal.SIGTERM)
                    stopping_at = now
                if stopping_at is not None and now - stopping_at > 0.5:
                    self._kill(process, signal.SIGKILL)
                    if now - stopping_at > 2:
                        break
                for key, _ in selector.select(0.05):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    remaining = MAX_OUTPUT - total
                    buffers[key.data].extend(data[:max(0, remaining)])
                    total += min(len(data), max(0, remaining))
                    if len(data) > remaining and reason is None:
                        reason = failure("OUTPUT_LIMIT", "CLI stdout + stderr exceeded 10 MiB; process group stopped")
            if process.poll() is None:
                self._kill(process, signal.SIGKILL)
            process.wait(timeout=3)
        except OSError as exc:
            reason = failure("SPAWN_ERROR", str(exc)[:2048])
        finally:
            if process is not None:
                self._kill(process, signal.SIGKILL)  # Do not leave descendants behind on successful exit either.
                if process.poll() is None:
                    process.wait(timeout=3)
                for name in buffers:
                    stream = getattr(process, name)
                    if stream:
                        stream.close()
            selector.close()
            self.process = None
        stdout = buffers["stdout"].decode("utf-8", "replace")
        stderr = buffers["stderr"].decode("utf-8", "replace")
        payload = None
        code = process.returncode if process is not None else None
        if reason is None:
            try:
                payload, completed = decode_cli(job, stdout, stderr, code)
            except (ValueError, RecursionError) as exc:
                reason = failure("INVALID_CLI_JSON", str(exc)[:1024])
            if reason is None and not completed:
                reason = failure("CLI_FAILED", "CLI returned a failure; see raw JSON evidence and exit code")
        status = "succeeded" if reason is None else {"CANCELLED": "cancelled", "TIMEOUT": "timed_out"}.get(reason["type"], "failed")
        with self.lock:
            job.update(status=status, finished_at=time.time(), exit_code=code, stdout=stdout,
                       stderr=stderr, result=payload, error=reason, output_bytes=total)
            job["summary"] = completion_summary(job)
            if status == "succeeded" or job["mode"] == "specialist":
                # A rejected/blocked study can still have real bounded failure reports.
                self._register_artifacts(job, produced)
            log_path = safe_local(self.root / "jobs" / job["id"] / "stderr.log", self.root)
            log_path.write_bytes(bytes(buffers["stderr"]))
            os.chmod(log_path, 0o600)
            LOG.info("Experimental CLI %s finished status=%s exit=%s stderr_bytes=%s", job["id"], status, code, len(buffers["stderr"]))
            self._persist(job)
            self._trim()

    def _register_specialist_artifacts(self, job, produced):
        """Never recurse into checkpoint directories or follow receipt-supplied paths."""
        workspace = safe_local(self.root / "jobs" / job["id"], self.root)
        job["directory_artifacts"] = []
        reports = []
        for target, role in produced:
            target = safe_local(target, workspace)
            if role == "specialist_study":
                if not target.is_dir():
                    continue
                for path, label in ((target, "study"), (target / "reconstructed", "unrecovered_control"),
                                    (target / "recovered", "recovered_candidate_not_certified")):
                    path = safe_local(path, workspace)
                    if path.is_dir():
                        job["directory_artifacts"].append({"role": label, "kind": "directory",
                            "path": str(path), "download_available": False,
                            "claim": "Directory exists; inspect manifest for completion/integrity. Native local reload, not a downloadable archive."})
                reports.extend((target / name, "specialist_manifest" if name == "manifest.json" else "specialist_report")
                               for name in SPECIALIST_REPORTS)
            elif role == "specialist_report":
                reports.append((target, role))
        for path, role in reports:
            path = safe_local(path, workspace)
            if not path.is_file() or path.stat().st_size > SPECIALIST_REPORT_LIMIT:
                continue
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                total = 0
                while True:
                    chunk = stream.read(min(1024 * 1024, SPECIALIST_REPORT_LIMIT + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > SPECIALIST_REPORT_LIMIT:
                        break
                    checksum.update(chunk)
            if total > SPECIALIST_REPORT_LIMIT:
                continue
            job["artifacts"].append({"id": str(len(job["artifacts"])), "role": role, "name": path.name,
                "path": str(path.relative_to(self.root)), "bytes": total, "sha256": checksum.hexdigest()})

    def _register_artifacts(self, job, produced):
        if job["mode"] == "specialist":
            self._register_specialist_artifacts(job, produced)
            return
        # Only known produced roles. No generic workspace/file browser or recursive JSON path extraction.
        if (job["mode"], job["operation"]) == ("compose", "run"):
            result = (job.get("result") or {}).get("result") or {}
            run_id = result.get("id")
            if isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
                folder = self.root / "compose" / "outputs" / run_id
                if folder.is_dir():
                    produced += [(p, "compose_audio") for p in folder.glob("*.wav")]
        for target, role in produced:
            target = safe_local(target, self.root)
            files = target.rglob("*") if target.is_dir() else iter([target])
            for path in islice(files, 256):
                path = safe_local(path, self.root)
                if not path.is_file():
                    continue
                if role == "compiler_candidate" and path.suffix not in (".json", ".safetensors", ".model", ".txt"):
                    continue
                checksum = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(chunk)
                job["artifacts"].append({"id": str(len(job["artifacts"])), "role": role,
                                         "name": path.name, "path": str(path.relative_to(self.root)),
                                         "bytes": path.stat().st_size, "sha256": checksum.hexdigest()})

    def artifact(self, job_id, artifact_id):
        job = self.get(job_id)
        item = next((x for x in job["artifacts"] if x["id"] == artifact_id), None)
        if not item or item["role"] not in {"compiler_candidate", "compiler_evidence", "compiler_certificate",
                                              "compiler_audit", "validation_evidence", "compose_audio", "deployment_export",
                                              "specialist_report", "specialist_manifest"}:
            raise StudioError("ARTIFACT_NOT_FOUND", "No confirmed produced artifact with this ID", 404)
        path = safe_local(self.root / item["path"], self.root)
        if item["role"].startswith("specialist_"):
            workspace = safe_local(self.root / "jobs" / job_id, self.root)
            path = safe_local(path, workspace)
            allowed = set()
            for produced in job.get("specialist_produced", []):
                target = safe_local(self.root / produced["path"], workspace)
                if produced["role"] == "specialist_study":
                    allowed.update(target / name for name in SPECIALIST_REPORTS)
                elif produced["role"] == "specialist_report":
                    allowed.add(target)
            if path not in allowed or not path.is_file() or path.stat().st_size > SPECIALIST_REPORT_LIMIT:
                raise StudioError("ARTIFACT_NOT_FOUND", "Only bounded specifically produced specialist reports may be served", 404)
            if path.stat().st_size != item["bytes"]:
                raise StudioError("ARTIFACT_CHANGED", "Produced report size changed; not serving stale evidence", 409)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                remaining = SPECIALIST_REPORT_LIMIT + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    digest.update(chunk)
            if remaining == 0 or digest.hexdigest() != item["sha256"]:
                raise StudioError("ARTIFACT_CHANGED", "Produced report changed; not serving stale hash evidence", 409)
        if not path.is_file():
            raise StudioError("ARTIFACT_NOT_FOUND", "Produced artifact no longer exists", 404)
        return path, item

    def close(self):
        self.stopping.set()
        self.worker.join(timeout=6)
        if not self.worker.is_alive():
            self.lockfile.close()


_manager = None
_manager_lock = threading.Lock()


def manager():
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager(os.environ.get("SILT_EXPERIMENTAL_ROOT", ".experimental-studio"))
    return _manager


def authenticated(request: Request):
    local_gate(request)
    service = manager()
    token = request.headers.get("X-SILT-Experimental-Token", "")
    if not hmac.compare_digest(token.encode("utf-8"), service.token.encode("ascii")):
        raise StudioError("AUTH_REQUIRED", "Open /experimental locally and use its session token", 403)
    return service


@router.get("/experimental", dependencies=[Depends(local_gate)], response_class=HTMLResponse)
def experimental_ui():
    template = (Path(__file__).parent / "static" / "experimental.html").read_text()
    nonce = secrets.token_urlsafe(20)
    response = HTMLResponse(template.replace("__SILT_TOKEN__", manager().token).replace("__SILT_NONCE__", nonce))
    response.headers["Content-Security-Policy"] = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'nonce-" + nonce + "'; connect-src 'self'; img-src 'self' blob:; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
    return response


@router.get("/api/experimental/schema")
def schema(service=Depends(authenticated)):
    return {"ok": True, "actions": ACTIONS, "request_schema": JobRequest.model_json_schema(),
            "workspace": str(service.root), "queue_limit": MAX_QUEUE, "retained_jobs": MAX_JOBS,
            "max_output_bytes": MAX_OUTPUT, "max_timeout_seconds": 900,
            "specialist_build_max_timeout_seconds": 3600, "specialist_report_max_bytes": SPECIALIST_REPORT_LIMIT,
            "worker_count": 1, "specialist_finalization": "Requires FINALIZE confirmation; frozen final is consumed once even on failure",
            "trusted_operator_only": True, "activation": "Explicit ACTIVATE confirmation; reviewed high-risk approval forwards --approve-high-risk and named --approver"}


@router.post("/api/experimental/jobs", status_code=202)
async def submit(request: Request, service=Depends(authenticated)):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BODY:
            raise StudioError("REQUEST_TOO_LARGE", "Request exceeds 128 KiB", 413)
    try:
        body = json.loads(data)
    except (ValueError, UnicodeError):
        raise StudioError("INVALID_REQUEST", "Body must be a JSON object", 422)
    job_request = JobRequest.model_validate(body)
    return {"ok": True, "job": project_job(service.submit(job_request))}


@router.get("/api/experimental/jobs")
def list_jobs(service=Depends(authenticated)):
    # Summary/argv/error fields of inherited records can contain sensitive text.
    keys = ("id", "mode", "operation", "operator", "status", "created_at", "exit_code")
    return {"ok": True, "jobs": [{k: job.get(k) for k in keys} for job in service.list()]}


@router.get("/api/experimental/jobs/{job_id}")
def get_job(job_id: str, request: Request, service=Depends(authenticated)):
    return {"ok": True, "job": project_job(service.get(job_id), viewer_reveal(request))}


@router.post("/api/experimental/jobs/{job_id}/cancel")
def cancel_job(job_id: str, service=Depends(authenticated)):
    return {"ok": True, "job": project_job(service.cancel(job_id))}


@router.get("/api/experimental/jobs/{job_id}/artifacts/{artifact_id}")
def download(job_id: str, artifact_id: str, request: Request, service=Depends(authenticated)):
    path, item = service.artifact(job_id, artifact_id)
    if (item["role"] == "deployment_export" or item["role"].startswith("specialist_")) and not viewer_reveal(request):
        raise StudioError("CONFIRM_SENSITIVE_REVEAL", "Archives and original specialist reports require explicit viewer download consent", 403)
    return FileResponse(path, filename=item["name"], media_type="application/octet-stream")


@router.on_event("shutdown")
def shutdown():
    global _manager
    if _manager is not None:
        _manager.close()
        _manager = None
_manager = None
