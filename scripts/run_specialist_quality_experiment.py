"""Run a reproducible SILT specialist quality experiment through public CLI.

This script is intentionally an orchestrator, not a model implementation.  It
collects environment evidence, records the public specialist CLI surface, then
invokes ``python -m asea.specialist build`` and, only when explicitly requested,
``python -m asea.specialist finalize``.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


DOCS = (
    "LOCAL_SETUP.md",
    "docs/CAPABILITIES.md",
    "docs/SPECIALIST_RECONSTRUCTION.md",
    "docs/SPECIALIST_RECOVERY.md",
    "docs/SPECIALIST_WORKFLOW.md",
)
HELP_COMMANDS = ("", "reconstruct", "recover", "build", "evaluate", "finalize")
DEFAULT_CANDIDATE = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_BASELINE = "SmolLM2-360M-Instruct or another local native Llama-family compact checkpoint"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def guarded_snapshot(path):
    # Fixed in-tree primitive, also used by planner and public CPU core.
    source_root = str(repo_root_from_script() / 'src')
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from asea.hardware.data import snapshot
    return snapshot(path, single_link=True)


def sha256_file(path: Path) -> dict:
    _, pin = guarded_snapshot(path)
    return {'sha256': pin['sha256'], 'bytes': pin['size']}


CAPTURE_LIMIT = 1024 * 1024


def _capture(raw):
    import base64
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="surrogatepass")
    raw = raw or b""
    bounded = raw[:CAPTURE_LIMIT]
    try:
        text = bounded.decode("utf-8")
        invalid = False
    except UnicodeDecodeError:
        text, invalid = bounded.decode("utf-8", errors="replace"), True
    meta = {"bytes": len(raw), "captured_bytes": len(bounded), "truncated": len(raw) > len(bounded),
            "sha256": hashlib.sha256(raw).hexdigest(), "encoding": "utf-8-replace", "invalid_utf8": invalid}
    if invalid:
        meta["raw_sample_base64"] = base64.b64encode(bounded).decode("ascii")
    return text, meta


def run(argv: list[str], cwd: Path, timeout: int = 60, env: dict | None = None) -> dict:
    """Linux PDEATH-bound direct child; bounded streams and kill-before-reap.

    No poll()/wait() until killpg: the unreaped leader pins the group identity.
    Inherited pipes get only the original deadline, never an unbounded join.
    Native Windows has no job-object implementation and is explicitly blocked.
    """
    import selectors
    import signal
    import math
    started = time.monotonic()
    result = {"argv": argv, "cwd": str(cwd), "returncode": None, "stdout": "", "stderr": ""}
    if sys.platform != "linux":
        result.update(error="kernel containment unsupported: Linux PDEATHSIG required; Windows job objects not implemented",
                      launch_error=True, containment="UNSUPPORTED", elapsed_seconds=0)
        return result
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("timeout must be finite, positive and at most 3600 seconds")
    worker = Path(__file__).resolve().parents[1] / "src/asea/specialist/controller_worker.py"
    command = [sys.executable, "-I", "-S", str(worker), str(os.getpid())] + list(argv)
    result["containment"] = "linux_pdeathsig_direct_child_known_group_only"
    deadline = started + timeout
    run_deadline = deadline - min(.1, timeout / 10)
    selector = selectors.DefaultSelector()
    samples = {name: bytearray() for name in ("stdout", "stderr")}
    digests = {name: hashlib.sha256() for name in samples}
    counts = dict.fromkeys(samples, 0)
    process = None
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        process = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True, shell=False)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        for name in samples:
            pipe = getattr(process, name)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while True:
            exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exited is not None and not selector.get_map():
                break
            if time.monotonic() >= run_deadline:
                result.update(timeout=True, error="timeout expired")
                break
            for key, _ in selector.select(min(.05, max(0, run_deadline - time.monotonic()))):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                digests[name].update(chunk)
                counts[name] += len(chunk)
                samples[name].extend(chunk[:max(0, CAPTURE_LIMIT - len(samples[name]))])
    except subprocess.TimeoutExpired as exc:
        result.update(timeout=True, error="timeout expired")
        for name in samples:
            result[name], result[name + "_capture"] = _capture(getattr(exc, name, None))
    except OSError as exc:
        result.update(error=str(exc), launch_error=True, error_type=type(exc).__name__)
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    result["returncode"] = process.wait(timeout=max(.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    result.update(error="cleanup deadline exceeded (SIGKILL sent)", capture_incomplete=True)
                process.stdout.close()
                process.stderr.close()
            selector.close()
            for name, sample in samples.items():
                if name + "_capture" not in result:
                    result[name], meta = _capture(bytes(sample))
                    meta.update(bytes=counts[name], sha256=digests[name].hexdigest(), truncated=counts[name] > len(sample))
                    result[name + "_capture"] = meta
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def physical_ram_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return None
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    return None


def disk_inventory(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {"path": str(path), "total_bytes": usage.total, "free_bytes": usage.free}


def command_output(argv: list[str], timeout: int = 10) -> dict:
    return run(argv, Path.cwd(), timeout=timeout)


def torch_inventory(python_exe: str) -> dict:
    probe = (
        "import importlib.util, json\n"
        "spec=importlib.util.find_spec('torch')\n"
        "result={'installed': bool(spec)}\n"
        "if spec:\n"
        " import torch\n"
        " result.update(version=torch.__version__, cuda_available=torch.cuda.is_available(), "
        "torch_cuda_version=str(torch.version.cuda), device_count=torch.cuda.device_count())\n"
        " result['devices']=[]\n"
        " for i in range(torch.cuda.device_count()):\n"
        "  p=torch.cuda.get_device_properties(i)\n"
        "  result['devices'].append({'index':i,'name':torch.cuda.get_device_name(i),"
        "'total_memory_bytes':p.total_memory,'capability':torch.cuda.get_device_capability(i)})\n"
        "print(json.dumps(result, sort_keys=True))\n"
    )
    observed = run([python_exe, "-c", probe], Path.cwd(), timeout=30)
    try:
        parsed = json.loads(observed["stdout"])
    except json.JSONDecodeError:
        parsed = {"installed": False, "probe_error": observed["stderr"] or observed["stdout"]}
    parsed["probe"] = {k: observed[k] for k in ("argv", "returncode", "elapsed_seconds")}
    return parsed


def git_inventory(repo: Path) -> dict:
    fields = {
        "top_level": ["git", "rev-parse", "--show-toplevel"],
        "branch": ["git", "branch", "--show-current"],
        "head": ["git", "log", "-1", "--format=%H %s"],
        "status": ["git", "status", "--short", "--branch"],
        "diff_stat": ["git", "diff", "--stat"],
    }
    return {name: run(argv, repo, timeout=20) for name, argv in fields.items()}


def docs_inventory(repo: Path) -> dict:
    result = {}
    for name in DOCS:
        path = repo / name
        result[name] = sha256_file(path) if path.is_file() else {"missing": True}
    return result


def specialist_help(repo: Path, python_exe: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    result = {}
    for command in HELP_COMMANDS:
        argv = [python_exe, "-m", "asea.specialist"]
        if command:
            argv.append(command)
        argv.append("--help")
        result[command or "root"] = run(argv, repo, timeout=30, env=env)
    return result


def build_argv(python_exe: str, recipe: Path, workspace: Path, expected_recipe_sha256=None, expected_data_binding_file=None, expected_data_binding_sha256=None) -> list[str]:
    argv = [python_exe, "-m", "asea.specialist", "build", "--recipe", str(recipe), "--workspace", str(workspace)]
    if expected_recipe_sha256 is not None:
        argv += ["--expected-recipe-sha256", expected_recipe_sha256]
    if expected_data_binding_file is not None:
        argv += ['--expected-data-binding-file', str(expected_data_binding_file),
                 '--expected-data-binding-sha256', expected_data_binding_sha256]
    return argv


def finalize_argv(python_exe: str, study: Path, suite: Path, output: Path,
                  expected_recipe_sha256=None, expected_config_sha256=None, expected_data_binding_sha256=None) -> list[str]:
    argv = [python_exe, "-m", "asea.specialist", "finalize", "--study", str(study), "--suite", str(suite), "--output", str(output)]
    for key, value in (("expected-recipe-sha256", expected_recipe_sha256), ("expected-config-sha256", expected_config_sha256), ("expected-data-binding-sha256", expected_data_binding_sha256)):
        if value is not None:
            argv += ["--" + key, value]
    return argv


def cli_receipt(result: dict) -> dict | None:
    try:
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError("duplicate receipt key")
                value[key] = item
            return value
        def constant(_):
            raise ValueError("nonfinite receipt")
        parsed = json.loads(result.get("stdout") or "{}", object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError, UnicodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def infrastructure_block_from_receipt(receipt: dict | None) -> bool:
    if not receipt or receipt.get("command") != "build":
        return False
    payload = receipt.get("result")
    if not isinstance(payload, dict) or payload.get("qualified_source") is not False:
        return False
    # Exact unavailable-file diagnostic only; corruption mentioning config/missing keys is NOT infrastructure.
    error = payload.get("error")
    return isinstance(error, str) and error.startswith("not a regular local file: ")


def classify_cli(result: dict, expected_command: str | None = None) -> tuple[str, str]:
    receipt = cli_receipt(result)
    if result.get("timeout") or result.get("launch_error") or result.get("capture_incomplete"):
        return "BLOCKED", "PROCESS_UNAVAILABLE"
    if result.get("returncode") is None or result.get("returncode", 0) < 0:
        return "BLOCKED", "PROCESS_INTERRUPTED"
    if any(result.get(name + "_capture", {}).get("truncated") or
           result.get(name + "_capture", {}).get("invalid_utf8") for name in ("stdout",)):
        return "BLOCKED", "INVALID_RECEIPT_TRANSPORT"
    if receipt is None:
        return "BLOCKED", "INVALID_JSON_RECEIPT"
    if receipt.get("status") == "BLOCKED" or receipt.get("operational_failure") is True:
        return "BLOCKED", "CLI_BLOCKED"
    if infrastructure_block_from_receipt(receipt):
        return "BLOCKED", "LOCAL_INPUT_UNAVAILABLE"
    if receipt.get("status") == "REJECTED":
        return "REJECTED", "CLI_REJECTED"
    payload = receipt.get("result")
    if not isinstance(payload, dict):
        return "REJECTED", "INVALID_RECEIPT_SCHEMA"
    if receipt.get("error") or payload.get("error") or result.get("error"):
        return "REJECTED", "RECEIPT_ERROR"
    if result["returncode"] != 0:
        return "REJECTED", "NONZERO_EXIT"
    command = receipt.get("command")
    if (receipt.get("schema_version") != 1 or type(receipt.get("schema_version")) is not int
            or command not in ("build", "finalize") or (expected_command and command != expected_command)
            or type(receipt.get("completed")) is not bool):
        return "REJECTED", "INVALID_RECEIPT_SCHEMA"
    if receipt["completed"] is not True or receipt.get("status") != "BUILT_UNCERTIFIED":
        return "REJECTED", "INCOMPLETE_OR_CONTRADICTORY_RECEIPT"
    if payload.get("engineering_complete") is not True or type(payload.get("quality_pass")) is not bool:
        return "REJECTED", "MISSING_ENGINEERING_FLAGS"
    if command == "build" and (type(payload.get("qualified_source")) is not bool or not receipt.get("workspace") or not receipt.get("manifest")):
        return "REJECTED", "MISSING_BUILD_FLAGS"
    if command == "finalize" and (payload.get("completed") is not True or payload.get("status") != receipt["status"]
                                   or payload.get("final_consumed") is not True or payload.get("training_on_final") is not False):
        return "REJECTED", "MISSING_FINAL_FLAGS"
    return "COMPLETED", "ENGINEERING_COMPLETE_NOT_CERTIFIED"


def build_allows_final(result: dict) -> bool:
    return (classify_cli(result, "build")[0] == "COMPLETED"
            and cli_receipt(result)["result"].get("quality_pass") is True
            and cli_receipt(result)["result"].get("qualified_source") is True)


def acceptance_criteria(candidate: str, baseline: str) -> dict:
    return {
        "candidate": candidate,
        "compact_baseline": baseline,
        "status": "proposed_not_approved_by_code",
        "criteria": [
            "Use disjoint training, development and final data; calibration is an exact TRAIN-only subset with a selection lock.",
            "Do not train on consumed final tasks or reuse them as a new unseen final set.",
            "Run source validation first; source_quality_floor is a recipe field and defaults to 1.0.",
            "Compare source teacher, unrepaired reconstruction, recovered model and compact baseline under one fixed protocol.",
            "Treat blocked infrastructure as BLOCKED and failed graded tasks as failures in the original denominator.",
            "Require recovered dev quality to beat unrepaired reconstruction without control regressions before finalization.",
            "Freeze the candidate and verify complete bundle loading with the original teacher unavailable before final.",
            "Require the recovered whole bundle to fit the declared output budget and have fewer safetensors bytes than the source.",
            "Record full receipts, hashes, wall time, memory observations, runtime settings, failures and skips.",
            "Never weaken gates, alter scores or claim unsupported hardware compatibility to obtain a pass.",
        ],
    }


def environment_report(repo: Path, python_exe: str, candidate: str, baseline: str) -> dict:
    nvidia = command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                            "--format=csv,noheader"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    probe = run([python_exe, "-m", "asea.hardware", "probe", "--path", str(repo)], repo, timeout=60, env=env)
    return {
        "hardware_probe": {"command": probe, "inventory": cli_receipt(probe)},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "experiment_python": command_output([python_exe, "--version"]),
        "cpu": platform.processor(),
        "physical_ram_bytes": physical_ram_bytes(),
        "disk": disk_inventory(repo),
        "nvidia_smi": nvidia,
        "torch": torch_inventory(python_exe),
        "git": git_inventory(repo),
        "docs": docs_inventory(repo),
        "acceptance_criteria": acceptance_criteria(candidate, baseline),
    }


def status_from_cli(result: dict) -> str:
    return classify_cli(result)[0]


def checked_path(value: Path, *, new=False) -> Path:
    """Reject symlinks in every component BEFORE resolving; never create parents."""
    path = Path(os.path.abspath(value))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("symlink path forbidden: " + str(part))
    if not path.parent.is_dir():
        raise ValueError("existing parent directory required: " + str(path))
    if new and path.exists():
        raise FileExistsError("output already exists: " + str(path))
    if new and (not os.access(path.parent, os.W_OK) or not path.parent.stat().st_mode & 0o222):
        raise PermissionError("output parent not writable: " + str(path.parent))
    return path


class ReservedReport:
    """Exclusive durable reservation; final writes address only the owned inode.

    Never use replace(path): even a checked replace can race and overwrite another
    owner's replacement. A torn terminal write leaves an owned incomplete report,
    never an unsafe successful publication. Callers retain the reservation until exit.
    """
    def __init__(self, path):
        self.path = checked_path(path, new=True)
        directory = os.open(self.path.anchor, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            for component in self.path.parent.parts[1:]:
                child = os.open(component, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                                getattr(os, "O_NOFOLLOW", 0), dir_fd=directory)
                os.close(directory)
                directory = child
            self.fd = os.open(self.path.name, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                              getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory)
            self.identity = os.fstat(self.fd)
            self.write({"schema_version": 1, "status": "RESERVED", "completed": False})
            os.fsync(directory)
        finally:
            os.close(directory)

    def write(self, report):
        current = self.path.lstat()
        if (current.st_dev, current.st_ino) != (self.identity.st_dev, self.identity.st_ino):
            raise FileExistsError("report reservation replaced; refusing publication")
        payload = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()
        os.lseek(self.fd, 0, os.SEEK_SET)
        view = memoryview(payload)
        while view:
            count = os.write(self.fd, view)
            view = view[count:]
        os.ftruncate(self.fd, len(payload))
        os.fsync(self.fd)

    def close(self):
        os.close(self.fd)


def publish_terminal(path, report):
    """Atomically link a fully fsynced terminal record into a NEW pathname.

    No replace/rename-over-existing race. The immutable terminal snapshot records
    the outcome, NOT acknowledgement that the subsequent directory fsync returned.
    The controller publishes a separate hash-bound acknowledgement only afterwards;
    neither non-atomic acknowledgement nor a successful exit guarantees survival
    of arbitrary power/storage failure. Already-linked snapshots are never replaced.
    """
    import secrets
    path = checked_path(path, new=True)
    directory = os.open(path.anchor, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    temporary = "." + path.name + "." + secrets.token_hex(16)
    created = False
    try:
        for component in path.parent.parts[1:]:
            child = os.open(component, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                            getattr(os, "O_NOFOLLOW", 0), dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory)
        created = True
        with os.fdopen(fd, "wb") as handle:
            handle.write((json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
    finally:
        if created:
            os.unlink(temporary, dir_fd=directory)
        os.close(directory)


def write_report(path: Path, report: dict) -> None:
    reserved = ReservedReport(path)
    try:
        reserved.write(report)
    finally:
        reserved.close()


def check_output_paths(args):
    outputs = {"report": checked_path(args.report, new=True),
               "terminal_report": checked_path(Path(str(args.report) + ".terminal.json"), new=True),
               "acknowledgement": checked_path(Path(str(args.report) + ".ack.json"), new=True)}
    if args.recipe and not args.preflight_only:
        outputs["data_binding"] = checked_path(Path(str(args.report) + ".data-binding.json"), new=True)
        outputs["execution_recipe"] = checked_path(Path(str(args.report) + ".execution-recipe.json"), new=True)
    for name in ("workspace", "final_output"):
        if getattr(args, name) is not None:
            outputs[name] = checked_path(getattr(args, name), new=True)
    for name, path in outputs.items():
        for other, value in outputs.items():
            if name != other and (path == value or path in value.parents or value in path.parents):
                raise ValueError("report, workspace and final output must be disjoint")
    inputs = [getattr(args, name) for name in ("recipe", "final_suite", "acceptance_record", "exclude_ledger", "deployment_receipt", "reviewed_plan")]
    for value in inputs:
        if value is not None:
            path = Path(os.path.abspath(value))
            if any(path == out or out in path.parents or path in out.parents for out in outputs.values()):
                raise ValueError("input/output alias forbidden")
    if args.final_suite is not None:
        final_path = Path(os.path.abspath(args.final_suite))
        for value in inputs:
            if value is not None and value is not args.final_suite and (Path(os.path.abspath(value)) == final_path or
                    (Path(value).exists() and final_path.exists() and os.path.samefile(value, final_path))):
                raise ValueError("final suite cannot alias a pre-final input")
    return outputs


def read_json(path, fingerprint=None):
    path = checked_path(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("bounded regular JSON file required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON")
    raw, pin = guarded_snapshot(path)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if fingerprint is not None:
        fingerprint.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def hardware_plan(args, repo, env):
    # Always execute the public hardware CLI in the SELECTED Python environment.
    argv = [args.python, "-m", "asea.hardware", "plan", "--recipe", str(args.recipe.absolute())]
    if args.workspace:
        argv += ["--workspace-parent", str(args.workspace.absolute().parent)]
    observed = run(argv, repo, timeout=min(args.timeout_seconds, 120), env=env)
    plan = cli_receipt(observed)
    if (observed.get("returncode") != 0 or observed.get("timeout") or observed.get("error")
            or observed.get("stdout_capture", {}).get("truncated")
            or observed.get("stdout_capture", {}).get("invalid_utf8")
            or not plan or plan.get("schema_version") != 1
            or plan.get("status") not in ("READY", "BLOCKED", "PLANNING_ONLY")):
        # A BLOCKED public plan may legitimately exit nonzero: preserve, never promote it.
        if not plan or plan.get("status") not in ("BLOCKED", "PLANNING_ONLY"):
            plan = {"schema_version": 1, "status": "BLOCKED", "reasons": ["hardware CLI plan unavailable/invalid"], "execution_device": None}
    if plan.get("status") == "READY":
        import re
        try:
            digest = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "plan_sha256"},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            valid = (type(plan.get("schema_version")) is int and plan.get("schema_version") == 1
                and re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", plan.get("execution_device", ""))
                and plan.get("plan_sha256") == digest and plan.get("reasons") == []
                and isinstance(plan.get("model"), dict) and plan["model"].get("checkpoint_bound") is True
                and isinstance(plan.get("hardware"), dict) and isinstance(plan.get("phases"), list)
                and isinstance(plan.get("estimates"), dict) and isinstance(plan.get("bindings"), dict))
        except (ValueError, TypeError):
            valid = False
        if not valid:
            plan = {"schema_version": 1, "status": "BLOCKED", "execution_device": None,
                    "reasons": ["READY plan schema/hash/inventory is inconsistent"]}
    return plan, observed


def reviewed_acceptance(args, plan, recipe):
    reviewed = read_json(args.reviewed_plan) if args.reviewed_plan else plan
    reviewed = reviewed.get("plan", reviewed)
    if not args.reviewed_plan_sha256 or args.reviewed_plan_sha256 != reviewed.get("plan_sha256"):
        raise ValueError("--run-final requires the explicit reviewed plan SHA256")
    unhashed = {k: v for k, v in reviewed.items() if k != "plan_sha256"}
    digest = hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != args.reviewed_plan_sha256 or reviewed.get("status") != "READY":
        raise ValueError("reviewed plan hash/status invalid")
    if (reviewed.get("execution_device") != plan.get("execution_device")
            or reviewed.get("bindings") != plan.get("bindings")):
        # Hardware observation hashes are volatile; capability is checked again by
        # the new READY plan and each execution stage, not silently waived.
        before = {k: v for k, v in reviewed.get("bindings", {}).items() if k != "hardware_sha256"}
        now = {k: v for k, v in plan.get("bindings", {}).items() if k != "hardware_sha256"}
        if reviewed.get("execution_device") != plan.get("execution_device") or not before or before != now:
            raise ValueError("reviewed recipe/inventory/workspace/device binding changed")
    if not args.acceptance_record or not args.exclude_ledger:
        raise ValueError("--run-final requires reviewed acceptance record and authoritative exclude ledger")
    acceptance = read_json(args.acceptance_record)
    if (acceptance.get("schema_version") != 1 or acceptance.get("status") != "APPROVED"
            or acceptance.get("plan_sha256") != args.reviewed_plan_sha256
            or acceptance.get("recipe_sha256") != sha256_file(args.recipe)["sha256"]
            or acceptance.get("policy") != "engineering_all_pass_v1"
            or acceptance.get("compact_baseline") != "not_run_engineering_only"
            or acceptance.get("exclude_ledger_sha256") != sha256_file(args.exclude_ledger)["sha256"]):
        raise ValueError("acceptance binding/policy mismatch (no quality certificate policy implemented)")
    ledger = read_json(args.exclude_ledger)
    if ledger.get("schema_version") != 1 or ledger.get("authoritative") is not True or not isinstance(ledger.get("consumed"), list):
        raise ValueError("authoritative versioned consumed-task ledger required")
    lock_value = recipe.get("selection_lock")
    if not lock_value:
        raise ValueError("explicit answer-free selection_lock required for automatic final")
    lock_path = Path(lock_value)
    if not lock_path.is_absolute():
        lock_path = args.recipe.absolute().parent / lock_path
    if lock_path.name != "selection-lock.json" or (args.final_suite and lock_path.absolute() == args.final_suite.absolute()):
        raise ValueError("only designated answer-free selection-lock.json may be read before final")
    lock = read_json(lock_path)
    consumed = ledger["consumed"]
    if any(not isinstance(r, dict) or not isinstance(r.get("id"), str) or not isinstance(r.get("family"), str) for r in consumed):
        raise ValueError("invalid consumed-task ledger entries")
    ids, families = {r["id"] for r in consumed}, {r["family"] for r in consumed}
    rows = lock.get("selection")
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("invalid selection metadata")
    if any(r.get("id") in ids or r.get("family") in families for r in rows):
        raise ValueError("authoritative ledger overlaps selected task/family")
    return acceptance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--python", default=sys.executable, help="Python executable used for SILT CLI calls")
    parser.add_argument("--recipe", type=Path, help="Fresh specialist build recipe")
    parser.add_argument("--workspace", type=Path, help="New specialist study workspace")
    parser.add_argument("--final-suite", type=Path, help="Prerecorded final suite for one-shot finalize")
    parser.add_argument("--final-output", type=Path, help="New final result path")
    parser.add_argument("--report", type=Path, default=Path("specialist_quality_experiment_report.json"))
    parser.add_argument("--candidate", help="Expected recipe canonical source ID; mismatch blocks")
    parser.add_argument("--compact-baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-final", action="store_true", help="Consume final data through SILT finalize after build succeeds")
    parser.add_argument("--reviewed-plan-sha256")
    parser.add_argument("--reviewed-plan", type=Path, help="Saved reviewed READY plan (or preflight report); live capabilities are rechecked")
    parser.add_argument("--acceptance-record", type=Path)
    parser.add_argument("--exclude-ledger", type=Path)
    parser.add_argument("--deployment-receipt", type=Path, help="Reviewed teacher-unavailable exercised-task receipt, bound to frozen models")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = check_output_paths(args)
        reserved = ReservedReport(outputs["report"])
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": "REPORT_RESERVATION_FAILED", "error": str(exc)}), file=sys.stderr)
        return 2  # No build, final, workspace creation or inventory occurred.
    report = {"schema_version": 1, "status": "BLOCKED", "commands": [], "certificate": False,
              "compact_baseline_status": "BASELINE_NOT_RUN", "deployment_status": "AWAITING_EXERCISED_RECEIPT",
              "terminal_report": str(outputs["terminal_report"]), "publication_contract": "atomic_exclusive_terminal; owned_inode_summary; completion is not durability acknowledgement; separate ack is non-atomic and cannot guarantee power-failure survival",
              "final_acknowledgement": {"state": "REQUIRES_SEPARATE_ACK", "path": str(outputs["acknowledgement"])}}
    reserved.write(dict(report, status="RESERVED"))
    try:
        repo = args.repo_root.resolve()
        if not 0 < args.timeout_seconds <= 3600:
            raise ValueError("timeout must be positive and at most 3600 seconds")
        candidate = "UNSPECIFIED_NO_RECIPE"
        recipe = None
        if args.recipe:
            fingerprint = {}
            recipe = read_json(args.recipe, fingerprint=fingerprint)
            report["input_recipe"] = fingerprint
            canonical = recipe.get("source_metadata", {}).get("canonical_id")
            source = recipe.get("source_path")
            if not isinstance(source, str) or not source:
                raise ValueError("recipe source_path required")
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = args.recipe.absolute().parent / source_path
            candidate = canonical or str(source_path.absolute())
            if args.candidate and args.candidate != candidate:
                raise ValueError("--candidate does not match recipe source identity")
            if any(source_path == out or source_path in out.parents or out in source_path.parents for out in outputs.values()):
                raise ValueError("source/output alias forbidden")
        report.update(environment_report(repo, args.python, candidate, args.compact_baseline))
        report["candidate_identity"] = candidate
        report["specialist_help"] = specialist_help(repo, args.python)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo / "src")
        if recipe is None:
            report.update(status="PREFLIGHT_ONLY" if args.preflight_only else "BLOCKED",
                          execution_readiness="NOT_READY_FOR_EXECUTION", reason_code="RECIPE_REQUIRED")
            return 0 if args.preflight_only else 2
        plan, observed = hardware_plan(args, repo, env)
        if sha256_file(args.recipe) != report["input_recipe"]:
            raise ValueError("recipe changed while planning")
        report.update(plan=plan, planning_command=observed, execution_readiness=plan.get("status"))
        if args.preflight_only:
            report["status"] = "PREFLIGHT_ONLY"
            return 0 if plan.get("status") in ("READY", "PLANNING_ONLY") else 2
        if plan.get("status") != "READY":
            report.update(status="BLOCKED", reason_code="PLAN_NOT_READY")
            return 2
        if not args.workspace:
            raise ValueError("--workspace required for build")
        if args.run_final:
            if not args.final_suite or not args.final_output:
                raise ValueError("--final-suite and --final-output required")
            report["reviewed_acceptance"] = reviewed_acceptance(args, plan, recipe)
        # Freeze only device selection; no implicit budget/training-policy changes.
        planned_device = plan.get("execution_device")
        import re
        if not isinstance(planned_device, str) or not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", planned_device):
            raise ValueError("READY plan lacks concrete execution device")
        effective_recipe = dict(recipe, execution_device=planned_device)
        for key in ("source_path", "calibration", "training", "validation_data", "validation_suite", "data_manifest", "selection_lock"):
            if key in effective_recipe:
                path = Path(effective_recipe[key])
                effective_recipe[key] = str(path if path.is_absolute() else args.recipe.absolute().parent / path)
        # Fixed in-tree normalizer, not a caller-selected module or ML import.
        source_root = str(repo_root_from_script() / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from asea.specialist.workflow import recipe_config, json_hash
        from asea.artifacts import digest as plan_digest
        expected_config = recipe_config(effective_recipe)
        original_config = dict(expected_config, execution_device=recipe.get("execution_device", "cpu"))
        bindings = plan.get("bindings") or {}
        planned_files = (plan.get("model") or {}).get("files")
        if (bindings.get("recipe_sha256") != plan_digest(original_config)
                or bindings.get("workspace_parent") != str(args.workspace.absolute().parent)
                or bindings.get("requested_device") != recipe.get("execution_device", "auto")
                or not isinstance(planned_files, dict) or not planned_files
                or bindings.get("inventory_sha256") != (plan.get("model") or {}).get("inventory_sha256")):
            raise ValueError("READY plan recipe/source/workspace binding mismatch")
        from asea.hardware.data import binding_from_plan, compare_binding
        expected_data = binding_from_plan(plan)
        binding_path = outputs['data_binding']
        expected_data_sha256 = hashlib.sha256((json.dumps(expected_data, indent=2, sort_keys=True,
            ensure_ascii=True, allow_nan=False) + '\n').encode()).hexdigest()
        write_report(binding_path, expected_data)
        if sha256_file(binding_path)['sha256'] != expected_data_sha256:
            raise ValueError('approved data binding changed before build')
        report['data_binding'] = {'path': str(binding_path), 'sha256': expected_data_sha256}
        effective_path = outputs["execution_recipe"]
        # Hash the intended serialization BEFORE publishing; a writer cannot redefine approval.
        expected_recipe_sha256 = hashlib.sha256((json.dumps(effective_recipe, indent=2, sort_keys=True,
            ensure_ascii=True, allow_nan=False) + "\n").encode()).hexdigest()
        expected_config_sha256 = json_hash(expected_config)
        write_report(effective_path, effective_recipe)
        if sha256_file(effective_path)["sha256"] != expected_recipe_sha256:
            raise ValueError("execution recipe changed before build")
        report["effective_recipe"] = {"path": str(effective_path), "sha256": expected_recipe_sha256,
                                      "config_sha256": expected_config_sha256,
                                      "original": sha256_file(args.recipe), "only_policy_change": "resolved_execution_device"}
        reserved.write(dict(report, status="BUILD_STARTING"))
        build = run(build_argv(args.python, effective_path, args.workspace.absolute(), expected_recipe_sha256, binding_path, expected_data_sha256), repo,
                    timeout=args.timeout_seconds, env=env)
        if sha256_file(effective_path)["sha256"] != expected_recipe_sha256:
            raise ValueError("execution recipe changed during build")
        receipt = cli_receipt(build)
        build["cli_status"] = receipt.get("status") if receipt else None
        build["stage_status"], build["reason_code"] = classify_cli(build, "build")
        report["commands"].append(build)
        if build["stage_status"] != "COMPLETED":
            report["status"] = build["stage_status"]
            return 1
        # The public source manifest is authority, not this wrapper's label or proposal.
        def bound_manifest():
            if sha256_file(effective_path)["sha256"] != expected_recipe_sha256:
                raise ValueError("execution recipe changed after approval")
            actual = read_json(args.workspace / "manifest.json")
            if (actual.get("config") != expected_config or actual.get("config_sha256") != expected_config_sha256
                    or actual.get("recipe_binding") != {"path": str(effective_path), "sha256": expected_recipe_sha256}
                    or actual.get("source", {}).get("path") != expected_config["source_path"]
                    or actual.get("source", {}).get("files") != planned_files):
                raise ValueError("built config/source differs from approved execution recipe and plan")
            compare_binding(expected_data, actual.get('data', {}).get('hashes', {}))
            if actual.get('expected_data_binding_sha256') != expected_data_sha256:
                raise ValueError('built data binding receipt differs from approved plan')
            return actual
        manifest = bound_manifest()
        if manifest.get("execution_device", "cpu") != planned_device:
            report.update(status="BLOCKED", reason_code="EXECUTION_PLAN_DRIFT")
            return 1
        planned_files = (plan.get("model") or {}).get("files")
        if planned_files is not None and manifest.get("source", {}).get("files") != planned_files:
            report.update(status="BLOCKED", reason_code="SOURCE_INVENTORY_PLAN_DRIFT")
            return 1
        report["execution_device"] = planned_device
        report["source_identity_binding"] = {"candidate": candidate,
            "inventory_sha256": (plan.get("model") or {}).get("inventory_sha256"),
            "source_matches_plan": planned_files is not None}
        if not args.run_final:
            report.update(status="BUILT_NOT_FINALIZED", finalization="not run; explicit reviewed engineering policy required")
            return 0
        if not build_allows_final(build):
            report.update(status="REJECTED", reason_code="PRE_FINAL_QUALITY_GATE", finalization="not run; source qualification or all-pass development gate failed")
            return 1
        if not args.deployment_receipt:
            report.update(status="BLOCKED", reason_code="AWAITING_DEPLOYMENT_RECEIPT", finalization="not run; teacher-unavailable task exercise not proved")
            return 1
        deployment = read_json(args.deployment_receipt)
        # Recheck approval/freshness inputs after build; never parse final answers here.
        if reviewed_acceptance(args, plan, recipe) != report["reviewed_acceptance"]:
            raise ValueError("acceptance record changed during build")
        if (deployment.get("schema_version") != 1 or deployment.get("status") != "REVIEWED"
                or deployment.get("teacher_unavailable") is not True
                or deployment.get("passed") is not True or type(deployment.get("tasks_exercised")) is not int
                or deployment["tasks_exercised"] < 1
                or deployment.get("frozen_models_sha256") != manifest.get("frozen_models_sha256")
                or deployment.get("execution_device") != planned_device):
            raise ValueError("teacher-unavailable exercised deployment receipt missing/mismatched")
        report["deployment_status"] = "REVIEWED_RECEIPT_SUPPLIED_NOT_AUTOMATED"
        report["deployment_receipt"] = sha256_file(args.deployment_receipt)
        reserved.write(dict(report, status="FINAL_STARTING"))
        if bound_manifest() != manifest:
            raise ValueError("built manifest changed before final")
        final = run(finalize_argv(args.python, args.workspace.absolute(), args.final_suite.absolute(), args.final_output.absolute(),
                                 expected_recipe_sha256, expected_config_sha256, expected_data_sha256),
                    repo, timeout=args.timeout_seconds, env=env)
        receipt = cli_receipt(final)
        final["cli_status"] = receipt.get("status") if receipt else None
        final["stage_status"], final["reason_code"] = classify_cli(final, "finalize")
        report["commands"].append(final)
        report["status"] = final["stage_status"]
        if final["stage_status"] == "COMPLETED" and receipt["result"].get("quality_pass") is not True:
            report.update(status="REJECTED", reason_code="FINAL_ALL_PASS_GATE")
        return 0 if report["status"] == "COMPLETED" else 1
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="BLOCKED", reason_code="WRAPPER_GUARD", error=str(exc), error_type=type(exc).__name__)
        return 2
    finally:
        try:
            publish_terminal(outputs["terminal_report"], report)
            reserved.write(report)
            publish_terminal(outputs["acknowledgement"], {"schema_version": 1,
                "state": "TERMINAL_AND_SUMMARY_FSYNC_RETURNED", "terminal_report": str(outputs["terminal_report"]),
                "terminal_sha256": sha256_file(outputs["terminal_report"])["sha256"],
                "power_failure_guarantee": False,
                "contract": "separate non-atomic acknowledgement; absence means unacknowledged, not failed model quality"})
        finally:
            reserved.close()


if __name__ == "__main__":
    raise SystemExit(main())
