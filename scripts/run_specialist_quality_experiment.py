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


def sha256_file(path: Path) -> dict:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": size}


def run(argv: list[str], cwd: Path, timeout: int = 60, env: dict | None = None) -> dict:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            check=False,
        )
        return {
            "argv": argv,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "cwd": str(cwd),
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "elapsed_seconds": time.monotonic() - started,
            "timeout": True,
            "error": "timeout expired",
        }


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
    try:
        out = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=timeout, check=False)
        return {"argv": argv, "returncode": out.returncode, "stdout": out.stdout, "stderr": out.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": str(exc), "available": False}


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


def build_argv(python_exe: str, recipe: Path, workspace: Path) -> list[str]:
    return [python_exe, "-m", "asea.specialist", "build", "--recipe", str(recipe), "--workspace", str(workspace)]


def finalize_argv(python_exe: str, study: Path, suite: Path, output: Path) -> list[str]:
    return [python_exe, "-m", "asea.specialist", "finalize", "--study", str(study), "--suite", str(suite), "--output", str(output)]


def cli_receipt(result: dict) -> dict | None:
    try:
        parsed = json.loads(result.get("stdout") or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def infrastructure_block_from_receipt(receipt: dict | None) -> bool:
    if not receipt or receipt.get("command") != "build":
        return False
    payload = receipt.get("result")
    if not isinstance(payload, dict) or payload.get("qualified_source") is not False:
        return False
    error_text = str(payload.get("error") or "").lower()
    missing_markers = (
        "not a regular local file",
        "no such file",
        "not found",
        "missing",
        "config.json",
        "local file",
        "local directory",
    )
    return any(marker in error_text for marker in missing_markers)


def build_allows_final(result: dict) -> bool:
    receipt = cli_receipt(result)
    if not receipt or receipt.get("completed") is not True:
        return False
    payload = receipt.get("result")
    if not isinstance(payload, dict):
        return False
    return payload.get("engineering_complete") is True and payload.get("quality_pass") is True


def acceptance_criteria(candidate: str, baseline: str) -> dict:
    return {
        "candidate": candidate,
        "compact_baseline": baseline,
        "status": "proposed_not_approved_by_code",
        "criteria": [
            "Use fresh disjoint calibration, training, development and final data with a selection lock.",
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
    return {
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
    if result.get("returncode") is None:
        return "BLOCKED"
    if not result.get("stdout"):
        return "BLOCKED"
    receipt = cli_receipt(result)
    if receipt is None:
        return "BLOCKED"
    if receipt.get("completed") is True:
        return "COMPLETED"
    if receipt.get("status") == "BLOCKED" or receipt.get("operational_failure") is True:
        return "BLOCKED"
    if infrastructure_block_from_receipt(receipt):
        return "BLOCKED"
    return "REJECTED"


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--python", default=sys.executable, help="Python executable used for SILT CLI calls")
    parser.add_argument("--recipe", type=Path, help="Fresh specialist build recipe")
    parser.add_argument("--workspace", type=Path, help="New specialist study workspace")
    parser.add_argument("--final-suite", type=Path, help="Prerecorded final suite for one-shot finalize")
    parser.add_argument("--final-output", type=Path, help="New final result path")
    parser.add_argument("--report", type=Path, default=Path("specialist_quality_experiment_report.json"))
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--compact-baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-final", action="store_true", help="Consume final data through SILT finalize after build succeeds")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    report = environment_report(repo, args.python, args.candidate, args.compact_baseline)
    report["specialist_help"] = specialist_help(repo, args.python)
    report["commands"] = []

    missing = []
    if not args.preflight_only:
        for name in ("recipe", "workspace"):
            if getattr(args, name) is None:
                missing.append(name)
        if args.run_final:
            for name in ("final_suite", "final_output"):
                if getattr(args, name) is None:
                    missing.append(name.replace("_", "-"))
    if missing:
        report["status"] = "BLOCKED"
        report["error"] = "missing required argument(s): " + ", ".join(missing)
        write_report(args.report, report)
        return 2

    if args.preflight_only:
        report["status"] = "PREFLIGHT_ONLY"
        write_report(args.report, report)
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    build = run(build_argv(args.python, args.recipe.resolve(), args.workspace.resolve()),
                repo, timeout=args.timeout_seconds, env=env)
    receipt = cli_receipt(build)
    if receipt:
        build["cli_status"] = receipt.get("status")
    build["stage_status"] = status_from_cli(build)
    report["commands"].append(build)

    if build["stage_status"] != "COMPLETED":
        report["status"] = build["stage_status"]
        write_report(args.report, report)
        return 1

    if args.run_final:
        if not build_allows_final(build):
            report["status"] = "REJECTED"
            report["finalization"] = "not run; pre-final quality gate did not pass"
            report["pre_final_gate"] = {
                "required_engineering_complete": True,
                "required_quality_pass": True,
                "observed_cli_status": build.get("cli_status"),
            }
            write_report(args.report, report)
            return 1
        final = run(finalize_argv(args.python, args.workspace.resolve(), args.final_suite.resolve(),
                                  args.final_output.resolve()), repo, timeout=args.timeout_seconds, env=env)
        receipt = cli_receipt(final)
        if receipt:
            final["cli_status"] = receipt.get("status")
        final["stage_status"] = status_from_cli(final)
        report["commands"].append(final)
        report["status"] = final["stage_status"]
        write_report(args.report, report)
        return 0 if final["stage_status"] == "COMPLETED" else 1

    report["status"] = "BUILT_NOT_FINALIZED"
    report["finalization"] = "not run; pass --run-final to consume final through SILT finalize"
    write_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
