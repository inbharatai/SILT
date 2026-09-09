"""Separate public specialist namespace; existing compiler commands are unchanged."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys


def _memory_mib(value):
    """CLI MiB is converted once; API/recipe contracts always use bytes."""
    from decimal import Decimal, DecimalException, localcontext
    try:
        number = Decimal(value)
        with localcontext() as context:
            context.prec = 64
            size = number * 1024**2
        if not number.is_finite() or size <= 0 or size > 2**63 - 1 or size != size.to_integral_value():
            raise ValueError()
        return int(size)
    except (DecimalException, ValueError, OverflowError):
        raise argparse.ArgumentTypeError("memory budget must be positive MiB with an integral byte value")


def _operational_error(error_type):
    return error_type in {"ImportError", "ModuleNotFoundError", "RuntimeError", "MemoryError",
                          "OutOfMemoryError", "OSError", "FileNotFoundError", "PermissionError",
                          "ReconstructionBlocked", "TimeoutExpired", "TimeoutError"}


def parser():
    root = argparse.ArgumentParser(prog="python -m asea.specialist", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    reconstruct = commands.add_parser("reconstruct", help="physically smaller standalone native weights")
    for name in ("source-dir", "output-dir", "calibration-path"):
        reconstruct.add_argument("--" + name, required=True)
    reconstruct.add_argument("--family", choices=("auto", "qwen2", "switch_transformers"), default="auto")
    reconstruct.add_argument("--method", choices=("activation", "magnitude", "uniform"), default="activation")
    reconstruct.add_argument("--retention", type=float, default=0.75)
    reconstruct.add_argument("--max-samples", type=int, default=32)
    recover = commands.add_parser("recover", help="real LoRA training with explicit native merge or factor-preserving export")
    for name in ("teacher-dir", "student-dir", "output-dir", "training-path", "validation-path"):
        recover.add_argument("--" + name, required=True)
    recover.add_argument("--export-mode", choices=("native_merged", "factor_preserving"), default="native_merged")
    recover.add_argument("--steps", type=int, default=64)
    recover.add_argument("--rank", type=int, default=8)
    recover.add_argument("--learning-rate", type=float, default=0.0001)
    recover.add_argument("--kd-weight", type=float, default=0.7)
    recover.add_argument("--method", choices=("lora_kd", "supervised_lora"), default="lora_kd")
    recover.add_argument("--max-samples", type=int, default=256)
    recover.add_argument("--teacher-mode", choices=("cache", "cached", "resident"),
                         type=lambda v: "cached" if v == "cache" else v, default="cached")
    for command in (reconstruct, recover):
        command.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
        command.add_argument("--max-length", type=int, default=256)
        command.add_argument("--seed", type=int, default=17)
        command.add_argument("--report", help="new JSON receipt file (also preserves recovery rejection)")
        command.add_argument("--receipt-mode", choices=("full", "compact"), default="full",
                             help="compact stdout requires --report; full evidence stays on disk")
    infer = commands.add_parser("infer", help="native exact-recovery-prefix greedy inference")
    infer.add_argument("--prompt", required=True)
    evaluate = commands.add_parser("evaluate", help="generate and externally test a function-IO suite")
    evaluate.add_argument("--suite", required=True)
    evaluate.add_argument("--output", required=True)
    for command in (infer, evaluate):
        command.add_argument("--model", required=True)
        command.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
        command.add_argument("--max-new-tokens", type=int, default=256)
    for command in (reconstruct, recover, infer, evaluate):
        command.add_argument("--memory-budget-mib", dest="memory_budget_bytes", type=_memory_mib, default=None,
                             help="optional operator ceiling; never overrides observed CPU RAM headroom")
    validate = commands.add_parser("validate", help="externally compare previously generated code; no model load")
    validate.add_argument("--generations", required=True, help="JSON array of {id, status, text, generation} native records")
    validate.add_argument("--suite", required=True)
    validate.add_argument("--output", required=True)
    for command in (evaluate, validate):
        command.add_argument("--trace-policy", choices=("none", "digest", "value"), default="digest")
    build = commands.add_parser("build", help="one immutable staged study; never reads final answers")
    build.add_argument("--recipe", required=True)
    build.add_argument("--workspace", required=True)
    finalize = commands.add_parser("finalize", help="one-shot final evaluation of already frozen controls")
    finalize.add_argument("--study", required=True)
    finalize.add_argument("--suite", required=True)
    finalize.add_argument("--output", required=True)
    return root


def main(argv=None):
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    receipt = {"schema_version": 1, "command": command, "status": "REJECTED", "completed": False}
    receipt_mode = args.pop("receipt_mode", "full")
    durable_pointer = {}
    try:
        if type(receipt_mode) is not str or receipt_mode not in ("full", "compact"):
            raise ValueError("receipt_mode must be full or compact")
        # Libraries sometimes print diagnostics: stdout stays exactly one compact
        # JSON object; no candidate-controlled stdout is promoted to a receipt.
        with contextlib.redirect_stdout(sys.stderr):
            from .workflow import write_json
            if command in ("reconstruct", "recover"):
                report_path = args.pop("report")
                if receipt_mode == "compact" and not report_path:
                    raise ValueError("compact receipt requires --report")
                if report_path:
                    from asea.artifacts import safe_path
                    report_target, output_target = safe_path(report_path), safe_path(args["output_dir"])
                    if report_target == output_target or output_target in report_target.parents:
                        raise ValueError("receipt/history must stay outside the strict output directory")
                    if safe_path(report_path).exists():
                        raise ValueError("report already exists")
                if command == "reconstruct":
                    from .reconstruction import reconstruct
                    result = reconstruct(**args)
                    completed = result.get("status") == "RECONSTRUCTED_UNVALIDATED"
                else:
                    from .recovery import recover
                    result = recover(**args)
                    from .workflow import recovery_complete
                    completed = recovery_complete(result, args["export_mode"])
                receipt.update(status=result.get("status"), completed=completed, result=result,
                               engineering_complete=completed)
                error = result.get("error")
                if not completed and isinstance(error, dict) and _operational_error(error.get("type")):
                    receipt.update(status="BLOCKED", operational_failure=True)
                if report_path:
                    # Write the complete immutable evidence BEFORE projecting stdout.
                    write_json(report_path, receipt)
                    if receipt_mode == "compact":
                        from .workflow import compact_receipt, receipt_pointer
                        durable_pointer = receipt_pointer("full_receipt", report_path)
                    history = result.get("training_history", result.get("recovery_history"))
                    if history is not None:
                        history_path = str(report_path) + ".history.json"
                        write_json(history_path, {"schema_version": 1, "history": history,
                            "actual_steps": result.get("actual_steps"), "status": result.get("status")})
                    if receipt_mode == "compact":
                        receipt = compact_receipt(receipt, command, report_path, args["output_dir"])
                    else:
                        # Preserve legacy --report behavior and default stdout shape.
                        if history is not None:
                            compact = {k: v for k, v in result.items() if k not in ("training_history", "recovery_history")}
                            compact["recovery_history_file"] = history_path
                            compact["recovery_history_rows"] = len(history)
                            receipt["result"] = compact
                        receipt["full_receipt_file"] = str(report_path)
            elif command == "infer":
                from .evaluation import infer
                result = infer(**args)
                receipt.update(result=result, completed=result["completed"], status=result["status"])
            elif command in ("evaluate", "validate"):
                from .evaluation import evaluate, validate
                from .recovery import _bounded_json
                from asea.artifacts import safe_path, file_hash
                output = args.pop("output")
                if safe_path(output).exists() or not safe_path(output).parent.is_dir():
                    raise ValueError("output must be new with existing parent")
                if command == "validate":
                    args["generations"] = _bounded_json(safe_path(args["generations"]))
                    result = validate(**args)
                else:
                    result = evaluate(**args)
                write_json(output, result)
                receipt.update(status=result["status"], completed=result["completed"], output=str(safe_path(output)),
                    output_sha256=file_hash(output)["sha256"], result={k: result.get(k) for k in
                    ("tasks_total", "tasks_passed", "tasks_failed", "tasks_blocked", "tasks_graded",
                     "operational_failures", "generation_completed", "oracle_executed",
                     "engineering_complete", "pass_rate", "quality_pass", "certificate", "resources")})
            elif command == "build":
                from .workflow import build
                result = build(**args)
                receipt.update(status=result["status"], completed=result["completed"], workspace=result["workspace"],
                    manifest=result["workspace"] + "/manifest.json", result={k: result.get(k) for k in
                    ("engineering_complete", "quality_pass", "qualified_source", "source_baseline", "certificate", "comparison", "error")})
            else:
                from .workflow import finalize
                result = finalize(**args)
                receipt.update(status=result["status"], completed=result["completed"], output=args["output"], result=result)
    except Exception as exc:
        blocked = _operational_error(type(exc).__name__)
        if receipt_mode == "compact":
            # Never re-emit a huge/NaN API result after projection/publication failure.
            receipt = {"schema_version": 1, "command": command,
                       "stdout_contract": command + "_compact_v1", "transport_error": True,
                       "error_truncated": len(str(exc)) > 2048, **durable_pointer}
        receipt.update(status="BLOCKED" if blocked else "REJECTED", completed=False,
                       engineering_complete=False, operational_failure=blocked,
                       error=str(exc)[:2048] if receipt_mode == "compact" else str(exc), error_type=type(exc).__name__)
    if receipt_mode == "compact":
        from .workflow import compact_payload
        payload = compact_payload(receipt)
    else:
        payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print(payload)
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
