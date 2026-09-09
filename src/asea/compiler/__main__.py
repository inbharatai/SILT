"""JSON-only public CLI: python -m asea.compiler."""
import argparse
from contextlib import redirect_stdout
import json
import sys

from .core import (SCHEMA, SCOPE, CompilerError, fail, inspect_model, prune_model,
                   evaluate_model, infer_model, certify_model)
from .diagnostics import (diagnose_model, roundtrip_model, diagnostic_receipt,
                          DEFAULT_WORKER_TIMEOUT_SECONDS, DEFAULT_WORKER_OUTPUT_LIMIT_BYTES,
                          DEFAULT_WORKER_ADDRESS_SPACE_MIB)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        fail("INVALID_ARGUMENT", message)


def parser():
    p = Parser(description="Local safetensors Switch compiler; seq2seq-only certification, no repair training")
    commands = p.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--model", required=True)
    prune = commands.add_parser("prune")
    prune.add_argument("--output", required=True)
    prune.add_argument("--keep-experts", type=int, required=True)
    prune.add_argument("--calibration", required=True)
    prune.add_argument("--scorer", choices=["probability_mass", "reap_dispatched"], default="probability_mass")
    prune.add_argument("--minimum-expert-observations", type=int, default=1)
    prune.add_argument("--preserve-router-fp32", action="store_true")
    prune.add_argument("--repair-training", action="store_true", help="Explicitly unsupported; returns supported=false")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--suite", required=True)
    evaluate.add_argument("--reference-model")
    evaluate.add_argument("--evidence-output", help="Optional new JSON file outside model trees; stdout always contains complete evidence")
    certify = commands.add_parser("certify")
    certify.add_argument("--suite", required=True)
    certify.add_argument("--reference-model", required=True)
    certify.add_argument("--certificate-output", required=True, help="New exclusive file outside both model directories")
    certify.add_argument("--minimum-cases", type=int, default=20)
    certify.add_argument("--max-loss", type=float, default=0.02)
    certify.add_argument("--minimum-reference-score", type=float, default=0.5)
    certify.add_argument("--minimum-candidate-score", type=float, default=0.5)
    certify.add_argument("--minimum-size-reduction", type=float, default=0.1)
    infer = commands.add_parser("infer")
    infer.add_argument("--prompt", required=True)
    for command in (prune, evaluate, infer, certify):
        command.add_argument("--model", required=True)
        command.add_argument("--dtype", choices=(["float32", "float16"] if command is certify else
                                                 ["float32", "float16", "bfloat16"]), default="float32")
        command.add_argument("--max-length", type=int, default=128)
        command.add_argument("--memory-budget-mib", type=int)
    for command in (evaluate, infer, certify):
        command.add_argument("--max-new-tokens", type=int, default=32)
    diagnose = commands.add_parser("diagnose", help="Fresh-process native/wrapper audit; never certification")
    diagnose.add_argument("--suite", required=True)
    diagnose.add_argument("--reference-model", help="Optional same-dtype source comparison, independently loaded")
    roundtrip = commands.add_parser("roundtrip", help="Identity expert mapping and save/fresh-reload audit; NOT pruning")
    roundtrip.add_argument("--suite", help="Diagnostic suite; defaults to one mechanics-only span probe")
    for command in (diagnose, roundtrip):
        command.add_argument("--model", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--dtype", choices=["float16", "bfloat16"], required=True)
        command.add_argument("--preserve-router-fp32", action="store_true")
        command.add_argument("--max-length", type=int, default=64)
        command.add_argument("--max-new-tokens", type=int, default=16)
        command.add_argument("--teacher-tokens", type=int, default=16)
        command.add_argument("--memory-budget-mib", type=int, help="Estimated resident-memory preflight budget, not an OS limit")
        command.add_argument("--worker-timeout-seconds", type=float, default=DEFAULT_WORKER_TIMEOUT_SECONDS)
        command.add_argument("--worker-output-limit-bytes", type=int, default=DEFAULT_WORKER_OUTPUT_LIMIT_BYTES,
                             help="Combined stdout/stderr ceiling per worker; logs are hashed, not echoed")
        command.add_argument("--worker-address-space-mib", type=int, default=DEFAULT_WORKER_ADDRESS_SPACE_MIB,
                             help="Per-worker RLIMIT_AS; not RSS or aggregate memory enforcement")
        command.add_argument("--atol", type=float, default=0.001)
        command.add_argument("--rtol", type=float, default=0.001)
    return p


def rejection(code, message):
    return {"command": "certify", "status": "REJECTED", "admission": "UNADMITTED",
            "reasons": [{"code": code, "message": message}], "evidence_scope": SCOPE,
            "certifies_coding_skills": False, "autoactivated": False}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    certifying = bool(argv and argv[0] == "certify")
    try:
        args = vars(parser().parse_args(argv))
        command = args.pop("command")
        if args.pop("repair_training", False):
            fail("UNSUPPORTED_REPAIR", "Repair training and Phase 6 dense reconstruction are not implemented")
        # Dependency logging must never contaminate the machine-readable stdout.
        with redirect_stdout(sys.stderr):
            result = {"inspect": inspect_model, "prune": prune_model,
                      "evaluate": evaluate_model, "infer": infer_model,
                      "certify": certify_model, "diagnose": diagnose_model,
                      "roundtrip": roundtrip_model}[command](**args)
        if command in ("diagnose", "roundtrip"):
            result = diagnostic_receipt(result)
        rejected = result.get("status") == "REJECTED"
        try:
            print(json.dumps({"ok": not rejected, **result}, sort_keys=True, allow_nan=False), flush=True)
        except BrokenPipeError:
            # Publication succeeded; delivery failed. Never auto-retry exclusive output.
            print(json.dumps({"ok": False, "error": {"type": "STDOUT_DELIVERY_FAILED"},
                              "publication": result.get("publication", "UNKNOWN"),
                              "output": result.get("output"), "report_sha256": result.get("report_sha256"),
                              "retry_safe": False}), file=sys.stderr)
            return 4
        return 2 if rejected else 0
    except CompilerError as exc:
        print(json.dumps({"ok": False, "schema": SCHEMA, "error": {"type": exc.code, "message": str(exc)}, "supported": False,
                          **(rejection(exc.code, str(exc)) if certifying else {})}, allow_nan=False))
        return 2
    except Exception as exc:  # Certification fails closed, including third-party errors.
        print(json.dumps({"ok": False, "schema": SCHEMA, "error": {"type": "RUNTIME_ERROR", "exception": type(exc).__name__, "message": str(exc)}, "supported": False,
                          **(rejection("RUNTIME_ERROR", str(exc)) if certifying else {})}, allow_nan=False))
        return 2 if certifying else 3


if __name__ == "__main__":
    sys.exit(main())
