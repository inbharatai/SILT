"""Public JSON-only entry point: python -m asea.compose --workspace PATH COMMAND."""
import argparse
import contextlib
import json
import os
import sys

from asea.artifacts import Blocked, Workspace, safe_file
from asea.certification import activate, evaluate, export_deployment, rollback, select_measured, plan_graph, recover_active
from .runtime import inspect_spec, run_spec
from .schema import CompositionSpec, EvaluationSuite, load_json


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        raise Blocked(message)


def parser():
    result = JsonParser(prog="python -m asea.compose", add_help=False,
                        description="Local fixed composition; JSON stdout; no runtime downloads or automatic activation.")
    result.add_argument("--workspace")  # required below except for read-only preview
    commands = result.add_subparsers(dest="command", required=True, parser_class=JsonParser)
    preview = commands.add_parser("preview", add_help=False)
    preview.add_argument("--spec", required=True)
    preview.add_argument("--output-format", required=True, choices=["raw_python", "fenced_python"])
    inspect = commands.add_parser("inspect", add_help=False)
    inspect.add_argument("--spec", required=True)
    run = commands.add_parser("run", add_help=False)
    run.add_argument("--spec", required=True)
    run.add_argument("--input")
    run.add_argument("--input-file")
    evaluate_parser = commands.add_parser("evaluate", add_help=False)
    evaluate_parser.add_argument("--spec", required=True)
    evaluate_parser.add_argument("--suite", required=True)
    evaluate_parser.add_argument("--resource-profile", choices=["observe_only", "process_as", "cgroup_v2"], default="observe_only")
    evaluate_parser.add_argument("--memory-mib", type=int, default=512)
    evaluate_parser.add_argument("--case-timeout", type=float, default=60.0)
    evaluate_parser.add_argument("--trace-policy", choices=["none", "digest", "value"], default="none")
    activation = commands.add_parser("activate", add_help=False)
    activation.add_argument("--evaluation", required=True)
    activation.add_argument("--approve-high-risk", action="store_true")
    activation.add_argument("--approver")
    select = commands.add_parser("select", add_help=False)
    select.add_argument("--evaluations", nargs="+", required=True)
    select.add_argument("--input-type", choices=["text", "audio", "image"], required=True)
    select.add_argument("--output-type", choices=["text", "audio"], required=True)
    select.add_argument("--max-wall-seconds", type=float, required=True)
    select.add_argument("--max-peak-rss-mb", type=float, required=True)
    select.add_argument("--suite-hash")
    select.add_argument("--max-disk-bytes", type=int)
    select.add_argument("--graph-hash")
    plan = commands.add_parser("plan", add_help=False)
    plan.add_argument("--spec", required=True)
    rollback_parser = commands.add_parser("rollback", add_help=False)
    rollback_parser.add_argument("--deployment", required=True)
    export = commands.add_parser("export", add_help=False)
    export.add_argument("--deployment", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--include-dependencies", action="store_true")
    export.add_argument("--include-sensitive-traces", action="store_true")
    commands.add_parser("list", add_help=False)
    return result


def preview_spec(spec, output_format):
    """No Workspace, artifact registration, model inventory, or model loading."""
    from .generation_contract import migration_preview
    return {"schema_version": 1, "read_only": True, "applied": False,
            "output_format": output_format,
            "nodes": [{"node_id": node.id, "adapter": node.component.kind,
                       "support": "supported" if node.component.kind == "hf_text" else "unsupported_adapter",
                       **migration_preview(node, output_format)} for node in spec.nodes]}


def main(argv=None):
    """Return an integer exit code. All stdout, including help/errors, is one JSON object."""
    argv = list(sys.argv[1:] if argv is None else argv)
    result_parser = parser()
    code = 0
    try:
        if "--help" in argv or "-h" in argv:
            result = {"ok": True, "usage": result_parser.format_help(), "commands": ["preview --spec FILE --output-format raw_python|fenced_python (no workspace required)", "inspect --spec FILE", "run --spec FILE [--input TEXT] [--input-file PCM_WAV_OR_IMAGE]", "evaluate --spec FILE --suite FILE [--resource-profile observe_only|process_as|cgroup_v2] [--memory-mib N] [--case-timeout N] [--trace-policy none|digest|value]", "activate --evaluation ID [--approver NAME] [--approve-high-risk]", "select --evaluations ID ... --input-type text/audio/image --output-type text/audio --max-wall-seconds N --max-peak-rss-mb N [--suite-hash SHA256] [--max-disk-bytes N] [--graph-hash SHA256]", "plan --spec FILE", "rollback --deployment ID", "export --deployment ID --output ZIP [--include-dependencies] [--include-sensitive-traces]", "list"]}
        else:
            with contextlib.redirect_stdout(sys.stderr):
                args = result_parser.parse_args(argv)
                if args.command != "preview" and not args.workspace:
                    raise Blocked("--workspace is required except for read-only preview")
                workspace = None if args.command == "preview" else Workspace(args.workspace)
                if args.command == "preview":
                    payload = preview_spec(load_json(args.spec, CompositionSpec), args.output_format)
                elif args.command == "inspect":
                    payload = inspect_spec(workspace, load_json(args.spec, CompositionSpec))
                elif args.command == "run":
                    payload = run_spec(workspace, load_json(args.spec, CompositionSpec), args.input, args.input_file)
                elif args.command == "evaluate":
                    payload = evaluate(workspace, load_json(args.spec, CompositionSpec), load_json(args.suite, EvaluationSuite),
                                       resource_profile=args.resource_profile, memory_mib=args.memory_mib,
                                       case_timeout=args.case_timeout, trace_policy=args.trace_policy)
                    if payload["status"] != "admitted":
                        code = 2
                elif args.command == "activate":
                    payload = activate(workspace, args.evaluation, args.approve_high_risk, args.approver)
                elif args.command == "select":
                    payload = select_measured(workspace, args.evaluations, args.input_type, args.output_type,
                                              args.max_wall_seconds, args.max_peak_rss_mb, args.suite_hash,
                                              args.max_disk_bytes, args.graph_hash)
                    if payload["status"] != "selected":
                        code = 2
                elif args.command == "plan":
                    payload = plan_graph(load_json(args.spec, CompositionSpec))
                elif args.command == "rollback":
                    payload = rollback(workspace, args.deployment)
                elif args.command == "export":
                    payload = export_deployment(workspace, args.deployment, args.output, args.include_dependencies,
                                                include_sensitive_traces=args.include_sensitive_traces)
                else:
                    with workspace.writer():
                        recover_active(workspace)
                        payload = workspace.list_records()
                        pointer = workspace.root / "active.json"
                        payload["active"] = json.loads(safe_file(pointer).read_text()) if pointer.exists() else None
                        if payload["active"]:
                            workspace.read_record("deployments", payload["active"]["deployment_id"])
            result = {"ok": code == 0, "command": args.command, "result": payload}
            if args.command == "run":
                # Supervisor PID binding only; the run payload must independently
                # match its same-workspace HMAC record, never trust stdout alone.
                result["execution_pid"] = os.getpid()
    except Exception as exc:
        code = 2 if isinstance(exc, (ValueError, OSError, KeyError, TypeError)) else 1
        result = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
