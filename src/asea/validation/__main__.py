"""python -m asea.validation: local governance and pending voice evidence only."""
import argparse
import json
import sys

from . import (ValidationRegistry, exclusive_json, load_json, prepare_listening_batch,
               prepare_voice_evidence, proportion_summary, validate_listening_review)
from . import schema


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", help="candidate workspace (registry is a disjoint sibling by default)")
    parser.add_argument("--registry", help="explicit disjoint validation registry directory")
    commands = parser.add_subparsers(dest="command", required=True)
    governed = commands.add_parser("governed-evaluate", help="observe the public compose gate; function_io_pilot_v1")
    governed.add_argument("--spec", required=True)
    governed.add_argument("--suite", required=True)
    governed.add_argument("--purpose", choices=["development", "final"], required=True)
    governed.add_argument("--compose-workspace", required=True, help="new empty root, disjoint from validation store")
    governed.add_argument("--dataset-manifest", help="pilot-v2 manifest.json; defaults to suite sibling")
    governed.add_argument("--resource-profile", choices=["process_as"], required=True)
    governed.add_argument("--memory-mib", type=int, required=True)
    governed.add_argument("--case-timeout", type=float, required=True)
    governed.add_argument("--max-seconds", type=float, default=3600.0, help="overall public CLI wall budget, plus at most 10s cleanup")
    governed.add_argument("--workspace", default=argparse.SUPPRESS, help="validation store (also accepted before command)")
    register = commands.add_parser("register")
    register.add_argument("--suite", required=True, help="closed data-only suite JSON")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--suite", required=True, help="registered suite ID")
    freeze.add_argument("--candidate-hash", required=True)
    freeze.add_argument("--policy", required=True)
    begin = commands.add_parser("begin")
    begin.add_argument("--reservation", required=True)
    begin.add_argument("--candidate-hash", required=True)
    begin.add_argument("--policy", required=True)
    finish = commands.add_parser("finish", help="record unverified result input, never issue a certificate")
    finish.add_argument("--run", required=True)
    finish.add_argument("--result", required=True)
    commands.add_parser("history")
    commands.add_parser("recover")
    voice = commands.add_parser("voice-check", help="measure PCM facts and leave quality pending_human")
    voice.add_argument("--audio", required=True)
    voice.add_argument("--text", required=True)
    voice.add_argument("--language", required=True)
    voice.add_argument("--model-manifest", required=True)
    voice.add_argument("--output", help="optional exclusive-write JSON output")
    batch = commands.add_parser("export-listen-batch", help="propose a local coordinator packet; no upload or approval")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--output", required=True)
    review = commands.add_parser("check-review", help="pure completeness check; never grants human approval")
    review.add_argument("--review", required=True)
    review.add_argument("--packet", required=True)
    stats = commands.add_parser("proportion")
    stats.add_argument("--successes", type=int, required=True)
    stats.add_argument("--cases", type=int, required=True)
    schemas = commands.add_parser("schema")
    schemas.add_argument("name", choices=["Suite", "Policy", "ResultInput", "VoiceModelManifest", "ListeningReview", "ListeningBatch"])
    args = parser.parse_args(argv)
    try:
        if args.command == "governed-evaluate":
            from .governed import governed_evaluate
            if not args.workspace:
                parser.error("--workspace is required for governed-evaluate")
            result = governed_evaluate(workspace=args.workspace, registry=args.registry,
                compose_workspace=args.compose_workspace, spec=args.spec, suite=args.suite,
                purpose=args.purpose, dataset_manifest=args.dataset_manifest,
                resource_profile=args.resource_profile, memory_mib=args.memory_mib,
                case_timeout=args.case_timeout, max_seconds=args.max_seconds)
            print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
            interrupted = result["receipt"].get("interruption_signal")
            if interrupted is not None:
                return 128 + interrupted
            code = result["receipt"]["exit_code"]
            return code if isinstance(code, int) and 0 < code < 256 else (0 if result["outcome"]["quality_admission"] else 2)
        elif args.command == "voice-check":
            result = prepare_voice_evidence(args.audio, args.text, args.language, args.model_manifest)
        elif args.command == "export-listen-batch":
            result = prepare_listening_batch(args.manifest)
        elif args.command == "check-review":
            result = validate_listening_review(load_json(args.review), load_json(args.packet))
        elif args.command == "proportion":
            result = proportion_summary(args.successes, args.cases)
        elif args.command == "schema":
            result = getattr(schema, args.name).model_json_schema()
        else:
            if not args.workspace:
                parser.error("--workspace is required for registry commands")
            registry = ValidationRegistry(args.workspace, args.registry)
            if args.command == "register":
                result = registry.register(load_json(args.suite))
            elif args.command == "freeze":
                result = registry.freeze(args.suite, args.candidate_hash, load_json(args.policy))
            elif args.command == "begin":
                result = registry.begin(args.reservation, args.candidate_hash, load_json(args.policy))
            elif args.command == "finish":
                result = registry.finish(args.run, load_json(args.result))
            elif args.command == "history":
                result = registry.history()
            else:
                result = registry.recover()
        if getattr(args, "output", None):
            exclusive_json(args.output, result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc), "quality_admission": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
