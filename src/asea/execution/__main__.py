"""JSON-only resource CLI: python -m asea.execution probe|run."""
import argparse
import json
import sys

from .controls import SCHEMA, probe, run


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parser():
    p = Parser(prog="python -m asea.execution", add_help=False)
    commands = p.add_subparsers(dest="command", required=True, parser_class=Parser)
    inspect = commands.add_parser("probe", add_help=False)
    inspect.add_argument("--delegated-root")
    execute = commands.add_parser("run", add_help=False)
    execute.add_argument("--profile", choices=("process_as", "cgroup_v2"), required=True)
    execute.add_argument("--memory-mib", type=int, required=True)
    execute.add_argument("--timeout", type=float, required=True)
    execute.add_argument("--cpu-seconds", type=int)
    execute.add_argument("--operation", choices=("compose", "compiler", "specialist"), required=True)
    execute.add_argument("--workspace")
    execute.add_argument("--delegated-root")
    execute.add_argument("arguments", nargs=argparse.REMAINDER)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    try:
        if argv in (["--help"], ["-h"]):
            print(json.dumps({"schema": SCHEMA, "usage": p.format_help(), "examples": [
                "python -m asea.execution probe",
                "python -m asea.execution run --profile process_as --memory-mib 256 --timeout 10 --operation compose --workspace ./workspace -- list",
                "python -m asea.execution run --profile process_as --memory-mib 256 --timeout 10 --operation compiler -- --help"]}))
            return 0
        args = vars(p.parse_args(argv))
        command = args.pop("command")
        if command == "probe":
            result = probe(**args)
            code = 0
        else:
            forwarded = args.pop("arguments")
            if not forwarded or forwarded[0] != "--":
                raise ValueError("Use -- before the fixed module's arguments")
            result = run(arguments=forwarded[1:], **args)
            code = 0 if result["status"] == "OK" else 2
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return code
    except (ValueError, TypeError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "BLOCKED", "launched": False,
                          "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
