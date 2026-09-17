"""Hardware/tokenizer preflight CLI. Never starts model execution stages."""
import argparse
import json
from pathlib import Path

from . import probe_hardware, plan_specialist


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only hardware/planning; no model loading, training, execution or promotion')
    commands = parser.add_subparsers(dest='command', required=True)
    probe = commands.add_parser('probe', help='Bounded selected-Python hardware primitive probe; no models')
    probe.add_argument('--python', dest='python_executable')
    probe.add_argument('--path', action='append', default=[])
    probe.add_argument('--output')
    plan = commands.add_parser('plan', help='Local TRAIN/DEV tokenizer preflight; no models or execution approval')
    plan.add_argument('--no-data-profile', action='store_true', help='Header-only PLANNING_ONLY estimates; no train/dev reads or tokenizer loads')
    plan.add_argument('--recipe', required=True)
    plan.add_argument('--device', default=None)
    plan.add_argument('--workspace-parent', default=None)
    plan.add_argument('--python', dest='python_executable')
    plan.add_argument('--output')
    args = parser.parse_args(argv)
    if args.command=='probe':
        value = probe_hardware(args.python_executable, args.path)
        value.update(no_execution=True, model_weights_loaded=False)
    else:
        hardware = None
        if args.python_executable:
            # Include exactly the volume on which the future workspace will live.
            hardware = probe_hardware(args.python_executable, [args.workspace_parent or str(Path.cwd())])
        value = plan_specialist(args.recipe, hardware, workspace_parent=args.workspace_parent, requested_device=args.device,
                                python_executable=args.python_executable, profile_data=not args.no_data_profile)
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output:
        from asea.specialist.workflow import write_json
        write_json(args.output,value)  # exclusive private report; never overwrite evidence
    print(raw,end='')
    return 0 if args.command=='probe' or value['status']=='READY' else 2


if __name__=='__main__':
    raise SystemExit(main())
