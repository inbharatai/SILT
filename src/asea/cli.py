"""Command line interface.

    python -m asea.cli suites
    python -m asea.cli modalities
    python -m asea.cli run     --config configs/assamese_transfer.json --workspace .work
    python -m asea.cli report  --workspace .work
    python -m asea.cli approve --workspace .work --packet <id> --approver you@example.org
    python -m asea.cli rollback --workspace .work --token <snapshot>
    python -m asea.cli audit   --workspace .work
    python -m asea.cli export  --workspace .work
    python -m asea.cli export  --workspace .work --base-model Qwen/Qwen2.5-7B-Instruct
    python -m asea.cli diff    --config configs/assamese_transfer.json --workspace .work \
                               --token-a <snapshot> --token-b <snapshot>
    python -m asea.cli diff-verify --workspace .work --report diff_report.json
    python -m asea.cli unlearn --config configs/assamese_transfer.json --workspace .work \
                               --suite assamese_english_v1 \
                               --token-before <snapshot> --token-after <snapshot>
    python -m asea.cli unlearn-verify --workspace .work --report unlearn_cert.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmarks.harness import load_all
from .capability_diff import CapabilityDiffer
from .config import build_pipeline, load_config
from .core.errors import SignatureMismatchError, SigningKeyError
from .core.pipeline import Pipeline
from .core.plugins import default_registry
from .distill.export import build_job_spec, export_artifact_bundle, export_dataset
from .unlearning import UnlearningVerifier

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "benchmarks"


def _emit(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def cmd_suites(args) -> int:
    suites = load_all(Path(args.data_dir))
    _emit({
        sid: {
            "task_type": s.task_type,
            "modality": s.modality.value,
            "domain": s.domain.value,
            "language": s.language,
            "splits": s.counts(),
        }
        for sid, s in suites.items()
    })
    return 0


def cmd_modalities(args) -> int:
    _emit(default_registry().modalities())
    return 0


def cmd_run(args) -> int:
    config = load_config(Path(args.config))
    pipeline, suites, adapter_id = build_pipeline(
        config, Path(args.workspace), Path(args.data_dir)
    )
    report = pipeline.run(adapter_id, suites=suites, human_approver=args.approver)
    _emit(report.to_dict())
    return 0


def cmd_report(args) -> int:
    pipeline = Pipeline(workspace=Path(args.workspace))
    _emit({
        "store": pipeline.store.stats(),
        "snapshots": pipeline.rollback.list_snapshots(),
        "audit": pipeline.audit.verify(),
        "approved_packets": [
            {
                "packet_id": p.packet_id,
                "capability": p.sender_capability.as_str(),
                "target": p.target_module,
                "packet_type": p.packet_type.value if p.packet_type else None,
                "evaluator_score": p.evaluator_score,
                "approved_by": p.human_approved_by,
                "rollback_token": p.rollback_token,
                "is_mock": p.provenance.is_mock,
            }
            for p in pipeline.store.list("approved")
        ],
        "pending_human": [
            {"packet_id": p.packet_id, "domain": p.domain.value}
            for p in pipeline.store.list("candidate")
            if p.promotion_status.value == "pending_human_approval"
        ],
    })
    return 0


def cmd_approve(args) -> int:
    pipeline = Pipeline(workspace=Path(args.workspace))
    _emit(pipeline.approve_pending(args.packet, approver=args.approver))
    return 0


def cmd_rollback(args) -> int:
    pipeline = Pipeline(workspace=Path(args.workspace))
    _emit(pipeline.rollback_to(args.token))
    return 0


def cmd_audit(args) -> int:
    pipeline = Pipeline(workspace=Path(args.workspace))
    entries = pipeline.audit.entries()
    if args.packet:
        entries = [e for e in entries if e.get("packet_id") == args.packet]
    _emit({
        "integrity": pipeline.audit.verify(),
        "entries": [
            {"index": e["index"], "event": e["event"], "actor": e["actor"],
             "packet_id": e.get("packet_id"), "timestamp": e["timestamp"]}
            for e in entries
        ],
    })
    return 0


def cmd_export(args) -> int:
    pipeline = Pipeline(workspace=Path(args.workspace))
    out_dir = Path(args.workspace) / "export"
    approved = pipeline.store.list("approved")
    audit_path = pipeline.workspace / "audit" / "audit.jsonl"

    if args.no_bundle:
        # Legacy loose-file behaviour: dataset + optional job spec only.
        manifest = export_dataset(approved, out_dir, args.name, include_mock=args.include_mock)
        spec = None
        if args.base_model is not None:
            spec = build_job_spec(manifest, base_model=args.base_model,
                                  policy=pipeline.gate.policy)
            with open(out_dir / "{}.job.json".format(args.name), "w", encoding="utf-8") as fh:
                json.dump(spec, fh, indent=2, ensure_ascii=False)
        _emit({"manifest": manifest, "job_spec": spec})
        return 0

    zip_path = export_artifact_bundle(
        approved, out_dir, name=args.name,
        base_model=args.base_model, include_mock=args.include_mock,
        audit_path=audit_path, policy=pipeline.gate.policy,
    )
    _emit({
        "bundle": str(zip_path),
        "name": args.name,
        "base_model": args.base_model,
        "include_mock": args.include_mock,
        "approved_packets": len(approved),
    })
    return 0


def cmd_diff(args) -> int:
    """Capability Diff (B1a): measure the receiver under two approved-set
    snapshots on held-out data and emit a locally HMAC-signed report.

    Reuses the evaluator's scoring path (``harness.run`` with the receiver's
    snapshot approved set), so the per-capability delta is attributable to the
    skill-set delta alone. The signature is local HMAC -- tamper-evident to the
    local key holder, NOT a portable third-party attestation (the report says
    so verbatim). Patent pending (India); local only.
    """
    config = load_config(Path(args.config))
    pipeline, suites, adapter_id = build_pipeline(
        config, Path(args.workspace), Path(args.data_dir)
    )
    binding = pipeline.adapters.get(adapter_id)
    receiver = pipeline.modules.get(binding.receiver_id)
    differ = CapabilityDiffer(
        harness=pipeline.harness,
        rollback=pipeline.rollback,
        workspace=pipeline.workspace,
        regression_tolerance=pipeline.evaluator.regression_tolerance,
        max_control_movement=pipeline.evaluator.max_control_movement,
    )
    report = differ.diff(receiver, suites, args.token_a, args.token_b)
    out = report.to_dict()
    payload = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    _emit(out)
    return 0


def cmd_diff_verify(args) -> int:
    """Verify a capability-diff report's local HMAC signature.

    A match prints ``{"valid": true, ...}`` and exits 0. A tampered report
    prints ``{"valid": false, ...}`` and exits 1. A missing/unreadable local
    signing key is a hard error (prints the reason, exits 2) -- a report whose
    key has vanished cannot be verified, and that is NOT a silent pass
    (adversarial audit 2026-08-17, I1).
    """
    pipeline = Pipeline(workspace=Path(args.workspace))
    differ = CapabilityDiffer(
        harness=pipeline.harness,
        rollback=pipeline.rollback,
        workspace=pipeline.workspace,
    )
    with open(args.report, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    try:
        result = differ.verify(report)
    except SignatureMismatchError as exc:
        _emit({"valid": False, "reason": str(exc)})
        return 1
    except SigningKeyError as exc:
        _emit({"valid": False, "unverifiable": True, "reason": str(exc)})
        return 2
    _emit(result)
    return 0


def cmd_unlearn(args) -> int:
    """Verified unlearning (B3): certify that a capability conferred by an
    approved skill packet is GONE after a rollback, on held-out data, and emit
    a locally HMAC-signed erasure certificate.

    Measures the receiver under three conditions (alone / with the before-set /
    with the after-set) and certifies adapter_removed AND capability_gone. The
    certificate's honesty_note states the SKILL-LAYER boundary verbatim: this
    is NOT weight-level forgetting. Patent pending (India); local only.
    """
    config = load_config(Path(args.config))
    pipeline, suites, adapter_id = build_pipeline(
        config, Path(args.workspace), Path(args.data_dir)
    )
    binding = pipeline.adapters.get(adapter_id)
    receiver = pipeline.modules.get(binding.receiver_id)
    if args.suite not in suites:
        _emit({"error": "unknown suite '{}'".format(args.suite),
               "available": sorted(suites)})
        return 1
    suite = suites[args.suite]
    verifier = UnlearningVerifier(
        harness=pipeline.harness,
        rollback=pipeline.rollback,
        workspace=pipeline.workspace,
        tolerance=pipeline.evaluator.regression_tolerance,
    )
    cert = verifier.verify(receiver, suite, args.token_before, args.token_after)
    out = cert.to_dict()
    payload = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    _emit(out)
    return 0


def cmd_unlearn_verify(args) -> int:
    """Verify an erasure certificate's local HMAC signature.

    A match prints ``{"valid": true, ...}`` and exits 0. A tampered certificate
    prints ``{"valid": false, ...}`` and exits 1. A missing/unreadable local
    signing key (``unlearn.key``) is a hard error (prints the reason, exits 2)
    -- a certificate whose key has vanished cannot be verified, and that is NOT
    a silent pass (adversarial audit 2026-08-17, I1).
    """
    pipeline = Pipeline(workspace=Path(args.workspace))
    verifier = UnlearningVerifier(
        harness=pipeline.harness,
        rollback=pipeline.rollback,
        workspace=pipeline.workspace,
    )
    with open(args.report, "r", encoding="utf-8") as fh:
        cert = json.load(fh)
    try:
        result = verifier.verify_certificate(cert)
    except SignatureMismatchError as exc:
        _emit({"valid": False, "reason": str(exc)})
        return 1
    except SigningKeyError as exc:
        _emit({"valid": False, "unverifiable": True, "reason": str(exc)})
        return 2
    _emit(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asea", description="Adaptive Skill Extraction Adapter"
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("suites", help="list benchmark suites").set_defaults(func=cmd_suites)
    sub.add_parser("modalities", help="list registered plugins").set_defaults(
        func=cmd_modalities
    )

    run = sub.add_parser("run", help="run a transfer from a config file")
    run.add_argument("--config", required=True)
    run.add_argument("--workspace", required=True)
    run.add_argument("--approver", default=None, help="named human approver (high-risk domains)")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="summarise a workspace")
    report.add_argument("--workspace", required=True)
    report.set_defaults(func=cmd_report)

    approve = sub.add_parser("approve", help="apply human approval to a pending packet")
    approve.add_argument("--workspace", required=True)
    approve.add_argument("--packet", required=True)
    approve.add_argument("--approver", required=True)
    approve.set_defaults(func=cmd_approve)

    rollback = sub.add_parser("rollback", help="restore an approved-set snapshot")
    rollback.add_argument("--workspace", required=True)
    rollback.add_argument("--token", required=True)
    rollback.set_defaults(func=cmd_rollback)

    audit = sub.add_parser("audit", help="verify and print the audit chain")
    audit.add_argument("--workspace", required=True)
    audit.add_argument("--packet", default=None)
    audit.set_defaults(func=cmd_audit)

    export = sub.add_parser(
        "export",
        help="bundle approved skill packets into a downloadable zip "
             "(L0-L3 skill packets +, optionally, an L4/L5 dataset and job spec)",
    )
    export.add_argument("--workspace", required=True)
    export.add_argument("--name", default="skill_bundle")
    export.add_argument(
        "--base-model", default=None,
        help="base model for an L4/L5 training job spec; omit for L0-L3 skill "
             "packets (no job spec is written, honestly -- SILT trains no weights)",
    )
    export.add_argument("--include-mock", action="store_true")
    export.add_argument(
        "--no-bundle", action="store_true",
        help="legacy mode: write loose dataset + job spec files instead of a zip",
    )
    export.set_defaults(func=cmd_export)

    diff = sub.add_parser(
        "diff",
        help="capability diff between two approved-set snapshots (B1a): a "
             "locally HMAC-signed, held-out per-capability delta. Reuses the "
             "evaluator's scoring path; NOT a portable attestation.",
    )
    diff.add_argument("--config", required=True)
    diff.add_argument("--workspace", required=True)
    diff.add_argument("--token-a", required=True, help="snapshot token (before)")
    diff.add_argument("--token-b", required=True, help="snapshot token (after)")
    diff.add_argument("--out", default=None, help="write the signed report JSON to this path too")
    diff.set_defaults(func=cmd_diff)

    diff_verify = sub.add_parser(
        "diff-verify",
        help="verify a capability-diff report's local HMAC signature",
    )
    diff_verify.add_argument("--workspace", required=True)
    diff_verify.add_argument("--report", required=True, help="path to a signed diff report JSON")
    diff_verify.set_defaults(func=cmd_diff_verify)

    unlearn = sub.add_parser(
        "unlearn",
        help="verified unlearning (B3): a locally HMAC-signed erasure "
             "certificate that a capability conferred by an approved skill "
             "packet is GONE after a rollback, measured on held-out data. "
             "SKILL-LAYER unlearning, NOT weight-level forgetting.",
    )
    unlearn.add_argument("--config", required=True)
    unlearn.add_argument("--workspace", required=True)
    unlearn.add_argument("--suite", required=True, help="suite_id of the capability to verify gone")
    unlearn.add_argument("--token-before", required=True, help="snapshot token that CONTAINED the packet")
    unlearn.add_argument("--token-after", required=True, help="snapshot token AFTER the rollback (packet removed)")
    unlearn.add_argument("--out", default=None, help="write the signed certificate JSON to this path too")
    unlearn.set_defaults(func=cmd_unlearn)

    unlearn_verify = sub.add_parser(
        "unlearn-verify",
        help="verify an erasure certificate's local HMAC signature",
    )
    unlearn_verify.add_argument("--workspace", required=True)
    unlearn_verify.add_argument("--report", required=True, help="path to a signed erasure certificate JSON")
    unlearn_verify.set_defaults(func=cmd_unlearn_verify)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
