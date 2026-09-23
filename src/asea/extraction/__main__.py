"""JSON-only public CLI: python -m asea.extraction (``silt-extract``).

Exit-code contract (extraction-brief C5; DIFFERENT from
``silt-compile``/``silt-capability`` -- do not mix them):

  * 0 -- successful operation
  * 2 -- typed refusal / rejected request
  * 3 -- invalid evidence or schema
  * 4 -- resource blocked (exact requirement + remedy; never a
    substituted estimate or fixture result)
  * 5 -- experimental execution failure

stdout is ALWAYS exactly one JSON object; diagnostics go to stderr.
Nothing auto-activates and no command deploys anything.

Commands (brief section W), all 18 wired: ``spec validate``,
``source inventory``, ``source verify``, ``dataset validate``,
``baseline``, ``trace``, ``intervene``, ``graph``, ``plan``,
``build``, ``provenance verify``, ``standalone verify``, ``recover``,
``evaluate``, ``compare``, ``reduce``, ``certify``, ``receipt``.
Stages whose real-model implementation is still ahead of this build
report an honest NOT_IMPLEMENTED refusal (exit 2) with the status
vocabulary -- a missing surface is stated, never faked; and once a
stage IS implemented, an 8 GB host that cannot hold the source model
reports BLOCKED_RESOURCE (exit 4), which is what will happen here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

from asea.extraction import (
    SealedSplit,
    SOURCE_MANIFEST_SCHEMA,
    SPEC_SCHEMA,
)
from asea.extraction.errors import (
    EXIT_BLOCKED,
    EXIT_EXECUTION_FAILURE,
    EXIT_INVALID_EVIDENCE,
    EXIT_REFUSED,
    EXIT_SUCCESS,
    ExtractionError,
    InvalidEvidence,
    Refused,
)
from asea.extraction.schema import (
    CapabilityExtractionSpec,
    CheckpointFile,
    SourceCheckpointManifest,
    source_manifest_fingerprint,
)

PROG = "silt-extract"

#: The status vocabulary (binding): exactly one status per item, and a
#: fixture result can never upgrade a real-model status.
STATUS_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"

#: Every command's output carries this line verbatim.
HONESTY_NOTE = (
    "Frequency is correlation; only controlled interventions support "
    "causal language. Fixture evidence is FIXTURE-verified mechanism "
    "evidence, never real-model evidence."
)

#: The thresholds are preregistered before any measurement and are
#: never weakened after seeing final results (brief section A).
RETENTION_THRESHOLDS_NOTE = (
    "thresholds were preregistered before any measurement and will not "
    "be weakened after seeing final results"
)

#: Real-model stages whose implementation is still ahead of this build.
#: Each entry: command -> (what the stage will do, what it needs).
_NOT_IMPLEMENTED_COMMANDS = {
    "baseline": (
        "measure the source model's functional baseline on the "
        "training/development cases",
        "the isolated GLM worker on a host its preflight admits; on the "
        "current 8 GB host the real run would be BLOCKED_RESOURCE",
    ),
    "trace": (
        "collect judged routing/activation observations joined to "
        "host-oracle outcomes (C2)",
        "the isolated GLM worker plus the host functional oracle",
    ),
    "intervene": (
        "run the controlled mask/measure/restore intervention on one "
        "canonical decoder component",
        "the isolated GLM worker plus the host functional oracle",
    ),
    "graph": (
        "build the causal component graph from verified intervention "
        "attempts with matched-control arms",
        "intervention attempts recorded by the intervene stage",
    ),
    "plan": (
        "derive the conservative ExtractionPlan (retain tokenizer, "
        "embeddings, dense layers, attention, head; causally selected "
        "routed-expert subset; rebuilt routers)",
        "a completed causal component graph",
    ),
    "build": (
        "physically extract and transform the GLM parameters into the "
        "standalone CompiledCapabilityModel",
        "an approved ExtractionPlan plus a host that can hold the source "
        "checkpoint in memory; on the current 8 GB host the real run "
        "would be BLOCKED_RESOURCE",
    ),
    "provenance verify": (
        "verify tensor provenance against the exact source manifest and "
        "account for every stored tensor",
        "a built CompiledCapabilityModel manifest",
    ),
    "standalone verify": (
        "prove the extracted artifact loads and generates OFFLINE with "
        "no source path and no network",
        "a built CompiledCapabilityModel",
    ),
    "recover": (
        "repair extraction damage with recovery training on the "
        "extracted artifact only (training-split data only)",
        "a built CompiledCapabilityModel plus training compute",
    ),
    "evaluate": (
        "run the untouched host-oracle functional evaluation; the "
        "final-split arm opens the sealed artifact exactly once (C6)",
        "a built or recovered CompiledCapabilityModel plus the Linux "
        "code sandbox for oracle grading",
    ),
    "compare": (
        "compare source vs compiled model on targets AND controls with "
        "the identical generation policy",
        "completed evaluations of both models",
    ),
    "reduce": (
        "search the reduction space under the preregistered acceptance "
        "rule (parameter count alone is never evidence)",
        "a verified standalone CompiledCapabilityModel",
    ),
    "certify": (
        "issue the capability-extraction receipt with the final verdict "
        "against the preregistered thresholds",
        "a sealed-final evaluation consumed exactly once plus the "
        "compare results",
    ),
}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refused("invalid argument: %s" % message,
                      "run `%s --help`" % PROG)


def _emit(payload: Dict[str, Any]) -> int:
    try:
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
    except BrokenPipeError:
        # stdout delivery failed after the operation: exit 5 (execution
        # failure), never 4 (4 means BLOCKED_RESOURCE here -- a different
        # contract than silt-compile/silt-capability).
        return 5
    return 0


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_spec(path: str) -> Dict[str, Any]:
    """Load and validate a CapabilityExtractionSpec. A malformed spec is
    invalid evidence (exit 3), not a crash."""
    resolved = Path(path)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise InvalidEvidence(
            "spec file is not valid JSON: %s" % exc,
            "fix the spec file; it must be a silt.extraction.spec.v1 "
            "JSON document")
    if not isinstance(raw, dict) or raw.get("schema") != SPEC_SCHEMA:
        raise InvalidEvidence(
            "spec file does not declare schema %r" % SPEC_SCHEMA,
            "regenerate the spec from the program's spec template; an "
            "unversioned document is not a spec")
    try:
        spec = CapabilityExtractionSpec.model_validate(raw)
    except Exception as exc:
        raise InvalidEvidence(
            "spec failed schema validation: %s" % exc,
            "correct the fields listed in the validation errors; "
            "thresholds are preregistered and must be set deliberately, "
            "never defaulted")
    return {
        "spec": spec,
        "raw": raw,
        "spec_sha256": _sha256_text(resolved.read_text(encoding="utf-8")),
    }


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Implemented commands
# ---------------------------------------------------------------------------

def _cmd_spec_validate(args) -> Dict[str, Any]:
    loaded = _load_spec(args.config)
    spec = loaded["spec"]
    return {
        "ok": True,
        "command": "spec validate",
        "status": "completed",
        "capability_id": spec.capability_id,
        "spec_sha256": loaded["spec_sha256"],
        "preregistered_thresholds": {
            "minimum_retention_ratio":
                spec.thresholds.minimum_retention_ratio,
            "maximum_stored_parameter_reduction":
                spec.thresholds.maximum_stored_parameter_reduction,
            "note": RETENTION_THRESHOLDS_NOTE,
        },
        "sealed_final_path": spec.sealed_final_path,
    }


def _checkpoint_tree(checkpoint: Path):
    if not checkpoint.is_dir():
        raise InvalidEvidence(
            "checkpoint %s is not an existing directory" % checkpoint,
            "point --checkpoint at the unpacked source checkpoint "
            "directory (config.json at its root)")
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            yield path.relative_to(checkpoint).as_posix(), path


def _cmd_source_inventory(args) -> Dict[str, Any]:
    """Disk-only inventory: names and sizes, NO hashing and NO model
    load -- takable on a host that cannot hold the model. This is a
    RECONNAISSANCE artifact, never exact identity (that is
    ``source verify``)."""
    checkpoint = Path(args.checkpoint)
    entries = []
    total_bytes = 0
    for name, path in _checkpoint_tree(checkpoint):
        size = path.stat().st_size
        total_bytes += size
        entries.append({"filename": name, "bytes": size})
    required_present = {
        "config.json": any(e["filename"] == "config.json" for e in entries),
        "generation_config.json": any(
            e["filename"] == "generation_config.json" for e in entries),
        "tokenizer_config.json": any(
            e["filename"] == "tokenizer_config.json" for e in entries),
        "weight_shards": any(
            e["filename"].endswith(".safetensors") for e in entries),
    }
    return {
        "ok": True,
        "command": "source inventory",
        "status": "completed",
        "checkpoint": str(checkpoint),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "required_present": required_present,
        "note": (
            "disk-only inventory (no hashing, no model load); EXACT "
            "identity is source verify"
        ),
    }


def _cmd_source_verify(args) -> Dict[str, Any]:
    """The C3 exact-identity manifest: sha256 EVERY file in the
    checkpoint tree and bind it to the declared source card
    (repository/model/variant/commit_sha/license). A host that cannot
    hold the model can still take this manifest -- it never loads the
    model."""
    card_path = Path(args.source_card)
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise InvalidEvidence(
            "source card is not valid JSON: %s" % exc,
            "the source card declares repository/model/variant/"
            "commit_sha/license of the exact checkpoint being verified")
    for field in ("repository", "model", "variant", "commit_sha",
                  "license"):
        if not isinstance(card.get(field), str) or not card[field]:
            raise InvalidEvidence(
                "source card missing required string field %r" % field,
                "fill every field: exact identity is claimed only when "
                "the source card AND the shard digests are complete")
    checkpoint = Path(args.checkpoint)
    files: Dict[str, CheckpointFile] = {}
    for name, path in _checkpoint_tree(checkpoint):
        import hashlib

        digest = hashlib.sha256()
        with open(str(path), "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[name] = CheckpointFile(
            filename=name, bytes=path.stat().st_size,
            sha256=digest.hexdigest())
    names = set(files)
    for required in ("config.json", "generation_config.json",
                     "tokenizer_config.json"):
        if required not in names:
            raise InvalidEvidence(
                "checkpoint %s has no %r; exact identity is claimed only "
                "when every required file is verified" % (checkpoint,
                                                         required),
                "unpack the FULL checkpoint (all shards, tokenizer, "
                "processor and configs) and re-run")
    if not any(name.endswith(".safetensors") for name in names):
        raise InvalidEvidence(
            "checkpoint %s contains no .safetensors shard" % checkpoint,
            "the exact-identity contract hashes only safetensors "
            "checkpoints; convert or download the safetensors release")
    chat_template_source = ("chat_template.jinja"
                           if "chat_template.jinja" in names
                           else "tokenizer_config.json")
    manifest = SourceCheckpointManifest(
        repository=card["repository"],
        model=card["model"],
        variant=card["variant"],
        commit_sha=card["commit_sha"],
        license=card["license"],
        files=files,
        chat_template_source=chat_template_source,
        aggregate_sha256=source_manifest_fingerprint(files),
    )
    payload = manifest.model_dump(mode="json", by_alias=True)
    written = None
    if not args.dry_run:
        workspace = Path(args.workspace)
        out = workspace / "source-manifests" / ("%s.json" % card["model"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, sort_keys=True) + "\n",
                       encoding="utf-8")
        written = str(out)
    return {
        "ok": True,
        "command": "source verify",
        "status": "completed",
        "schema": SOURCE_MANIFEST_SCHEMA,
        "model": card["model"],
        "commit_sha": card["commit_sha"],
        "file_count": len(files),
        "aggregate_sha256": manifest.aggregate_sha256,
        "chat_template_source": chat_template_source,
        "artifact": written,
        "dry_run": bool(args.dry_run),
    }


def _cmd_dataset_validate(args) -> Dict[str, Any]:
    """Validate one FunctionalCaseManifest. Near-duplicate freedom,
    prompt-hash uniqueness and the split vocabulary are enforced at
    schema level; unjudged telemetry is not the dataset's concern."""
    resolved = Path(args.dir)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise InvalidEvidence(
            "case manifest is not valid JSON: %s" % exc,
            "case manifests are silt.extraction.cases.v1 JSON documents")
    if not isinstance(raw, dict) or raw.get("schema") != "silt.extraction.cases.v1":
        raise InvalidEvidence(
            "case manifest does not declare schema "
            "silt.extraction.cases.v1",
            "regenerate the manifest from the dataset builder")
    from asea.extraction.schema import FunctionalCaseManifest

    try:
        manifest = FunctionalCaseManifest.model_validate(raw)
    except Exception as exc:
        raise InvalidEvidence(
            "case manifest failed schema validation: %s" % exc,
            "correct the manifest; near-duplicate freedom and prompt "
            "uniqueness are prerequisites, not afterthoughts")
    return {
        "ok": True,
        "command": "dataset validate",
        "status": "completed",
        "capability_id": manifest.capability_id,
        "split": manifest.split,
        "cases": len(manifest.cases),
        "families": sorted({case.family for case in manifest.cases}),
    }


def _cmd_receipt(args) -> Dict[str, Any]:
    """Materialise and sign a CapabilityExtractionReceipt from a JSON
    payload plus the workspace's failed-attempt ledger. The receipt's
    failure history comes from the ledger (C7) -- failed attempts stay
    visible forever -- and the verdict is validated against the
    preregistered thresholds echoed in the receipt itself."""
    resolved = Path(args.receipt)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise InvalidEvidence(
            "receipt payload is not valid JSON: %s" % exc,
            "receipt payloads are silt.extraction.receipt.v1 documents")
    from asea.extraction.ledger import FailedAttemptLedger
    from asea.extraction.schema import CapabilityExtractionReceipt

    try:
        receipt = CapabilityExtractionReceipt.model_validate(raw)
    except Exception as exc:
        raise InvalidEvidence(
            "receipt failed schema validation: %s" % exc,
            "a CERTIFIED verdict requires measured retention above the "
            "preregistered thresholds AND exactly-once final-split "
            "evaluation; anything else must be recorded as "
            "FAILED_HYPOTHESIS / PARTIAL_RESULT / NOT_MEASURED / "
            "BLOCKED_RESOURCE, never estimated")
    ledger = FailedAttemptLedger(Path(args.workspace) / "failed-attempts.jsonl")
    payload = receipt.model_dump(mode="json", by_alias=True)
    payload["failure_history"] = (payload.get("failure_history")
                                  or ledger.failure_history())
    written = None
    if not args.dry_run:
        workspace = Path(args.workspace)
        out = workspace / "receipts" / ("%s.json"
                                        % receipt.capability_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, sort_keys=True) + "\n",
                       encoding="utf-8")
        written = str(out)
    return {
        "ok": True,
        "command": "receipt",
        "status": "completed",
        "capability_id": receipt.capability_id,
        "final_verdict": receipt.final_verdict,
        "retained_ratio": payload["retained_ratio"],
        "stored_parameter_reduction": payload["stored_parameter_reduction"],
        "ledger_chain_ok": ledger.verify_chain(),
        "artifact": written,
        "dry_run": bool(args.dry_run),
    }


# ---------------------------------------------------------------------------
# Honest refusals
# ---------------------------------------------------------------------------

def _cmd_not_implemented(command: str):
    def handler(args) -> Dict[str, Any]:
        action, needs = _NOT_IMPLEMENTED_COMMANDS[command]
        sealed_note = None
        if command == "evaluate" and getattr(args, "split", None) == "final":
            # C6 discipline: a development-side invocation naming the
            # final split never gets to touch the sealed artifact, and
            # the attempt is recorded in the seal's access log.
            sealed = SealedSplit(Path(args.sealed_dir
                                      if getattr(args, "sealed_dir", None)
                                      else "."))
            try:
                sealed.access(command="silt-extract evaluate --split final",
                               authority="development")
            except Refused:
                sealed_note = (
                    "the sealed final split exists at %s; this NOT_IMPLEMENTED "
                    "refusal did NOT open it, and the access attempt was "
                    "recorded in its access log" % sealed.seal_path
                ) if sealed.seal_path.exists() else (
                    "no sealed final split present; nothing was opened"
                )
        return {
            "ok": False,
            "command": command,
            "status": STATUS_NOT_IMPLEMENTED,
            "reason": (
                "%s: not implemented in this build; the stage needs %s. "
                "This surface refuses honestly rather than emitting a "
                "placeholder result." % (action, needs)
            ),
            "sealed_split_note": sealed_note,
        }

    return handler


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parser() -> Parser:
    p = Parser(
        prog=PROG,
        description=(
            "SILT GLM-5.3-Flash capability extraction and model "
            "conversion: extract a CompiledCapabilityModel and prove "
            "retention on a sealed final split. Nothing auto-activates "
            "and no command deploys anything."
        ),
    )
    commands = p.add_subparsers(dest="command", required=True)

    spec_cmd = commands.add_parser("spec", help="Capability extraction spec operations")
    spec_sub = spec_cmd.add_subparsers(dest="spec_command", required=True)
    spec_validate = spec_sub.add_parser("validate")
    spec_validate.add_argument("--config", required=True)
    _common(spec_validate)

    source_cmd = commands.add_parser("source", help="Source checkpoint identity (C3)")
    source_sub = source_cmd.add_subparsers(dest="source_command", required=True)
    inventory = source_sub.add_parser("inventory",
                                      help="Disk-only reconnaissance (no hashing)")
    inventory.add_argument("--checkpoint", required=True)
    _common(inventory)
    verify = source_sub.add_parser("verify",
                                   help="Exact identity: sha256 the whole tree")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--source-card", required=True)
    _common(verify)

    dataset_cmd = commands.add_parser("dataset", help="Functional case manifests")
    dataset_sub = dataset_cmd.add_subparsers(dest="dataset_command", required=True)
    dataset_validate = dataset_sub.add_parser("validate")
    dataset_validate.add_argument("--dir", required=True)
    _common(dataset_validate)

    for command in ("baseline", "trace", "intervene", "graph", "plan",
                    "build", "recover", "evaluate", "compare", "reduce",
                    "certify"):
        entry = commands.add_parser(
            command,
            help="[real-model stage] %s"
                 % _NOT_IMPLEMENTED_COMMANDS[command][0])
        entry.add_argument("--config",
                           help="capability extraction spec (JSON)")
        entry.add_argument("--spec", help=argparse.SUPPRESS)
        if command == "evaluate":
            entry.add_argument("--split",
                               choices=["training", "development", "heldout",
                                        "final", "controls"],
                               default=None)
            entry.add_argument("--sealed-dir")
        if command in ("trace", "intervene", "baseline", "evaluate"):
            entry.add_argument("--checkpoint")
        _common(entry)

    provenance_cmd = commands.add_parser("provenance",
                                         help="Tensor provenance operations")
    provenance_sub = provenance_cmd.add_subparsers(dest="provenance_command",
                                                   required=True)
    provenance_verify = provenance_sub.add_parser("verify")
    _common(provenance_verify)

    standalone_cmd = commands.add_parser("standalone",
                                         help="Standalone artifact operations")
    standalone_sub = standalone_cmd.add_subparsers(dest="standalone_command",
                                                  required=True)
    standalone_verify = standalone_sub.add_parser("verify")
    _common(standalone_verify)

    receipt_cmd = commands.add_parser("receipt", help="Materialise and sign the receipt")
    receipt_cmd.add_argument("--receipt", required=True)
    _common(receipt_cmd)
    return p


def _common(entry):
    """Every command accepts the brief's universal flags. ``--json`` is
    accepted for interface uniformity -- stdout is ALWAYS JSON."""
    entry.add_argument("--dry-run", action="store_true",
                       help="compute without writing any artifact")
    entry.add_argument("--workspace", default=".extraction")
    entry.add_argument("--json", action="store_true", default=True,
                       help=argparse.SUPPRESS)
    entry.add_argument("--resume", action="store_true",
                       help="resume an interrupted run when resumable "
                            "state exists; without it the command starts "
                            "fresh")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command_for_error = argv[0] if argv else "unknown"
    try:
        args = vars(parser().parse_args(argv))
        command = args.pop("command")
        spec_command = args.pop("spec_command", None)
        source_command = args.pop("source_command", None)
        dataset_command = args.pop("dataset_command", None)
        provenance_command = args.pop("provenance_command", None)
        standalone_command = args.pop("standalone_command", None)
        namespace = argparse.Namespace(**args)
        if command == "spec" and spec_command == "validate":
            handler = _cmd_spec_validate
        elif command == "source" and source_command == "inventory":
            handler = _cmd_source_inventory
        elif command == "source" and source_command == "verify":
            handler = _cmd_source_verify
        elif command == "dataset" and dataset_command == "validate":
            handler = _cmd_dataset_validate
        elif command == "provenance" and provenance_command == "verify":
            handler = _cmd_not_implemented("provenance verify")
        elif command == "standalone" and standalone_command == "verify":
            handler = _cmd_not_implemented("standalone verify")
        elif command == "receipt":
            handler = _cmd_receipt
        elif command in _NOT_IMPLEMENTED_COMMANDS:
            handler = _cmd_not_implemented(command)
        else:
            raise Refused("unknown command: %s" % command,
                         "run `%s --help`" % PROG)
        with redirect_stdout(sys.stderr):
            result = handler(namespace)
        result.setdefault("honesty_note", HONESTY_NOTE)
        result.setdefault("autoactivated", False)
        delivered = _emit(result)
        if result.get("ok") is False:
            # An honest typed refusal (NOT_IMPLEMENTED, rejected) is
            # exit 2 per C5 -- completed operations exit 0 only.
            return EXIT_REFUSED
        return delivered
    except ExtractionError as exc:
        _emit({
            "ok": False,
            "command": command_for_error,
            "status": {
                EXIT_REFUSED: "refused",
                EXIT_INVALID_EVIDENCE: "invalid_evidence",
                EXIT_BLOCKED: "BLOCKED_RESOURCE",
                EXIT_EXECUTION_FAILURE: "execution_failure",
            }.get(exc.exit_code, "refused"),
            "error": exc.as_dict(),
            "autoactivated": False,
        })
        return exc.exit_code
    except Exception as exc:  # fail closed, including third-party errors
        _emit({
            "ok": False,
            "command": command_for_error,
            "status": "execution_failure",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "autoactivated": False,
        })
        return EXIT_EXECUTION_FAILURE


if __name__ == "__main__":
    sys.exit(main())