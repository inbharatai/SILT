"""JSON-only public CLI: python -m asea.capability_build (``silt-capability``).

Contract (mirrors ``silt-compile``):

  * stdout is ALWAYS exactly one JSON object; diagnostics go to stderr.
  * exit 0 = completed; 2 = rejected/typed failure (incl. BLOCKED_RESOURCE
    with its requirement+remedy); 3 = unexpected runtime error; 4 = stdout
    delivery failed after a side effect (never auto-retried).
  * Nothing auto-activates, nothing silently falls back between evidence
    classes, and a remote teacher connector runs only with explicit
    per-run consent (``--allow-remote``).

Commands wired in this build: ``spec validate``, ``teacher baseline``,
``trace``, ``footprint``, ``intervene``, ``evaluate``, ``receipt``,
``receipt-verify``. Commands for the later build phases (``student
baseline``, ``distill``, ``reduce``, ``search``, ``certify``) exist and
report ``rejected`` with an explicit not-yet-implemented reason -- a
missing surface is stated, never faked.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import (
    BLOCKED_RESOURCE,
    HONESTY_NOTE,
    TRACE_CLASS_BEHAVIOURAL,
    TRACE_CLASS_INTERNAL,
)
from .errors import (
    BlockedResource,
    CapabilityBuildError,
    RemoteConsentRequired,
)
from .schema import CapabilityBuildReceipt, ReceiptStatus
from .spec import load_spec

PROG = "silt-capability"

#: Commands whose backing modules land in later build phases. They are
#: registered so the surface is discoverable, and refuse honestly.
_PENDING_COMMANDS = {
    "reduce": "Phase 8: structural reduction candidate production (needs an adapter on loaded weights)",
    "search": "Phase 8: minimum-capability search loop (needs a candidate generator + host-oracle evaluator)",
    "certify": "Phase 8: SiltSpring per-skill state certification (full/int8/int4/int2)",
}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CapabilityBuildError("invalid argument: %s" % message)


def _emit(payload: Dict[str, Any]) -> int:
    try:
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
    except BrokenPipeError:
        return 4
    return 0


def _load_cases(path: str) -> List[Dict[str, Any]]:
    resolved = Path(path)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise CapabilityBuildError("cases file is not valid JSON: %s" % exc)
    if not isinstance(raw, list) or not raw:
        raise CapabilityBuildError("cases file must be a non-empty JSON list")
    for case in raw:
        if not isinstance(case, dict):
            raise CapabilityBuildError("every case must be a JSON object")
        for field in ("sample_id", "group", "prompt"):
            if field not in case:
                raise CapabilityBuildError("case missing required field %r" % field)
        if case["group"] not in ("target", "control"):
            raise CapabilityBuildError("case group must be target/control")
    return raw


def _receipt(
    command: str,
    status: ReceiptStatus,
    spec: Optional[Dict[str, Any]],
    evidence_class: str,
    *,
    workspace: Optional[Path] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> CapabilityBuildReceipt:
    limitations = [
        "Implementation success is not model-quality success.",
        "Failed experiments remain recorded and visible.",
    ]
    if evidence_class == TRACE_CLASS_BEHAVIOURAL:
        limitations.append(
            "Evidence is behavioural (remote/API): no internal router or "
            "expert claims are made."
        )
    receipt = CapabilityBuildReceipt(
        command=command,
        status=status,
        capability_id=(spec or {}).get("capability_id", "unknown"),
        spec_sha256=(spec or {}).get("spec_sha256", "0" * 64),
        evidence_class=evidence_class,
        teacher=(spec or {}).get("teacher", {}),
        error=error,
        limitations=limitations,
    )
    if extra:
        for key, value in extra.items():
            object.__setattr__(receipt, key, value)
    if workspace is not None:
        from .receipt import sign_receipt
        from .store import CapabilityStore

        signed = sign_receipt(workspace, receipt)
        store = CapabilityStore(workspace)
        name = "%s-%d" % (command.replace(" ", "_"), len(store.list("receipts")) + 1)
        store.put("receipts", name, signed)
    return receipt


def _cmd_spec_validate(args) -> Dict[str, Any]:
    loaded = load_spec(args.spec)
    spec = loaded["spec"]
    return {
        "ok": True,
        "command": "spec validate",
        "status": "completed",
        "capability_id": spec.capability_id,
        "spec_sha256": loaded["spec_sha256"],
        "spec_fingerprint": loaded["spec_fingerprint"],
        "remote_connector_selected": spec.remote_connector_selected,
        "thresholds_are_operator_configuration": True,
    }


def _behavioural_teacher(args, spec):
    from .teacher import BehaviouralOllamaTeacher

    return BehaviouralOllamaTeacher(
        spec,
        args.model or spec.teacher.model,
        allow_remote=args.allow_remote,
        host=args.host,
        max_new_tokens=args.max_new_tokens,
    )


def _cmd_teacher_baseline(args) -> Dict[str, Any]:
    loaded = load_spec(args.spec)
    spec = loaded["spec"]
    if spec.teacher.access != TRACE_CLASS_BEHAVIOURAL:
        raise BlockedResource(
            requirement=(
                "spec pins evidence class %s; the behavioural Ollama "
                "teacher cannot serve it" % spec.teacher.access
            ),
            remedy="run the isolated worker path for internal_open_weight "
            "evidence, or change the spec teacher pin explicitly",
        )
    if not (args.allow_remote or spec.remote_connector_selected):
        raise RemoteConsentRequired(
            "remote teacher requested without per-run consent; pass "
            "--allow-remote (consent never carries over between runs)"
        )
    teacher = _behavioural_teacher(args, spec)
    health = teacher.health()
    if not health["model_present"]:
        raise BlockedResource(
            requirement="Ollama model %r is not present at %s" % (teacher.model, teacher.host),
            remedy=health["hint"] or "pull the model first",
        )
    cases = _load_cases(args.cases)
    from .teacher import ask_teacher

    records = [ask_teacher(teacher, case) for case in cases]
    workspace = Path(args.workspace)
    from .store import CapabilityStore

    store = CapabilityStore(workspace)
    baseline = {
        "schema": "silt.capability_baseline.v1",
        "capability_id": spec.capability_id,
        "teacher": {"provider": spec.teacher.provider, "model": teacher.model,
                    "revision": spec.teacher.revision},
        "cases": [r["behavioural"].model_dump(mode="json") for r in records],
        "outcome_note": "outcomes are UNJUDGED here; judging is host-owned "
                        "(functional oracle), never the teacher grading itself",
    }
    name = "%s-baseline" % spec.capability_id
    written = store.put("baselines", name, baseline)
    _receipt(
        "teacher baseline",
        "completed",
        {"capability_id": spec.capability_id, "spec_sha256": loaded["spec_sha256"], "teacher": baseline["teacher"]},
        TRACE_CLASS_BEHAVIOURAL,
        workspace=workspace,
        extra={"data_split_hashes": {"cases": args.cases}},
    )
    return {
        "ok": True,
        "command": "teacher baseline",
        "status": "completed",
        "teacher": baseline["teacher"],
        "cases_asked": len(records),
        "artifact": written,
    }


def _cmd_trace(args) -> Dict[str, Any]:
    loaded = load_spec(args.spec)
    spec = loaded["spec"]
    if args.mode == "internal":
        return _trace_internal(args, spec, loaded)
    if not (args.allow_remote or spec.remote_connector_selected):
        raise RemoteConsentRequired(
            "remote behavioural tracing requires --allow-remote for THIS run"
        )
    teacher = _behavioural_teacher(args, spec)
    health = teacher.health()
    if not health["model_present"]:
        raise BlockedResource(
            requirement="Ollama model %r is not present" % teacher.model,
            remedy=health["hint"] or "pull the model first",
        )
    cases = _load_cases(args.cases)
    from .teacher import ask_teacher
    from .trace import make_behavioural_trace

    workspace = Path(args.workspace)
    from .store import CapabilityStore

    store = CapabilityStore(workspace)
    written = []
    for case in cases:
        record = ask_teacher(teacher, case)
        trace = make_behavioural_trace(
            capability_id=spec.capability_id,
            sample_id=case["sample_id"],
            model_revision=spec.teacher.revision,
            prompt=record["behavioural"].prompt,
            group=case["group"],
            outcome=record["outcome"],
            behavioural=record["behavioural"],
        )
        payload = trace.model_dump(mode="json", by_alias=True)
        name = "%s-%s" % (spec.capability_id, case["sample_id"])
        written.append(store.put("traces", name, payload))
    _receipt(
        "trace",
        "completed",
        {"capability_id": spec.capability_id, "spec_sha256": loaded["spec_sha256"]},
        TRACE_CLASS_BEHAVIOURAL,
        workspace=workspace,
    )
    return {
        "ok": True,
        "command": "trace",
        "status": "completed",
        "evidence_class": TRACE_CLASS_BEHAVIOURAL,
        "traces_written": len(written),
        "artifacts": written,
    }


def _trace_internal(args, spec, loaded) -> Dict[str, Any]:
    """Mode-2 path: instrumented routing traces via the isolated worker.
    On a host that cannot hold the teacher the worker client raises
    BlockedResource with the exact requirement+remedy -- never a behavioural
    substitute (no silent fallback between evidence classes)."""
    import os

    checkpoint = args.checkpoint or os.environ.get("GLM_CHECKPOINT")
    if not checkpoint:
        raise BlockedResource(
            requirement="internal tracing needs the open-weight checkpoint "
                        "(--checkpoint or GLM_CHECKPOINT env var)",
            remedy="mount the BF16 checkpoint (zai-org/GLM-5.3-Flash-BF16) "
                   "and run on a box the worker's preflight admits",
        )
    from .schema import InternalRecord, TraceOutcome
    from .store import CapabilityStore
    from .trace import make_internal_trace
    from .worker import GlmWorkerClient

    cases = _load_cases(args.cases)
    client = GlmWorkerClient(
        repo_root=Path(__file__).resolve().parents[3],
        checkpoint=checkpoint,
    )
    hello = client.hello()
    result = client.request("routing", {
        "prompts": [
            {
                "sample_id": case["sample_id"],
                "group": case["group"],
                "prompt": case["prompt"],
            }
            for case in cases
        ],
        "max_length": 2048,
    })
    workspace = Path(args.workspace)
    store = CapabilityStore(workspace)
    written = []
    for entry in result.get("per_prompt", []):
        layers: Dict[str, Any] = {}
        for layer_id, row in (entry.get("routing") or {}).items():
            experts = {}
            for index, mass in enumerate(row.get("sigmoid_mass") or []):
                experts[str(index)] = {
                    "usage_mass": float(mass),
                    "topk_count": int((row.get("topk_count") or [0] * (index + 1))[index]),
                }
            layers[str(layer_id)] = {
                "usage_mass": float(sum(row.get("sigmoid_mass") or [])),
                "tokens": int(row.get("tokens", 0)),
                "experts": experts,
            }
        case = next(
            (c for c in cases if c["sample_id"] == entry.get("sample_id")), None
        )
        if case is None:
            raise CapabilityBuildError(
                "worker reported a sample_id we never sent: %r"
                % entry.get("sample_id")
            )
        trace = make_internal_trace(
            capability_id=spec.capability_id,
            sample_id=case["sample_id"],
            model_revision=spec.teacher.revision,
            prompt=case["prompt"],
            group=case["group"],
            outcome=TraceOutcome(success=None),
            internal=InternalRecord(layers=layers),
        )
        name = "%s-%s" % (spec.capability_id, case["sample_id"])
        written.append(store.put(
            "traces", name, trace.model_dump(mode="json", by_alias=True)))
    _receipt(
        "trace",
        "completed",
        {"capability_id": spec.capability_id, "spec_sha256": loaded["spec_sha256"]},
        TRACE_CLASS_INTERNAL,
        workspace=workspace,
        extra={"teacher": {"provider": spec.teacher.provider, "model": spec.teacher.model,
                           "revision": spec.teacher.revision}},
    )
    return {
        "ok": True,
        "command": "trace",
        "status": "completed",
        "evidence_class": TRACE_CLASS_INTERNAL,
        "traces_written": len(written),
        "worker": hello,
        "artifacts": written,
    }


def _cmd_footprint(args) -> Dict[str, Any]:
    loaded = load_spec(args.spec)
    spec = loaded["spec"]
    workspace = Path(args.workspace)
    from .footprint import build_footprint, compute_enrichment
    from .store import CapabilityStore
    from .trace import load_trace, traces_same_class

    store = CapabilityStore(workspace)
    names = store.list("traces")
    if not names:
        raise BlockedResource(
            requirement="no traces recorded in this workspace",
            remedy="run `silt-capability trace` first (or the worker path "
                   "for internal evidence)",
        )
    traces = [load_trace(store.root / "traces" / ("%s.json" % n)) for n in names]
    trace_class = traces_same_class(traces)
    target_cases = sum(1 for t in traces if t.group == "target")
    control_cases = sum(1 for t in traces if t.group == "control")
    if trace_class == TRACE_CLASS_INTERNAL:
        enrichment = compute_enrichment(traces)
        layers = {}
        experts = {}
        for component, entry in enrichment["components"].items():
            if component.startswith("layer:"):
                layers[component] = entry
            else:
                experts[component] = entry
    else:
        # Behavioural evidence: no internal component claims AT ALL.
        layers, experts = {}, {}
    footprint = build_footprint(
        capability=spec.capability_id,
        teacher="%s/%s" % (spec.teacher.provider, spec.teacher.model),
        teacher_revision=spec.teacher.revision,
        spec_fingerprint=loaded["spec_fingerprint"],
        evidence_class=trace_class or TRACE_CLASS_BEHAVIOURAL,
        target_cases=target_cases,
        control_cases=control_cases,
        layers=layers,
        experts=experts,
    )
    payload = footprint.model_dump(mode="json", by_alias=True)
    name = "%s-%s" % (spec.capability_id, trace_class)
    written = store.put("footprints", name, payload)
    _receipt(
        "footprint",
        "completed",
        {"capability_id": spec.capability_id, "spec_sha256": loaded["spec_sha256"]},
        trace_class or TRACE_CLASS_BEHAVIOURAL,
        workspace=workspace,
    )
    return {
        "ok": True,
        "command": "footprint",
        "status": "completed",
        "evidence_class": trace_class,
        "target_cases": target_cases,
        "control_cases": control_cases,
        "correlation_only": True,
        "artifact": written,
    }


def _cmd_student_baseline(args) -> Dict[str, Any]:
    """Local-student baseline: run every case through a LOCAL Ollama model
    and judge each output against the case's exact ``expected`` output when
    present (deterministic textual verdict -- a MECHANISM, not the host
    oracle). Cases without ``expected`` are recorded UNJUDGED.

    Both the availability check and every inference go through the REAL
    OllamaConnector transport, which refuses redirects at the transport
    level -- a 3xx from the daemon is an error naming the target, never a
    silent second hop to an unvalidated host."""
    loaded = load_spec(args.spec)
    spec = loaded["spec"]
    from .student import local_ollama_student, measure_student_baseline

    student = local_ollama_student(
        args.model or "qwen2.5-coder:0.5b", host=args.host,
        max_new_tokens=args.max_new_tokens,
    )
    cases = _load_cases(args.cases)
    from asea.modules.real.ollama import OllamaConnector, OllamaConnectionError

    connector = OllamaConnector(
        student["model"], [], host=student["host"],
        max_new_tokens=student["max_new_tokens"],
    )
    try:
        health = connector.health()
    except OllamaConnectionError as exc:
        raise BlockedResource(
            requirement="local Ollama daemon unreachable at %s (%s)"
            % (student["host"], exc),
            remedy="start it with `ollama serve`; student runs are local "
                   "only (no remote student path exists)",
        )
    if not health["model_present"]:
        raise BlockedResource(
            requirement="student model %r not present locally" % student["model"],
            remedy="ollama pull %s" % student["model"],
        )

    def infer(_student, case):
        return connector._chat([{"role": "user", "content": case["prompt"]}])

    def judge(case, output):
        expected = case.get("expected")
        if expected is None:
            return False  # UNJUDGED cases never count as passes
        return output.strip() == expected.strip()

    summary = measure_student_baseline(student, cases, infer=infer, judge=judge)
    unjudged = sum(
        1 for case in cases if case.get("expected") is None
    )
    workspace = Path(args.workspace)
    from .store import CapabilityStore

    store = CapabilityStore(workspace)
    baseline = dict(summary)
    baseline["student"] = student
    baseline["capability_id"] = spec.capability_id
    baseline["unjudged_cases"] = unjudged
    baseline["judge"] = "exact_textual_match_MECHANISM_not_host_oracle"
    written = store.put(
        "baselines", "%s-student" % spec.capability_id, baseline
    )
    return {
        "ok": True,
        "command": "student-baseline",
        "status": "completed",
        "student": student["model"],
        "groups": summary["groups"],
        "unjudged_cases": unjudged,
        "artifact": written,
    }


def _cmd_distill(args) -> Dict[str, Any]:
    """Build sequence-level KD pairs from stored JUDGED traces, bound to an
    APPROVED five-split dataset, then emit the DeepApply hand-off
    descriptor. No training happens here; DeepApply + Gate 2 own training
    and admission.

    Binding rules (enforced, not advisory):

      * ``--dataset`` is required: pairs may only come from traces whose
        sample ids appear in that dataset's TRAINING split. A trace from
        outside the approved training split is refused -- unverifiable is
        not safe.
      * The dataset manifest must validate (hashes, counts, disjointness)
        and must carry the SAME capability id, teacher pin and spec
        fingerprint as ``--spec`` -- identity is verified, never assumed
        from a path.
      * The protected splits (development/heldout/final/controls) supply
        protected sample ids, content hashes, PROMPT hashes, PROMPT
        token-shapes and families to the pair builder; any collision
        poisons the whole build.
      * Every trace's capability id and teacher revision must match the
        spec -- traces from another teacher are refused.
      * A behavioural trace's PROMPT must match, byte for byte, the
        approved training row bearing that sample id: a trace under a
        training id whose prompt is not the frozen training prompt is
        refused. This closes the "training id, different content" hole
        (the sample id is an identity claim, not a free pass).
    """
    import hashlib
    import json as _json

    from .distillation import _token_shape

    loaded = load_spec(args.spec)
    spec = loaded["spec"]
    workspace = Path(args.workspace)
    from .distillation import build_sequence_pairs, hand_to_deepapply
    from .dataset import SPLITS, validate_dataset
    from .store import CapabilityStore
    from .trace import load_trace

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        raise BlockedResource(
            requirement="dataset directory %s does not exist; KD pairs may "
                        "only be built from traces bound to an approved "
                        "five-split dataset" % dataset_dir,
            remedy="build the dataset first: silt-capability dataset build "
                   "--spec spec.json --cases cases.jsonl --out "
                   "data/capability_v1",
        )
    validated = validate_dataset(dataset_dir)
    if validated["capability_id"] != spec.capability_id:
        raise CapabilityBuildError(
            "dataset identity mismatch: manifest was built for capability "
            "%r but the spec names %r" % (
                validated["capability_id"], spec.capability_id
            )
        )
    if validated["spec_fingerprint"] != loaded["spec_fingerprint"]:
        raise CapabilityBuildError(
            "dataset identity mismatch: manifest spec fingerprint %r does "
            "not match this spec (%r); the dataset was frozen for a "
            "different contract" % (
                validated["spec_fingerprint"], loaded["spec_fingerprint"]
            )
        )
    dataset_teacher = validated["teacher"]
    if (
        dataset_teacher.get("model") != spec.teacher.model
        or dataset_teacher.get("revision") != spec.teacher.revision
    ):
        raise CapabilityBuildError(
            "dataset teacher %r does not match the spec teacher %r@%r"
            % (
                dataset_teacher,
                spec.teacher.model,
                spec.teacher.revision,
            )
        )

    # Training split: the ONLY samples teaching material may come from.
    training_rows = []
    with (dataset_dir / "training.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                training_rows.append(_json.loads(line))
    training_ids = {row["sample_id"] for row in training_rows}
    training_prompts = {row["sample_id"]: row["prompt"] for row in training_rows}
    sample_families = {row["sample_id"]: row["family_id"] for row in training_rows}

    # Protected splits: every non-training split contributes ids, content
    # hashes, prompt hashes, PROMPT TOKEN SHAPES and families that must
    # never appear in pairs.
    protected_ids: set = set()
    protected_hashes: set = set()
    protected_prompts: set = set()
    protected_prompt_shapes: set = set()
    protected_families: set = set()
    for split in SPLITS:
        if split == "training":
            continue
        with (dataset_dir / ("%s.jsonl" % split)).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _json.loads(line)
                protected_ids.add(row["sample_id"])
                protected_hashes.add(row.get("content_sha256"))
                protected_prompts.add(
                    hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
                )
                protected_prompt_shapes.add(_token_shape(row["prompt"]))
                protected_families.add(row["family_id"])

    store = CapabilityStore(workspace)
    names = store.list("traces")
    if not names:
        raise BlockedResource(
            requirement="no traces recorded in this workspace",
            remedy="run `silt-capability trace` first",
        )
    traces = [
        load_trace(store.root / "traces" / ("%s.json" % n)) for n in names
    ]
    for trace in traces:
        if trace.capability_id != spec.capability_id:
            raise CapabilityBuildError(
                "trace %r belongs to capability %r, not %r; traces from "
                "another capability are never teaching material"
                % (trace.sample_id, trace.capability_id, spec.capability_id)
            )
        if trace.model_revision != spec.teacher.revision:
            raise CapabilityBuildError(
                "trace %r was produced by teacher revision %r, but the spec "
                "pins %r; traces from another teacher revision are never "
                "teaching material"
                % (trace.sample_id, trace.model_revision, spec.teacher.revision)
            )
        if trace.behavioural is not None and trace.sample_id not in training_ids:
            raise CapabilityBuildError(
                "trace %r is not part of the dataset's approved training "
                "split; only training-split traces may become pairs"
                % trace.sample_id
            )
        if trace.behavioural is not None:
            approved_prompt = training_prompts.get(trace.sample_id)
            if approved_prompt is None or trace.behavioural.prompt != approved_prompt:
                raise CapabilityBuildError(
                    "trace %r claims training sample %r but its prompt does "
                    "not match the approved training row byte for byte; a "
                    "sample id is an identity claim, not a free pass -- the "
                    "frozen training prompt is the only teaching material"
                    % (trace.sample_id, trace.sample_id)
                )
    pairs = build_sequence_pairs(
        traces,
        protected_sample_ids=protected_ids,
        protected_content_hashes={h for h in protected_hashes if h},
        protected_families=protected_families,
        protected_prompts=protected_prompts,
        protected_prompt_shapes=protected_prompt_shapes,
        sample_families=sample_families,
    )
    handoff = hand_to_deepapply(pairs) if pairs["positives"] else None
    payload = dict(pairs)
    payload["capability_id"] = spec.capability_id
    payload["dataset"] = {
        "dir": str(dataset_dir),
        "spec_fingerprint": validated["spec_fingerprint"],
        "training_samples": len(training_ids),
        "protected_splits_enforced": [
            s for s in SPLITS if s != "training"
        ],
    }
    payload["deepapply_handoff"] = handoff
    written = store.put(
        "candidates", "%s-kd-pairs" % spec.capability_id, payload
    )
    return {
        "ok": True,
        "command": "distill",
        "status": "completed",
        "positives": len(pairs["positives"]),
        "negatives": len(pairs["negatives"]),
        "unjudged_excluded": pairs["unjudged_excluded"],
        "dataset": payload["dataset"],
        "deepapply_handoff": handoff,
        "artifact": written,
    }


def _cmd_intervene(args) -> Dict[str, Any]:
    loaded = load_spec(args.spec)
    spec = loaded["spec"]
    if spec.teacher.access != TRACE_CLASS_INTERNAL:
        raise CapabilityBuildError(
            "interventions need internal_open_weight evidence; teacher %r "
            "exposes no internals to intervene on" % spec.teacher.model
        )
    import os

    checkpoint = getattr(args, "checkpoint", None) or os.environ.get("GLM_CHECKPOINT")
    if not checkpoint:
        raise BlockedResource(
            requirement=(
                "no GLM checkpoint given (--checkpoint or GLM_CHECKPOINT "
                "env var); causal intervention needs the open weights"
            ),
            remedy="mount the BF16 checkpoint (zai-org/GLM-5.3-Flash-BF16) "
                   "and pass --checkpoint /path/to/GLM-5.3-Flash-BF16",
        )
    from .worker import GlmWorkerClient

    repo_root = Path(__file__).resolve().parents[3]
    client = GlmWorkerClient(
        repo_root=repo_root,
        checkpoint=checkpoint,
        use_docker=None if getattr(args, "use_docker", None) is None else True,
    )
    # The handshake runs the worker's hardware preflight; on a host that
    # cannot hold the teacher this raises BlockedResource with the exact
    # requirement + remedy -- the honest outcome, never a local weaker
    # path.
    hello = client.hello()
    raise CapabilityBuildError(
        "worker admitted the checkpoint (preflight: %r) but the full "
        "mask-measure-restore intervention loop requires the generation+"
        "judging stage, which is not wired in this build; refusing to "
        "report an intervention that was not measured" % hello
    )


def _cmd_evaluate(args) -> Dict[str, Any]:
    from .evaluation import evaluate_code_cases

    source = Path(args.source).read_text(encoding="utf-8")
    cases = _load_cases(args.cases)
    result = evaluate_code_cases(source, cases)
    return {
        "ok": True,
        "command": "evaluate",
        "status": "BLOCKED_RESOURCE" if result["blocked"] else "completed",
        "platform": platform.system(),
        "result": result,
    }


def _cmd_receipt(args) -> Dict[str, Any]:
    workspace = Path(args.workspace)
    from .receipt import sign_receipt
    from .store import CapabilityStore

    raw = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CapabilityBuildError("receipt must be a JSON object")
    if raw.get("schema") != "silt.capability_receipt.v1":
        raise CapabilityBuildError("input is not a capability receipt")
    receipt = CapabilityBuildReceipt.model_validate(
        {k: v for k, v in raw.items() if k not in ("signature", "signature_alg", "key_fingerprint")}
    )
    signed = sign_receipt(workspace, receipt)
    store = CapabilityStore(workspace)
    name = Path(args.receipt).stem
    written = store.put("receipts", name, signed)
    return {"ok": True, "command": "receipt", "status": "completed", "artifact": written}


def _cmd_receipt_verify(args) -> Dict[str, Any]:
    from .receipt import verify_receipt

    workspace = Path(args.workspace)
    raw = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    verified = verify_receipt(workspace, raw)
    return {"ok": True, "command": "receipt-verify", "status": "completed", "verified": verified}


def _cmd_dataset_build(args) -> Dict[str, Any]:
    """Build the fresh five-split dataset (data/capability_v1 namespace).

    The frozen September sets are quarantined as inputs; the build writes
    a manifest + selection lock frozen BEFORE any model sees a case. The
    manifest binds capability + teacher + spec-fingerprint identity: a
    dataset that cannot prove what it was built for may never back
    training pairs."""
    loaded = load_spec(args.spec)
    spec = loaded["spec"]
    from .dataset import build_dataset, read_cases

    cases = read_cases(Path(args.cases))
    manifest = build_dataset(cases, output_dir=Path(args.out), spec=spec)
    return {
        "ok": True,
        "command": "dataset build",
        "status": "completed",
        "capability_id": manifest["capability_id"],
        "teacher": manifest["teacher"],
        "counts": manifest["counts"],
        "output": str(args.out),
        "frozen_before_model_generation": True,
        "note": manifest["note"],
    }


def _cmd_dataset_validate(args) -> Dict[str, Any]:
    from .dataset import validate_dataset

    result = validate_dataset(Path(args.dir))
    result["command"] = "dataset validate"
    result["status"] = "completed"
    return result


def _cmd_pending(command: str) -> Any:
    def handler(args) -> Dict[str, Any]:
        return {
            "ok": False,
            "command": command,
            "status": "rejected",
            "reason": (
                "not implemented in this build; scheduled: %s. This surface "
                "refuses honestly rather than emitting a placeholder result."
                % _PENDING_COMMANDS[command]
            ),
        }

    return handler


def parser() -> Parser:
    p = Parser(prog=PROG, description="SILT capability-build: teacher footprinting, causal intervention, minimum-capability search. Nothing auto-activates.")
    commands = p.add_subparsers(dest="command", required=True)

    spec_cmd = commands.add_parser("spec", help="Capability spec operations")
    spec_sub = spec_cmd.add_subparsers(dest="spec_command", required=True)
    validate = spec_sub.add_parser("validate")
    validate.add_argument("--spec", required=True)

    dataset_cmd = commands.add_parser("dataset", help="Build/validate the fresh capability case sets")
    dataset_sub = dataset_cmd.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_sub.add_parser("build")
    dataset_build.add_argument("--spec", required=True)
    dataset_build.add_argument("--cases", required=True)
    dataset_build.add_argument("--out", required=True)
    dataset_validate = dataset_sub.add_parser("validate")
    dataset_validate.add_argument("--dir", required=True)

    teacher = commands.add_parser("teacher-baseline", help="Measure teacher outcomes on a case set (behavioural)")
    trace_cmd = commands.add_parser("trace", help="Collect capability traces (behavioural or internal)")
    footprint_cmd = commands.add_parser("footprint", help="Aggregate traces into a CapabilityFootprint")
    intervene_cmd = commands.add_parser("intervene", help="Causal mask/measure/restore on one teacher component")
    evaluate_cmd = commands.add_parser("evaluate", help="Host-oracle functional evaluation (Linux sandbox)")
    student_cmd = commands.add_parser("student-baseline", help="Measure a LOCAL student model on the same cases as the teacher")
    distill_cmd = commands.add_parser("distill", help="Build sequence-level KD pairs from judged traces and hand off to DeepApply")
    reduce_cmd = commands.add_parser("reduce", help="[later phase] Structural reduction candidate")
    search_cmd = commands.add_parser("search", help="[later phase] Minimum-capability search")
    certify_cmd = commands.add_parser("certify", help="[later phase] SiltSpring state certification")
    student_cmd.add_argument("--cases", required=True)
    student_cmd.add_argument("--model", default="qwen2.5-coder:0.5b")
    student_cmd.add_argument("--host", default="http://localhost:11434")
    student_cmd.add_argument("--max-new-tokens", type=int, default=512)
    receipt_cmd = commands.add_parser("receipt", help="Sign and store a capability-build receipt")
    verify_cmd = commands.add_parser("receipt-verify", help="Verify a signed receipt against the local key")

    for command in (teacher, trace_cmd, footprint_cmd, intervene_cmd,
                    student_cmd, distill_cmd, reduce_cmd, search_cmd, certify_cmd):
        command.add_argument("--spec", required=True)
        command.add_argument("--workspace", default=".capability-build")
    for command in (teacher, trace_cmd):
        command.add_argument("--cases", required=True)
        command.add_argument("--model")
        command.add_argument("--host", default="http://localhost:11434")
        command.add_argument("--allow-remote", action="store_true",
                              help="EXPLICIT per-run consent for the remote/cloud teacher; never carries over")
        command.add_argument("--max-new-tokens", type=int, default=1024)
    trace_cmd.add_argument("--mode", choices=["behavioural", "internal"], default="behavioural")
    trace_cmd.add_argument("--checkpoint", help="Open-weight checkpoint for internal tracing (or GLM_CHECKPOINT)")
    distill_cmd.add_argument("--dataset", required=True,
                             help="Approved five-split dataset directory; pairs may come ONLY from its training split")
    intervene_cmd.add_argument("--component")
    intervene_cmd.add_argument("--seed", type=int, default=0)
    intervene_cmd.add_argument("--checkpoint", help="Open-weight checkpoint for interventions (or GLM_CHECKPOINT)")
    evaluate_cmd.add_argument("--cases", required=True)
    evaluate_cmd.add_argument("--source", required=True)
    evaluate_cmd.add_argument("--spec")
    evaluate_cmd.add_argument("--workspace", default=".capability-build")
    receipt_cmd.add_argument("--receipt", required=True)
    receipt_cmd.add_argument("--workspace", default=".capability-build")
    verify_cmd.add_argument("--receipt", required=True)
    verify_cmd.add_argument("--workspace", default=".capability-build")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = vars(parser().parse_args(argv))
        command = args.pop("command")
        spec_command = args.pop("spec_command", None)
        dataset_command = args.pop("dataset_command", None)
        namespace = argparse.Namespace(**args)
        handlers = {
            "spec validate": _cmd_spec_validate,
            "dataset build": _cmd_dataset_build,
            "dataset validate": _cmd_dataset_validate,
            "teacher-baseline": _cmd_teacher_baseline,
            "trace": _cmd_trace,
            "footprint": _cmd_footprint,
            "intervene": _cmd_intervene,
            "evaluate": _cmd_evaluate,
            "student-baseline": _cmd_student_baseline,
            "distill": _cmd_distill,
            "receipt": _cmd_receipt,
            "receipt-verify": _cmd_receipt_verify,
        }
        for pending in _PENDING_COMMANDS:
            handlers[pending] = _cmd_pending(pending)
        key = ("%s %s" % (command, spec_command)) if command == "spec" else (
            ("%s %s" % (command, dataset_command)) if command == "dataset" else command
        )
        handler = handlers[key]
        with redirect_stdout(sys.stderr):
            result = handler(namespace)
        result.setdefault("honesty_note", HONESTY_NOTE)
        result.setdefault("autoactivated", False)
        return _emit(result)
    except BlockedResource as exc:
        _emit({
            "ok": False,
            "command": (argv[0] if argv else "unknown"),
            "status": BLOCKED_RESOURCE,
            "error": str(exc),
            "requirement": exc.requirement,
            "remedy": exc.remedy,
            "autoactivated": False,
        })
        return 2
    except (RemoteConsentRequired, CapabilityBuildError) as exc:
        _emit({
            "ok": False,
            "command": (argv[0] if argv else "unknown"),
            "status": "rejected",
            "error": str(exc),
            "autoactivated": False,
        })
        return 2
    except Exception as exc:  # Fail closed, including third-party errors.
        _emit({
            "ok": False,
            "command": (argv[0] if argv else "unknown"),
            "status": "rejected",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "autoactivated": False,
        })
        return 3


if __name__ == "__main__":
    sys.exit(main())