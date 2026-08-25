"""Level 4/5 export.

This system does not train models. It cannot: there is no trainer here, and
running one inside a request/response adapter would be the wrong architecture
even if there were. What it *can* do honestly is produce the two artefacts a
human needs in order to run training themselves:

  * a validated JSONL dataset built only from APPROVED packets, and
  * a job specification recording base model, adapter hyper-parameters,
    provenance summary and the evaluation gate the resulting adapter must pass
    before anyone ships it.

The gate refuses to promote L4/L5 packets to a live receiver (see
promotion/gate.py, ``applicable_learning_level``). Export is the only path.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.protocol import LearningLevel, PromotionStatus, SkillPacket
from ..promotion.gate import PromotionPolicy

# A bundle/dataset name is interpolated into filesystem paths and zip arcnames,
# so it must be path-safe (adversarial audit 2026-08-13: a name containing '..'
# wrote outside out_dir and produced a zip-slip entry). Path separators and
# parent traversal are rejected.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name):
        raise ValueError(
            "invalid bundle/dataset name {!r}: must match ^[A-Za-z0-9._-]+$ "
            "(no path separators or '..')".format(name)
        )
    # The regex above allows '.' and '..' as whole names (every char is in the
    # class), which resolve to this/parent directory when interpolated into a
    # path -- exactly the traversal the regex comment claims to block. Reject
    # them explicitly so the comment is honest.
    if name in (".", ".."):
        raise ValueError(
            "invalid bundle/dataset name {!r}: '.' and '..' are directory "
            "traversals, not names".format(name)
        )
    return name


def _eval_gate_from_policy(policy: PromotionPolicy) -> Dict[str, Any]:
    """Serialise the gate's ACTUAL thresholds into the job spec's eval_gate.

    Adversarial audit (2026-08-13): build_job_spec used to hardcode an invented
    eval_gate (improvement 0.02 vs the gate's 0.01, human review unconditional
    vs high-risk-only, plus a non-existent native-speaker check) that diverged
    from the gate the source packet actually passed. A consumer re-benchmarking
    against that invented bar would reject adapters the source gate accepted.
    This derives the bar from the real policy instead.
    """
    gate = {
        "held_out_improvement_min": policy.min_improvement,
        "min_evaluator_score": policy.min_evaluator_score,
        "min_safety_score": policy.min_safety_score,
        "max_case_regression_ratio": policy.max_case_regression_ratio,
        "max_synthetic_depth": policy.max_synthetic_depth,
        "strict_no_mock": policy.strict_no_mock,
        "no_self_transfer": True,  # hard gate check, not a policy knob
        # The gate requires a human approver for HIGH-risk domains only, not
        # unconditionally -- so the bar says the same.
        "require_human_review_for_high_risk_domains": True,
        "source": "derived_from_gate_policy",
    }
    # Control-movement bound (audit 2026-08-17): BOTH gates enforce a symmetric
    # no_control_movement hard check (a non-target control suite may not MOVE in
    # either direction, not merely drop). The bound lives in different places on
    # the two gates, so the disclosure reflects where the THRESHOLD KNOB actually
    # sits -- never implying a knob that does not exist on the policy serialised:
    #   * Gate 2 (DeepApplyPolicy): the knob is a policy field, disclosed here.
    #   * Gate 1 (base PromotionPolicy): the knob is on the Evaluator
    #     (Evaluator.max_control_movement), NOT on the policy -- so the base
    #     policy has no max_control_movement to disclose, and we omit it. The
    #     CHECK still fires on Gate 1; only the policy-level threshold is absent.
    if hasattr(policy, "max_control_movement"):
        gate["max_control_movement"] = policy.max_control_movement
    return gate


# Fallback used only when no policy is supplied (e.g. bare CLI --no-bundle
# without a workspace gate). Labelled so no one mistakes it for the real bar.
_ILLUSTRATIVE_EVAL_GATE = {
    "held_out_improvement_min": 0.01,  # matches PromotionPolicy default
    "min_evaluator_score": 0.6,
    "min_safety_score": 0.7,
    "max_case_regression_ratio": 1.0,
    "max_synthetic_depth": 2,
    "strict_no_mock": True,
    "no_self_transfer": True,
    "require_human_review_for_high_risk_domains": True,
    "source": "illustrative_fallback_not_from_gate",
}


_BUNDLE_README = """\
SILT skill bundle -- the "trained model" download
=================================================

SILT trains no weights. There is no trainer in this repository and there never
will be one inside the adapter. The promoted artefact is an inspectable SKILL
PACKET (JSON: a lexicon, glossary, rule list, or exemplar set) that a receiver
model conditions on at inference time. So "the trained model" is really
<receiver model> + <approved skill packet(s) in this bundle>.

What is in this archive
-----------------------
  approved/<packet_id>.json   the raw promoted skill packet(s) -- the primary
                              artefact; this is what the receiver learns from.
  manifest.json               bundle index: per-packet capability, target,
                              learning level, provenance chain, gate verdict.
  <name>.jsonl                a supervised dataset flattened from the approved
                              packets (one row per entry/rule/example).
  <name>.manifest.json         dataset manifest (row/packet counts, skips).
  <name>.job.json              OPTIONAL -- only present if a base model was
                              supplied. A NOT_EXECUTED training-job spec for an
                              external L4/L5 trainer (LoRA / sequence KD). It is
                              a recipe, not a trained adapter.
  audit.jsonl                  OPTIONAL -- the hash-chained audit trail of the
                              run that produced these packets, if available.

How to use the skill packet (L0-L3, the primary path)
-----------------------------------------------------
At inference time the receiver injects the packet's redacted payload into its
system prompt via render_skills -- exactly the path the gate measured:

    skills = [packet.redacted_for_receiver() for packet in approved]
    answer = receiver.infer_with_skills(capability, prompt, skills)

No training, no weight surgery. The receiver is "conditioned", not retrained.
To consume the packet from a different runtime, parse approved/<id>.json and
emit its `distilled_skill` payload as the prompt prefix your model expects.

How to use the dataset + job spec (L4/L5, opt-in)
-------------------------------------------------
Take <name>.jsonl and <name>.job.json to an external trainer (HuggingFace TRL /
PEFT, an Ollama Modelfile, etc.). The job spec's `eval_gate` is the bar the
resulting adapter must clear before anyone ships it. When the bundle was built
through the Studio/CLI, `eval_gate.source` is "derived_from_gate_policy" -- it
mirrors the thresholds of the actual promotion gate the source packet passed
(improvement floor, evaluator/safety minima, case-regression ratio, synthetic-
depth ceiling, strict-no-mock, and human-review for high-risk domains). When
no policy was supplied it is "illustrative_fallback_not_from_gate" and must NOT
be treated as authoritative. Either way, re-enter the trained adapter as a NEW
receiver module and re-benchmark it through SILT before use.

Honesty
--------
No "% of knowledge transferred" appears anywhere because no such measurement
exists. The only defensible claim is the one the gate made: the receiver's
held-out score rose by a measured amount after conditioning on this packet, with
no uncontrolled regression, on real (non-mock) weights.
"""


def _packet_summary(packet: SkillPacket) -> Dict[str, Any]:
    return {
        "packet_id": packet.packet_id,
        "capability": packet.sender_capability.as_str(),
        "target_module": packet.target_module,
        "packet_type": packet.packet_type.value if packet.packet_type else None,
        "learning_level": int(packet.learning_level),
        "evaluator_score": packet.evaluator_score,
        "provenance_chain": list(packet.provenance.chain),
        "provenance_is_mock": packet.provenance.is_mock,
        "synthetic_depth": packet.provenance.synthetic_depth,
        "gate_verdict": packet.promotion_status.value,
        "human_approved_by": packet.human_approved_by,
        "rollback_token": packet.rollback_token,
        # Per-packet payload hash so a consumer can detect that
        # approved/<id>.json was edited after export (audit 2026-08-13: the
        # bundle previously carried no integrity anchor for the payload).
        "content_hash": packet.content_hash(),
    }


def export_artifact_bundle(
    packets: List[SkillPacket],
    out_dir: Path,
    name: str = "skill_bundle",
    base_model: Optional[str] = None,
    include_mock: bool = False,
    audit_path: Optional[Path] = None,
    policy: Optional[PromotionPolicy] = None,
) -> Path:
    """Bundle the approved skill packet(s) into a single downloadable zip.

    This is the "download the trained model" artefact. SILT trains no weights,
    so the bundle is the approved skill packet(s) a receiver conditions on at
    inference time, plus -- for L4/L5 -- a dataset and a NOT_EXECUTED training
    job spec a human runs themselves.

    Refuses anything not PROMOTED, and by default refuses mock-derived content
    (the same guard as :func:`export_dataset`: a bundle assembled from placeholder
    data looks legitimate once on disk, which is worse than no bundle). Returns
    the path to ``<out_dir>/<name>.zip``.

    ``policy``: when supplied (the production path passes the pipeline's gate
    policy), the job spec's ``eval_gate`` is derived from the gate's ACTUAL
    thresholds. When omitted, an explicitly-labelled illustrative fallback is
    used -- never passed off as the real bar.
    """
    _validate_name(name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted: List[SkillPacket] = []
    skipped: List[Dict[str, str]] = []
    for packet in packets:
        if packet.promotion_status != PromotionStatus.PROMOTED:
            skipped.append({"packet_id": packet.packet_id, "reason": "not promoted"})
            continue
        if packet.provenance.is_mock and not include_mock:
            skipped.append({"packet_id": packet.packet_id, "reason": "mock provenance"})
            continue
        accepted.append(packet)

    # Reuse the existing validated dataset writer (writes <name>.jsonl +
    # <name>.manifest.json into out_dir, applying the same PROMOTED/mock guard).
    dataset_manifest = export_dataset(accepted, out_dir, name, include_mock=include_mock)
    # The bundle has already filtered; export_dataset therefore sees nothing to
    # skip and would report skipped:[]. Surface the bundle-level skips on the
    # dataset manifest too, so a consumer reading it alone sees the truth
    # (audit 2026-08-13 #38: the on-disk manifest must agree with the bundle
    # manifest, not just the in-memory dict). Rewrite the file in place.
    dataset_manifest["skipped"] = skipped
    with open(out_dir / "{}.manifest.json".format(name), "w", encoding="utf-8") as fh:
        json.dump(dataset_manifest, fh, indent=2, ensure_ascii=False)

    job_spec: Optional[Dict[str, Any]] = None
    if base_model is not None:
        job_spec = build_job_spec(dataset_manifest, base_model=base_model, policy=policy)
        # Make the bundled job spec portable: the dataset ships beside it as
        # <name>.jsonl, so point 'dataset' at the bundle-relative name, not the
        # exporter's absolute host path (audit 2026-08-13). The CLI --no-bundle
        # path keeps the absolute path because it calls build_job_spec directly.
        job_spec["dataset"] = "{}.jsonl".format(name)
        with open(out_dir / "{}.job.json".format(name), "w", encoding="utf-8") as fh:
            json.dump(job_spec, fh, indent=2, ensure_ascii=False)

    has_audit = audit_path is not None and Path(audit_path).exists()
    # Anchor the audit trail to the bundle by its content hash, so a consumer
    # can detect the trail was swapped/truncated after export.
    audit_sha256 = None
    if has_audit:
        audit_sha256 = hashlib.sha256(Path(audit_path).read_bytes()).hexdigest()  # type: ignore[arg-type]

    bundle_manifest = {
        "kind": "silt_skill_bundle",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "packets": [_packet_summary(p) for p in accepted],
        "dataset": "{}.jsonl".format(name),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "dataset_manifest": "{}.manifest.json".format(name),
        "job_spec": ("{}.job.json".format(name) if job_spec is not None else None),
        "audit": ("audit.jsonl" if has_audit else None),
        "audit_sha256": audit_sha256,
        "skipped": skipped,
        "contains_mock_data": any(p.provenance.is_mock for p in accepted),
        "note": (
            "SILT trains no weights. This bundle is the approved skill packet(s) "
            "the receiver conditions on at inference time via render_skills. For "
            "L4/L5, take the dataset + job spec to an external trainer."
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(bundle_manifest, fh, indent=2, ensure_ascii=False)
    with open(out_dir / "README.txt", "w", encoding="utf-8") as fh:
        fh.write(_BUNDLE_README)

    zip_path = out_dir / "{}.zip".format(name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for packet in accepted:
            zf.writestr(
                "approved/{}.json".format(packet.packet_id),
                packet.model_dump_json(indent=2),
            )
        zf.write(out_dir / "manifest.json", "manifest.json")
        zf.write(out_dir / "README.txt", "README.txt")
        dataset_path = out_dir / "{}.jsonl".format(name)
        if dataset_path.exists():
            zf.write(dataset_path, dataset_path.name)
        ds_manifest_path = out_dir / "{}.manifest.json".format(name)
        if ds_manifest_path.exists():
            zf.write(ds_manifest_path, ds_manifest_path.name)
        if job_spec is not None:
            # Use the basename as the arcname (defense-in-depth against zip-slip
            # even though name is validated).
            job_path = out_dir / "{}.job.json".format(name)
            zf.write(job_path, job_path.name)
        if has_audit:
            zf.write(Path(audit_path), "audit.jsonl")  # type: ignore[arg-type]

    return zip_path


def _rows_from_packet(packet: SkillPacket) -> List[Dict[str, Any]]:
    """Flatten a distilled payload into supervised training rows.

    Malformed entries (missing keys, wrong types) are skipped rather than
    crashing the whole export (audit 2026-08-13): a single bad entry in one
    packet must not prevent the rest of an approved set from exporting.
    """
    payload = packet.distilled_skill or {}
    rows: List[Dict[str, Any]] = []
    meta = {
        "packet_id": packet.packet_id,
        "capability": packet.sender_capability.as_str(),
        "language": packet.language,
        "domain": packet.domain.value,
        "origin_kind": packet.provenance.origin_kind.value,
        "synthetic_depth": packet.provenance.synthetic_depth,
        "is_mock": packet.provenance.is_mock,
    }

    def _row(in_key, out_key, item):
        if not isinstance(item, dict):
            return None
        inp, outp = item.get(in_key), item.get(out_key)
        if inp is None or outp is None:
            return None
        return {"input": inp, "output": outp, **meta}

    for entry in payload.get("entries", []) or []:
        if isinstance(entry, dict) and "source" in entry:
            row = _row("source", "target", entry)
        elif isinstance(entry, dict) and "grapheme" in entry:
            row = _row("grapheme", "phoneme", entry)
        else:
            row = None
        if row is not None:
            rows.append(row)
    for example in payload.get("examples", []) or []:
        row = _row("buggy", "fixed", example)
        if row is not None:
            rows.append(row)
    for rule in payload.get("rules", []) or []:
        row = _row("condition", "action", rule)
        if row is not None:
            rows.append(row)
    return rows


def export_dataset(
    packets: List[SkillPacket],
    out_dir: Path,
    name: str,
    include_mock: bool = False,
) -> Dict[str, Any]:
    """Write approved packets to JSONL. Returns a manifest.

    Refuses anything not PROMOTED, and by default refuses mock-derived content:
    a training set assembled from placeholder data is worse than no training set
    because it looks legitimate once it is on disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_name(name)

    accepted, skipped = [], []
    for packet in packets:
        if packet.promotion_status != PromotionStatus.PROMOTED:
            skipped.append((packet.packet_id, "not promoted"))
            continue
        if packet.provenance.is_mock and not include_mock:
            skipped.append((packet.packet_id, "mock provenance"))
            continue
        accepted.append(packet)

    rows: List[Dict[str, Any]] = []
    for packet in accepted:
        rows.extend(_rows_from_packet(packet))

    path = out_dir / "{}.jsonl".format(name)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "dataset_path": str(path),
        "dataset_sha256": dataset_sha256,
        "row_count": len(rows),
        "packet_count": len(accepted),
        "skipped": [{"packet_id": p, "reason": r} for p, r in skipped],
        "contains_mock_data": any(p.provenance.is_mock for p in accepted),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "{}.manifest.json".format(name), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def build_job_spec(
    manifest: Dict[str, Any],
    base_model: str,
    level: LearningLevel = LearningLevel.L4_PEFT_CANDIDATE,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 1e-4,
    epochs: int = 2,
    target_modules: Optional[List[str]] = None,
    eval_gate: Optional[Dict[str, Any]] = None,
    policy: Optional[PromotionPolicy] = None,
) -> Dict[str, Any]:
    """Describe the training run a human would launch. Nothing is executed.

    ``eval_gate`` is the bar the resulting adapter must clear before anyone
    ships it. When ``policy`` is supplied it is derived from the gate's ACTUAL
    thresholds; when neither ``eval_gate`` nor ``policy`` is supplied, an
    explicitly-labelled illustrative fallback is used (never the invented,
    divergent default this function used to hardcode -- audit 2026-08-13).
    """
    if level not in (LearningLevel.L4_PEFT_CANDIDATE, LearningLevel.L5_DISTILL_DATASET):
        raise ValueError("job specs are only meaningful for L4/L5")

    if eval_gate is not None:
        gate = eval_gate
    elif policy is not None:
        gate = _eval_gate_from_policy(policy)
    else:
        gate = _ILLUSTRATIVE_EVAL_GATE

    return {
        "status": "NOT_EXECUTED",
        "note": (
            "This system does not train. Run this job yourself, then re-enter the "
            "resulting adapter as a new receiver module and re-benchmark it before "
            "using it for anything."
        ),
        "level": int(level),
        "base_model": base_model,
        "dataset": manifest["dataset_path"],
        "row_count": manifest["row_count"],
        "contains_mock_data": manifest["contains_mock_data"],
        "method": "lora" if level == LearningLevel.L4_PEFT_CANDIDATE else "sequence_kd",
        "hyperparameters": {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "target_modules": target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "eval_gate": gate,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
