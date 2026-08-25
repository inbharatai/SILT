"""L4/L5 export: dataset + job spec, and the refusal to train."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from asea.core.protocol import (
    LearningLevel,
    OriginKind,
    PacketType,
    PromotionStatus,
    Provenance,
)
from asea.distill.export import build_job_spec, export_artifact_bundle, export_dataset


def _promoted(capability, provenance, factory, entries):
    return factory(
        capability, provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": entries},
        rollback_token="snap",
        promotion_status=PromotionStatus.PROMOTED,
    )


def test_export_writes_only_promoted_packets(tmp_path, capability, clean_provenance, packet_factory):
    promoted = _promoted(
        capability, clean_provenance, packet_factory,
        [{"source": "ভাত", "target": "rice"}, {"source": "পানী", "target": "water"}],
    )
    draft = packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "x", "target": "y"}]},
    )

    manifest = export_dataset([promoted, draft], tmp_path, "as_en")
    assert manifest["row_count"] == 2
    assert manifest["packet_count"] == 1
    assert manifest["skipped"][0]["reason"] == "not promoted"

    rows = [json.loads(l) for l in open(manifest["dataset_path"], encoding="utf-8")]
    assert rows[0]["input"] == "ভাত"
    assert rows[0]["output"] == "rice"
    assert rows[0]["synthetic_depth"] == 0


def test_export_refuses_mock_data_by_default(tmp_path, capability, packet_factory):
    mock_prov = Provenance(
        origin_kind=OriginKind.MODEL_GENERATED, chain=["qwen-mock"], is_mock=True
    )
    packet = _promoted(capability, mock_prov, packet_factory,
                       [{"source": "a", "target": "b"}])
    manifest = export_dataset([packet], tmp_path, "mocky")
    assert manifest["row_count"] == 0
    assert manifest["skipped"][0]["reason"] == "mock provenance"

    forced = export_dataset([packet], tmp_path, "mocky_forced", include_mock=True)
    assert forced["row_count"] == 1
    assert forced["contains_mock_data"] is True


def test_job_spec_is_explicitly_not_executed(tmp_path, capability, clean_provenance, packet_factory):
    packet = _promoted(capability, clean_provenance, packet_factory,
                       [{"source": "ভাত", "target": "rice"}])
    manifest = export_dataset([packet], tmp_path, "as_en")
    spec = build_job_spec(manifest, base_model="Qwen/Qwen2.5-7B-Instruct")

    assert spec["status"] == "NOT_EXECUTED"
    assert "does not train" in spec["note"]
    assert spec["method"] == "lora"
    # No policy supplied -> the labelled illustrative fallback (audit 2026-08-13:
    # build_job_spec no longer hardcodes an invented, divergent gate). Human
    # review is required for high-risk domains (the gate's actual rule), not
    # unconditionally, and the invented native-speaker key is gone.
    assert spec["eval_gate"]["source"] == "illustrative_fallback_not_from_gate"
    assert spec["eval_gate"]["require_human_review_for_high_risk_domains"] is True
    assert "require_native_speaker_review_for_language_tasks" not in spec["eval_gate"]
    assert spec["eval_gate"]["held_out_improvement_min"] == 0.01  # matches gate default


def test_job_spec_eval_gate_derived_from_policy(tmp_path, capability, clean_provenance,
                                                packet_factory):
    """When a policy is supplied, eval_gate mirrors the gate's actual thresholds,
    not an invented default (audit 2026-08-13)."""
    from asea.promotion.gate import PromotionPolicy
    packet = _promoted(capability, clean_provenance, packet_factory,
                       [{"source": "ভাত", "target": "rice"}])
    manifest = export_dataset([packet], tmp_path, "as_en")
    policy = PromotionPolicy(min_improvement=0.05, min_evaluator_score=0.7,
                             max_case_regression_ratio=0.2, max_synthetic_depth=3)
    spec = build_job_spec(manifest, base_model="Qwen/Qwen2.5-7B-Instruct", policy=policy)
    gate = spec["eval_gate"]
    assert gate["source"] == "derived_from_gate_policy"
    assert gate["held_out_improvement_min"] == 0.05
    assert gate["min_evaluator_score"] == 0.7
    assert gate["max_case_regression_ratio"] == 0.2
    assert gate["max_synthetic_depth"] == 3
    assert gate["strict_no_mock"] is True
    assert gate["require_human_review_for_high_risk_domains"] is True


def test_job_spec_rejects_applicable_levels(tmp_path, capability, clean_provenance, packet_factory):
    packet = _promoted(capability, clean_provenance, packet_factory,
                       [{"source": "a", "target": "b"}])
    manifest = export_dataset([packet], tmp_path, "x")
    with pytest.raises(ValueError, match="only meaningful for L4/L5"):
        build_job_spec(manifest, "some-model", level=LearningLevel.L3_SKILL_PACKET)


# -- artifact bundle (the "download the trained model" path) --------------------


def _bundle_entries(zip_bytes):
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    return zf, zf.namelist()


def test_bundle_contains_approved_packet_dataset_manifest_and_readme(
    tmp_path, capability, clean_provenance, packet_factory
):
    packet = _promoted(
        capability, clean_provenance, packet_factory,
        [{"source": "ভাত", "target": "rice"}],
    )
    zip_path = export_artifact_bundle([packet], tmp_path, name="as_en")
    assert zip_path.exists()
    zf, names = _bundle_entries(zip_path.read_bytes())

    assert "manifest.json" in names
    assert "README.txt" in names
    assert "as_en.jsonl" in names
    assert "as_en.manifest.json" in names
    assert "approved/{}.json".format(packet.packet_id) in names
    # No base_model => no job spec (honest about L0-L3 skill packets).
    assert "as_en.job.json" not in names
    assert not any(n == "audit.jsonl" for n in names)  # no audit path supplied

    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["kind"] == "silt_skill_bundle"
    assert manifest["packets"][0]["packet_id"] == packet.packet_id
    assert manifest["packets"][0]["gate_verdict"] == "promoted"
    assert manifest["job_spec"] is None
    assert manifest["contains_mock_data"] is False
    assert "SILT trains no weights" in manifest["note"]

    readme = zf.read("README.txt").decode("utf-8")
    assert "SILT trains no weights" in readme

    rows = [json.loads(l) for l in zf.read("as_en.jsonl").decode("utf-8").splitlines()]
    assert rows[0]["input"] == "ভাত" and rows[0]["output"] == "rice"


def test_bundle_refuses_mock_data_by_default(tmp_path, capability, packet_factory):
    mock_prov = Provenance(
        origin_kind=OriginKind.MODEL_GENERATED, chain=["qwen-mock"], is_mock=True
    )
    packet = _promoted(capability, mock_prov, packet_factory, [{"source": "a", "target": "b"}])
    zip_path = export_artifact_bundle([packet], tmp_path, name="mocky")
    zf, names = _bundle_entries(zip_path.read_bytes())
    assert not any(n.startswith("approved/") for n in names)
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["skipped"][0]["reason"] == "mock provenance"

    # Opting in to mock data lets it through, flagged honestly.
    zip_path = export_artifact_bundle([packet], tmp_path, name="mocky_forced",
                                      include_mock=True)
    zf, names = _bundle_entries(zip_path.read_bytes())
    assert any(n.startswith("approved/") for n in names)
    assert json.loads(zf.read("manifest.json"))["contains_mock_data"] is True


def test_bundle_skips_non_promoted_packets(tmp_path, capability, clean_provenance, packet_factory):
    promoted = _promoted(capability, clean_provenance, packet_factory,
                         [{"source": "a", "target": "b"}])
    draft = packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "x", "target": "y"}]},
    )  # status DISTILLED, not promoted
    zip_path = export_artifact_bundle([promoted, draft], tmp_path, name="mixed")
    zf, names = _bundle_entries(zip_path.read_bytes())
    assert "approved/{}.json".format(promoted.packet_id) in names
    assert "approved/{}.json".format(draft.packet_id) not in names
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["skipped"] == [{"packet_id": draft.packet_id, "reason": "not promoted"}]


def test_bundle_writes_job_spec_only_when_base_model_supplied(
    tmp_path, capability, clean_provenance, packet_factory
):
    packet = _promoted(capability, clean_provenance, packet_factory,
                       [{"source": "a", "target": "b"}])
    zip_path = export_artifact_bundle(
        [packet], tmp_path, name="withjob", base_model="Qwen/Qwen2.5-7B-Instruct",
    )
    zf, names = _bundle_entries(zip_path.read_bytes())
    assert "withjob.job.json" in names
    spec = json.loads(zf.read("withjob.job.json"))
    assert spec["status"] == "NOT_EXECUTED"
    assert spec["base_model"] == "Qwen/Qwen2.5-7B-Instruct"


def test_bundle_includes_audit_when_path_exists(
    tmp_path, capability, clean_provenance, packet_factory
):
    packet = _promoted(capability, clean_provenance, packet_factory,
                       [{"source": "a", "target": "b"}])
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    audit_path = audit_dir / "audit.jsonl"
    audit_path.write_text('{"index": 0, "event": "run_complete"}\n', encoding="utf-8")
    zip_path = export_artifact_bundle([packet], tmp_path, name="aud", audit_path=audit_path)
    zf, names = _bundle_entries(zip_path.read_bytes())
    assert "audit.jsonl" in names
    assert "run_complete" in zf.read("audit.jsonl").decode("utf-8")
