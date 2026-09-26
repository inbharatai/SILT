"""Extraction-package tests (master brief sections D/W/X; FIXTURE_VERIFIED).

Every test here is a MECHANISM test on fixtures: a fake checkpoint tree,
a JSON spec, a tmp-path workspace. Nothing in this file is GLM evidence
and no passing test upgrades any real-model status.

Exit codes are asserted against the ACTUAL subprocess exit status
(brief C5: "test the ACTUAL subprocess exit status, not returned
dicts").
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from asea.extraction import (
    ActivationObservation,
    BlockedResource,
    CapabilityExtractionReceipt,
    CapabilityExtractionSpec,
    CausalComponentGraph,
    CheckpointFile,
    CompiledCapabilityConfig,
    CompiledCapabilityManifest,
    ExtractionPlan,
    FunctionalCase,
    FunctionalCaseManifest,
    FunctionalJoin,
    InvalidEvidence,
    Refused,
    RoutingObservation,
    SealedSplit,
    SourceCheckpointManifest,
    judged_only,
    source_manifest_fingerprint,
)
from asea.extraction.ledger import FailedAttemptLedger
from asea.extraction.schema import GraphNode

_CLI = [sys.executable, "-m", "asea.extraction"]


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _join(source_revision: str = "abc123") -> FunctionalJoin:
    return FunctionalJoin(
        capability_id="repo_repair_v1",
        case_id="case-1",
        prompt_hash=_hash("prompt"),
        source_revision=source_revision,
        generation_policy_hash=_hash("policy"),
        oracle_artifact_hash=_hash("oracle"),
        verdict=True,
    )


def _routing(join=None, layer=3, expert=17) -> RoutingObservation:
    return RoutingObservation(
        capability_id="repo_repair_v1",
        layer_id=layer,
        expert_id=expert,
        dispatched_count=4,
        sigmoid_mass=0.25,
        source_revision="abc123",
        functional_join=join,
    )


def _spec_dict(**overrides):
    base = {
        "schema": "silt.extraction.spec.v1",
        "capability_id": "repo_repair_v1",
        "description": "repository-level software engineering",
        "source_model": {
            "repository": "zai-org",
            "model": "GLM-5.3-Flash",
            "variant": "BF16",
            "commit_sha": "0" * 40,
            "license": "MIT",
        },
        "thresholds": {
            "minimum_retention_ratio": 0.9,
            "maximum_stored_parameter_reduction": 0.25,
        },
        "target_task_families": ["repo_debug"],
        "control_task_families": ["math", "translation"],
        "evaluation": {"final": "sealed final split"},
        "sealed_final_path": "sealed/",
    }
    base.update(overrides)
    return base


def _fixture_checkpoint(root: Path) -> Path:
    """A minimal but SHAPE-correct checkpoint tree (FIXTURE, never a GLM
    checkpoint claim)."""
    checkpoint = root / "GLM-5.3-Flash-BF16"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "generation_config.json").write_text("{}")
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "…"}))
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"weights-a")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"weights-b")
    (checkpoint / "preprocessor_config.json").write_text("{}")
    return checkpoint


# ---------------------------------------------------------------------------
# Schemas (brief section X: "schemas; malformed evidence")
# ---------------------------------------------------------------------------

class TestCapabilityExtractionSpec:
    def test_valid_spec_round_trips(self):
        spec = CapabilityExtractionSpec.model_validate(_spec_dict())
        assert spec.thresholds.minimum_retention_ratio == 0.9

    def test_malformed_spec_rejected(self):
        bad = _spec_dict()
        bad["schema"] = "silt.extraction.spec.v2"
        with pytest.raises(Exception):
            CapabilityExtractionSpec.model_validate(bad)

    def test_target_control_overlap_rejected(self):
        bad = _spec_dict()
        bad["control_task_families"] = ["repo_debug"]
        with pytest.raises(Exception):
            CapabilityExtractionSpec.model_validate(bad)

    def test_missing_final_evaluation_rejected(self):
        bad = _spec_dict()
        bad["evaluation"] = {"development": "dev split"}
        with pytest.raises(Exception):
            CapabilityExtractionSpec.model_validate(bad)


class TestSourceCheckpointManifest:
    def _files(self) -> dict:
        return {
            "config.json": CheckpointFile(filename="config.json", bytes=2,
                                          sha256=_hash("config")),
            "generation_config.json": CheckpointFile(
                filename="generation_config.json", bytes=2,
                sha256=_hash("gen")),
            "tokenizer_config.json": CheckpointFile(
                filename="tokenizer_config.json", bytes=2,
                sha256=_hash("tok")),
            "model.safetensors": CheckpointFile(
                filename="model.safetensors", bytes=10,
                sha256=_hash("weights")),
        }

    def _manifest(self, files=None, **overrides):
        files = files if files is not None else self._files()
        payload = {
            "repository": "zai-org",
            "model": "GLM-5.3-Flash",
            "variant": "BF16",
            "commit_sha": "0" * 40,
            "license": "MIT",
            "files": {name: entry.model_dump()
                      for name, entry in files.items()},
            "chat_template_source": "tokenizer_config.json",
            "aggregate_sha256": source_manifest_fingerprint(files),
        }
        payload.update(overrides)
        return SourceCheckpointManifest.model_validate(payload)

    def test_exact_identity_requires_every_required_file(self):
        files = self._files()
        del files["generation_config.json"]
        with pytest.raises(ValueError, match="generation_config.json"):
            self._manifest(files=files)

    def test_exact_identity_requires_a_weight_shard(self):
        files = self._files()
        del files["model.safetensors"]
        with pytest.raises(ValueError, match="safetensors"):
            self._manifest(files=files)

    def test_aggregate_is_order_independent(self):
        files = self._files()
        flipped = dict(reversed(list(files.items())))
        assert (source_manifest_fingerprint(files)
                == source_manifest_fingerprint(flipped))

    def test_wrong_aggregate_is_rejected(self):
        with pytest.raises(Exception):
            self._manifest(aggregate_sha256=_hash("not the aggregate"))


# ---------------------------------------------------------------------------
# C2: unjudged telemetry may be stored but excluded from evidence
# ---------------------------------------------------------------------------

class TestJudgedOnly:
    def test_unjudged_traces_excluded_from_evidence(self):
        judged = _routing(join=_join())
        unjudged = _routing()
        kept = judged_only([judged, unjudged])
        assert kept == [judged]

    def test_null_verdict_is_not_a_join(self):
        # FunctionalJoin.verdict is a required bool: success=null cannot
        # even be represented.
        with pytest.raises(Exception):
            FunctionalJoin(
                capability_id="repo_repair_v1", case_id="case-1",
                prompt_hash=_hash("prompt"), source_revision="abc123",
                generation_policy_hash=_hash("policy"),
                oracle_artifact_hash=_hash("oracle"), verdict=None,
            )

    def test_cross_revision_join_is_rejected(self):
        with pytest.raises(ValueError, match="revisions"):
            _routing(join=_join(source_revision="different"))

    def test_activation_joins_same_contract(self):
        judged = ActivationObservation(
            capability_id="repo_repair_v1", layer_id=3, expert_id=17,
            activation_l2_mean=1.5, observations=10,
            source_revision="abc123", functional_join=_join())
        unjudged = ActivationObservation(
            capability_id="repo_repair_v1", layer_id=3, expert_id=17,
            activation_l2_mean=1.5, observations=10,
            source_revision="abc123")
        assert judged_only([judged, unjudged]) == [judged]


# ---------------------------------------------------------------------------
# Causal-component graph honesty (frequency is correlation)
# ---------------------------------------------------------------------------

class TestCausalComponentGraph:
    def test_required_role_without_causal_evidence_is_rejected(self):
        with pytest.raises(ValueError, match="causal"):
            CausalComponentGraph.model_validate({
                "capability_id": "repo_repair_v1",
                "source_revision": "abc123",
                "nodes": [
                    {"component": "expert:3/17", "role": "REQUIRED",
                     "evidence_count": 500, "causal_evidence": False},
                ],
                "limitations": ["frequency is correlation"],
            })

    def test_correlation_only_role_is_allowed_without_cause(self):
        graph = CausalComponentGraph.model_validate({
            "capability_id": "repo_repair_v1",
            "source_revision": "abc123",
            "nodes": [
                {"component": "expert:3/17", "role": "TARGET_ENRICHED",
                 "evidence_count": 500, "causal_evidence": False},
            ],
            "limitations": ["frequency is correlation"],
        })
        assert graph.nodes[0].role == "TARGET_ENRICHED"


# ---------------------------------------------------------------------------
# Receipt verdict discipline (Section T)
# ---------------------------------------------------------------------------

def _receipt_dict(**overrides) -> dict:
    base = {
        "schema": "silt.extraction.receipt.v1",
        "capability_id": "repo_repair_v1",
        "command": "certify",
        "spec_sha256": _hash("spec"),
        "source_manifest_sha256": _hash("manifest"),
        "preregistered_thresholds": {
            "minimum_retention_ratio": 0.9,
            "maximum_stored_parameter_reduction": 0.25,
        },
        "retained_ratio": "NOT_MEASURED",
        "stored_parameter_reduction": "NOT_MEASURED",
        "final_verdict": "NOT_MEASURED",
        "final_split_evaluated_exactly_once": False,
        "limitations": ["fixture test"],
    }
    base.update(overrides)
    return base


class TestReceipt:
    def test_not_measured_verdict_honest(self):
        receipt = CapabilityExtractionReceipt.model_validate(_receipt_dict())
        assert receipt.retained_ratio == "NOT_MEASURED"
        assert receipt.final_verdict == "NOT_MEASURED"

    def test_certified_requires_measured_values(self):
        with pytest.raises(ValueError, match="measured"):
            CapabilityExtractionReceipt.model_validate(
                _receipt_dict(final_verdict="CERTIFIED",
                              final_split_evaluated_exactly_once=True))

    def test_certified_below_preregistered_threshold_is_rejected(self):
        with pytest.raises(ValueError, match="preregistered"):
            CapabilityExtractionReceipt.model_validate(
                _receipt_dict(final_verdict="CERTIFIED",
                              final_split_evaluated_exactly_once=True,
                              retained_ratio=0.5,
                              stored_parameter_reduction=0.4))

    def test_certified_at_thresholds_passes(self):
        receipt = CapabilityExtractionReceipt.model_validate(
            _receipt_dict(final_verdict="CERTIFIED",
                          final_split_evaluated_exactly_once=True,
                          retained_ratio=0.91,
                          stored_parameter_reduction=0.30))
        assert receipt.final_verdict == "CERTIFIED"

    def test_verdict_requires_exactly_once_evaluation(self):
        with pytest.raises(ValueError, match="exactly once"):
            CapabilityExtractionReceipt.model_validate(
                _receipt_dict(final_verdict="FAILED_HYPOTHESIS",
                              retained_ratio=0.4,
                              final_split_evaluated_exactly_once=False))


class TestCompiledCapabilityManifest:
    def test_parameter_reduction_arithmetic(self):
        manifest = CompiledCapabilityManifest(
            name="CompiledCapabilityModel-repo-repair-v1",
            capability_id="repo_repair_v1",
            config=CompiledCapabilityConfig(
                num_hidden_layers=45, hidden_size=4096,
                n_routed_experts_per_layer={"3": 40}, num_experts_per_tok=8,
                first_k_dense_replace=3, vocab_size=151552),
            plan_sha256=_hash("plan"), provenance_sha256=_hash("prov"),
            source_manifest_sha256=_hash("source"),
            stored_parameters=1_000_000, source_stored_parameters=2_000_000)
        assert manifest.stored_parameter_reduction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# C6: the sealed final split
# ---------------------------------------------------------------------------

class TestSealedSplit:
    def _sealed(self, tmp_path, spec_sha=None):
        sealed = SealedSplit(tmp_path / "sealed")
        cases = {"cases": [
            {"case_id": "f-1", "family": "repo_debug", "group": "target",
             "content_sha256": _hash("c"), "prompt_sha256": _hash("p"),
             "license": "MIT"},
        ]}
        sealed.seal(capability_id="repo_repair_v1",
                     spec_sha256=spec_sha or _hash("spec"), cases=cases)
        return sealed, cases

    def test_development_access_is_refused_and_recorded(self, tmp_path):
        sealed, _ = self._sealed(tmp_path)
        with pytest.raises(Refused, match="development"):
            sealed.access(command="silt-extract trace --split final")
        log = sealed.access_log()
        assert len(log) == 1
        assert log[0]["verdict"] == "REFUSED_DEVELOPMENT_ACCESS"
        assert not sealed.is_consumed()

    def test_final_evaluation_opens_exactly_once(self, tmp_path):
        sealed, cases = self._sealed(tmp_path)
        opened = sealed.open_for_final_evaluation(
            command="silt-extract evaluate --split final",
            spec_sha256=_hash("spec"))
        assert opened == cases
        assert sealed.is_consumed()
        with pytest.raises(Refused, match="already been opened"):
            sealed.open_for_final_evaluation(
                command="silt-extract evaluate --split final",
                spec_sha256=_hash("spec"))
        verdicts = [entry["verdict"] for entry in sealed.access_log()]
        assert verdicts == ["PERMITTED", "REFUSED_ALREADY_CONSUMED"]

    def test_spec_mismatch_is_refused_without_consuming(self, tmp_path):
        sealed, _ = self._sealed(tmp_path)
        with pytest.raises(Refused, match="pinned"):
            sealed.open_for_final_evaluation(command="evaluate",
                                              spec_sha256=_hash("other"))
        assert not sealed.is_consumed()

    def test_resealing_is_refused(self, tmp_path):
        sealed, _ = self._sealed(tmp_path)
        with pytest.raises(Refused, match="reseal"):
            sealed.seal(capability_id="repo_repair_v1",
                        spec_sha256=_hash("spec"), cases={})

    def test_tampered_seal_is_refused(self, tmp_path):
        sealed, _ = self._sealed(tmp_path)
        raw = json.loads(sealed.seal_path.read_text(encoding="utf-8"))
        raw["cases"]["cases"].append({"case_id": "smuggled"})
        sealed.seal_path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(Refused, match="modified"):
            sealed.open_for_final_evaluation(
                command="evaluate", spec_sha256=_hash("spec"))


# ---------------------------------------------------------------------------
# C7: durable failed-attempt ledger
# ---------------------------------------------------------------------------

class TestFailedAttemptLedger:
    def test_every_required_field_recorded(self, tmp_path):
        ledger = FailedAttemptLedger(tmp_path / "attempts.jsonl")
        entry = ledger.record(
            stage="intervene",
            source_fingerprint=_hash("source"),
            request={"component": "expert:3/17"},
            component_ids=["expert:3/17"],
            case_hashes=[_hash("case")],
            error_class="worker_crashed",
            timeout_info={"timeout": 300, "deadline_hit": False},
            restoration_status="RESTORED_AND_VERIFIED",
            stderr_tail="boom\n" * 100,
            generated_artifacts={"partial.json": _hash("partial")},
        )
        assert entry["stderr_tail"] == "boom\n" * 100  # under the 4000 cap
        assert entry["generated_artifacts"]["partial.json"] == _hash("partial")
        assert ledger.verify_chain()
        assert ledger.failure_history() == [
            "0: stage=intervene error=worker_crashed "
            "restoration=RESTORED_AND_VERIFIED"
        ]

    def test_long_stderr_tail_is_capped(self, tmp_path):
        ledger = FailedAttemptLedger(tmp_path / "attempts.jsonl")
        entry = ledger.record(stage="trace", source_fingerprint=_hash("s"),
                              request={}, stderr_tail="x" * 10_000)
        assert len(entry["stderr_tail"]) == 4000

    def test_tampered_history_breaks_the_chain(self, tmp_path):
        path = tmp_path / "attempts.jsonl"
        ledger = FailedAttemptLedger(path)
        ledger.record(stage="trace", source_fingerprint=_hash("s"),
                      request={})
        ledger.record(stage="build", source_fingerprint=_hash("s"),
                      request={})
        assert ledger.verify_chain()
        lines = path.read_text(encoding="utf-8").splitlines()
        edited = json.loads(lines[0])
        edited["stage"] = "never-happened"
        lines[0] = json.dumps(edited, sort_keys=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert not FailedAttemptLedger(path).verify_chain()

    def test_unknown_error_class_rejected(self, tmp_path):
        ledger = FailedAttemptLedger(tmp_path / "attempts.jsonl")
        with pytest.raises(ValueError):
            ledger.record(stage="trace", source_fingerprint=_hash("s"),
                          request={}, error_class="made_up_class")


# ---------------------------------------------------------------------------
# CLI exit codes (C5): ACTUAL subprocess exit status
# ---------------------------------------------------------------------------

def _run(*argv, cwd=None):
    completed = subprocess.run(
        list(_CLI) + list(argv), capture_output=True, text=True,
        cwd=str(cwd) if cwd else None, timeout=120)
    assert completed.stdout.strip(), "stdout must always be one JSON object"
    return (completed.returncode, json.loads(completed.stdout))


class TestCliExitCodes:
    def test_spec_validate_success_exit_0(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps(_spec_dict()), encoding="utf-8")
        code, payload = _run("spec", "validate", "--config", str(spec))
        assert code == 0
        assert payload["ok"] is True
        assert payload["capability_id"] == "repo_repair_v1"

    def test_malformed_spec_is_invalid_evidence_exit_3(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{not json", encoding="utf-8")
        code, payload = _run("spec", "validate", "--config", str(spec))
        assert code == 3
        assert payload["status"] == "invalid_evidence"

    def test_wrong_schema_version_is_exit_3(self, tmp_path):
        spec = tmp_path / "spec.json"
        bad = _spec_dict()
        bad["schema"] = "silt.extraction.spec.v2"
        spec.write_text(json.dumps(bad), encoding="utf-8")
        code, payload = _run("spec", "validate", "--config", str(spec))
        assert code == 3

    def test_source_inventory_exit_0_no_hashing(self, tmp_path):
        checkpoint = _fixture_checkpoint(tmp_path)
        code, payload = _run("source", "inventory",
                             "--checkpoint", str(checkpoint))
        assert code == 0
        assert payload["file_count"] == 6
        assert payload["required_present"] == {
            "config.json": True,
            "generation_config.json": True,
            "tokenizer_config.json": True,
            "weight_shards": True,
        }

    def test_source_verify_exit_0_and_dry_run_writes_nothing(self, tmp_path):
        checkpoint = _fixture_checkpoint(tmp_path)
        card = tmp_path / "card.json"
        card.write_text(json.dumps({
            "repository": "zai-org", "model": "GLM-5.3-Flash",
            "variant": "BF16", "commit_sha": "0" * 40, "license": "MIT",
        }), encoding="utf-8")
        workspace = tmp_path / "ws"
        code, payload = _run(
            "source", "verify", "--checkpoint", str(checkpoint),
            "--source-card", str(card), "--workspace", str(workspace),
            "--dry-run")
        assert code == 0
        assert payload["file_count"] == 6
        assert payload["artifact"] is None
        assert not (workspace / "source-manifests").exists()

    def test_source_verify_missing_required_file_exit_3(self, tmp_path):
        checkpoint = _fixture_checkpoint(tmp_path)
        (checkpoint / "generation_config.json").unlink()
        card = tmp_path / "card.json"
        card.write_text(json.dumps({
            "repository": "zai-org", "model": "GLM-5.3-Flash",
            "variant": "BF16", "commit_sha": "0" * 40, "license": "MIT",
        }), encoding="utf-8")
        code, payload = _run(
            "source", "verify", "--checkpoint", str(checkpoint),
            "--source-card", str(card))
        assert code == 3
        assert "generation_config.json" in payload["error"]["message"]

    def test_dataset_validate_exit_0(self, tmp_path):
        manifest = {
            "schema": "silt.extraction.cases.v1",
            "capability_id": "repo_repair_v1",
            "split": "development",
            "near_duplicate_free": True,
            "cases": [
                {"case_id": "c-1", "family": "repo_debug",
                 "group": "target", "content_sha256": _hash("c"),
                 "prompt_sha256": _hash("p1"), "license": "MIT"},
                {"case_id": "c-2", "family": "math",
                 "group": "control", "content_sha256": _hash("c2"),
                 "prompt_sha256": _hash("p2"), "license": "MIT"},
            ],
        }
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        code, payload = _run("dataset", "validate", "--dir", str(path))
        assert code == 0
        assert payload["cases"] == 2

    def test_dataset_duplicate_prompts_exit_3(self, tmp_path):
        manifest = {
            "schema": "silt.extraction.cases.v1",
            "capability_id": "repo_repair_v1",
            "split": "development",
            "near_duplicate_free": True,
            "cases": [
                {"case_id": "c-1", "family": "repo_debug",
                 "group": "target", "content_sha256": _hash("c"),
                 "prompt_sha256": _hash("same"), "license": "MIT"},
                {"case_id": "c-2", "family": "math",
                 "group": "control", "content_sha256": _hash("c2"),
                 "prompt_sha256": _hash("same"), "license": "MIT"},
            ],
        }
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        code, _ = _run("dataset", "validate", "--dir", str(path))
        assert code == 3

    def test_not_implemented_stage_is_typed_refusal_exit_2(self):
        # baseline/trace/intervene/graph/plan are IMPLEMENTED now (their
        # own tests below); the remaining real-model stages still refuse
        # honestly.
        for command in ("build", "recover", "compare", "reduce", "certify"):
            code, payload = _run(command)
            assert code == 2, command
            assert payload["status"] == "NOT_IMPLEMENTED"
            assert payload["ok"] is False

    def test_evaluate_final_does_not_open_the_seal(self, tmp_path):
        sealed = SealedSplit(tmp_path / "sealed")
        sealed.seal(
            capability_id="repo_repair_v1", spec_sha256=_hash("spec"),
            cases={"cases": []},
        )
        code, payload = _run("evaluate", "--split", "final",
                             "--sealed-dir", str(tmp_path / "sealed"))
        assert code == 2
        assert payload["status"] == "NOT_IMPLEMENTED"
        assert payload["sealed_split_note"] is not None
        assert "NOT open" in payload["sealed_split_note"] \
            or "did NOT open" in payload["sealed_split_note"]
        assert not sealed.is_consumed()
        assert any(entry["verdict"] == "REFUSED_DEVELOPMENT_ACCESS"
                   for entry in sealed.access_log())

    def test_receipt_success_exit_0(self, tmp_path):
        receipt = tmp_path / "receipt.json"
        receipt.write_text(json.dumps(_receipt_dict()), encoding="utf-8")
        code, payload = _run("receipt", "--receipt", str(receipt),
                             "--workspace", str(tmp_path / "ws"),
                             "--dry-run")
        assert code == 0
        assert payload["final_verdict"] == "NOT_MEASURED"

    def test_receipt_certified_without_evidence_exit_3(self, tmp_path):
        bad = _receipt_dict(final_verdict="CERTIFIED")
        receipt = tmp_path / "receipt.json"
        receipt.write_text(json.dumps(bad), encoding="utf-8")
        code, _ = _run("receipt", "--receipt", str(receipt),
                       "--workspace", str(tmp_path / "ws"))
        assert code == 3

    def test_unknown_command_is_typed_refusal_exit_2(self):
        code, _ = _run("definitely-not-a-command")
        assert code == 2


# ---------------------------------------------------------------------------
# ExtractionPlan / provenance honesty guards
# ---------------------------------------------------------------------------

class TestPlanAndProvenanceGuards:
    def test_plan_must_be_conservative(self):
        payload = {
            "schema": "silt.extraction.plan.v1",
            "capability_id": "repo_repair_v1",
            "source_manifest_sha256": _hash("m"),
            "graph_sha256": _hash("g"),
            "retained_dense": False,
            "retained_tokenizer": True,
            "retained_attention": True,
            "retained_head": True,
            "expert_retention": [
                {"layer_id": 3, "retained_experts": [1, 2],
                 "selection_basis": "causal"}],
            "selection_arm": "causal",
        }
        with pytest.raises(ValueError, match="conservative"):
            ExtractionPlan.model_validate(payload)

    def test_provenance_needs_a_copied_source(self):
        from asea.extraction.schema import TensorProvenance

        with pytest.raises(ValueError, match="copied source"):
            TensorProvenance.model_validate({
                "schema": "silt.extraction.provenance.v1",
                "manifest_sha256": _hash("m"),
                "source_manifest_sha256": _hash("s"),
                "transformations": [
                    {"source_tensor": "mlp.3.experts.17.gate",
                     "target_tensor": "model.experts.17.gate",
                     "method": "RECOVERY_TRAINED"}],
            })


class TestErrors:
    def test_exit_code_contract(self):
        assert Refused("x", "y").exit_code == 2
        assert InvalidEvidence("x", "y").exit_code == 3
        assert BlockedResource("GPU", "use a bigger box").exit_code == 4
        assert BlockedResource("GPU", "use a bigger box").requirement == "GPU"
        from asea.extraction.errors import ExecutionFailure

        assert ExecutionFailure("x", "y").exit_code == 5


# ---------------------------------------------------------------------------
# Sections E-M: implemented real-model stages (FIXTURE-verified mechanisms;
# the admission gate reads REAL config arithmetic and REAL host memory)
# ---------------------------------------------------------------------------

#: A shape-correct GLM-5.3-Flash TEXT config for admission/architecture
#: fixtures. FIXTURE ONLY: no weights exist here and no test is GLM
#: evidence -- the config arithmetic is verified against the WORKER's
#: own formula, never against a claimed model measurement.
GLM_TEXT_CONFIG = {
    "model_type": "glm5_next",
    "num_hidden_layers": 45,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "hidden_size": 4096,
    "scoring_func": "sigmoid",
    "max_position_embeddings": 1_048_576,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "vocab_size": 151552,
}


def _glm_checkpoint(root: Path, **config_overrides) -> Path:
    checkpoint = root / "glm-config-fixture"
    checkpoint.mkdir()
    text = dict(GLM_TEXT_CONFIG)
    text.update(config_overrides)
    (checkpoint / "config.json").write_text(json.dumps(text),
                                           encoding="utf-8")
    return checkpoint


def _attempt(component, target_drop, control_drop, *, seed=7,
             revision="abc123", restored=True):
    """One InterventionAttempt record shaped exactly as the intervene
    stage records them (FIXTURE, never a real intervention)."""
    return {
        "schema": "silt.extraction.intervention.v1",
        "capability_id": "repo_repair_v1",
        "component": component,
        "target_drop": target_drop,
        "control_drop": control_drop,
        "target_cases": 12,
        "control_cases": 12,
        "seed": seed,
        "restored_and_verified": restored,
        "source_revision": revision,
        "functional_joins": [
            {"case_id": "case-1", "capability_id": "repo_repair_v1",
             "prompt_hash": _hash("prompt-" + component),
             "source_revision": revision,
             "generation_policy_hash": _hash("policy"),
             "oracle_artifact_hash": _hash("oracle"), "verdict": True},
        ],
    }


class TestStageAdmission:
    """The admission gate runs REAL config arithmetic against REAL host
    memory; on every realistic test host the GLM-sized requirement
    (~1.2 TiB) exceeds availability and the stage reports
    BLOCKED_RESOURCE (exit 4) -- the honest outcome, never a
    substituted estimate."""

    def _spec(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps(_spec_dict()), encoding="utf-8")
        return spec

    def test_baseline_is_blocked_resource_exit_4(self, tmp_path):
        spec = self._spec(tmp_path)
        checkpoint = _glm_checkpoint(tmp_path)
        code, payload = _run("baseline", "--config", str(spec),
                             "--checkpoint", str(checkpoint))
        assert code == 4
        assert payload["status"] == "BLOCKED_RESOURCE"
        error = payload["error"]
        assert error["exit_code"] == 4
        assert "GiB" in error["message"]
        assert "host memory" in error["remedy"]

    def test_trace_and_intervene_share_the_blocked_path(self, tmp_path):
        spec = self._spec(tmp_path)
        checkpoint = _glm_checkpoint(tmp_path)
        for command in ("trace", "intervene"):
            code, payload = _run(command, "--config", str(spec),
                                "--checkpoint", str(checkpoint))
            assert code == 4, command
            assert payload["status"] == "BLOCKED_RESOURCE"

    def test_architecture_mismatch_is_invalid_evidence_exit_3(self, tmp_path):
        spec = self._spec(tmp_path)
        checkpoint = _glm_checkpoint(tmp_path, num_hidden_layers=44)
        code, payload = _run("baseline", "--config", str(spec),
                             "--checkpoint", str(checkpoint))
        assert code == 3
        assert payload["status"] == "invalid_evidence"
        assert "num_hidden_layers" in payload["error"]["message"]

    def test_missing_checkpoint_argument_is_typed_refusal(self, tmp_path):
        spec = self._spec(tmp_path)
        code, payload = _run("baseline", "--config", str(spec))
        assert code == 2
        assert payload["status"] == "refused"

    def test_admitted_host_reports_not_implemented_honestly(
            self, tmp_path, monkeypatch, capsys):
        """On a host the gate ADMITS (fixture: memory probe patched to
        2 TiB -- the gate arithmetic itself is NOT patched), the
        beyond-admission measurement orchestration is stated as
        NOT_IMPLEMENTED rather than faked."""
        from asea.extraction import stages as extraction_stages
        from asea.extraction.__main__ import main

        monkeypatch.setattr(extraction_stages, "available_bytes",
                            lambda: 2 * 1024 ** 4)
        spec = self._spec(tmp_path)
        checkpoint = _glm_checkpoint(tmp_path)
        code = main(["baseline", "--config", str(spec),
                     "--checkpoint", str(checkpoint)])
        assert code == 2
        out = json.loads(capsys.readouterr().out.strip())
        assert out["status"] == "NOT_IMPLEMENTED"
        assert out["admission"]["admitted"] is True
        assert out["admission"]["estimated_parameters"] > 300e9


class TestStagesUnits:
    def test_parameter_estimate_mirrors_the_worker_formula(self):
        from asea.extraction.stages import estimate_parameters

        estimate = estimate_parameters(GLM_TEXT_CONFIG)
        per_expert = 3 * 4096 * 2048  # gate/up/down
        moe = (288 + 1) * per_expert * (45 - 3)
        dense_mlp = 45 * 3 * 4096 * 12288
        vocab = 151552 * 4096
        attention = 4 * 45 * 4096 * 4096
        assert estimate == moe + dense_mlp + vocab + attention
        assert 300e9 < estimate < 350e9

    def test_text_config_nesting_is_merged(self, tmp_path):
        from asea.extraction.stages import load_checkpoint_config

        checkpoint = tmp_path / "vision-composite"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps({
            "model_type": "glm5_next",
            "text_config": dict(GLM_TEXT_CONFIG),
        }), encoding="utf-8")
        config = load_checkpoint_config(checkpoint)
        assert config["num_hidden_layers"] == 45
        assert config["moe_intermediate_size"] == 2048

    def test_validate_source_architecture_flags_each_mismatch(self):
        from asea.extraction.stages import validate_source_architecture

        report = validate_source_architecture(
            dict(GLM_TEXT_CONFIG, scoring_func="softmax"))
        assert report["valid"] is False
        assert report["checks"]["scoring_func"]["ok"] is False
        assert report["checks"]["hidden_size"]["ok"] is True

    def test_position_floor_is_a_floor_not_an_exact_match(self):
        from asea.extraction.stages import validate_source_architecture

        bigger = validate_source_architecture(
            dict(GLM_TEXT_CONFIG, max_position_embeddings=2_000_000))
        smaller = validate_source_architecture(
            dict(GLM_TEXT_CONFIG, max_position_embeddings=131_072))
        assert bigger["valid"] is True
        assert smaller["valid"] is False

    def test_power_analysis_is_monotone_and_degenerate_safe(self):
        from asea.extraction.errors import InvalidEvidence
        from asea.extraction.stages import required_sample_size

        small = required_sample_size(
            baseline_pass_rate=0.9, minimum_detectable_effect=0.3)
        larger = required_sample_size(
            baseline_pass_rate=0.9, minimum_detectable_effect=0.15)
        assert 0 < small < larger
        with pytest.raises(InvalidEvidence):
            required_sample_size(
                baseline_pass_rate=1.0, minimum_detectable_effect=0.1)
        with pytest.raises(InvalidEvidence):
            required_sample_size(
                baseline_pass_rate=0.5, minimum_detectable_effect=0.5)


class TestGraphCommand:
    def _artifacts(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps(_spec_dict()), encoding="utf-8")
        attempts = tmp_path / "attempts.jsonl"
        records = [
            # verified target effect above its matched-control arm
            _attempt("expert:3/17", 0.8, 0.0),
            # control regression under intervention: harmful component
            _attempt("expert:3/18", 0.0, 0.9, seed=8),
            # verified but no measurable effect either arm
            _attempt("expert:3/19", 0.0, 0.0, seed=9),
        ]
        attempts.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        enrichment = tmp_path / "enrichment.json"
        enrichment.write_text(json.dumps({
            "expert:3/20": {"target_cases": 30, "control_cases": 0},
            "expert:3/21": {"target_cases": 5, "control_cases": 5},
            "expert:3/22": {"target_cases": 4, "control_cases": 2,
                            "control_unobserved": True},
        }), encoding="utf-8")
        return spec, attempts, enrichment

    def test_graph_classifies_roles_offline_exit_0(self, tmp_path):
        spec, attempts, enrichment = self._artifacts(tmp_path)
        workspace = tmp_path / "ws"
        code, payload = _run(
            "graph", "--config", str(spec),
            "--interventions", str(attempts),
            "--enrichment", str(enrichment),
            "--workspace", str(workspace), "--dry-run")
        assert code == 0
        assert payload["ok"] is True
        assert payload["nodes"] == 6
        assert payload["roles"] == {
            "REQUIRED": 1, "NEGATIVE_OR_HARMFUL": 1, "UNKNOWN": 2,
            "TARGET_ENRICHED": 1, "SHARED_GENERAL": 1,
        }
        assert payload["causal_evidence_components"] == [
            "expert:3/17", "expert:3/18", "expert:3/19"]
        assert payload["artifact"] is None
        assert not (workspace / "graphs").exists()

    def test_graph_writes_artifact_and_it_round_trips(self, tmp_path):
        spec, attempts, enrichment = self._artifacts(tmp_path)
        workspace = tmp_path / "ws"
        code, payload = _run(
            "graph", "--config", str(spec),
            "--interventions", str(attempts),
            "--enrichment", str(enrichment),
            "--workspace", str(workspace))
        assert code == 0
        artifact = Path(payload["artifact"])
        assert artifact == workspace / "graphs" / "repo_repair_v1.json"
        graph = CausalComponentGraph.model_validate(
            json.loads(artifact.read_text(encoding="utf-8")))
        assert graph.source_revision == "abc123"
        roles = {node.component: node.role for node in graph.nodes}
        assert roles["expert:3/17"] == "REQUIRED"
        assert roles["expert:3/18"] == "NEGATIVE_OR_HARMFUL"
        assert roles["expert:3/21"] == "SHARED_GENERAL"

    def test_unrestorable_attempt_is_invalid_evidence_exit_3(self, tmp_path):
        spec, attempts, _ = self._artifacts(tmp_path)
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps(
            _attempt("expert:3/17", 0.5, 0.0, restored=False)) + "\n",
            encoding="utf-8")
        code, payload = _run("graph", "--config", str(spec),
                             "--interventions", str(bad))
        assert code == 3
        assert "ledger" in payload["error"]["message"]

    def test_missing_intervention_artifact_is_exit_3(self, tmp_path):
        spec, _, _ = self._artifacts(tmp_path)
        code, payload = _run("graph", "--config", str(spec),
                             "--interventions",
                             str(tmp_path / "never-recorded.jsonl"))
        assert code == 3

    def test_enrichment_only_needs_source_revision_exit_3_without_it(
            self, tmp_path):
        spec, attempts, _ = self._artifacts(tmp_path)
        enrichment = tmp_path / "enrichment.json"
        enrichment.write_text(json.dumps({
            "expert:3/20": {"target_cases": 30, "control_cases": 0}}),
            encoding="utf-8")
        code, payload = _run("graph", "--config", str(spec),
                             "--interventions", str(attempts),
                             "--enrichment", str(enrichment))
        # attempts carry a revision, so this succeeds; the refusal case
        # is the enrichment-ONLY graph:
        assert code == 0
        only_enrichment = tmp_path / "only-enrichment.jsonl"
        only_enrichment.write_text("", encoding="utf-8")
        code, payload = _run("graph", "--config", str(spec),
                             "--interventions", str(only_enrichment),
                             "--enrichment", str(enrichment))
        assert code == 3
        assert "source-revision" in payload["error"]["message"]

    def test_correlation_only_roles_never_become_causal(self, tmp_path):
        """A massively enriched component (correlation) can never hold
        REQUIRED -- only a verified intervention can (schema-enforced)."""
        spec, _, _ = self._artifacts(tmp_path)
        empty_attempts = tmp_path / "empty.jsonl"
        empty_attempts.write_text("", encoding="utf-8")
        enrichment = tmp_path / "enrichment.json"
        enrichment.write_text(json.dumps({
            "expert:3/20": {"target_cases": 10_000, "control_cases": 0}}),
            encoding="utf-8")
        code, payload = _run("graph", "--config", str(spec),
                             "--interventions", str(empty_attempts),
                             "--enrichment", str(enrichment),
                             "--source-revision", "abc123")
        assert code == 0
        assert payload["roles"] == {"TARGET_ENRICHED": 1}
        assert payload["causal_evidence_components"] == []


class TestPlanCommand:
    def _chain(self, tmp_path):
        """The full offline chain: source verify -> graph -> plan."""
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps(_spec_dict()), encoding="utf-8")
        checkpoint = _fixture_checkpoint(tmp_path)
        card = tmp_path / "card.json"
        card.write_text(json.dumps({
            "repository": "zai-org", "model": "GLM-5.3-Flash",
            "variant": "BF16", "commit_sha": "0" * 40, "license": "MIT",
        }), encoding="utf-8")
        workspace = tmp_path / "ws"
        code, verify_payload = _run(
            "source", "verify", "--checkpoint", str(checkpoint),
            "--source-card", str(card), "--workspace", str(workspace))
        assert code == 0
        attempts = tmp_path / "attempts.jsonl"
        records = [
            _attempt("expert:3/17", 0.8, 0.0),
            _attempt("expert:4/5", 0.6, 0.1, seed=11),
            _attempt("expert:3/18", 0.0, 0.9, seed=8),  # harmful: NOT retained
        ]
        attempts.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8")
        code, graph_payload = _run(
            "graph", "--config", str(spec),
            "--interventions", str(attempts),
            "--workspace", str(workspace))
        assert code == 0
        return (spec, workspace, Path(graph_payload["artifact"]),
                Path(verify_payload["artifact"]))

    def test_plan_retains_only_causally_required_experts(self, tmp_path):
        spec, workspace, graph_artifact, manifest_artifact = self._chain(tmp_path)
        code, payload = _run(
            "plan", "--config", str(spec),
            "--graph", str(graph_artifact),
            "--manifest", str(manifest_artifact),
            "--workspace", str(workspace))
        assert code == 0
        assert payload["ok"] is True
        assert payload["selection_arm"] == "causal"
        assert payload["retained_dense"] is True
        assert payload["retained_tokenizer"] is True
        assert payload["retained_attention"] is True
        assert payload["retained_head"] is True
        assert payload["layers_with_retained_experts"] == 2
        assert payload["retained_experts"] == 2
        plan = ExtractionPlan.model_validate(
            json.loads(Path(payload["artifact"]).read_text(encoding="utf-8")))
        retained = {entry.layer_id: entry.retained_experts
                    for entry in plan.expert_retention}
        assert retained == {3: [17], 4: [5]}

    def test_plan_without_required_components_is_exit_3(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps(_spec_dict()), encoding="utf-8")
        workspace = tmp_path / "ws"
        attempts = tmp_path / "attempts.jsonl"
        attempts.write_text(json.dumps(
            _attempt("expert:3/19", 0.0, 0.0)) + "\n", encoding="utf-8")
        code, graph_payload = _run(
            "graph", "--config", str(spec),
            "--interventions", str(attempts),
            "--workspace", str(workspace))
        assert code == 0
        checkpoint = _fixture_checkpoint(tmp_path)
        card = tmp_path / "card.json"
        card.write_text(json.dumps({
            "repository": "zai-org", "model": "GLM-5.3-Flash",
            "variant": "BF16", "commit_sha": "0" * 40, "license": "MIT",
        }), encoding="utf-8")
        code, verify_payload = _run(
            "source", "verify", "--checkpoint", str(checkpoint),
            "--source-card", str(card), "--workspace", str(workspace))
        assert code == 0
        code, payload = _run(
            "plan", "--config", str(spec),
            "--graph", str(graph_payload["artifact"]),
            "--manifest", str(verify_payload["artifact"]),
            "--workspace", str(workspace))
        assert code == 3
        assert "REQUIRED" in payload["error"]["message"]

    def test_plan_with_missing_graph_artifact_is_exit_3(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps(_spec_dict()), encoding="utf-8")
        code, payload = _run(
            "plan", "--config", str(spec),
            "--graph", str(tmp_path / "no-graph.json"),
            "--manifest", str(tmp_path / "no-manifest.json"))
        assert code == 3
        assert payload["status"] == "invalid_evidence"