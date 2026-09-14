"""Numeric-admission regression tests, NOT model capability/quality evidence.

ScriptedLossModel executes a tiny local layer through the real disk-bank and
streamer, but deliberately returns synthetic scalar losses. No pretrained
weights, training, network, held-out selection, or benchmark claims are involved.
"""
from types import SimpleNamespace
import json
import math
import sys

import pytest

torch = pytest.importorskip("torch")

from asea.deepapply.backends.siltstream_vendor import hf_real
from asea.deepapply.backends.siltstream_vendor.errors import SiltStreamError
from asea.deepapply.backends.siltstream_vendor.spring import SpringModel


class ScriptedLossModel(torch.nn.Module):
    """Loss-output test double; local layer is real, scalar losses are NOT NLL."""

    def __init__(self, losses):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.losses = iter(losses)
        self.calls = 0

    def forward(self, input_ids, labels):
        self.model.layers[0](torch.ones(1, 2))
        self.calls += 1
        value = next(self.losses)
        dtype = torch.bool if isinstance(value, bool) else torch.float64
        return SimpleNamespace(loss=torch.tensor(value, dtype=dtype))


def hf_call(tmp_path, losses, tolerance=0.02, levels=("int8",), suites=None):
    model = ScriptedLossModel(losses)
    suites = {"synthetic": torch.zeros(1, 2, dtype=torch.long)} if suites is None else suites
    return hf_real.certify_hf_states(
        model, model.model.layers, suites, levels, str(tmp_path), tolerance=tolerance
    )


def toy_double(losses):
    """Exercise actual SpringModel.certify without constructing/training a model."""
    spring = SpringModel.__new__(SpringModel)
    spring.levels = ["full", "int8"]
    spring._qstates = {"int8": []}
    spring._bytes = {"full": 16, "int8": 4}
    spring._bytes_packed = dict(spring._bytes)
    spring.certificates = {}
    spring._certified_lora_fp = None
    spring.lora_fingerprint = lambda: "synthetic-fixed-fingerprint"
    values = iter(losses)
    def synthetic_loss(batch, state):
        value = next(values)
        if isinstance(value, bool):
            return torch.tensor(value, dtype=torch.bool)
        if isinstance(value, (int, float)):
            return torch.tensor(value, dtype=torch.float64)
        return value  # Deliberately malformed evaluator output for type guards.

    spring.loss = synthetic_loss
    return spring


NONFINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("value", NONFINITE + [True, False])
@pytest.mark.parametrize("position", ["reference", "candidate"])
def test_hf_rejects_invalid_loss(tmp_path, value, position):
    losses = [value, 1.0] if position == "reference" else [1.0, value]
    with pytest.raises(SiltStreamError, match="finite real"):
        hf_call(tmp_path, losses)


@pytest.mark.parametrize("value", NONFINITE + [True, False, None, "0.02", 1 + 0j])
def test_hf_rejects_invalid_tolerance_before_evaluation(tmp_path, value):
    model = ScriptedLossModel([])
    with pytest.raises(SiltStreamError, match="tolerance.*finite real"):
        hf_real.certify_hf_states(model, model.model.layers, {}, (), str(tmp_path), value)
    assert model.calls == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", NONFINITE + [True, False, None, "1.0", 1 + 0j])
def test_hf_validates_evaluator_output_before_coercion(tmp_path, monkeypatch, value):
    monkeypatch.setattr(hf_real, "suite_loss", lambda *args: value)
    with pytest.raises(SiltStreamError, match="finite real"):
        hf_call(tmp_path, [])


@pytest.mark.parametrize("suites", [[1.0], {True: None}, {1: None}, {None: None}])
def test_hf_rejects_malformed_suite_mapping(tmp_path, suites):
    with pytest.raises(SiltStreamError, match="suite"):
        hf_call(tmp_path, [1.0, 1.0], suites=suites)


@pytest.mark.parametrize("reference,candidate", [
    (-sys.float_info.max, sys.float_info.max),
    (sys.float_info.max, -sys.float_info.max),
    (0.0, sys.float_info.max),
    (-0.0, -sys.float_info.max),
])
def test_hf_rejects_overflow_in_delta_or_ratio(tmp_path, reference, candidate):
    with pytest.raises(SiltStreamError, match="finite real"):
        hf_call(tmp_path, [reference, candidate])


@pytest.mark.parametrize("level", ["int8", "int4", "int2"])
@pytest.mark.parametrize("reference,candidate,tolerance", [
    (1.0, 1.01, .02), (1.0, 1.03, .02), (1.0, 1.04, .05),
    (1.0, 1.06, .05), (1.0, .9, .02),
    (100.0, 102.0, .02), (100.0, 105.0, .05),
    (1.0, 1.0, -0.0), (1.0, 1.0, -0.01),
    (0.0, 0.0, .02), (-0.0, 0.0, .02),
    (0.0, -0.0, .02), (-0.0, -0.0, .02),
    (0.0, 1e-14, .02), (0.0, 3e-14, .02),
    (sys.float_info.max, sys.float_info.max, .02),
])
def test_hf_preserves_finite_comparison_and_reference_convention(
    tmp_path, reference, candidate, tolerance, level
):
    result = hf_call(tmp_path, [reference, candidate], tolerance, (level,))
    expected = (candidate - reference) / max(abs(reference), 1e-12)
    assert result["full"]["loss"] == {"synthetic": reference}
    assert result["full"]["degradation"] == {"synthetic": 0.0}
    assert result["full"]["certified"] == ["synthetic"]
    assert result["full"]["revoked"] == []
    assert result["full"]["bytes_packed"] is None
    state = result[level]
    assert state["degradation"] == {"synthetic": expected}
    assert state["certified"] == (["synthetic"] if expected <= tolerance else [])
    assert state["revoked"] == (["synthetic"] if expected > tolerance else [])
    assert state["bytes_packed"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", NONFINITE + [True, False, None, "1.0", 1 + 0j])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_toy_rejects_invalid_reference_full_and_quantized_loss(value, position):
    losses = [1.0, 1.0, 1.0]
    losses[position] = value
    spring = toy_double(losses)
    with pytest.raises(SiltStreamError, match="finite real"):
        spring.certify({"synthetic": None})
    assert spring.certificates == {}
    assert spring._certified_lora_fp is None


@pytest.mark.parametrize("value", NONFINITE + [True, False, None, "0.02"])
def test_toy_rejects_invalid_tolerance(value):
    spring = toy_double([])
    with pytest.raises(SiltStreamError, match="tolerance.*finite real"):
        spring.certify({"synthetic": None}, value)
    assert spring.certificates == {}


@pytest.mark.parametrize("reference,candidate", [
    (-sys.float_info.max, sys.float_info.max),
    (sys.float_info.max, -sys.float_info.max),
    (0.0, sys.float_info.max),
])
def test_toy_rejects_derived_overflow(reference, candidate):
    spring = toy_double([reference, reference, candidate])
    with pytest.raises(SiltStreamError, match="finite real"):
        spring.certify({"synthetic": None})
    assert spring.certificates == {}


@pytest.mark.parametrize("reference,candidate,tolerance", [
    (100.0, 102.0, .02), (100.0, 105.0, .05), (1.0, 1.06, .05),
    (1.0, .9, .02), (1.0, 1.0, -.01), (0.0, 0.0, .02),
    (-0.0, -0.0, .02), (0.0, 3e-14, .02),
])
def test_toy_preserves_finite_results(reference, candidate, tolerance):
    spring = toy_double([reference, reference, candidate])
    result = spring.certify({"synthetic": None}, tolerance)
    for level, loss in [("full", reference), ("int8", candidate)]:
        expected = (loss - reference) / max(abs(reference), 1e-12)
        assert result[level].relative_degradation == {"synthetic": expected}
        assert result[level].certified_skills == (["synthetic"] if expected <= tolerance else [])
        assert result[level].revoked_skills == (["synthetic"] if expected > tolerance else [])


def test_toy_failed_recertification_drops_old_and_partial_certificates():
    from asea.spring.certifier import CompressionCertifier

    spring = toy_double([1.0, 1.0, 1.0, 1.0, 1.0, float("-inf")])
    wrapper = CompressionCertifier(spring)
    wrapper.certify({"synthetic": None})
    assert wrapper.serve("int8", "synthetic") == "int8"
    with pytest.raises(SiltStreamError, match="finite real"):
        wrapper.certify({"synthetic": None})
    assert spring.certificates == {}
    assert spring._certified_lora_fp is None
    with pytest.raises(SiltStreamError):
        wrapper.serve("int8", "synthetic")
    with pytest.raises(SiltStreamError):
        spring.choose_state(100, ["synthetic"])


def test_hf_invalid_candidate_restores_full_weights(tmp_path):
    model = ScriptedLossModel([1.0, float("nan")])
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(SiltStreamError, match="finite real"):
        hf_real.certify_hf_states(model, model.model.layers, {"synthetic": None},
                                  ("int8",), str(tmp_path))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    layer = model.model.layers[0]
    assert not layer._forward_hooks
    assert not layer._forward_pre_hooks


def test_spring_job_propagates_actual_numeric_refusal_without_model_loading(tmp_path, monkeypatch):
    from asea.studio.spring_jobs import SpringJob

    job = SpringJob({}, tmp_path)
    # Only the loading boundary is replaced. The actual certifier raises and
    # SpringJob's real worker/status/SSE code must propagate its typed refusal.
    monkeypatch.setattr(job, "_execute", lambda: hf_call(tmp_path / "banks", [float("nan")]))
    job._run()
    assert job.status == "failed"
    assert job.error_type == "CertificationError"
    assert job.report is None
    assert job.to_dict()["error"] == "CertificationError"
    telemetry = "".join(job.telemetry())
    assert "job_failed" in telemetry
    assert "CertificationError" in telemetry
    assert "job_done" not in telemetry
    json.dumps(job.to_dict(), allow_nan=False)


@pytest.mark.parametrize("value", NONFINITE + [True, False])
@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_hf_checks_every_skill_in_reference_and_candidate_map(tmp_path, value, position):
    losses = [1.0, 1.0, 1.0, 1.0]
    losses[position] = value
    # A valid neighbouring skill must not mask an invalid value in a map/max.
    with pytest.raises(SiltStreamError, match="finite real"):
        hf_call(tmp_path, losses, suites={"first": None, "second": None})


@pytest.mark.parametrize("level", ["int8", "int4", "int2"])
@pytest.mark.parametrize("value", NONFINITE)
def test_hf_all_quantized_levels_fail_closed(tmp_path, level, value):
    with pytest.raises(SiltStreamError, match="finite real"):
        hf_call(tmp_path, [1.0, value], levels=(level,))


@pytest.mark.parametrize("value", NONFINITE + [True, False])
def test_hf_reference_only_still_requires_valid_loss(tmp_path, value):
    with pytest.raises(SiltStreamError, match="finite real"):
        hf_call(tmp_path, [value], levels=("full",))


@pytest.mark.parametrize("candidate", [math.nextafter(102.0, -math.inf),
                                        math.nextafter(102.0, math.inf)])
def test_hf_adjacent_finite_threshold_values_unchanged(tmp_path, candidate):
    result = hf_call(tmp_path, [100.0, candidate])
    expected = (candidate - 100.0) / 100.0
    assert result["int8"]["certified"] == (["synthetic"] if expected <= .02 else [])
    assert result["int8"]["revoked"] == (["synthetic"] if expected > .02 else [])


@pytest.mark.parametrize("suites", [[1.0], {True: None}, {1: None}, {None: None}])
def test_toy_rejects_malformed_suite_mapping(suites):
    with pytest.raises(SiltStreamError, match="suite"):
        toy_double([1.0, 1.0, 1.0]).certify(suites)


@pytest.mark.parametrize("value", [torch.tensor(True), torch.tensor(complex(1, 0)),
                                  torch.tensor([1.0, 2.0]), 10 ** 400])
def test_hf_rejects_invalid_scalar_kinds(tmp_path, monkeypatch, value):
    monkeypatch.setattr(hf_real, "suite_loss", lambda *args: value)
    with pytest.raises(SiltStreamError, match="finite real"):
        hf_call(tmp_path, [])


def test_spring_get_list_and_sse_routes_preserve_typed_failure(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from asea.studio import server
    from asea.studio.spring_jobs import SpringJob, SpringManager

    job = SpringJob({}, tmp_path)
    monkeypatch.setattr(job, "_execute", lambda: hf_call(tmp_path / "banks", [float("nan")]))
    job._run()
    manager = SpringManager()
    manager._jobs[job.job_id] = job
    monkeypatch.setattr(server, "spring_manager", manager)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    response = client.get("/api/spring/" + job.job_id)
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "CertificationError"
    assert "report" not in response.json()
    listing = client.get("/api/spring")
    assert listing.status_code == 200
    assert listing.json()["jobs"][0]["error"] == "CertificationError"
    sse = client.get("/api/spring/" + job.job_id + "/telemetry")
    assert sse.status_code == 200
    assert "job_failed" in sse.text and "CertificationError" in sse.text
    assert "job_done" not in sse.text and "spring_state" not in sse.text
