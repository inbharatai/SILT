"""Streamer state correctness on tiny local tensors, NOT pretrained/quality evidence.

Actual HFDiskBank/quantization/hooks/certification are exercised on CPU. The CUDA
case is optional. Nonpersistent runtime caches are intentionally outside rollback.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from asea.deepapply.backends.siltstream_vendor.hf_real import (
    HFDiskBank, HFStreamer, certify_hf_states,
)
from asea.deepapply.backends.siltstream_vendor.errors import UnsupportedModelError
from asea.deepapply.backends.siltstream_vendor.spring import CertificationError


class TinyTensorLayer(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.231, .917], [.347, .719]]))
        self.tied_weight = self.weight
        self.register_buffer("matrix", torch.tensor([[.137, .823], [.479, .923]], dtype=dtype))
        self.register_buffer("tied_matrix", self.matrix)
        self.nested = torch.nn.Module()
        self.nested.register_buffer("offset", torch.tensor([.173, .619], dtype=dtype))
        self.register_buffer("counter", torch.tensor(16777219, dtype=torch.int64))
        self.register_buffer("cache", torch.zeros(256, 256), persistent=False)
        self.lora_live = torch.nn.Parameter(torch.tensor([.123, .456]))
        self.seen = []
        self.raise_in_forward = False

    def forward(self, x):
        self.seen.append((self.matrix.detach().clone(), self.weight.detach().clone()))
        self.cache.add_(1)  # Runtime cache mutation is not a banked-state change.
        if self.raise_in_forward:
            raise RuntimeError("synthetic layer failure")
        return (torch.nn.functional.linear(x, self.weight)
                + x @ self.matrix.to(x.dtype)
                + self.nested.offset.to(x.dtype))


class TinyTensorModule(torch.nn.Module):
    def __init__(self, dtype=torch.float64, invalid_candidate=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyTensorLayer(dtype)])
        # Layer-local ties and aliases outside the streamed stack stay the SAME
        # Parameter/Tensor objects: restoration must not replace registrations.
        self.external_weight = self.model.layers[0].weight
        self.external_matrix = self.model.layers[0].matrix
        self.calls = 0
        self.invalid_candidate = invalid_candidate

    def forward(self, input_ids=None, labels=None):
        layer = self.model.layers[0]
        y = layer(torch.ones(1, 2, device=layer.lora_live.device, dtype=layer.weight.dtype))
        self.calls += 1
        loss = y.square().mean()
        if self.invalid_candidate and self.calls == 2:
            loss = torch.tensor(float("nan"), device=loss.device)
        return SimpleNamespace(loss=loss)


def snapshot(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


def assert_restored(model, original, weight, matrix):
    after = model.state_dict()
    assert after.keys() == original.keys()
    for name, expected in original.items():
        assert after[name].dtype == expected.dtype, name
        assert after[name].device == expected.device, name
        assert torch.equal(after[name], expected), name
    layer = model.model.layers[0]
    assert layer.weight is layer.tied_weight is model.external_weight is weight
    assert layer.matrix is layer.tied_matrix is model.external_matrix is matrix
    assert not layer._forward_hooks
    assert not layer._forward_pre_hooks


def expected_loss(state):
    x = torch.ones(1, 2)
    y = (torch.nn.functional.linear(x, state["weight"].float())
         + x @ state["matrix"].float() + state["nested.offset"].float())
    return y.square().mean().item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("level", ["int8", "int4", "int2"])
@pytest.mark.parametrize("invalid_candidate", [False, True])
def test_certification_restores_persistent_state(tmp_path, dtype, level, invalid_candidate):
    model = TinyTensorModule(dtype, invalid_candidate)
    layer = model.model.layers[0]
    original = snapshot(model)
    weight, matrix = layer.weight, layer.matrix
    # Independent expected finite score from the actual banked tensors, not a
    # mocked evaluator. Dtype conversions match the tiny layer's forward only.
    expected_bank = HFDiskBank(model.model.layers, str(tmp_path / "expected"), level)
    quantized = expected_bank.load(0)
    reference = expected_loss(layer.state_dict())
    candidate = expected_loss(quantized)
    assert not torch.equal(quantized["matrix"].double(), original["model.layers.0.matrix"].double())
    if invalid_candidate:
        with pytest.raises(CertificationError, match="finite real"):
            certify_hf_states(model, model.model.layers, {"tiny": None},
                              (level,), str(tmp_path / "cert"))
    else:
        result = certify_hf_states(model, model.model.layers, {"tiny": None},
                                   (level,), str(tmp_path / "cert"))
        degradation = (candidate - reference) / max(abs(reference), 1e-12)
        assert result["full"]["loss"] == {"tiny": reference}
        assert result[level]["loss"] == {"tiny": candidate}
        assert result[level]["degradation"] == {"tiny": degradation}
        assert result[level]["certified"] == (["tiny"] if degradation <= .02 else [])
        assert result[level]["revoked"] == (["tiny"] if degradation > .02 else [])
    assert model.calls == 2
    assert torch.equal(layer.seen[1][0], quantized["matrix"])
    assert_restored(model, original, weight, matrix)


def test_successive_quantized_candidates_start_from_original_full_state(tmp_path):
    model = TinyTensorModule()
    layer = model.model.layers[0]
    original = snapshot(model)
    expected = {
        level: HFDiskBank(model.model.layers, str(tmp_path / ("expected-" + level)), level).load(0)
        for level in ("int8", "int4", "int2")
    }
    result = certify_hf_states(model, model.model.layers, {"tiny": None},
                               ("int8", "int4", "int2"), str(tmp_path / "cert"))
    for index, (level, state) in enumerate(expected.items(), 1):
        assert torch.equal(layer.seen[index][0], state["matrix"])
        assert result[level]["loss"] == {"tiny": expected_loss(state)}
    assert_restored(model, original, layer.weight, layer.matrix)


@pytest.mark.parametrize("raises", [False, True])
def test_direct_streamer_full_bank_restore_and_live_cache_scope(tmp_path, monkeypatch, raises):
    model = TinyTensorModule()
    layer = model.model.layers[0]
    original = snapshot(model)
    weight, matrix, cache, adapter = layer.weight, layer.matrix, layer.cache, layer.lora_live
    cache_storage = cache.data_ptr()
    original_clone = torch.Tensor.clone

    def no_cache_clone(tensor, *args, **kwargs):
        assert tensor.data_ptr() != cache_storage, "nonpersistent cache must not be snapshotted"
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", no_cache_clone)
    full = HFDiskBank(model.model.layers, str(tmp_path / "full"))
    bank = HFDiskBank(model.model.layers, str(tmp_path / "quant"), "int4")
    assert "cache" not in full.load(0)
    assert "lora_live" not in full.load(0)
    streamer = HFStreamer(model, bank, restore_bank=full)
    layer.raise_in_forward = raises
    try:
        with streamer:
            with torch.no_grad():
                adapter.add_(1)
            model()
    except RuntimeError as exc:
        assert raises and str(exc) == "synthetic layer failure"
    else:
        assert not raises
    # Live adapters and nonpersistent caches intentionally survive runtime edits.
    original["model.layers.0.lora_live"].add_(1)
    assert_restored(model, original, weight, matrix)
    assert layer.lora_live is adapter
    assert layer.cache is cache and cache.data_ptr() == cache_storage
    assert torch.equal(cache, torch.ones_like(cache))
    assert not streamer._handles and not streamer._offloaded
    # Repeat restore must not use the already-quantized current buffer dtype.
    streamer.restore_all()
    assert_restored(model, original, weight, matrix)


def test_restore_bank_must_be_full_even_when_streaming_full(tmp_path):
    model = TinyTensorModule()
    full = HFDiskBank(model.model.layers, str(tmp_path / "full"))
    quant = HFDiskBank(model.model.layers, str(tmp_path / "quant"), "int8")
    with pytest.raises(UnsupportedModelError, match="restore_bank.*full"):
        HFStreamer(model, full, restore_bank=quant)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("level", ["full", "int4"])
def test_native_parameter_dtype_and_repeated_context_restoration(tmp_path, dtype, level):
    model = TinyTensorModule(dtype)
    layer = model.model.layers[0]
    layer.weight.data = layer.weight.to(dtype)
    original = snapshot(model)
    weight, matrix = layer.weight, layer.matrix
    full = HFDiskBank(model.model.layers, str(tmp_path / "full"))
    assert full.load(0)["weight"].dtype == dtype
    assert full.load(0)["matrix"].dtype == dtype
    bank = full if level == "full" else HFDiskBank(model.model.layers, str(tmp_path / level), level)
    streamer = HFStreamer(model, bank, restore_bank=full)
    for _ in range(2):
        with streamer:
            model()
        assert_restored(model, original, weight, matrix)


@pytest.mark.parametrize("raises", [False, True])
def test_cross_layer_object_ties_survive_restoration(tmp_path, raises):
    model = TinyTensorModule()
    first = model.model.layers[0]
    second = TinyTensorLayer()
    second.weight = second.tied_weight = first.weight
    second.matrix = second.tied_matrix = first.matrix
    model.model.layers.append(second)
    original = snapshot(model)
    weight, matrix = first.weight, first.matrix
    full = HFDiskBank(model.model.layers, str(tmp_path / "full"))
    quant = HFDiskBank(model.model.layers, str(tmp_path / "quant"), "int4")
    streamer = HFStreamer(model, quant, restore_bank=full)
    second.raise_in_forward = raises
    try:
        with streamer:
            first(torch.ones(1, 2))
            second(torch.ones(1, 2))
    except RuntimeError as exc:
        assert raises and str(exc) == "synthetic layer failure"
    else:
        assert not raises
    assert_restored(model, original, weight, matrix)
    assert second.weight is second.tied_weight is weight
    assert second.matrix is second.tied_matrix is matrix
    assert not second._forward_hooks and not second._forward_pre_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("compute", ["cpu", "cuda"])
def test_restore_original_devices_not_compute_device(tmp_path, raises, compute):
    # Deliberately heterogeneous original placement; no pretrained model loaded.
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    model = TinyTensorModule().to(device)
    layer = model.model.layers[0]
    layer.matrix.data = layer.matrix.cpu()
    layer.nested.offset.data = layer.nested.offset.cpu()
    # Module.to may replace registered buffers, so establish the intended
    # aliases AFTER preparing placement, before the streamer owns any state.
    layer.tied_matrix = layer.matrix
    model.external_matrix = layer.matrix
    original = snapshot(model)
    weight, matrix = layer.weight, layer.matrix
    full = HFDiskBank(model.model.layers, str(tmp_path / "full"))
    quant = HFDiskBank(model.model.layers, str(tmp_path / "quant"), "int4")
    compute_device = str(device) if compute == "cuda" else "cpu"
    streamer = HFStreamer(model, quant, restore_bank=full, device=compute_device)
    layer.raise_in_forward = raises
    try:
        with streamer:
            layer(torch.ones(1, 2, device=compute_device))
    except RuntimeError as exc:
        assert raises and str(exc) == "synthetic layer failure"
    else:
        assert not raises
    assert_restored(model, original, weight, matrix)
