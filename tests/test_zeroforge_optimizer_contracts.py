"""Synthetic/tiny mechanics only: native ZeroForge, not model-quality evidence.

No pretrained weights, downloads, CUDA, substituted train function or backward.
The LM is two layers / 16 hidden; runs use at most 8 small steps. Tensor tests
isolate the numerical SPSA and rollback contracts with an actual torch loss.
"""
from __future__ import annotations

import math

import pytest

# This suite deliberately requires real CPU tensor execution; no skip fallback.
import torch

from asea.deepapply.backends.siltstream_vendor.config import ModelConfig, StreamConfig
from asea.deepapply.backends.siltstream_vendor.errors import SiltStreamError
from asea.deepapply.backends.siltstream_vendor.model import StreamedCausalLM
from asea.deepapply.backends.siltstream_vendor.zeroforge import train_zeroforge


@pytest.fixture(autouse=True)
def forward_only(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        pytest.fail("ZeroForge called autograd.backward")

    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    try:
        yield
    finally:
        torch.set_num_threads(old_threads)


def tiny_lm():
    return StreamedCausalLM(
        ModelConfig(vocab_size=16, n_layers=2, n_heads=2, d_model=16,
                    d_ff=32, max_seq_len=8, lora_rank=1, lora_alpha=2.0,
                    lora_targets=("v",)),
        StreamConfig(storage_tier="ram", compute_device="cpu", seed=47),
    )


def batches():
    return [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 4, 3, 2]]),
            torch.tensor([[3, 3, 2, 1]])]


def state(model):
    return {name: (p.detach().clone(), p.requires_grad)
            for name, p in model.named_parameters()}


def assert_state(model, before):
    for name, p in model.named_parameters():
        expected, flag = before[name]
        torch.testing.assert_close(p, expected, rtol=0, atol=0, equal_nan=True)
        assert p.requires_grad is flag, name


class TensorQuadratic(torch.nn.Module):
    """Real differentiable tensor objective; explicit param list like HF target."""
    fingerprint = "synthetic-quadratic-only"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.7, -0.4], dtype=torch.float64))
        self.offset = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64),
                                         requires_grad=False)
        self.calls = []

    def trainable_parameters(self):
        # HFZeroForgeTarget returns its explicit list, NOT a requires_grad filter.
        return [self.weight, self.offset]

    def loss(self, batch, streamed=True):
        self.calls.append(batch.clone())
        assert not torch.is_grad_enabled()
        return ((self.weight - batch[:2]) ** 2).sum() + (self.offset - batch[2:]).square().sum()

    def audit_metadata(self):
        return {"evidence": "synthetic/tiny mechanics"}


def test_synthetic_native_lm_updates_counts_cycle_and_frozen_base(monkeypatch):
    model = tiny_lm()
    # Keep one adapter deliberately frozen: trainable_parameters filters it out.
    next(model.lora.parameters()).requires_grad_(False)
    before = state(model)
    bank_before = [model.bank.fetch(i, torch.device("cpu")) for i in range(2)]
    data = batches()
    data_before = [b.clone() for b in data]
    seen = []
    actual_loss = model.loss

    def observe(batch, streamed=True):
        seen.append(batch.clone())
        assert not torch.is_grad_enabled()
        return actual_loss(batch, streamed=streamed)

    monkeypatch.setattr(model, "loss", observe)
    report = train_zeroforge(model, data, steps=5, n_directions=2, seed=13)
    assert report.forward_passes == len(seen) == 1 + 5 * (2 * 2 + 1)
    assert report.backward_passes == 0
    assert report.steps == 5 and len(report.loss_curve) == 6
    assert all(math.isfinite(x) for x in report.loss_curve)
    assert report.initial_loss == report.loss_curve[0]
    assert report.final_loss == report.loss_curve[-1]
    assert report.config_fingerprint == model.fingerprint
    assert report.audit == model.audit_metadata()
    expected_batches = [data[0]] + [data[s % 3] for s in range(5) for _ in range(5)]
    assert all(torch.equal(a, b) for a, b in zip(seen, expected_batches))
    assert any(not torch.equal(p, before[n][0]) for n, p in model.named_parameters()
               if before[n][1])
    for name, p in model.named_parameters():
        assert p.requires_grad == before[name][1]
        assert p.grad is None
        if not before[name][1]:
            assert torch.equal(p, before[name][0])
    for i in range(2):
        assert all(torch.equal(t, model.bank.fetch(i, torch.device("cpu"))[key])
                   for key, t in bank_before[i].items())
    assert all(torch.equal(a, b) for a, b in zip(data, data_before))


@pytest.mark.parametrize("streamed", [True, False])
def test_synthetic_one_step_independent_spsa_reference(streamed):
    model = tiny_lm()
    reference = tiny_lm()
    params = list(reference.trainable_parameters())
    originals = [p.detach().clone() for p in params]
    lr, eps, seed, n_directions = 0.05, 0.001, 123, 4
    updates = [torch.zeros_like(p) for p in params]
    with torch.no_grad():
        expected_initial = reference.loss(batches()[0], streamed=streamed).item()
        for direction in range(n_directions):
            # Independent RNG/formula, not the production _directions helper.
            generator = torch.Generator(device="cpu").manual_seed(seed + direction)
            vectors = [torch.randn(p.shape, generator=generator, dtype=p.dtype) for p in params]
            for p, original, vector in zip(params, originals, vectors):
                p.copy_(original)
                p.add_(vector, alpha=eps)
            plus = reference.loss(batches()[0], streamed=streamed).item()
            for p, original, vector in zip(params, originals, vectors):
                p.copy_(original)
                p.add_(vector, alpha=-eps)
            minus = reference.loss(batches()[0], streamed=streamed).item()
            coefficient = (plus - minus) / (2 * eps * n_directions)
            for update, vector in zip(updates, vectors):
                update.add_(vector, alpha=coefficient)
        expected = [original.add(update, alpha=-lr) for original, update in zip(originals, updates)]
        for p, value in zip(params, expected):
            p.copy_(value)
        expected_final = reference.loss(batches()[0], streamed=streamed).item()
    report = train_zeroforge(model, batches(), steps=1, streamed=streamed, seed=seed)
    for actual, value in zip(model.lora.parameters(), expected):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    assert report.loss_curve == [expected_initial, expected_final]
    assert report.forward_passes == 10 and report.backward_passes == 0


def test_synthetic_seed_reproducibility_and_rng_isolation():
    a, b, c = tiny_lm(), tiny_lm(), tiny_lm()
    rng_before = torch.random.get_rng_state().clone()
    ra = train_zeroforge(a, batches(), steps=3, seed=17)
    rb = train_zeroforge(b, batches(), steps=3, seed=17)
    rc = train_zeroforge(c, batches(), steps=3, seed=18)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert ra.loss_curve == rb.loss_curve
    assert all(torch.equal(x, y) for x, y in zip(a.lora.parameters(), b.lora.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(a.lora.parameters(), c.lora.parameters()))
    assert rc.forward_passes == ra.forward_passes


def test_synthetic_repeat_training_preserves_selection():
    model = tiny_lm()
    next(model.lora.parameters()).requires_grad_(False)
    train_zeroforge(model, batches(), steps=1)
    before = state(model)
    train_zeroforge(model, batches(), steps=1)
    assert any(not torch.equal(p, before[n][0]) for n, p in model.named_parameters())
    assert [p.requires_grad for p in model.parameters()] == [v[1] for v in before.values()]


def test_synthetic_quadratic_loss_decrease_not_quality_evidence():
    model = TensorQuadratic()
    before = state(model)
    report = train_zeroforge(model, [torch.zeros(3, dtype=torch.float64)],
                             steps=8, lr=0.05, eps=0.001, n_directions=4, seed=17)
    assert report.final_loss < report.initial_loss
    assert report.improved
    assert report.forward_passes == 1 + 8 * 9
    # Explicit target lists can contain False flags; don't blanket-enable them.
    assert model.weight.requires_grad is before["weight"][1]
    assert model.offset.requires_grad is before["offset"][1]


@pytest.mark.parametrize("at_call", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_synthetic_failure_or_cancellation_rolls_back_current_step(monkeypatch, at_call, error_type):
    model = tiny_lm()
    next(model.lora.parameters()).requires_grad_(False)
    # D=1: initial=1; step0=2,3,4; step1=5,6,7; step2+=8.
    completed = max(0, (at_call - 2) // 3)
    reference = tiny_lm()
    next(reference.lora.parameters()).requires_grad_(False)
    if completed:
        train_zeroforge(reference, batches(), steps=completed, n_directions=1, seed=11)
    expected = state(reference)
    # Before fix reference flags are corrupted; expected caller flags are known.
    for n, (_, flag) in state(model).items():
        expected[n] = (expected[n][0], flag)
    real_loss = model.loss
    calls = 0
    sentinel = error_type("synthetic interruption")

    def interrupt(batch, streamed=True):
        nonlocal calls
        calls += 1
        result = real_loss(batch, streamed=streamed)
        if calls == at_call:
            raise sentinel
        return result

    monkeypatch.setattr(model, "loss", interrupt)
    with pytest.raises(error_type) as caught:
        train_zeroforge(model, batches(), steps=3, n_directions=1, seed=11)
    assert caught.value is sentinel
    assert_state(model, expected)


@pytest.mark.parametrize("kwargs", [
    {"eps": 0}, {"eps": -0.001}, {"eps": float("nan")}, {"eps": float("inf")},
    {"eps": True}, {"eps": "bad"}, {"lr": float("nan")}, {"lr": float("inf")},
    {"lr": -0.01}, {"lr": True}, {"n_directions": 0}, {"n_directions": -1},
    {"n_directions": 1.5}, {"n_directions": True}, {"steps": -1},
    {"steps": 1.5}, {"steps": True}, {"seed": 1.5}, {"seed": True},
    {"seed": 2**64}, {"seed": -(2**63) - 1}, {"seed": 2**64 - 1},
])
def test_synthetic_invalid_settings_fail_typed_before_forward_or_mutation(kwargs, monkeypatch):
    model = tiny_lm()
    before = state(model)
    calls = []
    real_loss = model.loss

    def observe(*args, **kw):
        calls.append(1)
        return real_loss(*args, **kw)

    monkeypatch.setattr(model, "loss", observe)
    with pytest.raises(SiltStreamError):
        train_zeroforge(model, batches(), **dict({"steps": 2}, **kwargs))
    assert not calls
    assert_state(model, before)


def test_synthetic_empty_batches_fail_typed_without_mutation():
    model = tiny_lm()
    before = state(model)
    with pytest.raises(SiltStreamError):
        train_zeroforge(model, [], steps=1)
    assert_state(model, before)


@pytest.mark.parametrize("at_call", [1, 2, 3, 4])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_synthetic_nonfinite_loss_rejected_and_rolled_back(monkeypatch, at_call, bad):
    model = tiny_lm()
    before = state(model)
    real_loss = model.loss
    calls = 0

    def corrupt(batch, streamed=True):
        nonlocal calls
        calls += 1
        result = real_loss(batch, streamed=streamed)
        return result.new_tensor(bad) if calls == at_call else result

    monkeypatch.setattr(model, "loss", corrupt)
    with pytest.raises(SiltStreamError):
        train_zeroforge(model, batches(), steps=1, n_directions=1)
    assert_state(model, before)


def test_synthetic_finite_loss_but_overflowed_update_rolls_back(monkeypatch):
    model = TensorQuadratic()
    with torch.no_grad():
        model.weight.fill_(100)
        model.offset.fill_(100)
    before = state(model)
    real_loss = model.loss

    def clipped(batch, streamed=True):
        # The actual quadratic derivative is large enough that finite fp64 lr
        # overflows the parameter update, even if a loss implementation clips it.
        return torch.nan_to_num(real_loss(batch, streamed=streamed), nan=1.0)

    monkeypatch.setattr(model, "loss", clipped)
    with pytest.raises(SiltStreamError):
        train_zeroforge(model, [torch.zeros(3, dtype=torch.float64)],
                         steps=1, lr=1e308, eps=1e-3)
    assert_state(model, before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_synthetic_nonfinite_starting_parameter_rejected(monkeypatch, bad):
    model = tiny_lm()
    with torch.no_grad():
        next(model.lora.parameters()).view(-1)[0] = bad
    before = state(model)
    called = []
    real_loss = model.loss

    def observe(*args, **kwargs):
        called.append(1)
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(model, "loss", observe)
    with pytest.raises(SiltStreamError):
        train_zeroforge(model, batches(), steps=1)
    assert not called
    assert_state(model, before)


@pytest.mark.parametrize("steps,lr", [(0, 0.05), (2, 0.0)])
def test_synthetic_zero_steps_or_zero_lr_no_update(steps, lr):
    model = tiny_lm()
    before = state(model)
    report = train_zeroforge(model, batches(), steps=steps, lr=lr)
    assert report.forward_passes == 1 + steps * 9
    assert report.backward_passes == 0
    assert_state(model, before)
