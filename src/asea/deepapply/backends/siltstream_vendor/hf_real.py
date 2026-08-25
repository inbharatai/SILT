"""Real-model bridge: SiltStream / ZeroForge / SiltSpring on actual
HuggingFace transformers -- no toys, no mocks.

Design (deliberately different from the toy path, and MORE general):
instead of reimplementing an architecture, we stream the HF model's OWN
decoder layers. Each layer's weights are offloaded (param.data replaced by
an empty tensor, storage freed) and re-loaded from a disk bank inside a
pre-forward hook just before the layer executes, then freed again after.
The model's own forward code runs -- correctness by construction; parity
against the fully-resident run is measured, not assumed.

The same hook mechanism serves SiltSpring: the bank stores quantized
(int8/int4/int2) containers per layer; the pre-forward hook dequantizes one
layer at a time. Certificates compare each state's loss on real suites
against the full-precision reference.

ZeroForge runs on the real model by injecting LoRA wrappers around chosen
attention projections; only those tiny A/B matrices are perturbed --
forward-only, zero backward passes, exactly as in zeroforge.py.

Everything here is exercised by scripts/run_real_validation.py which writes
a measured report. Honest scope: validated on SmolLM2-135M (Llama-family)
CPU fp32 in this repo; other families must be re-validated before trust.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from .errors import StorageError, UnsupportedModelError
from .quant import dequantize_state, packed_bytes, quantize_state, state_bytes


# --------------------------------------------------------------------------
# Layer access
# --------------------------------------------------------------------------


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder layer stack on common HF causal LMs."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    raise UnsupportedModelError(
        f"cannot locate decoder layers on {type(model).__name__}; "
        "supported stacks: model.layers / transformer.h / gpt_neox.layers"
    )


# --------------------------------------------------------------------------
# Disk bank of real layer weights (full precision or quantized)
# --------------------------------------------------------------------------


class HFDiskBank:
    """Per-layer weight storage on disk; full fp32 or quantized containers."""

    def __init__(self, layers: nn.ModuleList, disk_dir: str, level: str = "full"):
        self.disk_dir = disk_dir
        self.level = level
        self.n_layers = len(layers)
        os.makedirs(disk_dir, exist_ok=True)
        self._bytes = 0
        self._bytes_packed = 0
        for i, layer in enumerate(layers):
            # Trainable adapter params (lora_*) are NEVER banked: the bank
            # holds only the frozen base. Banking trainables would overwrite
            # live learning on every streamed forward (bug B1, found by the
            # gate rejecting a run whose train loss was bit-frozen).
            state = {
                k: v.detach().cpu().clone()
                for k, v in layer.state_dict().items()
                if "lora_" not in k
            }
            if level == "full":
                torch.save(state, self._path(i))
                b = sum(v.numel() * v.element_size() for v in state.values())
                self._bytes += b
                self._bytes_packed += b
            else:
                q = quantize_state(state, level)
                torch.save(q, self._path(i))
                self._bytes += state_bytes(q)
                self._bytes_packed += packed_bytes(q)

    def _path(self, i: int) -> str:
        return os.path.join(self.disk_dir, f"{self.level}_layer_{i:04d}.pt")

    def load(self, i: int) -> Dict[str, torch.Tensor]:
        path = self._path(i)
        if not os.path.exists(path):
            raise StorageError(f"missing bank file {path}")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if self.level == "full":
            return obj
        return dequantize_state(obj)


# --------------------------------------------------------------------------
# Hook-based streaming executor
# --------------------------------------------------------------------------


class HFStreamer:
    """Streams a real HF model: decoder-layer weights live on disk and are
    materialized one layer at a time via pre-forward hooks."""

    def __init__(self, model: nn.Module, bank: HFDiskBank,
                 restore_bank: Optional[HFDiskBank] = None,
                 device: str = "cpu"):
        self.model = model
        self.layers = get_decoder_layers(model)
        if len(self.layers) != bank.n_layers:
            raise UnsupportedModelError("bank/model layer count mismatch")
        self.bank = bank
        # Restoration source: exiting a QUANTIZED streamer must restore the
        # FULL-precision weights (bug B2: restoring from the quantized bank
        # silently left the model compressed -- a spring that cannot
        # re-expand). Streaming a full bank may restore from itself.
        self.restore_bank = restore_bank if restore_bank is not None else bank
        if self.restore_bank.level != "full" and bank.level != "full":
            raise UnsupportedModelError(
                "streaming a quantized bank requires restore_bank at level "
                "'full' -- exiting must re-expand to full precision")
        # Compute device for the resident layer during streaming. The bank
        # always stores on CPU disk (see HFDiskBank); ``_load_layer`` moves each
        # re-materialized layer onto ``device`` just before it runs. ``"cpu"``
        # is the historical default and a no-op, so every existing CPU test
        # path is unchanged. ``"cuda"`` enables GPU streamed LoRA (and, with a
        # 4-bit base, QLoRA on a heavy model): only one decoder layer is
        # resident on the GPU at a time, so an 8 GB card can stream a 7B model.
        self.device = device
        self._handles: List = []
        self._offloaded = False

    # -- weight materialization ------------------------------------------------

    def _load_layer(self, i: int) -> None:
        state = self.bank.load(i)
        layer = self.layers[i]
        for name, param in layer.named_parameters():
            if "lora_" in name:
                continue  # live trainables are never loaded from the bank
            if name in state:
                # Move the banked tensor onto the compute device (the bank
                # stores on CPU disk). For 4-bit Params4bit the param dtype is
                # the uint8 quantized storage, so ``.to(dtype)`` is a no-op and
                # ``.to(device)`` places the bytes on the GPU; the param's
                # ``quant_state`` (the dequant scale/shape) survives the
                # free/reload intact -- empirically verified bitwise on
                # SmolLM2-135M 4-bit (scripts/probe_4bit_streamer.py).
                param.data = state[name].to(param.dtype).to(self.device)
        for name, buf in layer.named_buffers():
            if name in state:
                buf.data = state[name].to(self.device)

    def _free_layer(self, i: int) -> None:
        for name, param in self.layers[i].named_parameters():
            if "lora_" in name:
                continue  # trainables stay resident (they are tiny and LIVE)
            param.data = torch.empty(0, dtype=param.dtype)

    def offload_all(self) -> None:
        for i in range(len(self.layers)):
            self._free_layer(i)
        self._offloaded = True

    def restore_all(self) -> None:
        for i in range(len(self.layers)):
            state = self.restore_bank.load(i)
            layer = self.layers[i]
            for name, param in layer.named_parameters():
                if "lora_" in name:
                    continue
                if name in state:
                    param.data = state[name].to(param.dtype).to(self.device)
        self._offloaded = False

    def resident_layer_bytes(self) -> int:
        return sum(
            p.numel() * p.element_size() for l in self.layers for p in l.parameters()
        )

    # -- streamed execution ------------------------------------------------------

    def __enter__(self):
        self.offload_all()
        for i, layer in enumerate(self.layers):
            pre = layer.register_forward_pre_hook(
                lambda module, args, idx=i: self._load_layer(idx)
            )
            post = layer.register_forward_hook(
                lambda module, args, output, idx=i: self._free_layer(idx)
            )
            self._handles += [pre, post]
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.restore_all()
        return False


# --------------------------------------------------------------------------
# LoRA injection for real models (ZeroForge target)
# --------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_A, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # On a 4-bit (nf4) base the upstream hidden states are the compute dtype
        # (bfloat16), while the trainable LoRA A/B stay float32 for stable
        # gradients. Cast the LoRA branch to one dtype for the matmul, then back
        # to y.dtype for the residual add -- standard QLoRA discipline. Both the
        # streamed and the resident forward use THIS path, so the cast is
        # identical on both sides and forward parity is preserved bitwise. The
        # .to() casts are differentiable, so gradients still reach lora_A/B.
        xa = x.to(self.lora_A.dtype)
        delta = nn.functional.linear(nn.functional.linear(xa, self.lora_A), self.lora_B)
        return y + (delta * self.scaling).to(y.dtype)


def inject_lora(
    model: nn.Module,
    targets: Sequence[str] = ("q_proj", "v_proj"),
    last_n_layers: Optional[int] = None,
    rank: int = 8,
    alpha: float = 16.0,
) -> List[nn.Parameter]:
    layers = get_decoder_layers(model)
    chosen = layers if last_n_layers is None else layers[-last_n_layers:]
    params: List[nn.Parameter] = []
    for layer in chosen:
        for module in layer.modules():
            for attr in list(targets):
                child = getattr(module, attr, None)
                if isinstance(child, nn.Linear):
                    wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                    setattr(module, attr, wrapped)
                    params += [wrapped.lora_A, wrapped.lora_B]
    if not params:
        raise UnsupportedModelError(f"no LoRA targets {targets} found")
    return params


class HFZeroForgeTarget:
    """Duck-typed target for zeroforge.train_zeroforge on a REAL HF model."""

    def __init__(self, model: nn.Module, lora_params: List[nn.Parameter], name: str):
        self.model = model
        self._params = lora_params
        self.fingerprint = f"hf:{name}"

    def trainable_parameters(self) -> List[nn.Parameter]:
        return self._params

    def loss(self, input_ids: torch.Tensor, streamed: bool = False) -> torch.Tensor:
        out = self.model(input_ids=input_ids, labels=input_ids)
        return out.loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._params:
            p.grad = None

    def audit_metadata(self) -> Dict[str, object]:
        return {
            "component": "hf_zeroforge_target",
            "config_fingerprint": self.fingerprint,
            "n_lora_params": sum(p.numel() for p in self._params),
        }


# --------------------------------------------------------------------------
# Real-suite evaluation helpers
# --------------------------------------------------------------------------


def texts_to_batch(tokenizer, texts: Sequence[str], max_len: int = 96) -> torch.Tensor:
    enc = tokenizer(
        list(texts), return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_len,
    )
    ids = enc["input_ids"]
    return ids


@torch.no_grad()
def suite_loss(model: nn.Module, input_ids: torch.Tensor) -> float:
    return float(model(input_ids=input_ids, labels=input_ids).loss.item())


def certify_hf_states(
    model: nn.Module,
    layers: nn.ModuleList,
    suites: Dict[str, torch.Tensor],
    levels: Sequence[str],
    disk_dir: str,
    tolerance: float = 0.02,
) -> Dict[str, Dict[str, object]]:
    """SiltSpring on a real model: evaluate every quantization state on real
    suites vs the full-precision reference, streaming one layer at a time.

    A quantized streamer must re-expand to FULL precision on exit (vendor guard
    B2: a spring that cannot re-expand is a silent compression trap). So we bank
    the full-precision layers ONCE as the ``restore_bank`` and pass it to every
    quantized streamer; without it the streamer refuses (the previous code
    omitted it and the real-HF certification path always raised for any
    quantized level -- it had never actually run). ``device`` is inferred from
    the model so a cuda model streams layers onto the GPU (else the freed
    layers would land on CPU and break the forward against cuda embeddings)."""
    # Infer the compute device from the model's first parameter.
    try:
        device = next(model.parameters()).device
        device = str(device.type)
    except StopIteration:
        device = "cpu"

    reference = {name: suite_loss(model, b) for name, b in suites.items()}
    results: Dict[str, Dict[str, object]] = {
        "full": {
            "loss": reference,
            "degradation": {k: 0.0 for k in suites},
            "certified": sorted(suites),
            "revoked": [],
            "bytes_packed": None,
        }
    }
    # Bank the full-precision layers once; this is the re-expand source for every
    # quantized streamer (B2 guard).
    full_bank = HFDiskBank(layers, disk_dir=os.path.join(disk_dir, "full"), level="full")
    for level in levels:
        if level == "full":
            continue
        bank = HFDiskBank(layers, disk_dir=os.path.join(disk_dir, level), level=level)
        streamer = HFStreamer(model, bank, restore_bank=full_bank, device=device)
        with streamer:
            losses = {name: suite_loss(model, b) for name, b in suites.items()}
        degradation = {
            k: (losses[k] - reference[k]) / max(abs(reference[k]), 1e-12) for k in suites
        }
        certified = sorted(k for k, d in degradation.items() if d <= tolerance)
        revoked = sorted(k for k, d in degradation.items() if d > tolerance)
        results[level] = {
            "loss": losses,
            "degradation": degradation,
            "certified": certified,
            "revoked": revoked,
            "bytes_packed": bank._bytes_packed,
        }
    return results
