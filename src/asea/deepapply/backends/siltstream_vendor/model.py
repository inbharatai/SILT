"""StreamedCausalLM -- a decoder-only causal LM whose blocks execute either
resident (all weights on the compute device for the whole pass) or streamed
(each block's frozen weights fetched from the LayerBank exactly when needed,
in forward AND in backward).

Streaming in backward is done with per-layer gradient checkpointing: the
forward keeps no intermediate activations for a block; the backward
re-fetches the block's weights from the bank and recomputes. Frozen weights
never require grad, so backward only produces grads for activations and the
resident LoRA parameters -- which is exactly what LoRA training needs.

Design guarantees:
- The SAME functional block code runs in both modes (see functional.py), so
  resident-vs-streamed parity is a test of the streaming machinery only.
- Dropout is deliberately absent (deterministic parity > regularization for
  a v1 trust component; document before adding).
- Embeddings, final norm and LM head stay resident in v1 (documented limit).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .bank import LayerBank
from .config import ModelConfig, StreamConfig, config_fingerprint
from .errors import BackendUnavailableError, UnsupportedModelError
from .functional import block_forward, init_block_state


class StreamedCausalLM(torch.nn.Module):
    def __init__(self, model_cfg: ModelConfig, stream_cfg: StreamConfig) -> None:
        super().__init__()
        if model_cfg.n_layers < 1:
            raise UnsupportedModelError("model must have at least one decoder layer")
        unknown = set(model_cfg.lora_targets) - {"q", "k", "v", "o", "fc1", "fc2"}
        if unknown:
            raise UnsupportedModelError(
                f"unsupported LoRA targets: {sorted(unknown)}; "
                "supported: q,k,v,o,fc1,fc2"
            )
        device = torch.device(stream_cfg.compute_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailableError(
                "compute_device='cuda' requested but torch.cuda.is_available() is False"
            )

        self.model_cfg = model_cfg
        self.stream_cfg = stream_cfg
        self.device_ = device
        self.scaling = model_cfg.lora_alpha / model_cfg.lora_rank
        self.fingerprint = config_fingerprint(model_cfg, stream_cfg)

        gen = torch.Generator().manual_seed(stream_cfg.seed)

        # Resident components -------------------------------------------------
        self.tok_emb = torch.nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.pos_emb = torch.nn.Embedding(model_cfg.max_seq_len, model_cfg.d_model)
        self.ln_f = torch.nn.LayerNorm(model_cfg.d_model)
        self.lm_head = torch.nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        with torch.no_grad():
            for p in (self.tok_emb.weight, self.pos_emb.weight, self.lm_head.weight):
                torch.nn.init.normal_(p, std=0.02, generator=gen)
        for p in self.parameters():
            p.requires_grad_(False)  # base is frozen; LoRA added below

        # Frozen block weights -> the bank ------------------------------------
        layer_states = [init_block_state(model_cfg, gen) for _ in range(model_cfg.n_layers)]
        self.bank = LayerBank(
            layer_states,
            storage_tier=stream_cfg.storage_tier,
            disk_dir=stream_cfg.disk_dir,
        )
        # Resident copy used ONLY by resident mode (parity reference). For a
        # real big model you would not build this; parity harnesses construct
        # it explicitly. Kept lazily: built on first resident call.
        self._resident_states: Optional[List[Dict[str, torch.Tensor]]] = None

        # LoRA parameters (trainable, resident, tiny) --------------------------
        self.lora = torch.nn.ModuleDict()
        for li in range(model_cfg.n_layers):
            layer_params = torch.nn.ParameterDict()
            for tgt in model_cfg.lora_targets:
                in_f = model_cfg.d_ff if tgt == "fc2" else model_cfg.d_model
                out_f = model_cfg.d_ff if tgt == "fc1" else model_cfg.d_model
                a = torch.nn.Parameter(torch.empty(model_cfg.lora_rank, in_f))
                b = torch.nn.Parameter(torch.zeros(out_f, model_cfg.lora_rank))
                torch.nn.init.normal_(a, std=0.02, generator=gen)
                layer_params[f"{tgt}_A"] = a
                layer_params[f"{tgt}_B"] = b
            self.lora[str(li)] = layer_params
        self.to(device)

    # -- helpers ---------------------------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.lora.parameters() if p.requires_grad]

    def _adapters_for(self, li: int) -> Dict[str, tuple]:
        params = self.lora[str(li)]
        return {
            tgt: (params[f"{tgt}_A"], params[f"{tgt}_B"])
            for tgt in self.model_cfg.lora_targets
        }

    def _ensure_resident_states(self) -> List[Dict[str, torch.Tensor]]:
        if self._resident_states is None:
            # fetch() returns isolated clones, so the resident reference can
            # never share memory with the bank (required for the parity
            # harness to detect bank corruption).
            self._resident_states = [
                self.bank.fetch(i, self.device_)
                for i in range(self.model_cfg.n_layers)
            ]
        return self._resident_states

    # -- forward ---------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, streamed: bool = True) -> torch.Tensor:
        bsz, seq = input_ids.shape
        if seq > self.model_cfg.max_seq_len:
            raise UnsupportedModelError(
                f"sequence length {seq} exceeds max_seq_len {self.model_cfg.max_seq_len}"
            )
        pos = torch.arange(seq, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]

        n_heads, scaling = self.model_cfg.n_heads, self.scaling
        if streamed:
            for li in range(self.model_cfg.n_layers):
                adapters = self._adapters_for(li)

                def run(x_in: torch.Tensor, _li: int = li, _ad=adapters) -> torch.Tensor:
                    # weights are re-fetched INSIDE the checkpointed function,
                    # so backward recompute streams the layer again.
                    w = self.bank.fetch(_li, self.device_)
                    return block_forward(x_in, w, _ad, n_heads, scaling)

                needs_grad = torch.is_grad_enabled() and (
                    x.requires_grad or any(p.requires_grad for p in self.lora[str(li)].values())
                )
                if needs_grad:
                    x = checkpoint(run, x, use_reentrant=False)
                else:
                    x = run(x)
        else:
            states = self._ensure_resident_states()
            for li in range(self.model_cfg.n_layers):
                x = block_forward(x, states[li], self._adapters_for(li), n_heads, scaling)

        x = self.ln_f(x)
        return F.linear(x, self.lm_head.weight)

    def loss(self, input_ids: torch.Tensor, streamed: bool = True) -> torch.Tensor:
        """Next-token cross-entropy over the sequence."""
        logits = self.forward(input_ids, streamed=streamed)
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.model_cfg.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )

    def audit_metadata(self) -> Dict[str, object]:
        return {
            "component": "siltstream",
            "config_fingerprint": self.fingerprint,
            "storage_tier": self.stream_cfg.storage_tier,
            "compute_device": str(self.device_),
            "seed": self.stream_cfg.seed,
            "n_layers": self.model_cfg.n_layers,
            "lora_targets": list(self.model_cfg.lora_targets),
            "bank_resident_bytes": self.bank.approx_bytes_resident(),
        }
