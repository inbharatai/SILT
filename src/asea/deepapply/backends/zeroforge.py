"""The ``zeroforge`` TrainerBackend -- forward-only zeroth-order LoRA.

The second of siltstream's two trainers, integrated as a deep-apply backend
mode beside ``streamed``. ZeroForge estimates LoRA gradients from FORWARD
PASSES ONLY (central-difference SPSA, MeZO-spirit): perturb the LoRA params by
+/- eps along a random direction, measure the loss twice, step along the
direction scaled by the loss difference. No autograd graph, no backward, no
optimizer moments -- ``backward_passes == 0`` is recorded to make the claim
auditable. This is the trainer that can run on top of a quantized inference
engine (torch CPU today; GGUF/llama.cpp is the intended target).

Binding rules (same as the streamed backend):

* Parity is the admission bar: a pre-train forward-parity check runs before
  any perturbation; failure aborts with :class:`DeepApplyBlocked`.
* Receiver-architecture honesty: real HF CausalLM stacks with a discoverable
  decoder layer list only; everything else raises :class:`DeepApplyBlocked`
  naming :class:`UnsupportedModelError`. Mock / non-HF receivers refused.
* Defense in depth: the dataset is re-checked for mock contamination / emptiness
  before any weight is touched.
* Gate 2 is unchanged: a zeroforge adapter is judged identically to a standard
  or streamed one (zero backend-conditional branches in the gate).

Honest scope: ZeroForge is a STOCHASTIC method -- its gradients are noisy
estimates. It is designed to sit behind Gate 2, which judges the trained
artifact's outcome. The backend reports the full loss curve and
``forward_passes`` / ``backward_passes=0``; it never claims a clean gradient.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Dict, List

from ...core.interfaces import ModuleAdapter
from ..dataset import TrainingDataset
from ..errors import DeepApplyBlocked
from ..trainer import (
    AdapterArtifact,
    CPU_PARAM_CEILING,
    STREAMING_CREDIT,
    _AdaptedHFModule,
    _estimate_params,
    _require_deep,
    _set_seeds,
)
from .streamed import (
    SiltStreamBackend,
    _assert_dataset_clean,
    _hf_config_fingerprint,
    _parity_report_hash,
)

#: Reuse the streamed backend's version bound.
SILTSTREAM_VERSION = "0.1.0"


class ZeroForgeArtifact(AdapterArtifact):
    """Adapter artifact produced by the zeroforge backend.

    The LoRA is siltstream's own :class:`LoRALinear` (not peft): A/B matrices
    wrapped around the chosen attention projections. ``attach`` reloads the
    base, re-injects LoRA with the SAME config, loads the trained A/B, and
    returns a real :class:`_AdaptedHFModule` (``is_mock=False``).
    """

    backend = "zeroforge"
    backend_version = "siltstream-zeroforge-{}".format(SILTSTREAM_VERSION)

    def __init__(
        self,
        model_id: str,
        adapter_path: str,
        capabilities: List[Any],
        lora_config: Dict[str, Any],
        trainable_param_count: int,
        training_loss: float,
        max_new_tokens: int,
        parity: Dict[str, Any],
        storage_tier: str,
        config_fingerprint: str,
        seed: int,
        forward_passes: int,
        backward_passes: int,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._capabilities = list(capabilities)
        self.lora_config = dict(lora_config)
        self.trainable_param_count = int(trainable_param_count)
        self.training_loss = float(training_loss)
        self.max_new_tokens = int(max_new_tokens)
        self.parity = dict(parity)
        self.storage_tier = str(storage_tier)
        self.config_fingerprint = str(config_fingerprint)
        self.seed = int(seed)
        self.parity_verified = bool(self.parity.get("parity_verified", False))
        self.parity_report_hash = _parity_report_hash(self.parity)
        self.forward_passes = int(forward_passes)
        self.backward_passes = int(backward_passes)
        # Whether the base was loaded in 4-bit; ``attach`` must reload it the
        # same way (a 7B fp16 base is 14 GB -> OOM on an 8 GB card, and the
        # trained A/B were calibrated to the 4-bit dequantized forward).
        self.load_in_4bit = bool(load_in_4bit)

    def attach(self, receiver: ModuleAdapter) -> ModuleAdapter:
        _require_deep()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .siltstream_vendor.hf_real import inject_lora

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.load_in_4bit:
            # The trained A/B were calibrated to the 4-bit dequantized forward
            # (see class docstring). Reloading fp16/fp32 on a CUDA-less eval
            # host would judge a different artifact -- a silent fallback the
            # binding rules forbid. Refuse; never silently reload full precision.
            if device != "cuda":
                raise DeepApplyBlocked(
                    "a 4-bit-trained zeroforge adapter cannot be attached without "
                    "CUDA; the trained A/B were calibrated to the 4-bit dequantized "
                    "forward and an fp16/fp32 base would judge a different artifact. "
                    "Refusing to silently reload full precision. " + STREAMING_CREDIT
                )
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb_cfg
            ).to(device)
        else:
            base = AutoModelForCausalLM.from_pretrained(self.model_id).to(device)
        base.config.use_cache = True
        targets = tuple(self.lora_config.get("target_modules", ("q_proj", "v_proj")))
        inject_lora(
            base, targets=targets,
            last_n_layers=self.lora_config.get("last_n_layers"),
            rank=int(self.lora_config.get("r", 8)),
            alpha=float(self.lora_config.get("lora_alpha", 16.0)),
        )
        base = base.to(device)  # place freshly-injected LoRA A/B on the device
        state = torch.load(self.adapter_path, map_location=device, weights_only=True)
        # Restore the trained A/B into the freshly-injected LoRALinear modules.
        with torch.no_grad():
            for name, param in base.named_parameters():
                if "lora_" in name and name in state:
                    param.data.copy_(state[name].to(param.dtype))
        base.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return _AdaptedHFModule(
            module_id="{}+lora-zeroforge".format(receiver.module_id),
            model_id=self.model_id,
            capabilities=self._capabilities,
            peft_model=base,
            tokenizer=tokenizer,
            torch=torch,
            max_new_tokens=self.max_new_tokens,
        )


class ZeroForgeBackend:
    """Forward-only zeroth-order LoRA training (siltstream's ``train_zeroforge``).

    Conforms to the :class:`TrainerBackend` ABC. Reuses :class:`SiltStreamBackend`
    for ``capabilities()`` and the HF forward-parity gate so the two backends
    share one honesty seam.
    """

    name = "zeroforge"
    version = "siltstream-zeroforge-{}".format(SILTSTREAM_VERSION)

    def capabilities(self) -> Dict[str, Any]:
        caps = SiltStreamBackend().capabilities()
        caps["backend"] = self.name
        caps["backend_version"] = self.version
        caps["backward_passes"] = 0
        caps["method"] = "forward-only zeroth-order (central-difference SPSA)"
        return caps

    def supports(self, receiver: ModuleAdapter) -> bool:
        try:
            _require_deep()
        except DeepApplyBlocked:
            return False
        return True

    def train(
        self,
        receiver: ModuleAdapter,
        dataset: TrainingDataset,
        config: Dict[str, Any],
        out_dir: Path,
    ) -> AdapterArtifact:
        _assert_dataset_clean(dataset)
        _require_deep()
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from .siltstream_vendor import UnsupportedModelError, train_zeroforge
        from .siltstream_vendor.hf_real import (
            HFDiskBank,
            HFStreamer,
            HFZeroForgeTarget,
            get_decoder_layers,
            inject_lora,
        )

        # Receiver-architecture honesty: real HF CausalLM only.
        model_id = getattr(receiver, "model_id", None)
        if not model_id:
            raise DeepApplyBlocked(
                "zeroforge backend needs a real HF receiver exposing model_id; "
                "receiver '{}' is not one (no model_id). {}".format(
                    getattr(receiver, "module_id", "?"), STREAMING_CREDIT
                )
            )
        model_id = str(model_id)

        try:
            cfg = AutoConfig.from_pretrained(model_id)
            arch = getattr(cfg, "model_type", "unknown")
        except Exception as exc:
            raise DeepApplyBlocked(
                "zeroforge backend could not read config for '{}': {}".format(model_id, exc)
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            n_params = _estimate_params(cfg)
            ceiling = int(config.get("cpu_param_ceiling", CPU_PARAM_CEILING))
            if n_params > ceiling:
                raise DeepApplyBlocked(
                    "model '{}' has ~{:.1f}B params; zeroforge CPU training is BLOCKED. "
                    "Use a CUDA GPU or a smaller model (CPU ceiling: {:.1f}B). {}".format(
                        model_id, n_params / 1e9, ceiling / 1e9, STREAMING_CREDIT
                    )
                )
        # 4-bit (QLoRA-style) base loading -- GPU-only. With a 4-bit base a 7B
        # model fits in ~4.5 GB resident, so the forward-parity reference pass
        # (which needs the whole base resident) fits in 8 GB. zeroforge's own
        # inject_lora (LoRALinear) wraps a bnb Linear4bit just fine (Linear4bit
        # subclasses nn.Linear), so the forward-only perturbation path works on
        # a 4-bit base without peft.
        load_in_4bit = bool(config.get("load_in_4bit", False))
        if load_in_4bit and device != "cuda":
            raise DeepApplyBlocked(
                "zeroforge backend was asked for a 4-bit base but no CUDA GPU is "
                "available; bitsandbytes 4-bit compute is GPU-only. Refusing to "
                "silently fall back. " + STREAMING_CREDIT
            )

        rank = int(config.get("lora_rank", 8))
        alpha = float(config.get("lora_alpha", 16))
        targets = tuple(config.get("target_modules") or ("q_proj", "v_proj"))
        seed = int(config.get("seed", 0))
        _set_seeds(seed)
        storage_tier = str(config.get("storage_tier", "disk"))

        audit_ctx = config.get("_audit") or {}
        audit = audit_ctx.get("log")
        session_id = str(audit_ctx.get("session_id", ""))
        adapter_id = str(audit_ctx.get("adapter_id", ""))
        actor = str(audit_ctx.get("actor", "deep_apply"))

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_cfg)
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)
            if device == "cuda":
                model = model.to(device)
        model.config.use_cache = False

        try:
            layers = get_decoder_layers(model)
        except UnsupportedModelError as exc:
            raise DeepApplyBlocked(
                "zeroforge backend does not support architecture '{}' for '{}': {}. "
                "Use the standard backend, or bigger hardware. {}".format(
                    arch, model_id, exc, STREAMING_CREDIT
                )
            ) from exc

        # Inject siltstream LoRA (A/B wrappers) around the chosen projections.
        try:
            lora_params = inject_lora(
                model, targets=targets, last_n_layers=None, rank=rank, alpha=alpha,
            )
        except UnsupportedModelError as exc:
            raise DeepApplyBlocked(
                "zeroforge backend found no LoRA targets {} in '{}': {}. ".format(
                    targets, model_id, exc
                )
                + "Use the standard backend, or bigger hardware. " + STREAMING_CREDIT
            ) from exc

        # Place the freshly-injected LoRA A/B on the compute device (inject_lora
        # creates them on CPU; the base is already on device). No-op on CPU.
        model = model.to(device)
        bank_dir = str(Path(out_dir) / "layer_bank")
        bank = HFDiskBank(layers, disk_dir=bank_dir, level="full")
        streamer = HFStreamer(model, bank, device=device)

        # Pre-train forward parity (shared with the streamed backend).
        sample_text = dataset.rows[0].get("input") or dataset.rows[0].get("output") or "x"
        sample_ids = tokenizer(str(sample_text), return_tensors="pt", truncation=True,
                               max_length=64).input_ids
        if device == "cuda":
            sample_ids = sample_ids.to(device)
        quant = "nf4" if load_in_4bit else ""
        fp = _hf_config_fingerprint(model_id, arch, storage_tier, device,
                                    {"r": rank, "lora_alpha": alpha, "target_modules": list(targets)},
                                    quant)
        parity_helper = SiltStreamBackend()
        parity_meta = parity_helper._hf_forward_parity(
            model, streamer, sample_ids, fp,
            storage_tier, device, 0.0,
            audit=audit, session_id=session_id, adapter_id=adapter_id, actor=actor,
        )
        if not parity_meta.get("parity_verified"):
            raise DeepApplyBlocked(
                "zeroforge backend parity unverified for '{}'; refusing to train".format(
                    model_id
                )
            )

        # Tokenize dataset rows into forward-only batches.
        batches: List[torch.Tensor] = []
        for row in dataset.rows:
            inp, outp = row.get("input"), row.get("output")
            text = "{}\n{}".format(inp, outp) if (inp is not None and outp is not None) else str(inp or outp or "x")
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).input_ids
            if device == "cuda":
                ids = ids.to(device)
            batches.append(ids)
        if not batches:
            raise DeepApplyBlocked("zeroforge backend: dataset yielded 0 usable batches")

        target = HFZeroForgeTarget(model, lora_params, name=model_id)
        steps = min(int(config.get("max_steps", 20)), int(config.get("max_steps_cap", 64)))
        lr = float(config.get("learning_rate", 5e-2))
        eps = float(config.get("zeroforge_eps", 1e-3))
        n_directions = int(config.get("zeroforge_n_directions", 4))

        # Streamed forward-only training: the streamer's hooks materialize one
        # layer at a time during every forward (ZeroForge does many forwards).
        with streamer:
            report = train_zeroforge(
                target, batches, steps=steps, lr=lr, eps=eps,
                n_directions=n_directions, seed=seed, streamed=False,
            )

        if not math.isfinite(report.final_loss):
            raise DeepApplyBlocked(
                "zeroforge training diverged (loss not finite) for '{}'; refusing to "
                "emit a broken adapter".format(model_id)
            )

        # Save the trained A/B (siltstream LoRA, not peft).
        adapter_path = Path(out_dir) / "zeroforge_lora.pt"
        lora_state = {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if "lora_" in name
        }
        torch.save(lora_state, str(adapter_path))
        # Release the streamer/bank FIRST (they hold the model), then the model,
        # so the 4-bit base does not stay resident through Gate 2's sequential
        # baseline+candidate evaluation. See streamed.train for the rationale.
        del streamer, bank, target, model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        trainable = sum(p.numel() for p in lora_params)
        caps = list(receiver.manifest().capabilities)
        lora_config = {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": list(targets),
            "last_n_layers": None,
            "method": "zeroforge",
        }
        return ZeroForgeArtifact(
            model_id=model_id,
            adapter_path=str(adapter_path),
            capabilities=caps,
            lora_config=lora_config,
            trainable_param_count=int(trainable),
            training_loss=float(report.final_loss),
            max_new_tokens=int(config.get("max_new_tokens", 48)),
            parity=parity_meta,
            storage_tier=storage_tier,
            config_fingerprint=fp,
            seed=seed,
            forward_passes=int(report.forward_passes),
            backward_passes=int(report.backward_passes),
            load_in_4bit=load_in_4bit,
        )