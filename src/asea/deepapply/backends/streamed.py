"""The ``streamed`` TrainerBackend -- siltstream-backed layer-streamed LoRA.

This is the canonical low-VRAM backend for deep-apply, backed by the vendored
``siltstream`` package (see :mod:`.siltstream_vendor`). It replaces the earlier
BETA placeholder ``StreamedTrainerBackend`` (which only ran on CUDA and was
never runtime-validated); that class is kept importable for test compatibility
but is no longer registered.

Binding rules enforced here (from the integration spec):

* **Parity is the admission bar.** Before any real training step, the exact
  configuration (model shape, storage tier, device) is parity-checked. A
  configuration whose parity is unverified is recorded as
  ``parity_verified=false`` in the artifact metadata; a parity FAILURE aborts
  the run with :class:`DeepApplyBlocked` (wrapping the vendor
  :class:`ParityError`) -- never a warning, never a fallback to the standard
  backend.
* **Receiver-architecture honesty.** siltstream v1 streams HF CausalLM stacks
  with a discoverable decoder layer list (``model.layers`` / ``transformer.h``
  / ``gpt_neox.layers``) via :func:`.siltstream_vendor.hf_real.get_decoder_layers`.
  For every other architecture the backend raises
  :class:`DeepApplyBlocked` naming :class:`UnsupportedModelError` and exactly
  what is unsupported. It does NOT pretend generic HF support. Mock / non-HF
  receivers (no ``model_id``) are refused with a named block.
* **Defense in depth.** The dataset is re-checked for mock contamination and
  emptiness BEFORE the receiver or any weight is touched -- a belt beside
  Gate 1's intake refusal.
* **Gate 2 is unchanged.** This backend produces an :class:`AdapterArtifact`
  judged identically to a standard one; the gate has zero backend-conditional
  branches.

CPU host: ``capabilities()`` reports ``cpu`` + ``ram``/``disk`` tiers. The
streamed backend runs on CPU for small models (the base is banked to a disk
tier and decoder layers are re-materialized one at a time via forward hooks;
HF gradient checkpointing makes backward re-fetch each layer, so LoRA grads
are correct without keeping the base resident). Requesting ``cuda`` on a host
without it raises :class:`DeepApplyBlocked` (no silent CPU fallback when the
caller explicitly asked for CUDA).
"""

from __future__ import annotations

import gc
import hashlib
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.interfaces import ModuleAdapter
from ..dataset import TrainingDataset
from ..errors import DeepApplyBlocked
from ..trainer import (
    AdapterArtifact,
    CPU_PARAM_CEILING,
    STREAMING_CREDIT,
    _AdaptedHFModule,
    _estimate_params,
    _fingerprint,
    _require_deep,
    _set_seeds,
)

#: Vendored siltstream version this backend is bound to.
SILTSTREAM_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parity_report_hash(meta: Dict[str, Any]) -> str:
    """Stable hash of the parity metadata block, for the audit chain + packet."""
    import json

    blob = json.dumps(meta, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _hf_config_fingerprint(
    model_id: str, arch: str, storage_tier: str, device: str, lora_config: Dict[str, Any],
    quant: str = "",
) -> str:
    """Stable short hash of the HF streamed configuration, for audit metadata.

    Mirrors siltstream's toy ``config_fingerprint`` (sha256[:16]) so a packet
    records exactly which configuration a parity check covered. ``quant`` is
    "" for a full-precision base or "nf4" for a 4-bit QLoRA base -- bound into
    the fingerprint because parity is configuration-specific (a 4-bit base is a
    different configuration from an fp16 one).
    """
    blob = "{}|{}|{}|{}|{}|{}".format(
        model_id, arch, storage_tier, device, repr(sorted(lora_config.items())), quant
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _assert_dataset_clean(dataset: TrainingDataset) -> None:
    """Defense in depth beside Gate 1: refuse mock-tainted or empty training data.

    Gate 1's intake (:func:`build_training_dataset`) already refuses non-PROMOTED
    and (under ``strict_no_mock``) mock-provenance packets. This re-checks the
    built dataset so a backend cannot be tricked into training on a mock or on
    nothing even if a future caller bypasses intake. Pure logic -- no torch.
    """
    if dataset is None:
        raise DeepApplyBlocked("streamed backend refused a None training dataset")
    if not dataset.rows:
        raise DeepApplyBlocked(
            "streamed backend refused an empty training dataset; only Gate-1-PROMOTED "
            "packets may enter training data and they yielded 0 rows"
        )
    if not dataset.source_packet_ids:
        raise DeepApplyBlocked(
            "streamed backend refused a training dataset with no source packet ids; "
            "training data must be traceable to Gate-1-PROMOTED packets"
        )
    if dataset.contains_mock:
        raise DeepApplyBlocked(
            "streamed backend refused a mock-contaminated training dataset "
            "(contains_mock=true); a mock cannot launder itself into weights -- "
            "defense in depth beside Gate 1"
        )


def _parity_metadata_from_toy(report) -> Dict[str, Any]:
    """Build the parity metadata block from a vendor :class:`ParityReport`
    (toy ``StreamedCausalLM`` path: full forward + backward parity, bitwise)."""
    return {
        "parity_verified": bool(report.passed),
        "forward_max_abs_diff": float(report.forward_max_abs_diff),
        "backward_max_abs_diff": float(report.backward_max_abs_diff),
        "forward_bitwise": bool(report.forward_bitwise),
        "backward_bitwise": bool(report.backward_bitwise),
        "device": str(report.device),
        "dtype": str(report.dtype),
        "tolerance": float(report.tolerance),
        "n_params_compared": int(report.n_params_compared),
        "config_fingerprint": str(report.config_fingerprint),
        "notes": list(report.notes),
    }


def _parity_metadata_from_hf(
    forward_max_abs_diff: float,
    config_fingerprint: str,
    device: str,
    dtype: str,
    storage_tier: str,
    tolerance: float,
) -> Dict[str, Any]:
    """Build the parity metadata block for the HF forward-parity path.

    Honest scope: the HF path verifies FORWARD parity (streamed logits vs
    resident logits, bitwise on CPU fp32). Backward LoRA-grad parity on a real
    HF model is not separately asserted here -- backward correctness rests on
    gradient-checkpoint recompute re-fetching each layer through the same hooks
    (construction), the same mechanism the toy harness proves bitwise. The
    notes record this so a trust layer reads "verified" honestly.
    """
    bitwise = forward_max_abs_diff == 0.0
    return {
        "parity_verified": bool(forward_max_abs_diff <= tolerance),
        "forward_max_abs_diff": float(forward_max_abs_diff),
        "backward_max_abs_diff": None,
        "forward_bitwise": bool(bitwise),
        "backward_bitwise": None,
        "device": str(device),
        "dtype": str(dtype),
        "tolerance": float(tolerance),
        "n_params_compared": 0,
        "config_fingerprint": str(config_fingerprint),
        "storage_tier": str(storage_tier),
        "notes": [
            "HF forward parity (streamed vs resident logits); "
            "backward LoRA-grad parity rests on gradient-checkpoint recompute "
            "(same hook path), not a separate bitwise assertion"
        ],
    }


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


class SiltStreamArtifact(AdapterArtifact):
    """Adapter artifact produced by the streamed backend.

    Carries the parity metadata block (recorded into the :class:`AdapterPacket`
    by the runner) alongside the standard adapter fields. ``attach`` reloads
    the base + the saved peft LoRA adapter and returns a real
    :class:`_AdaptedHFModule` (``is_mock=False``) for Gate 2 evaluation.
    """

    backend = "streamed"
    backend_version = "siltstream-{}".format(SILTSTREAM_VERSION)

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
        load_in_4bit: bool = False,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._capabilities = list(capabilities)
        self.lora_config = dict(lora_config)
        self.trainable_param_count = int(trainable_param_count)
        self.training_loss = float(training_loss)
        self.max_new_tokens = int(max_new_tokens)
        # Parity / streaming metadata -> AdapterPacket additions.
        self.parity = dict(parity)
        self.storage_tier = str(storage_tier)
        self.config_fingerprint = str(config_fingerprint)
        self.seed = int(seed)
        self.parity_verified = bool(self.parity.get("parity_verified", False))
        self.parity_report_hash = _parity_report_hash(self.parity)
        # Whether the base was loaded in 4-bit (QLoRA). ``attach`` MUST reload
        # the base the same way: a 7B adapter trained against a 4-bit base
        # cannot be evaluated against an fp16 base on an 8 GB card (14 GB ->
        # OOM), and the LoRA delta was calibrated to the 4-bit dequantized
        # forward, so the eval base must match the training base.
        self.load_in_4bit = bool(load_in_4bit)

    def attach(self, receiver: ModuleAdapter) -> ModuleAdapter:
        _require_deep()
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.load_in_4bit:
            # The LoRA was trained against a 4-bit (nf4) dequantized forward.
            # Reloading the base in fp16/fp32 on a CUDA-less eval host would
            # judge a *different* artifact (the logits differ by the propagated
            # quantization error) -- a silent fallback the binding rules forbid.
            # Refuse; never silently reload full precision.
            if device != "cuda":
                raise DeepApplyBlocked(
                    "a 4-bit-trained adapter cannot be attached without CUDA; "
                    "the LoRA delta was calibrated to the 4-bit dequantized forward "
                    "and an fp16/fp32 base would judge a different artifact. Refusing "
                    "to silently reload full precision. " + STREAMING_CREDIT
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
            base = AutoModelForCausalLM.from_pretrained(self.model_id)
            base = base.to(device)
        peft_model = PeftModel.from_pretrained(base, self.adapter_path)
        peft_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return _AdaptedHFModule(
            module_id="{}+lora-streamed".format(receiver.module_id),
            model_id=self.model_id,
            capabilities=self._capabilities,
            peft_model=peft_model,
            tokenizer=tokenizer,
            torch=torch,
            max_new_tokens=self.max_new_tokens,
        )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class SiltStreamBackend:
    """Layer-streamed LoRA training backed by the vendored siltstream package.

    Conforms to the :class:`TrainerBackend` ABC (``name``, ``version``,
    ``supports``, ``train``) and adds ``capabilities()`` and
    ``run_parity_check(...)`` -- the parity gate ``train`` runs before any
    weight is touched, also exposed for fast no-weights testing on the toy
    siltstream contract.
    """

    name = "streamed"
    version = "siltstream-{}".format(SILTSTREAM_VERSION)

    # ------------------------------------------------------------------ caps

    def capabilities(self) -> Dict[str, Any]:
        """Report what this backend can run on this host.

        CPU host -> ``cpu`` device + ``ram``/``disk`` tiers. CUDA detected at
        runtime if torch is importable and ``torch.cuda.is_available()``. If
        torch is missing, ``deep_extra_available`` is False and the device list
        is empty (the backend will raise :class:`DeepApplyBlocked` on train).
        """
        deep_ok = True
        devices: List[str] = []
        compute_device = "cpu"
        try:
            import torch  # noqa: F401
        except ImportError:
            deep_ok = False
        else:
            if torch.cuda.is_available():
                devices = ["cpu", "cuda"]
                compute_device = "cuda"
            else:
                devices = ["cpu"]
        return {
            "backend": self.name,
            "backend_version": self.version,
            "architectures": [
                "hf-causal-lm:model.layers",
                "hf-causal-lm:transformer.h",
                "hf-causal-lm:gpt_neox.layers",
                "siltstream-toy-block-contract",
            ],
            "storage_tiers": ["ram", "disk"],
            "devices": devices,
            "compute_device": compute_device,
            "deep_extra_available": deep_ok,
            "parity_required": True,
            "credit": STREAMING_CREDIT,
        }

    # ------------------------------------------------------------------ ABC

    def supports(self, receiver: ModuleAdapter) -> bool:
        try:
            _require_deep()
        except DeepApplyBlocked:
            return False
        return True

    # ----------------------------------------------------------- parity gate

    def _hf_forward_parity(
        self,
        model: Any,
        streamer: Any,
        sample_ids: Any,
        fingerprint: str,
        storage_tier: str,
        device: str,
        tolerance: float,
        audit: Any = None,
        session_id: str = "",
        adapter_id: str = "",
        actor: str = "deep_apply",
    ) -> Dict[str, Any]:
        """HF forward parity: streamed logits vs resident logits (bitwise on
        CPU fp32). Writes the ``parity_check`` audit event on success and
        failure; raises :class:`DeepApplyBlocked` on a parity mismatch. The
        caller supplies the bank-backed :class:`HFStreamer` (so ``train``
        banks once and reuses the same bank for parity and training)."""
        import torch

        from .siltstream_vendor import ParityError

        with torch.no_grad():
            resident = model(sample_ids).logits.detach().clone()
        try:
            with streamer:
                with torch.no_grad():
                    streamed = model(sample_ids).logits.detach().clone()
        except Exception as exc:  # bank/storage failure during streaming
            fail_meta = {
                "parity_verified": False,
                "config_fingerprint": fingerprint,
                "storage_tier": storage_tier,
                "device": device,
                "tolerance": tolerance,
                "error": type(exc).__name__,
                "reason": str(exc),
            }
            if audit is not None:
                audit.append(
                    "parity_check", actor=actor, session_id=session_id,
                    packet_id=adapter_id,
                    detail={"passed": False, "parity_report_hash": _parity_report_hash(fail_meta), **fail_meta},
                )
            raise DeepApplyBlocked(
                "streamed backend parity check could not stream '{}': {}".format(
                    fingerprint, exc
                )
            ) from exc

        fwd_diff = float((resident - streamed).abs().max().item())
        meta = _parity_metadata_from_hf(
            forward_max_abs_diff=fwd_diff,
            config_fingerprint=fingerprint,
            device=device,
            dtype=str(resident.dtype),
            storage_tier=storage_tier,
            tolerance=tolerance,
        )
        report_hash = _parity_report_hash(meta)
        if meta["parity_verified"]:
            if audit is not None:
                audit.append(
                    "parity_check", actor=actor, session_id=session_id,
                    packet_id=adapter_id,
                    detail={"passed": True, "parity_report_hash": report_hash, **meta},
                )
            return meta
        # Parity FAILED: audit then refuse.
        fail_meta = dict(meta)
        fail_meta["error"] = "ParityError"
        fail_meta["reason"] = (
            "forward max|diff|={:.3e} > tolerance {:.3e}".format(fwd_diff, tolerance)
        )
        if audit is not None:
            audit.append(
                "parity_check", actor=actor, session_id=session_id,
                packet_id=adapter_id,
                detail={"passed": False, "parity_report_hash": _parity_report_hash(fail_meta), **fail_meta},
            )
        raise DeepApplyBlocked(
            "streamed backend parity check FAILED for config '{}': forward "
            "max|diff|={:.3e} > tolerance {:.3e}. Run aborted -- the streamed "
            "backend refuses a configuration whose parity is unverified (never "
            "a warning, never a fallback). {}".format(
                fingerprint, fwd_diff, tolerance, STREAMING_CREDIT
            )
        ) from ParityError(fail_meta["reason"])

    def run_parity_check(
        self,
        model: Any,
        input_ids: Any,
        storage_tier: str = "ram",
        device: str = "cpu",
        tolerance: float = 0.0,
        audit: Any = None,
        session_id: str = "",
        adapter_id: str = "",
        actor: str = "deep_apply",
    ) -> Dict[str, Any]:
        """Parity gate. Returns the parity metadata block; writes a
        ``parity_check`` audit event (if ``audit`` given) on BOTH success and
        failure. On parity failure raises :class:`DeepApplyBlocked` wrapping
        the vendor :class:`ParityError` -- never a warning, never a fallback.

        Dispatches on the model type:

        * a vendor :class:`StreamedCausalLM` -> :func:`verify_parity` (full
          forward + backward bitwise parity; the validated siltstream path, no
          weights downloaded, <1s on CPU).
        * an HF CausalLM with a discoverable decoder layer list -> forward
          streamed-vs-resident parity via :class:`HFStreamer` (bitwise on CPU
          fp32); used by ``train()`` on real receivers.
        """
        _require_deep()
        from .siltstream_vendor import ParityError, UnsupportedModelError, verify_parity
        from .siltstream_vendor.hf_real import HFDiskBank, HFStreamer, get_decoder_layers

        meta: Optional[Dict[str, Any]] = None
        fingerprint = ""
        try:
            # Toy contract: full forward + backward parity.
            if hasattr(model, "fingerprint") and hasattr(model, "loss") and hasattr(
                model, "trainable_parameters"
            ):
                report = verify_parity(model, input_ids, tolerance=tolerance,
                                        raise_on_fail=True)
                meta = _parity_metadata_from_toy(report)
                fingerprint = str(report.config_fingerprint)
            else:
                # Real HF model: forward streamed-vs-resident parity via a
                # temporary bank (standalone path; train() reuses its own bank).
                import tempfile

                layers = get_decoder_layers(model)  # raises UnsupportedModelError
                arch = type(model).__name__
                fingerprint = _hf_config_fingerprint(
                    getattr(model, "name", arch), arch, storage_tier, device, {},
                )
                with tempfile.TemporaryDirectory() as tmpd:
                    bank = HFDiskBank(layers, disk_dir=tmpd, level="full")
                    streamer = HFStreamer(model, bank)
                    # _hf_forward_parity writes its own parity_check audit event
                    # (success and failure) and raises on mismatch; return early
                    # so the toy-path success-audit below is not duplicated.
                    return self._hf_forward_parity(
                        model, streamer, input_ids, fingerprint,
                        storage_tier, device, tolerance,
                        audit=audit, session_id=session_id, adapter_id=adapter_id, actor=actor,
                    )
        except ParityError as exc:
            # Parity FAILED: record the audit event, then refuse -- never warn.
            fail_meta = {
                "parity_verified": False,
                "config_fingerprint": fingerprint,
                "storage_tier": storage_tier,
                "device": device,
                "tolerance": tolerance,
                "error": "ParityError",
                "reason": str(exc),
            }
            if audit is not None:
                audit.append(
                    "parity_check",
                    actor=actor,
                    session_id=session_id,
                    packet_id=adapter_id,
                    detail={
                        "passed": False,
                        "parity_report_hash": _parity_report_hash(fail_meta),
                        **fail_meta,
                    },
                )
            raise DeepApplyBlocked(
                "streamed backend parity check FAILED for config '{}': {}. "
                "Run aborted -- the streamed backend refuses a configuration "
                "whose parity is unverified (never a warning, never a fallback). "
                "{}".format(fingerprint, exc, STREAMING_CREDIT)
            ) from exc
        except UnsupportedModelError as exc:
            if audit is not None:
                audit.append(
                    "parity_check",
                    actor=actor,
                    session_id=session_id,
                    packet_id=adapter_id,
                    detail={
                        "passed": False,
                        "error": "UnsupportedModelError",
                        "reason": str(exc),
                    },
                )
            raise DeepApplyBlocked(
                "streamed backend cannot parity-check {}: {}. ".format(
                    type(model).__name__, exc
                )
                + "Use the standard backend, or bigger hardware. " + STREAMING_CREDIT
            ) from exc

        # Success: record the audit event with the report hash.
        if meta is None:
            raise DeepApplyBlocked("streamed backend parity check produced no report")
        report_hash = _parity_report_hash(meta)
        if audit is not None:
            audit.append(
                "parity_check",
                actor=actor,
                session_id=session_id,
                packet_id=adapter_id,
                detail={"passed": True, "parity_report_hash": report_hash, **meta},
            )
        return meta

    # ------------------------------------------------------------- training

    def train(
        self,
        receiver: ModuleAdapter,
        dataset: TrainingDataset,
        config: Dict[str, Any],
        out_dir: Path,
    ) -> AdapterArtifact:
        # 1. defense in depth (pure logic, before any weight or receiver touch).
        _assert_dataset_clean(dataset)
        # 2. deps.
        _require_deep()
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from .siltstream_vendor import UnsupportedModelError
        from .siltstream_vendor.hf_real import HFDiskBank, HFStreamer, get_decoder_layers

        # 3. receiver-architecture honesty: the streamed backend trains a REAL
        #    HF CausalLM. A mock / non-HF receiver (no model_id) is refused with
        #    a named block -- never a silent fallback to a toy or to standard.
        model_id = getattr(receiver, "model_id", None)
        if not model_id:
            raise DeepApplyBlocked(
                "streamed backend needs a real HF receiver exposing model_id to stream "
                "and train; receiver '{}' is not one (no model_id). Use the standard "
                "backend for mock / non-HF receivers. {}".format(
                    getattr(receiver, "module_id", "?"), STREAMING_CREDIT
                )
            )
        model_id = str(model_id)

        try:
            cfg = AutoConfig.from_pretrained(model_id)
            arch = getattr(cfg, "model_type", "unknown")
        except Exception as exc:
            raise DeepApplyBlocked(
                "streamed backend could not read config for '{}': {}".format(model_id, exc)
            ) from exc

        # 4. hardware. The streamed backend runs on CPU for small models (disk
        #    tier) -- it is NOT pointless on CPU the way the BETA was, because
        #    the base is banked off-RAM-to-disk and re-materialized per layer.
        #    A big model on CPU is still BLOCKED (multi-day). CUDA is optional;
        #    explicitly requesting cuda on a host without it is a named block.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        requested_device = str(config.get("compute_device", device))
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise DeepApplyBlocked(
                "streamed backend was asked for cuda but no CUDA GPU is available; "
                "refusing to silently fall back to CPU. " + STREAMING_CREDIT
            )
        device = requested_device if requested_device in ("cpu", "cuda") else device
        # 4.5. optional 4-bit (QLoRA) base loading. bitsandbytes 4-bit compute is
        #       GPU-only, so a 4-bit request on a host without CUDA is a named
        #       block -- never a silent fp16/CPU fallback. With 4-bit the FULL
        #       model is resident on the GPU in ~0.6 bytes/param (a 7B base fits
        #       in ~4.5 GB), so the resident parity pass (which needs the whole
        #       model resident to compare streamed-vs-resident) fits in 8 GB --
        #       the reason 4-bit is required for a heavy model, not just nice to
        #       have. The streamer still banks each layer to disk and
        #       re-materializes one at a time on the GPU; LoRA grads flow via
        #       gradient-checkpoint recompute through the same hooks.
        load_in_4bit = bool(config.get("load_in_4bit", False))
        if load_in_4bit and device != "cuda":
            raise DeepApplyBlocked(
                "streamed backend was asked for 4-bit (QLoRA) training but no CUDA "
                "GPU is available; bitsandbytes 4-bit compute is GPU-only. Refusing "
                "to silently fall back. " + STREAMING_CREDIT
            )
        if device == "cpu":
            n_params = _estimate_params(cfg)
            ceiling = int(config.get("cpu_param_ceiling", CPU_PARAM_CEILING))
            if n_params > ceiling:
                raise DeepApplyBlocked(
                    "model '{}' has ~{:.1f}B params; streamed CPU training is BLOCKED. "
                    "Use a CUDA GPU, a smaller model, or the standard backend "
                    "(CPU ceiling: {:.1f}B). {}".format(
                        model_id, n_params / 1e9, ceiling / 1e9, STREAMING_CREDIT
                    )
                )

        # 5. LoRA config + seeds.
        lora_config = {
            "r": int(config.get("lora_rank", 8)),
            "lora_alpha": int(config.get("lora_alpha", 16)),
            "target_modules": list(config.get("target_modules") or ["q_proj", "v_proj"]),
            "lora_dropout": float(config.get("lora_dropout", 0.05)),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        seed = int(config.get("seed", 0))
        _set_seeds(seed)
        storage_tier = str(config.get("storage_tier", "disk"))

        # Audit plumbing (optional; the runner injects _audit so parity_check
        # events land in the same hash chain even on failure).
        audit_ctx = config.get("_audit") or {}
        audit = audit_ctx.get("log")
        session_id = str(audit_ctx.get("session_id", ""))
        adapter_id = str(audit_ctx.get("adapter_id", ""))
        actor = str(audit_ctx.get("actor", "deep_apply"))

        # 6. load base + tokenizer.
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

        # 7. architecture gate: locate the decoder layer list BEFORE banking.
        try:
            layers = get_decoder_layers(model)
        except UnsupportedModelError as exc:
            raise DeepApplyBlocked(
                "streamed backend does not support architecture '{}' for '{}': {}. "
                "It streams HF CausalLM stacks with a discoverable decoder layer "
                "list (model.layers / transformer.h / gpt_neox.layers). Use the "
                "standard backend, or bigger hardware. {}".format(
                    arch, model_id, exc, STREAMING_CREDIT
                )
            ) from exc

        # 8. inject LoRA (peft) BEFORE banking. peft wraps the target
        #    projections IN PLACE (q_proj -> q_proj.base_layer), which RENAMES
        #    the frozen base weights in ``layer.named_parameters()``
        #    (``self_attn.q_proj.weight`` -> ``self_attn.q_proj.base_layer.weight``).
        #    The disk bank and the streamer's restore are keyed by those names,
        #    so the bank MUST be captured after peft -- otherwise the streamer's
        #    name-keyed restore cannot find ``q_proj.base_layer.weight`` (the
        #    bank holds it under the old name ``q_proj.weight``), the freed base
        #    weight stays ``torch.empty(0)``, and the streamed forward/backward
        #    dies in ``F.linear`` with ``size mismatch ... vec (0)``. (Found by
        #    the real-AI e2e on SmolLM2-135M; the toy path never hit it because
        #    the vendor StreamedCausalLM is not peft-wrapped.)
        peft_model = get_peft_model(model, LoraConfig(**lora_config))
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        if trainable == 0:
            raise DeepApplyBlocked(
                "streamed LoRA produced 0 trainable params for '{}'; target_modules "
                "{} may not exist in this model".format(model_id, lora_config["target_modules"])
            )

        # 9. bank the frozen base (disk tier, POST-peft names) and build the
        #    streamer. The bank skips ``lora_*`` (vendor B1 guard) so only the
        #    frozen base is banked; the live LoRA A/B stay resident. ``layers``
        #    is the same ModuleList ref peft modified in place, so its
        #    ``state_dict()`` now yields ``q_proj.base_layer.weight`` -- matching
        #    what the streamer will restore.
        bank_dir = str(Path(out_dir) / "layer_bank")
        bank = HFDiskBank(layers, disk_dir=bank_dir, level="full")
        streamer = HFStreamer(model, bank, device=device)

        # 10. pre-train parity (forward streamed vs resident, bitwise on CPU
        #     fp32) using the SAME bank training will stream from. Run in EVAL
        #     mode so dropout is deterministic: with peft B=0 the LoRA path is
        #     zero, but base attention/hidden dropout in train mode would make
        #     the two passes differ and false-fail parity. On failure ->
        #     DeepApplyBlocked, no training, parity_check audited.
        sample_text = dataset.rows[0].get("input") or dataset.rows[0].get("output") or "x"
        sample_ids = tokenizer(str(sample_text), return_tensors="pt", truncation=True,
                               max_length=64).input_ids
        if device == "cuda":
            sample_ids = sample_ids.to(device)
        quant = "nf4" if load_in_4bit else ""
        fp = _hf_config_fingerprint(model_id, arch, storage_tier, device, lora_config, quant)
        peft_model.eval()
        parity_meta = self._hf_forward_parity(
            peft_model, streamer, sample_ids, fp,
            storage_tier, device, 0.0,
            audit=audit, session_id=session_id, adapter_id=adapter_id, actor=actor,
        )
        if not parity_meta.get("parity_verified"):
            # _hf_forward_parity would have raised already; belt-and-suspenders.
            raise DeepApplyBlocked(
                "streamed backend parity unverified for '{}'; refusing to train".format(
                    model_id
                )
            )

        # 11. enable gradient checkpointing so backward re-fetches each layer
        #     through the streamer hooks (genuine streamed backprop: the base
        #     stays banked, one layer resident at a time). use_reentrant=False
        #     so the forward hooks fire normally on BOTH the forward and the
        #     checkpoint recompute; enable_input_require_grads so the frozen
        #     base still has a grad-requiring input for the LoRA grads to flow
        #     through the checkpointed region.
        peft_model.train()
        try:
            peft_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except Exception:
            # Some configs do not support checkpointing; streamed backprop then
            # keeps activations resident (still correct, just more RAM). Not fatal.
            pass
        try:
            peft_model.enable_input_require_grads()
        except Exception:
            pass

        lr = float(config.get("learning_rate", 1e-4))
        max_steps_cap = int(config.get("max_steps_cap", 32))
        max_steps = min(int(config.get("max_steps", 16)), max_steps_cap)
        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad], lr=lr
        )

        last_loss = float("nan")
        rows = dataset.rows
        # Telemetry hook (optional, stashed in the train cfg under ``_on_step``
        # like ``_audit``). The Studio's live loss curve / step gauge reads REAL
        # per-step streamed losses from here -- never a fabricated number.
        on_step = config.get("_on_step")
        with streamer:
            for step in range(max_steps):
                row = rows[step % len(rows)]
                inp, outp = row.get("input"), row.get("output")
                if inp is None or outp is None:
                    continue
                enc = tokenizer(
                    "{}\n{}".format(inp, outp), return_tensors="pt", truncation=True,
                    max_length=256,
                )
                input_ids = enc["input_ids"]
                if device == "cuda":
                    input_ids = input_ids.to(device)
                in_enc = tokenizer(str(inp), return_tensors="pt", truncation=True,
                                   max_length=256)
                in_len = in_enc["input_ids"].shape[-1]
                labels = input_ids.clone()
                if in_len < labels.shape[-1]:
                    labels[:, :in_len] = -100
                out = peft_model(input_ids=input_ids, labels=labels)
                loss = out.loss
                if not torch.isfinite(loss):
                    last_loss = float("inf")
                    if on_step is not None:
                        on_step({"phase": "train_step", "backend": "streamed",
                                 "step": step + 1, "max_steps": max_steps,
                                 "loss": last_loss, "diverged": True})
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                if on_step is not None:
                    on_step({"phase": "train_step", "backend": "streamed",
                             "step": step + 1, "max_steps": max_steps,
                             "loss": last_loss})

        if not math.isfinite(last_loss):
            raise DeepApplyBlocked(
                "streamed training diverged (loss not finite) for '{}'; refusing to "
                "emit a broken adapter".format(model_id)
            )

        # 11. save the adapter (peft) and free. Release the streamer and bank
        # FIRST: HFStreamer holds ``self.model`` and HFDiskBank holds layer
        # references, so ``del model`` alone would not drop refcount to zero and
        # the 4-bit base (~5.6 GB for a 7B) would stay resident through Gate 2 --
        # where the evaluator loads baseline + candidate. Freeing here keeps the
        # Gate 2 peak to one candidate copy (sequential evaluator), not three.
        adapter_dir = Path(out_dir) / "adapter_model_streamed"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(adapter_dir))
        del streamer, bank, peft_model, model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        caps = list(receiver.manifest().capabilities)
        return SiltStreamArtifact(
            model_id=model_id,
            adapter_path=str(adapter_dir),
            capabilities=caps,
            lora_config=lora_config,
            trainable_param_count=int(trainable),
            training_loss=last_loss,
            max_new_tokens=int(config.get("max_new_tokens", 48)),
            parity=parity_meta,
            storage_tier=storage_tier,
            config_fingerprint=fp,
            seed=seed,
            load_in_4bit=load_in_4bit,
        )


# ---------------------------------------------------------------------------
# TrainerBackend protocol conformance -- SiltStreamBackend quacks like one but
# is defined here (not subclassing the ABC to avoid an import cycle through
# trainer.py, which imports from this package's parent). The registry in
# trainer.py treats it as a TrainerBackend.
# ---------------------------------------------------------------------------