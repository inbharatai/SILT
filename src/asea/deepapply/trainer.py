"""Trainer backends for deep-apply.

Two real implementations behind a pluggable :class:`TrainerBackend`:

* :class:`StandardTrainerBackend` -- model resident on device; the default and
  the path that can run on CPU for small models.
* :class:`StreamedTrainerBackend` -- low-VRAM training in the style of **Soup**
  (github.com/MakazhanAlpamys/Soup, Apache-2.0): the frozen base is kept in host
  RAM and decoder layers are streamed to the GPU one at a time, with LoRA params
  resident on device. Their published result is an 8B LoRA train in ~3.3 GB VRAM.

Prior-art honesty (binding): layer streaming/offloading is third-party published
work, credited here and in NOTICE, never claimed as ours. The backend used and
its version are recorded in :class:`AdapterPacket`.

Every backend is lazy: ``torch``/``transformers``/``peft`` are imported inside
methods, so importing this module on a machine without them costs nothing. If
deep-apply is actually invoked without the ``[deep]`` extra, the backend raises
the named :class:`DeepApplyBlocked` error -- never a silent mock, never a fake
training log.

**Gate 2 treats every backend identically.** The trainer produces an
:class:`AdapterArtifact`; the evaluator attaches it and measures the outcome.
The gate never sees which backend ran.
"""

from __future__ import annotations

import abc
import gc
import hashlib
import math
import random
from pathlib import Path
from typing import Any, Dict, List

from ..core.interfaces import ModuleAdapter
from ..core.protocol import CapabilityKey, CapabilityManifest, LearningLevel
from .dataset import TrainingDataset
from .errors import DeepApplyBlocked

# ---------------------------------------------------------------------------
# Attribution (Apache-2.0). The streamed technique is third-party work.
# ---------------------------------------------------------------------------
STREAMING_CREDIT = (
    "Layer-streamed low-VRAM LoRA training in the style of Soup "
    "(github.com/MakazhanAlpamys/Soup, Apache-2.0). Technique credited, not "
    "claimed as original work of this project."
)

#: Models above this parameter count are BLOCKED on CPU (training would take
#: days and is not what "CPU-graceful" means). Small models (e.g. SmolLM2-135M)
#: sit well under this and may train on CPU. Override via config.
CPU_PARAM_CEILING = 1_500_000_000


def _require_deep(extra_hint: str = "pip install -e '.[deep]'") -> None:
    """Raise DeepApplyBlocked if the [deep] training deps are missing.

    Never silently degrades to a mock. Called at the top of every real backend.
    """
    missing: List[str] = []
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise DeepApplyBlocked(
            "deep-apply needs the [deep] extra (missing: {}). Install with:\n  {}".format(
                ", ".join(missing), extra_hint
            )
        )


def _set_seeds(seed: int) -> None:
    import torch  # lazy

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_id_of(receiver: ModuleAdapter) -> str:
    mid = getattr(receiver, "model_id", None)
    if not mid:
        # Fall back to the manifest version (real connectors expose it there).
        try:
            mid = receiver.manifest().version
        except Exception:
            mid = None
    if not mid:
        raise DeepApplyBlocked(
            "cannot determine base model id from receiver '{}'; deep-apply needs a "
            "real connector exposing model_id".format(receiver.module_id)
        )
    return str(mid)


def _fingerprint(base_model: str, lora_config: Dict[str, Any]) -> str:
    blob = "{}|{}".format(base_model, repr(sorted(lora_config.items())))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _estimate_params(cfg) -> int:
    """Rough param-count estimate from a transformers config, without weights."""
    n_params = getattr(cfg, "num_params", None)
    if n_params:
        return int(n_params)
    n_layers = getattr(cfg, "num_hidden_layers", 0) or 0
    hidden = getattr(cfg, "hidden_size", 0) or 0
    vocab = getattr(cfg, "vocab_size", 0) or 0
    return n_layers * hidden * hidden * 12 + vocab * hidden


class AdapterArtifact(abc.ABC):
    """The thing a trainer returns. Carries metadata + an ``attach`` that
    produces the adapter-conditioned receiver module for Gate 2 evaluation."""

    backend: str = ""
    backend_version: str = ""
    trainable_param_count: int = 0
    training_loss: float = 0.0
    lora_config: Dict[str, Any] = {}

    @abc.abstractmethod
    def attach(self, receiver: ModuleAdapter) -> ModuleAdapter:
        """Return a ModuleAdapter whose ``infer`` reflects the adapter."""


# ---------------------------------------------------------------------------
# Standard backend
# ---------------------------------------------------------------------------


class _AdaptedHFModule(ModuleAdapter):
    """A real receiver model with a trained LoRA adapter attached for inference.

    ``is_mock = False``: this runs real weights + a real adapter. It reuses the
    same deterministic decoding (``do_sample=False``) as ``HFCausalConnector``
    so Gate 2's A/B is not noise.
    """

    is_mock = False

    def __init__(
        self,
        module_id: str,
        model_id: str,
        capabilities: List[CapabilityKey],
        peft_model,
        tokenizer,
        torch,
        max_new_tokens: int = 48,
    ) -> None:
        super().__init__(module_id, model_id)
        self.model_id = model_id
        self._capabilities = list(capabilities)
        self._model = peft_model
        self._tokenizer = tokenizer
        self._torch = torch
        self.max_new_tokens = max_new_tokens

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=["receiver"],
            capabilities=self._capabilities,
            max_learning_level=LearningLevel.L4_PEFT_CANDIDATE,
            is_mock=False,
            version="{}+lora".format(self.model_id),
        )

    def unload(self) -> None:
        """Free the attached peft/base weights.

        Lets the Gate 2 evaluator run baseline and candidate SEQUENTIALLY on a
        memory-bounded GPU (e.g. one 4-bit 7B copy ~5.6 GB on an 8 GB card)
        instead of requiring both co-resident (~11 GB). Score-equivalent: the
        harness re-loads via ``attach`` for the next phase.
        """
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def infer(self, capability: CapabilityKey, prompt: Any) -> Any:
        from ..modules.real.prompting import build_messages

        torch = self._torch
        messages = build_messages(capability, prompt)
        if getattr(self._tokenizer, "chat_template", None):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n".join(m["content"] for m in messages) + "\n"
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        seq = output[0][inputs.input_ids.shape[-1]:]
        return self._clean(self._tokenizer.decode(seq, skip_special_tokens=True).strip())

    def infer_with_skills(self, capability, prompt, skills):
        # The adapter is weights baked into the model, not a prompt-side skill.
        return self.infer(capability, prompt)

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.startswith("```")]
            text = "\n".join(lines).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        return text


class StandardLoRAArtifact(AdapterArtifact):
    backend = "standard"
    backend_version = "peft-lora-v1"

    def __init__(
        self,
        model_id: str,
        adapter_path: str,
        capabilities: List[CapabilityKey],
        lora_config: Dict[str, Any],
        trainable_param_count: int,
        training_loss: float,
        max_new_tokens: int = 48,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._capabilities = list(capabilities)
        self.lora_config = dict(lora_config)
        self.trainable_param_count = trainable_param_count
        self.training_loss = training_loss
        self.max_new_tokens = max_new_tokens

    def attach(self, receiver: ModuleAdapter) -> ModuleAdapter:
        _require_deep()
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(self.model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base = base.to(device)
        peft_model = PeftModel.from_pretrained(base, self.adapter_path)
        peft_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return _AdaptedHFModule(
            module_id="{}+lora".format(receiver.module_id),
            model_id=self.model_id,
            capabilities=self._capabilities,
            peft_model=peft_model,
            tokenizer=tokenizer,
            torch=torch,
            max_new_tokens=self.max_new_tokens,
        )


class TrainerBackend(abc.ABC):
    """Pluggable training backend. A trainer turns rows into an AdapterArtifact."""

    name: str = ""
    version: str = ""

    @abc.abstractmethod
    def supports(self, receiver: ModuleAdapter) -> bool:
        """True if this backend can train on ``receiver`` in this environment."""

    @abc.abstractmethod
    def train(
        self,
        receiver: ModuleAdapter,
        dataset: TrainingDataset,
        config: Dict[str, Any],
        out_dir: Path,
    ) -> AdapterArtifact:
        """Train and return an :class:`AdapterArtifact`."""


class StandardTrainerBackend(TrainerBackend):
    """Model resident on device; the default. CPU-graceful for small models."""

    name = "standard"
    version = "peft-lora-v1"

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
        _require_deep()
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = _model_id_of(receiver)
        lora_config = {
            "r": config.get("lora_rank", 8),
            "lora_alpha": config.get("lora_alpha", 16),
            "target_modules": config.get("target_modules") or ["q_proj", "v_proj"],
            "lora_dropout": config.get("lora_dropout", 0.05),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        seed = int(config.get("seed", 0))
        _set_seeds(seed)

        # Hardware ladder: a big model without CUDA is BLOCKED. A small model
        # may train on CPU. We estimate the param count from the config without
        # downloading weights; if above the CPU ceiling, refuse with a named
        # reason rather than silently starting a multi-day CPU run.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_id)
                n_params = _estimate_params(cfg)
                ceiling = int(config.get("cpu_param_ceiling", CPU_PARAM_CEILING))
                if n_params > ceiling:
                    raise DeepApplyBlocked(
                        "model '{}' has ~{:.1f}B params; CPU training is BLOCKED. "
                        "Deep-apply needs a CUDA GPU for a model this size, or use a "
                        "smaller model (CPU ceiling: {:.1f}B).".format(
                            model_id, n_params / 1e9, ceiling / 1e9
                        )
                    )
            except DeepApplyBlocked:
                raise
            except Exception:
                # Cannot estimate -> proceed on CPU (small models only reach here);
                # the run is simply slow, never faked.
                pass

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.config.use_cache = False
        peft_model = get_peft_model(model, LoraConfig(**lora_config))
        peft_model.train()

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        if trainable == 0:
            raise DeepApplyBlocked(
                "LoRA config produced 0 trainable params for '{}'; target_modules "
                "{} may not exist in this model".format(model_id, lora_config["target_modules"])
            )

        lr = float(config.get("learning_rate", 1e-4))
        max_steps_cap = int(config.get("max_steps_cap", 64))
        max_steps = min(
            int(config.get("max_steps", config.get("epochs", 1) * max(1, len(dataset.rows)))),
            max_steps_cap,
        )
        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad], lr=lr
        )

        last_loss = float("nan")
        rows = dataset.rows or [{}]
        # Telemetry hook (optional). Stashed in the train cfg under ``_on_step``
        # (same pattern as ``_audit``) so the signature is unchanged and every
        # existing caller/test is byte-identical when absent. The Studio's live
        # loss curve / step gauge reads REAL per-step losses from here -- never
        # a fabricated number.
        on_step = config.get("_on_step")
        for step in range(max_steps):
            row = rows[step % len(rows)]
            inp, outp = row.get("input"), row.get("output")
            if inp is None or outp is None:
                continue
            enc = tokenizer(
                "{}\n{}".format(inp, outp), return_tensors="pt", truncation=True, max_length=256
            )
            input_ids = enc["input_ids"].to(device)
            in_enc = tokenizer(str(inp), return_tensors="pt", truncation=True, max_length=256)
            in_len = in_enc["input_ids"].shape[-1]
            labels = input_ids.clone()
            if in_len < labels.shape[-1]:
                labels[:, :in_len] = -100
            out = peft_model(input_ids=input_ids, labels=labels)
            loss = out.loss
            if not torch.isfinite(loss):
                last_loss = float("inf")
                if on_step is not None:
                    on_step({"phase": "train_step", "backend": self.name,
                             "step": step + 1, "max_steps": max_steps,
                             "loss": last_loss, "diverged": True})
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            if on_step is not None:
                on_step({"phase": "train_step", "backend": self.name,
                         "step": step + 1, "max_steps": max_steps,
                         "loss": last_loss})

        adapter_dir = Path(out_dir) / "adapter_model"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(adapter_dir))
        del peft_model, model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        if not math.isfinite(last_loss):
            raise DeepApplyBlocked(
                "training diverged (loss not finite) for '{}'; refusing to emit a "
                "broken adapter".format(model_id)
            )

        caps = list(receiver.manifest().capabilities)
        return StandardLoRAArtifact(
            model_id=model_id,
            adapter_path=str(adapter_dir),
            capabilities=caps,
            lora_config=lora_config,
            trainable_param_count=int(trainable),
            training_loss=last_loss,
            max_new_tokens=int(config.get("max_new_tokens", 48)),
        )


# ---------------------------------------------------------------------------
# Streamed backend (low-VRAM, BETA)
# ---------------------------------------------------------------------------


_SUPPORTED_LAYER_ATTRS = ("layers", "h", "blocks")  # model.layers / transformer.h / ...


def _layer_module_list(model) -> Any:
    """Return the decoder layer list of a CausalLM, or None if unsupported."""
    base = model
    for attr in ("base_model", "model"):
        if hasattr(base, attr):
            base = getattr(base, attr)
            break
    for attr in _SUPPORTED_LAYER_ATTRS:
        lst = getattr(base, attr, None)
        if isinstance(lst, (list, tuple)) and len(lst) > 0:
            return lst
    return None


class StreamedTrainerBackend(TrainerBackend):
    """Low-VRAM streamed LoRA training (Soup-style). BETA -- RETIRED from the
    backend registry as of the siltstream integration (2026-08-16).

    The canonical ``streamed`` backend is now
    :class:`asea.deepapply.backends.SiltStreamBackend` (siltstream-backed,
    CPU-capable, parity-gated). This BETA class is kept DEFINED but is no
    longer returned by :func:`get_backend`; it remains importable for test
    compatibility (``from asea.deepapply import StreamedTrainerBackend``).

    The frozen base lives in host RAM; decoder layers are streamed to the GPU
    one at a time via forward hooks, and LoRA params stay resident on device.

    Honest status: this backend requires CUDA and a supported CausalLM
    architecture (a model with ``model.layers`` / ``transformer.h``). On this
    machine it raises :class:`DeepApplyBlocked` when invoked -- it is never
    silently substituted with the standard backend. Gate 2 judges a streamed
    adapter identically to a standard one.
    """

    name = "streamed"
    version = "soup-stream-v1-beta"

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
        _require_deep()
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise DeepApplyBlocked(
                "streamed backend needs a CUDA GPU (it streams layers to the GPU "
                "to fit big models in low VRAM). On CPU it is pointless. Use the "
                "standard backend, or bigger hardware. " + STREAMING_CREDIT
            )

        model_id = _model_id_of(receiver)
        try:
            cfg = AutoConfig.from_pretrained(model_id)
            arch = getattr(cfg, "model_type", "unknown")
        except Exception as exc:
            raise DeepApplyBlocked(
                "streamed backend could not read config for '{}': {}".format(model_id, exc)
            ) from exc

        # Architecture gate: only CausalLM stacks with a layer list are supported.
        base_for_check = AutoModelForCausalLM.from_config(cfg)
        if _layer_module_list(base_for_check) is None:
            raise DeepApplyBlocked(
                "streamed backend does not support architecture '{}' (no decoder "
                "layer list found). Use the standard backend, or bigger hardware. "
                "{}".format(arch, STREAMING_CREDIT)
            )
        del base_for_check

        # BETA streaming training loop. The base stays on CPU; per-layer forward
        # hooks move each decoder layer to CUDA for its compute and back to CPU
        # afterwards. LoRA params (added by peft) live on CUDA. This implements the
        # published Soup technique. NOTE: runtime correctness on a real GPU is not
        # verified in this environment (no CUDA); the gates above are.
        lora_config = {
            "r": config.get("lora_rank", 8),
            "lora_alpha": config.get("lora_alpha", 16),
            "target_modules": config.get("target_modules") or ["q_proj", "v_proj"],
            "lora_dropout": config.get("lora_dropout", 0.05),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        seed = int(config.get("seed", 0))
        _set_seeds(seed)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.use_cache = False

        layer_list = _layer_module_list(model)
        hooks = []

        def _make_hook(layer):
            def pre(_module, _inputs):
                layer.to("cuda")

            def post(_module, _inputs, _output):
                layer.to("cpu")

            return pre, post

        for layer in layer_list:
            pre, post = _make_hook(layer)
            hooks.append(layer.register_forward_pre_hook(pre))
            hooks.append(layer.register_forward_hook(post))

        peft_model = get_peft_model(model, LoraConfig(**lora_config))
        for p in peft_model.parameters():
            if p.requires_grad:
                p.data = p.data.to("cuda")
        peft_model.train()

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        if trainable == 0:
            for h in hooks:
                h.remove()
            raise DeepApplyBlocked(
                "streamed LoRA produced 0 trainable params for '{}'".format(model_id)
            )

        lr = float(config.get("learning_rate", 1e-4))
        max_steps = min(int(config.get("max_steps", 16)), int(config.get("max_steps_cap", 32)))
        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad], lr=lr
        )

        last_loss = float("nan")
        rows = dataset.rows or [{}]
        for step in range(max_steps):
            row = rows[step % len(rows)]
            inp, outp = row.get("input"), row.get("output")
            if inp is None or outp is None:
                continue
            enc = tokenizer(
                "{}\n{}".format(inp, outp), return_tensors="pt", truncation=True, max_length=256
            )
            input_ids = enc["input_ids"]  # stays on CPU; embedding lookup on CPU
            in_enc = tokenizer(str(inp), return_tensors="pt", truncation=True, max_length=256)
            in_len = in_enc["input_ids"].shape[-1]
            labels = input_ids.clone()
            if in_len < labels.shape[-1]:
                labels[:, :in_len] = -100
            out = peft_model(input_ids=input_ids, labels=labels.to("cuda"))
            loss = out.loss
            if not torch.isfinite(loss):
                last_loss = float("inf")
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

        for h in hooks:
            h.remove()

        if not math.isfinite(last_loss):
            raise DeepApplyBlocked(
                "streamed training diverged (loss not finite) for '{}'".format(model_id)
            )

        adapter_dir = Path(out_dir) / "adapter_model_streamed"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        for p in peft_model.parameters():
            if p.requires_grad:
                p.data = p.data.to("cpu")
        peft_model.save_pretrained(str(adapter_dir))
        del peft_model, model
        torch.cuda.empty_cache()

        caps = list(receiver.manifest().capabilities)
        return StandardLoRAArtifact(
            model_id=model_id,
            adapter_path=str(adapter_dir),
            capabilities=caps,
            lora_config=lora_config,
            trainable_param_count=int(trainable),
            training_loss=last_loss,
            max_new_tokens=int(config.get("max_new_tokens", 48)),
        )


BACKENDS = {
    "standard": StandardTrainerBackend,
}

# ``streamed`` and ``zeroforge`` are resolved LAZILY (imported on first
# get_backend call) to avoid an import cycle: the backend modules in
# ``.backends`` import helpers (``_AdaptedHFModule``, ``_require_deep`` ...)
# from THIS module, so they cannot be imported at module-load time here.
_LAZY_BACKENDS = {
    "streamed": ("asea.deepapply.backends", "SiltStreamBackend"),
    "zeroforge": ("asea.deepapply.backends", "ZeroForgeBackend"),
}


def get_backend(name: str) -> TrainerBackend:
    """Construct a trainer backend by name. Raises DeepApplyBlocked on unknown.

    ``standard`` resolves eagerly; ``streamed`` (siltstream layer-streamed
    LoRA) and ``zeroforge`` (forward-only zeroth-order) resolve lazily. The
    retired BETA ``StreamedTrainerBackend`` is intentionally NOT registered --
    the canonical ``streamed`` backend is the siltstream-backed
    :class:`asea.deepapply.backends.SiltStreamBackend`.
    """
    cls = BACKENDS.get(name)
    if cls is not None:
        return cls()
    lazy = _LAZY_BACKENDS.get(name)
    if lazy is not None:
        mod_path, attr = lazy
        import importlib

        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)()
    supported = sorted(set(BACKENDS) | set(_LAZY_BACKENDS))
    raise DeepApplyBlocked(
        "unknown trainer backend '{}'; supported: {}".format(name, supported)
    )