"""Real connector for HuggingFace causal LMs (Qwen, Gemma, Llama, Phi...).

REAL, not a mock: ``is_mock = False``. Loads actual weights and runs actual
generation.

Design notes that matter for correctness of the whole system:

* **Deterministic decoding by default** (``do_sample=False``). Sampling turns the
  evaluator's before/after A/B into noise, and you would promote packets on the
  strength of a lucky decode. If you must sample, set a fixed seed and expect
  wider error bars.
* **Lazy loading.** ``torch``/``transformers`` are imported inside methods, so
  importing this module on a machine without them costs nothing and raises a
  clear error only if you actually try to load. This keeps the package usable in
  environments where only the mock path is available.
* **Real confidence.** ``confidence()`` returns the mean per-token probability of
  the generated sequence, not a hardcoded constant. It is still a weak signal --
  a confidently wrong model scores high -- and the relevance filter treats it as
  a tiebreaker only.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional

from ...core.interfaces import ModuleAdapter
from ...core.protocol import CapabilityKey, CapabilityManifest, LearningLevel
from .prompting import build_messages


class HFCausalConnector(ModuleAdapter):
    is_mock = False

    def __init__(
        self,
        model_id: str,
        capabilities: List[CapabilityKey],
        roles: Optional[List[str]] = None,
        module_id: Optional[str] = None,
        display_name: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 64,
        max_learning_level: LearningLevel = LearningLevel.L4_PEFT_CANDIDATE,
        report_confidence: bool = True,
        trust_remote_code: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        super().__init__(
            module_id or model_id.split("/")[-1].lower(),
            display_name or model_id,
        )
        self.model_id = model_id
        self._capabilities = list(capabilities)
        self._roles = list(roles or ["sender", "receiver"])
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_learning_level = max_learning_level
        self.report_confidence = report_confidence
        self.trust_remote_code = trust_remote_code
        # 4-bit (nf4) inference load. GPU-only. Lets a 7B-class model fit in
        # ~3.5 GB resident so the Gate 2 A/B (baseline + candidate co-resident)
        # fits an 8 GB card (~7 GB). Ignored on CPU; ``load`` raises if 4-bit is
        # requested without CUDA rather than silently load fp16 and OOM.
        self.load_in_4bit = load_in_4bit

        self._tokenizer = None
        self._model = None
        self._torch = None
        self._last_confidence: Optional[float] = None

    # -- lifecycle --------------------------------------------------------

    def _resolve_device(self, torch) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch, device: str):
        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        if device == "cuda":
            return torch.float16
        if device == "mps":
            return torch.float16
        # CPU: float32. float16 matmul is poorly supported on CPU and bfloat16
        # is only partially supported (some ops upcast, older torch builds
        # error). Match HFSeq2SeqTranslator, which deliberately uses fp32 on
        # CPU -- a consistent, safe CPU default.
        return torch.float32

    def load(self) -> "HFCausalConnector":
        if self._model is not None:
            return self
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "HFCausalConnector needs torch and transformers. Install with:\n"
                "  pip install 'adaptive-skill-extraction-adapter[connectors]'\n"
                "or: pip install torch transformers"
            ) from exc

        self._torch = torch
        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if self.load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig

            # Compute dtype: honour an explicit self.dtype, but keep bfloat16
            # as the 4-bit compute default (the stable choice for nf4). The
            # ``dtype`` variable above already ran _resolve_dtype, which on
            # ``auto``+cuda yields float16 -- fine for full-precision load but
            # not what we want as the 4-bit *compute* dtype, so re-derive here
            # rather than reuse it.
            compute_dtype = dtype if self.dtype != "auto" else torch.bfloat16
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb_cfg,
                trust_remote_code=self.trust_remote_code,
            ).to(device)
        else:
            if self.load_in_4bit:
                # RuntimeError, not ImportError: torch/transformers ARE
                # importable here -- the failure is missing hardware, not a
                # missing dependency. ImportError would mislead a caller's
                # except-import-error fallback into a spurious "install
                # torch" path.
                raise RuntimeError(
                    "HFCausalConnector load_in_4bit=True requires a CUDA GPU; none "
                    "is available. Drop load_in_4bit or run on a CUDA host."
                )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype, trust_remote_code=self.trust_remote_code
            ).to(device)
        self._model.eval()
        self.resolved_device = device
        self.resolved_dtype = str(dtype)
        return self

    def unload(self) -> None:
        """Free weights. Useful when sender and receiver cannot co-reside in RAM."""
        self._model = None
        self._tokenizer = None
        self._last_confidence = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # -- identity ---------------------------------------------------------

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=self._roles,
            capabilities=self._capabilities,
            max_learning_level=self.max_learning_level,
            is_mock=False,
            version=self.model_id,
        )

    # -- inference --------------------------------------------------------

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        self.load()
        torch = self._torch
        tokenizer, model = self._tokenizer, self._model

        # Reset every call. confidence() reads self._last_confidence, so without
        # this a run with report_confidence=False (or one that produced no
        # scores) would return the *previous* run's confidence -- a stale,
        # cross-prompt signal that mislabels the current output. It is repopulated
        # below only when scores are actually produced.
        self._last_confidence = None

        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Base (non-chat) model: flatten to a plain prompt.
            text = "\n".join(m["content"] for m in messages) + "\n"

        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,           # determinism: see module docstring
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=self.report_confidence,
            )

        sequence = output.sequences[0][inputs.input_ids.shape[-1]:]
        decoded = tokenizer.decode(sequence, skip_special_tokens=True).strip()

        if self.report_confidence and getattr(output, "scores", None):
            probs = []
            for step, score in enumerate(output.scores):
                if step >= len(sequence):
                    break
                distribution = torch.softmax(score[0].float(), dim=-1)
                probs.append(float(distribution[sequence[step]].item()))
            self._last_confidence = sum(probs) / len(probs) if probs else None

        return self._clean(decoded)

    @staticmethod
    def _clean(text: str) -> str:
        """Strip the wrappers small instruct models add despite being told not to."""
        text = text.strip()
        if text.startswith("```"):
            lines = [l for l in text.splitlines() if not l.startswith("```")]
            text = "\n".join(lines).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        return text

    def infer(self, capability: CapabilityKey, prompt: Any) -> Any:
        return self._generate(build_messages(capability, prompt))

    def infer_with_skills(
        self, capability: CapabilityKey, prompt: Any, skills: List[Dict[str, Any]]
    ) -> Any:
        return self._generate(build_messages(capability, prompt, skills))

    def confidence(self, capability: CapabilityKey, prompt: Any, output: Any) -> float:
        if self._last_confidence is None:
            return 0.5
        return max(0.0, min(1.0, self._last_confidence))
