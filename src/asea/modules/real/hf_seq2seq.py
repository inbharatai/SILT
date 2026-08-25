"""Real connector for HuggingFace seq2seq translation models (NLLB, mBART, M2M100).

REAL: ``is_mock = False``.

This is the natural **sender** for a low-resource language flow. NLLB-200
genuinely covers Assamese (``asm_Beng``) and Manipuri (``mni_Beng``), which most
general instruct models handle poorly -- exactly the asymmetry the adapter is
designed to exploit.

Translation models take no skill injection: ``infer_with_skills`` falls back to
``infer`` (inherited). That is honest -- a dedicated MT model has no prompt
channel for a glossary -- and it means such a module should be registered as a
sender only.

Bodo (``brx``) is NOT in NLLB-200. Requesting it raises rather than silently
translating into a neighbouring language, which would quietly poison a packet.
"""

from __future__ import annotations

import gc
from typing import Any, List, Optional

from ...core.interfaces import ModuleAdapter
from ...core.protocol import CapabilityKey, CapabilityManifest, LearningLevel
from .prompting import split_pair

#: BCP-47-ish tag -> NLLB-200 FLORES code.
NLLB_CODES = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "mni": "mni_Beng",
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ne": "npi_Deva",
    "or": "ory_Orya",
}

#: Languages this family cannot do. Listed so failure is loud, not silent.
UNSUPPORTED = {"brx": "Bodo is not covered by NLLB-200"}


class HFSeq2SeqTranslator(ModuleAdapter):
    is_mock = False

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        capabilities: Optional[List[CapabilityKey]] = None,
        module_id: Optional[str] = None,
        display_name: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 96,
        lang_codes: Optional[dict] = None,
    ) -> None:
        super().__init__(
            module_id or model_id.split("/")[-1].lower(),
            display_name or model_id,
        )
        self.model_id = model_id
        self._capabilities = list(capabilities or [])
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.lang_codes = dict(lang_codes or NLLB_CODES)

        self._tokenizer = None
        self._model = None
        self._torch = None

    # -- lifecycle --------------------------------------------------------

    def load(self) -> "HFSeq2SeqTranslator":
        if self._model is not None:
            return self
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HFSeq2SeqTranslator needs torch, transformers and sentencepiece.\n"
                "  pip install torch transformers sentencepiece"
            ) from exc

        self._torch = torch
        device = (
            self.device
            if self.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = (
            getattr(torch, self.dtype)
            if self.dtype != "auto"
            else (torch.float16 if device == "cuda" else torch.float32)
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, dtype=dtype
        ).to(device)
        self._model.eval()
        self.resolved_device = device
        return self

    def unload(self) -> None:
        # Mirror HFCausalConnector.unload: free weights AND empty the CUDA
        # cache. The deep-apply evaluator runs baseline/candidate sequentially
        # and relies on unload() actually returning GPU memory to the
        # allocator; without empty_cache a freed NLLB leaves the cache
        # reserved and the next load on a tight card can OOM.
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # -- identity ---------------------------------------------------------

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=["sender"],  # no prompt channel for skills; sender only
            capabilities=self._capabilities,
            max_learning_level=LearningLevel.L0_INTERACTION,
            is_mock=False,
            version=self.model_id,
        )

    # -- inference --------------------------------------------------------

    def _code(self, tag: str) -> str:
        if tag in UNSUPPORTED:
            raise ValueError(
                "{} ({}). Use a model that covers it rather than accepting a "
                "near-miss language.".format(UNSUPPORTED[tag], tag)
            )
        try:
            return self.lang_codes[tag]
        except KeyError:
            raise ValueError(
                "no language code mapping for '{}'; known: {}".format(
                    tag, sorted(self.lang_codes)
                )
            )

    def infer(self, capability: CapabilityKey, prompt: Any) -> Any:
        self.load()
        torch = self._torch
        src, tgt = split_pair(capability.language)
        if not src or not tgt:
            raise ValueError(
                "translation capability needs a 'src->tgt' language tag, got {!r}".format(
                    capability.language
                )
            )

        self._tokenizer.src_lang = self._code(src)
        inputs = self._tokenizer(str(prompt), return_tensors="pt").to(self._model.device)
        forced_bos = self._tokenizer.convert_tokens_to_ids(self._code(tgt))

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,  # determinism
            )
        return self._tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

    def confidence(self, capability: CapabilityKey, prompt: Any, output: Any) -> float:
        # A dedicated MT model gives no cheap calibrated signal here. Returning a
        # neutral constant is more honest than inventing one; the relevance
        # filter's reference check is what actually guards quality.
        return 0.6
