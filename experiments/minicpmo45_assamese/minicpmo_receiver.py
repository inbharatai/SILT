"""MiniCPM-o 4.5 as a SILT receiver — real ModuleAdapter, original checkpoint.

Registers MiniCPM-o 4.5 in SILT's actual receiver abstraction (ModuleAdapter)
with the seven Assamese capabilities from capability_map.md. Uses the ORIGINAL
trainable HuggingFace checkpoint (NOT Ollama/GGUF), per the spec §2. This is the
receiver half of the SILT graph; the SILT gap engine measures its baseline on
each capability and the gate decides what to transfer in.

It is deliberately a honest skeleton on this CPU machine: it implements the full
ModuleAdapter contract and a checkpoint preflight, but `infer` raises a clear
BLOCKED error until the checkpoint is actually loadable on a CUDA box (blockers
B1-B3). It does NOT fake outputs. On the GPU box it lazy-loads
`AutoModel.from_pretrained(checkpoint, trust_remote_code=True, ...)` and the
omni generation pipeline; that body is filled in from the live run.

Why experiment-local, not in src/asea/studio/catalog.py: production code stays
untouched (no silent architecture change). `register_into_catalog()` shows the
exact additive catalog entry to add on the GPU box — same shape as the existing
`tts-learner-zero` / `glm-ollama` additions, no core/policy/gate edit.

  PYTHONPATH=src python -c "from minicpmo_receiver import MiniCPMOReceiver; \
      r = MiniCPMOReceiver('openbmb/MiniCPM-o-2_6'); print(r.manifest().model_dump_json(indent=2))"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from asea.core.interfaces import ModuleAdapter
from asea.core.protocol import (
    CapabilityKey, CapabilityManifest, Domain, Modality,
)


# -- Assamese capability keys (reuse existing modalities; no new core modality) -

def _text_as() -> CapabilityKey:
    return CapabilityKey(task_type="translate", modality=Modality.TEXT,
                          domain=Domain.TRANSLATION, language="as->en")


def _reason_as() -> CapabilityKey:
    return CapabilityKey(task_type="reason", modality=Modality.TEXT,
                          domain=Domain.GENERAL, language="as")


def _stt_as() -> CapabilityKey:
    return CapabilityKey(task_type="transcribe", modality=Modality.AUDIO_ASR,
                          domain=Domain.GENERAL, language="as")


def _g2p_as() -> CapabilityKey:
    return CapabilityKey(task_type="grapheme_to_phoneme", modality=Modality.SPEECH_TTS,
                          domain=Domain.PRONUNCIATION, language="as-ipa")


def _tts_as() -> CapabilityKey:
    return CapabilityKey(task_type="synthesize", modality=Modality.SPEECH_TTS,
                          domain=Domain.GENERAL, language="as")


def _sts_as() -> CapabilityKey:
    return CapabilityKey(task_type="sts", modality=Modality.SPEECH_TTS,
                          domain=Domain.GENERAL, language="as")


def _vision_as() -> CapabilityKey:
    return CapabilityKey(task_type="describe", modality=Modality.OCR,
                          domain=Domain.GENERAL, language="as")


def _tool_as() -> CapabilityKey:
    return CapabilityKey(task_type="tool_call", modality=Modality.TEXT,
                          domain=Domain.SOFTWARE, language="as")


def _agent_as() -> CapabilityKey:
    return CapabilityKey(task_type="agent_reason", modality=Modality.TEXT,
                          domain=Domain.GENERAL, language="as")


def assamese_capabilities() -> List[CapabilityKey]:
    """The seven Assamese capabilities from capability_map.md."""
    return [_text_as(), _reason_as(), _stt_as(), _g2p_as(), _tts_as(),
            _sts_as(), _vision_as(), _tool_as(), _agent_as()]


class MiniCPMOReceiver(ModuleAdapter):
    """MiniCPM-o 4.5 omni model as a SILT receiver.

    is_mock is False — this is a real model. The preflight + lazy-load guard mean
    it refuses to fabricate output when the checkpoint is unavailable.
    """

    is_mock = False

    def __init__(self, checkpoint: str = "openbmb/MiniCPM-o-2_6",
                 module_id: str = "minicpmo-4.5",
                 display_name: str = "MiniCPM-o 4.5 (OpenBMB omni, trainable checkpoint)",
                 capabilities: Optional[List[CapabilityKey]] = None) -> None:
        super().__init__(module_id=module_id, display_name=display_name)
        self.checkpoint = checkpoint
        self._capabilities = capabilities if capabilities is not None else assamese_capabilities()
        self._model = None  # lazy-loaded on first infer (GPU box)

    # -- identity ---------------------------------------------------------

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=["receiver"],
            capabilities=self._capabilities,
            is_mock=self.is_mock,
            version="4.5",
        )

    # -- checkpoint guard --------------------------------------------------

    def checkpoint_available(self) -> bool:
        """True if the checkpoint appears in the HF cache (no download here)."""
        hf = Path.home() / ".cache" / "huggingface" / "hub"
        if not hf.exists():
            return False
        slug = "models--" + self.checkpoint.replace("/", "--")
        return (hf / slug).exists()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not self.checkpoint_available():
            raise RuntimeError(
                "BLOCKED: MiniCPM-o checkpoint '{}' not present and torch is "
                "CPU-only with 8 GB VRAM (hardware.json B1-B3). Run on a CUDA "
                "box with >=16 GB VRAM after `huggingface-cli download {}`.".format(
                    self.checkpoint, self.checkpoint)
            )
        # GPU-box body (only reached when the checkpoint is downloaded on a
        # CUDA box). Load the omni model with trust_remote_code, set up the
        # audio-in / text / speech-out generation pipeline. Filled in from the
        # live run; never fabricated.
        from transformers import AutoModel  # noqa: WPS433
        self._model = AutoModel.from_pretrained(
            self.checkpoint, trust_remote_code=True, torch_dtype="auto",
        )

    # -- behaviour --------------------------------------------------------

    def infer(self, capability: CapabilityKey, prompt: Any) -> Any:
        self._ensure_loaded()
        # GPU-box body: dispatch by capability.modality to the omni pipeline
        # (text chat / audio-in ASR / speech-out synthesis / vision / ...).
        # Returns the model's real output for the held-out probe. Until the
        # checkpoint runs, _ensure_loaded raises BLOCKED above — no fake output.
        raise NotImplementedError(
            "infer for {} not implemented until the checkpoint runs on GPU".format(
                capability.as_str())
        )

    def infer_with_skills(
        self, capability: CapabilityKey, prompt: Any, skills: List[Dict[str, Any]]
    ) -> Any:
        """Conditioned inference. Explicitly guarded — does NOT rely on the
        inherited default (which delegates to ``self.infer``). Every capability
        path hits ``_ensure_loaded`` itself, so no skill-conditioned call can slip
        past the checkpoint guard even if a future edit overrides this method's
        body. Until the checkpoint runs on GPU this raises BLOCKED — no fake
        output, with or without injected skills.
        """
        self._ensure_loaded()
        raise NotImplementedError(
            "infer_with_skills for {} not implemented until the checkpoint "
            "runs on GPU".format(capability.as_str())
        )


def register_into_catalog():
    """The exact additive catalog entry to add on the GPU box.

    Add to src/asea/studio/catalog.py CATALOG (same shape as the existing
    `tts-learner-zero` / `glm-ollama` additions — ADDITIVE, no core/policy/gate
    edit)::

        from experiments.minicpmo45_assamese.minicpmo_receiver import MiniCPMOReceiver
        def _minicpmo():
            return MiniCPMOReceiver("openbmb/MiniCPM-o-2_6")
        CATALOG["minicpmo-4.5"] = {
            "factory": _minicpmo,
            "roles": ["receiver"],
            "description": "MiniCPM-o 4.5 (OpenBMB omni, trainable checkpoint)",
            "requires": "CUDA torch + >=16 GB VRAM + huggingface-cli download openbmb/MiniCPM-o-2_6",
            "builtin": False,
        }
    """
    return MiniCPMOReceiver()


if __name__ == "__main__":
    r = MiniCPMOReceiver()
    m = r.manifest()
    print("module:", m.module_id, "| is_mock:", m.is_mock, "| roles:", m.roles)
    print("checkpoint_available:", r.checkpoint_available())
    print("capabilities ({}):".format(len(m.capabilities)))
    for c in m.capabilities:
        print("  ", c.as_str())