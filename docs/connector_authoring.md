# SILT — Adding a real connector

A universal adapter is only as good as how cheaply you can plug something into
it. Replacing a mock with a real model means writing one subclass. Nothing in the
core, the filters, the gate or the audit layer changes.

## The interface

```python
from asea.core.interfaces import ModuleAdapter
from asea.core.protocol import CapabilityKey, CapabilityManifest, LearningLevel


class MyModel(ModuleAdapter):
    is_mock = False          # be honest; this rides into provenance

    def manifest(self) -> CapabilityManifest: ...
    def infer(self, capability, prompt): ...

    # optional
    def infer_with_skills(self, capability, prompt, skills): ...
    def confidence(self, capability, prompt, output): ...
```

Only `manifest` and `infer` are required. `infer_with_skills` defaults to
ignoring the skills, which correctly models a module that cannot consume them —
such a receiver will simply never show improvement and so will never pass the
gate. That is the right failure mode: silent, measurable, non-damaging.

## A real Qwen receiver

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from asea.core.interfaces import ModuleAdapter
from asea.core.protocol import CapabilityKey, CapabilityManifest, Domain, LearningLevel, Modality


class QwenConnector(ModuleAdapter):
    is_mock = False

    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
        super().__init__(module_id="qwen-2.5-7b", display_name="Qwen2.5 7B Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.device = device

    def manifest(self):
        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=["sender", "receiver"],
            capabilities=[
                CapabilityKey(task_type="translate", modality=Modality.TEXT,
                              domain=Domain.TRANSLATION, language="as->en"),
                CapabilityKey(task_type="bug_fix", modality=Modality.CODE,
                              domain=Domain.SOFTWARE, language="python"),
            ],
            max_learning_level=LearningLevel.L4_PEFT_CANDIDATE,
            is_mock=False,
        )

    def _generate(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = out[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def infer(self, capability, prompt):
        return self._generate([
            {"role": "system", "content": _system_for(capability)},
            {"role": "user", "content": str(prompt)},
        ])

    def infer_with_skills(self, capability, prompt, skills):
        # The ONLY thing you may inject is skill["distilled_skill"].
        # Never inject sender_output; it is not in the redacted view anyway.
        context = _render_skills(skills)
        return self._generate([
            {"role": "system", "content": _system_for(capability)},
            {"role": "system", "content": "Reference knowledge:\n" + context},
            {"role": "user", "content": str(prompt)},
        ])
```

**Determinism matters.** Use `do_sample=False` (or a fixed seed) for anything
feeding the evaluator. Sampling turns your before/after A/B into noise and you
will promote packets on the strength of a lucky decode.

## An AI4Bharat ASR sender

```python
class IndicConformerASR(ModuleAdapter):
    is_mock = False

    def manifest(self):
        return CapabilityManifest(
            module_id="ai4bharat-indicconformer",
            display_name="AI4Bharat IndicConformer",
            roles=["sender"],           # ASR is a source, not a learner
            capabilities=[
                CapabilityKey(task_type="transcribe", modality=Modality.AUDIO_ASR,
                              domain=Domain.LANGUAGE, language=lang)
                for lang in ("as", "bn", "brx", "hi", "mni")
            ],
            is_mock=False,
        )

    def infer(self, capability, prompt):
        # prompt is an audio path or array; return the transcript string
        return self.pipeline(prompt, language=capability.language)["text"]
```

Note `roles=["sender"]` — the registry enforces this, and binding it as a
receiver raises `RegistrationError`.

## Registering

Imperatively:

```python
pipeline.register_module(QwenConnector())
pipeline.bind_adapter("as-to-qwen", "assamese-corpus", "qwen-2.5-7b")
```

Or declaratively — add the factory to `CONNECTOR_FACTORIES` in `asea/config.py`
and reference it by `preset` in a config file.

## Adding a new modality

Register an extractor and a distiller; the core needs no edit. This is asserted
by `test_new_modality_needs_no_core_edit`.

```python
class OcrExtractor(BaseExtractor):
    modality = Modality.OCR

class OcrDistiller(BaseDistiller):
    modality = Modality.OCR
    packet_type = PacketType.CORRECTION_PAIR

    def build_payload(self, group):
        return {"pairs": [{"observed": p.notes["prompt"],
                           "corrected": self.taught_value(p)} for p in group]}

plugins = default_registry()
plugins.register_extractor(OcrExtractor())
plugins.register_distiller(OcrDistiller())
pipeline = Pipeline(workspace=ws, plugins=plugins)
```

Optionally add a `MetricPlugin` for that modality; the universal evaluator falls
back to similarity if you do not.

## Upgrading similarity to something real

The single highest-value swap in the whole system:

```python
from sentence_transformers import SentenceTransformer, util
from asea.core.interfaces import SimilarityBackend


class EmbeddingSimilarity(SimilarityBackend):
    def __init__(self, model_id="sentence-transformers/LaBSE"):
        self.model = SentenceTransformer(model_id)

    def similarity(self, a, b):
        ea, eb = self.model.encode([a, b], convert_to_tensor=True)
        return float(util.cos_sim(ea, eb).item())

    @property
    def is_semantic(self):
        return True


harness = BenchmarkHarness(plugins=default_registry(),
                           similarity=EmbeddingSimilarity())
pipeline = Pipeline(workspace=ws, harness=harness)
```

Re-measure every existing result afterwards. Thresholds calibrated against a
lexical proxy are meaningless against an embedding metric.

## Checklist before pointing this at anything real

- [ ] `is_mock = False` and it is actually true
- [ ] Deterministic decoding for evaluation paths
- [ ] Held-out split genuinely disjoint from extraction
- [ ] At least one regression suite covering capabilities you are not targeting
- [ ] Embedding similarity backend installed
- [ ] Native-speaker review for language tasks; test execution for code tasks
- [ ] `strict_no_mock` left at its default (`True`)
- [ ] Named human approvers configured for any high-risk domain
