"""Declarative wiring.

A run is described by a JSON file: which modules exist, which pair is bound,
which benchmark suites apply, and what promotion policy governs it. This keeps
"what was run" auditable as a file rather than as arguments someone typed once.

Both mock and REAL modules can be built from config. Presets ending in ``_mock``
are placeholders; ``qwen_ollama``, ``gemma_ollama``, ``qwen_hf`` and
``nllb_translator`` load real weights. Add your own by registering a factory in
``CONNECTOR_FACTORIES``; see docs/connector_authoring.md.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .benchmarks.harness import BenchmarkSuite, load_suite
from .core.interfaces import ModuleAdapter
from .core.pipeline import Pipeline
from .core.protocol import CapabilityKey, Domain, LearningLevel, Modality
from .modules.mock.base import MockModule
from .modules.mock.zoo import (
    make_ai4bharat_asr,
    make_ai4bharat_tts,
    make_gemma,
    make_qwen,
)
from .promotion.gate import PromotionGate, PromotionPolicy


def _real_factories() -> Dict[str, Callable[..., ModuleAdapter]]:
    """Real connectors, imported lazily.

    The import is guarded because the real package is usable without torch or
    transformers installed (they load on demand), but a broken install should
    degrade to "mock presets only" rather than making every config unusable.
    """
    try:
        from .modules.real import (
            make_gemma_ollama,
            make_nllb_translator,
            make_qwen_hf,
            make_qwen_ollama,
        )
    except ImportError:  # pragma: no cover - environment dependent
        # ImportError == the optional ML deps (torch/transformers/etc.) are not
        # installed, which is the documented "mock presets only" degradation.
        # A broader except would also swallow a real bug in modules/real
        # (SyntaxError, AttributeError from a bad refactor) and silently disable
        # every real preset with no signal -- the worst possible failure mode
        # for a config layer. Let non-import errors surface.
        return {}
    return {
        "qwen_ollama": make_qwen_ollama,
        "gemma_ollama": make_gemma_ollama,
        "qwen_hf": make_qwen_hf,
        "nllb_translator": make_nllb_translator,
    }


#: Named constructors usable from a config file.
#: ``*_mock`` entries are placeholders; the rest load real weights.
CONNECTOR_FACTORIES: Dict[str, Callable[..., ModuleAdapter]] = {
    "qwen_mock": make_qwen,
    "gemma_mock": make_gemma,
    "ai4bharat_asr_mock": make_ai4bharat_asr,
    "ai4bharat_tts_mock": make_ai4bharat_tts,
}
CONNECTOR_FACTORIES.update(_real_factories())


def load_suites(names: List[str], data_dir: Path) -> Dict[str, BenchmarkSuite]:
    return {name: load_suite(Path(data_dir) / "{}.json".format(name)) for name in names}


def _capability(spec: Dict[str, Any]) -> CapabilityKey:
    return CapabilityKey(
        task_type=spec["task_type"],
        modality=Modality(spec["modality"]),
        domain=Domain(spec.get("domain", "general")),
        language=spec.get("language"),
    )


def _knowledge(
    spec: Optional[Dict[str, Any]], suites: Dict[str, BenchmarkSuite]
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not spec:
        return None
    splits = tuple(spec.get("splits", ["extraction"]))
    only = set(spec["only_cases"]) if "only_cases" in spec else None
    overrides = spec.get("overrides", {})
    table: Dict[str, Dict[str, Any]] = {}
    for suite_name in spec.get("suites", []):
        suite = suites[suite_name]
        bucket = table.setdefault(suite.capability().as_str(), {})
        for case in suite.cases:
            if case.split not in splits:
                continue
            if only is not None and case.case_id not in only:
                continue
            bucket[str(case.prompt).strip()] = case.expected
        for prompt, answer in overrides.items():
            if prompt in bucket:
                bucket[prompt] = answer
    return table


def build_module(spec: Dict[str, Any], suites: Dict[str, BenchmarkSuite]) -> ModuleAdapter:
    knowledge = _knowledge(spec.get("knowledge"), suites)

    if "preset" in spec:
        factory = CONNECTOR_FACTORIES.get(spec["preset"])
        if factory is None:
            raise KeyError(
                "unknown preset '{}'; known: {}".format(
                    spec["preset"], sorted(CONNECTOR_FACTORIES)
                )
            )

        # Pass through only what this factory actually accepts, so one config
        # schema can drive both mock presets (knowledge, fallback) and real
        # connectors (model, host, dtype, ...) without special-casing either.
        accepted = set(inspect.signature(factory).parameters)
        kwargs: Dict[str, Any] = dict(spec.get("args", {}))
        if "capabilities" in spec and "capabilities" in accepted:
            kwargs["capabilities"] = [_capability(c) for c in spec["capabilities"]]
        if knowledge is not None and "knowledge" in accepted:
            kwargs["knowledge"] = knowledge
        if "fallback" in spec and "fallback" in accepted:
            kwargs["fallback"] = spec["fallback"]
        return factory(**kwargs)

    return MockModule(
        module_id=spec["id"],
        display_name=spec.get("name", spec["id"]),
        roles=spec["roles"],
        capabilities=[_capability(c) for c in spec["capabilities"]],
        knowledge=knowledge,
        fallback=spec.get("fallback", "echo"),
        consumes_skills=spec.get("consumes_skills", True),
        max_learning_level=LearningLevel(spec.get("max_learning_level", 3)),
    )


def build_pipeline(config: Dict[str, Any], workspace: Path, data_dir: Path):
    """Return (pipeline, suites, adapter_id) ready to run."""
    suites = load_suites(config["suites"], data_dir)
    policy = PromotionPolicy(**config.get("promotion_policy", {}))
    pipeline = Pipeline(workspace=workspace, gate=PromotionGate(policy))

    for spec in config["modules"]:
        pipeline.register_module(build_module(spec, suites))

    adapter = config["adapter"]
    pipeline.bind_adapter(
        adapter["id"], adapter["sender"], adapter["receiver"],
        description=adapter.get("description", ""),
    )
    return pipeline, list(suites.values()), adapter["id"]


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
