"""Benchmark harness.

Design rule that matters more than the code: the harness enforces a **split
discipline**. Extraction may only read the ``extraction`` split. Evaluation may
only read the ``heldout`` split. Regression checks run on the ``regression``
split, which covers capabilities the transfer is *not* trying to improve.

Without that separation you measure the extractor memorising its own probes and
call it learning. The harness raises if a caller asks for the wrong split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import (
    BatchedInferenceError,
    EvaluationError,
    InferenceCountMismatchError,
)
from ..core.interfaces import ModuleAdapter, SimilarityBackend
from ..core.plugins import PluginRegistry
from ..core.protocol import CapabilityKey, Domain, Modality
from ..evaluator import metrics
from ..evaluator.similarity import LexicalSimilarity

VALID_SPLITS = ("extraction", "heldout", "regression")


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    prompt: Any
    expected: Any
    split: str = "heldout"
    meta: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    description: str = ""
    task_type: str
    modality: Modality
    domain: Domain = Domain.GENERAL
    language: Optional[str] = None
    cases: List[BenchmarkCase] = Field(default_factory=list)

    def capability(self) -> CapabilityKey:
        return CapabilityKey(
            task_type=self.task_type,
            modality=self.modality,
            domain=self.domain,
            language=self.language,
        )

    def split(self, name: str) -> List[BenchmarkCase]:
        if name not in VALID_SPLITS:
            raise EvaluationError(
                "unknown split '{}'; expected one of {}".format(name, VALID_SPLITS)
            )
        return [c for c in self.cases if c.split == name]

    def counts(self) -> Dict[str, int]:
        return {s: len(self.split(s)) for s in VALID_SPLITS}


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    prompt: Any
    expected: Any
    actual: Any
    similarity: float
    task_success: float
    language_preservation: float
    hallucination_risk: float
    score: float


class SuiteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    module_id: str
    split: str
    with_skills: bool
    case_results: List[CaseResult] = Field(default_factory=list)
    similarity_is_semantic: bool = False

    @property
    def score(self) -> float:
        return metrics.mean(r.score for r in self.case_results)

    @property
    def task_success(self) -> float:
        return metrics.mean(r.task_success for r in self.case_results)

    @property
    def language_preservation(self) -> float:
        return metrics.mean(r.language_preservation for r in self.case_results)

    @property
    def hallucination_risk(self) -> float:
        return metrics.mean(r.hallucination_risk for r in self.case_results)

    @property
    def similarity(self) -> float:
        return metrics.mean(r.similarity for r in self.case_results)

    def summary(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "module_id": self.module_id,
            "split": self.split,
            "with_skills": self.with_skills,
            "n_cases": len(self.case_results),
            "score": round(self.score, 4),
            "task_success": round(self.task_success, 4),
            "language_preservation": round(self.language_preservation, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "similarity_is_semantic": self.similarity_is_semantic,
        }


class BenchmarkHarness:
    def __init__(
        self,
        plugins: Optional[PluginRegistry] = None,
        similarity: Optional[SimilarityBackend] = None,
        max_batch_size: int = 1,
    ) -> None:
        self.plugins = plugins
        self.similarity = similarity or LexicalSimilarity()
        #: Global ceiling on cases pushed through one forward pass (A2, audit
        #: 2026-08-17). Default 1 -> the harness loops the single-case infer
        #: path, byte-identical to pre-A2, and the only safe default for a
        #: 4-bit 7B on an 8 GB card where two cases co-resident OOM. The
        #: EFFECTIVE batch for a run is min(this, module.preferred_batch_size),
        #: so a module is never forced beyond what it declared safe.
        self.max_batch_size = max_batch_size

    def run(
        self,
        module: ModuleAdapter,
        suite: BenchmarkSuite,
        split: str = "heldout",
        skills: Optional[List[Dict[str, Any]]] = None,
        stop_callback: Optional[Any] = None,
    ) -> SuiteResult:
        """Score ``module`` on ``suite``'s ``split``.

        ``stop_callback`` (B2 SPRT, audit 2026-08-17) is an opt-in callable
        invoked after each ``CaseResult`` is appended, with the running
        ``results`` list; if it returns True, scoring stops and a PARTIAL
        ``SuiteResult`` is returned. Default ``None`` -> no callback, and the
        run is byte-identical to pre-B2. The callback is the SPRT's only
        early-stop hook; it may ONLY ever request a stop to REJECT (the SPRT's
        ``should_stop`` is reject-only), and the harness never decides
        promotion, so an early stop here always means "stop evaluating, this is
        failing" -- never "stop, this passed".
        """
        cases = suite.split(split)
        if not cases:
            raise EvaluationError(
                "suite '{}' has no cases in split '{}'".format(suite.suite_id, split)
            )
        capability = suite.capability()
        metric_plugin = self.plugins.metric(suite.modality) if self.plugins else None

        # --- produce one actual per case (A2: optionally batched). ---
        # Effective batch is the min of the harness cap and the module's
        # declared preferred batch -- a module is never forced to batch beyond
        # what it said was safe, and the harness cap is the global ceiling. At
        # effective <= 1 this is the single-case path, byte-identical to pre-A2.
        # At > 1 the chunked path calls infer_batch / infer_with_skills_batch;
        # any exception is wrapped in BatchedInferenceError (NO silent fallback
        # to single-case -- that would hide an OOM and change outputs) and a
        # wrong output COUNT raises InferenceCountMismatchError (NO silent
        # count-based misalignment). ORDER preservation (output i answers
        # prompts[i]) is a module obligation the harness CANNOT verify without
        # ground-truth labels, which would defeat the held-out split -- see
        # ModuleAdapter.infer_batch; the boundary is pinned by
        # tests/test_batched_inference.py::test_order_preservation_is_a_module_obligation_*.
        # The cap is read ONCE here at run() entry (not per-chunk), so a module
        # that mutates its own preferred_batch_size mid-run is honoured at the
        # original value for the whole run.
        effective_batch = min(
            self.max_batch_size, getattr(module, "preferred_batch_size", 1)
        )

        results: List[CaseResult] = []

        def _score(case, actual) -> CaseResult:
            sim = self.similarity.similarity(str(case.expected), str(actual))
            if metric_plugin is not None:
                task = metric_plugin.score(case.expected, actual, None)
            else:
                task = sim
            lang = metrics.language_preservation(actual, suite.language)
            halluc = metrics.hallucination_risk(actual, case.expected)
            # Schema compliance is a packet-level property, not a case-level one,
            # so a running module is credited 1.0 here and checked at gate time.
            score = metrics.aggregate(1.0, sim, task, lang, halluc)
            return CaseResult(
                case_id=case.case_id,
                prompt=case.prompt,
                expected=case.expected,
                actual=actual,
                similarity=sim,
                task_success=task,
                language_preservation=lang,
                hallucination_risk=halluc,
                score=score,
            )

        def _should_stop() -> bool:
            # Fire the SPRT callback after each scored case. With no callback
            # (the default) this is a constant False and run() is byte-identical
            # to pre-B2. Inference and scoring are interleaved per case (per
            # chunk for the batched path) so an early stop saves real inference
            # time, not just a post-hoc scoring loop.
            return stop_callback is not None and bool(stop_callback(results))

        # Inference + scoring interleaved (see _should_stop). The CaseResults
        # are built in suite order with the same math as the pre-B2 two-phase
        # path, so with no callback the SuiteResult is byte-identical.
        if effective_batch <= 1:
            for case in cases:
                actual = (
                    module.infer_with_skills(capability, case.prompt, skills)
                    if skills
                    else module.infer(capability, case.prompt)
                )
                results.append(_score(case, actual))
                if _should_stop():
                    return self._result(suite, module, split, skills, results)
        else:
            for start in range(0, len(cases), effective_batch):
                chunk = cases[start:start + effective_batch]
                prompts = [c.prompt for c in chunk]
                try:
                    if skills:
                        outs = module.infer_with_skills_batch(capability, prompts, skills)
                    else:
                        outs = module.infer_batch(capability, prompts)
                except Exception as exc:
                    raise BatchedInferenceError(
                        module.module_id, len(prompts), exc
                    ) from exc
                if len(outs) != len(prompts):
                    raise InferenceCountMismatchError(
                        module.module_id, expected=len(prompts), got=len(outs)
                    )
                for case, actual in zip(chunk, outs):
                    results.append(_score(case, actual))
                    if _should_stop():
                        return self._result(suite, module, split, skills, results)

        return self._result(suite, module, split, skills, results)

    def _result(self, suite, module, split, skills, results) -> SuiteResult:
        return SuiteResult(
            suite_id=suite.suite_id,
            module_id=module.module_id,
            split=split,
            with_skills=bool(skills),
            case_results=results,
            similarity_is_semantic=self.similarity.is_semantic,
        )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_suite(path: Path) -> BenchmarkSuite:
    with open(path, "r", encoding="utf-8") as fh:
        return BenchmarkSuite.model_validate(json.load(fh))


def load_all(directory: Path) -> Dict[str, BenchmarkSuite]:
    suites: Dict[str, BenchmarkSuite] = {}
    for path in sorted(Path(directory).glob("*.json")):
        suite = load_suite(path)
        suites[suite.suite_id] = suite
    return suites
