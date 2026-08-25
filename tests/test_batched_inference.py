"""Bounded per-model batch size (A2, audit 2026-08-17).

The harness runs a module over a suite's cases. By default it loops the
single-case ``infer`` path -- byte-identical to pre-A2, and the only safe
default for a 4-bit 7B on an 8 GB card where two cases co-resident OOM. A2 adds
an opt-in batched path: the harness caps the effective batch at
``min(max_batch_size, module.preferred_batch_size)`` and, when > 1, calls
``infer_batch`` / ``infer_with_skills_batch`` in chunks.

Honesty contract pinned here:
  * batch=1 (default) never calls the batched API -> bit-identical, no OOM risk;
  * the batched path produces the SAME SuiteResult as the single-case path
    (correctness, not just speedup), INCLUDING for a non-divisible case count
    whose final chunk is short (12 cases / batch 5 -> [5, 5, 2], never dropped);
  * effective batch is capped by BOTH the harness and the module (a module is
    never forced beyond what it declared safe);
  * a batched call that raises is wrapped in a typed BatchedInferenceError --
    NO silent fallback to single-case (that would hide an OOM and change
    outputs);
  * a batched call returning the wrong COUNT raises InferenceCountMismatchError
    -- NO silent count-based misalignment (outputs zipped to cases 1:1 by
    position). ORDER preservation (output i answers prompts[i]) is a MODULE
    obligation (see ModuleAdapter.infer_batch) that the harness CANNOT verify
    without ground-truth labels, which would defeat the held-out split -- it is
    load-bearing on the module, and the boundary is PINNED, not hidden, by
    test_order_preservation_is_a_module_obligation below;
  * the batch=1 path does NOT wrap exceptions (pre-A2 behaviour preserved).
"""

from __future__ import annotations

import pytest

from asea.benchmarks.harness import BenchmarkHarness
from asea.core.errors import BatchedInferenceError, InferenceCountMismatchError
from asea.core.plugins import default_registry
from asea.core.protocol import CapabilityKey, Domain, Modality
from asea.evaluator.similarity import LexicalSimilarity
from asea.modules.mock.zoo import make_generic_receiver, text_cap


_AS_CAP = CapabilityKey(
    task_type="translate", modality=Modality.TEXT,
    domain=Domain.TRANSLATION, language="as->en",
)


class _BatchMock:
    """A minimal module that records how it was called.

    Knows a {prompt: answer} table (so results are deterministic and
    score-equivalent to the single-case path). Tracks single ``infer`` calls
    and the sizes of every ``infer_batch`` / ``infer_with_skills_batch`` chunk.
    Can be made to raise on the batched path (simulated OOM) or to return a
    short list (count-mismatch bug)."""

    is_mock = True
    consumes_skills = True

    def __init__(self, knowledge, preferred_batch_size=1, fail_batch=False, short_batch=False):
        self.module_id = "batch-mock"
        self.display_name = "Batch Mock"
        self._knowledge = knowledge
        self.preferred_batch_size = preferred_batch_size
        self.fail_batch = fail_batch
        self.short_batch = short_batch
        self.infer_calls = 0
        self.batch_sizes = []
        self.skills_batch_sizes = []

    # -- identity ---------------------------------------------------------

    def manifest(self):
        from asea.core.protocol import CapabilityManifest, LearningLevel

        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=["sender"],
            capabilities=[_AS_CAP],
            max_learning_level=LearningLevel.L3_SKILL_PACKET,
            is_mock=True,
        )

    # -- behaviour --------------------------------------------------------

    def _answer(self, prompt):
        return self._knowledge.get(str(prompt).strip(), "unknown")

    def infer(self, capability, prompt):
        self.infer_calls += 1
        return self._answer(prompt)

    def infer_batch(self, capability, prompts):
        self.batch_sizes.append(len(prompts))
        if self.fail_batch:
            raise RuntimeError("simulated CUDA OOM")
        outs = [self._answer(p) for p in prompts]
        if self.short_batch:
            return outs[:-1]
        return outs

    def infer_with_skills(self, capability, prompt, skills):
        self.infer_calls += 1
        # Apply the skill's exact-match entries, else the knowledge table, so the
        # conditioned path is also deterministic and score-equivalent.
        from asea.modules.mock.base import lookup_in_skills

        hit = lookup_in_skills(skills, prompt)
        return hit if hit is not None else self._answer(prompt)

    def infer_with_skills_batch(self, capability, prompts, skills):
        self.skills_batch_sizes.append(len(prompts))
        if self.fail_batch:
            raise RuntimeError("simulated CUDA OOM")
        outs = [self.infer_with_skills(capability, p, skills) for p in prompts]
        if self.short_batch:
            return outs[:-1]
        return outs


def _harness(max_batch_size=1, similarity=None):
    return BenchmarkHarness(
        plugins=default_registry(),
        similarity=similarity or LexicalSimilarity(),
        max_batch_size=max_batch_size,
    )


def _as_en_suite():
    """A suite with > 4 extraction cases so batching into chunks > 1 is
    exercised. Reuses the bundled Assamese data (12 extraction cases)."""
    from asea.benchmarks.harness import load_suite
    from pathlib import Path

    return load_suite(Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "assamese_english.json")


def _knowledge():
    cases = _as_en_suite().split("extraction")
    return {str(c.prompt): str(c.expected) for c in cases}


def test_default_batch_size_one_never_calls_batched_api():
    """batch=1 (the default) is the single-case path: infer is called per case,
    infer_batch / infer_with_skills_batch are NEVER called. Byte-identical to
    pre-A2, and the only safe default for a memory-bounded GPU."""
    suite = _as_en_suite()
    module = _BatchMock(_knowledge(), preferred_batch_size=8)
    harness = _harness(max_batch_size=1)  # default
    result = harness.run(module, suite, split="extraction")
    assert module.infer_calls == len(suite.split("extraction"))
    assert module.batch_sizes == []
    assert module.skills_batch_sizes == []
    # And it produced real results (not empty).
    assert len(result.case_results) == len(suite.split("extraction"))


def test_module_preferred_batch_one_overrides_harness_cap():
    """Even if the harness allows batching, a module that declares
    preferred_batch_size=1 is never batched -- a 4-bit 7B that knows it has no
    headroom stays on the single-case path regardless of the harness cap."""
    suite = _as_en_suite()
    module = _BatchMock(_knowledge(), preferred_batch_size=1)
    harness = _harness(max_batch_size=8)
    harness.run(module, suite, split="extraction")
    assert module.batch_sizes == [], "module preferred_batch_size=1 must win"
    assert module.infer_calls == len(suite.split("extraction"))


def test_batched_path_is_score_equivalent_to_single_case():
    """The batched path produces the SAME SuiteResult as the single-case path
    -- batching is an optimisation, not a behaviour change. Compares the full
    case-result list (case_id, actual, similarity, score) element-wise."""
    suite = _as_en_suite()
    single = _BatchMock(_knowledge(), preferred_batch_size=1)
    batched = _BatchMock(_knowledge(), preferred_batch_size=4)

    r1 = _harness(max_batch_size=1).run(single, suite, split="extraction")
    r2 = _harness(max_batch_size=4).run(batched, suite, split="extraction")

    assert batched.batch_sizes, "the batched path must actually be used"
    # Chunks of 4 over 12 cases -> [4, 4, 4].
    assert max(batched.batch_sizes) <= 4
    assert single.batch_sizes == []

    def signature(res):
        return [(c.case_id, c.actual, round(c.similarity, 6), round(c.score, 6))
                for c in res.case_results]

    assert signature(r1) == signature(r2), "batched results must equal single-case"


def test_effective_batch_is_min_of_harness_and_module():
    """effective_batch = min(harness.max_batch_size, module.preferred_batch_size).
    Harness 4 + module 2 -> chunks of 2; harness 2 + module 4 -> chunks of 2."""
    suite = _as_en_suite()
    m1 = _BatchMock(_knowledge(), preferred_batch_size=2)
    _harness(max_batch_size=4).run(m1, suite, split="extraction")
    assert max(m1.batch_sizes) == 2, "module cap (2) wins over harness cap (4)"

    m2 = _BatchMock(_knowledge(), preferred_batch_size=4)
    _harness(max_batch_size=2).run(m2, suite, split="extraction")
    assert max(m2.batch_sizes) == 2, "harness cap (2) wins over module cap (4)"


def test_skills_path_is_also_batched():
    """The skills-conditioned path (candidate evaluation) is batched too -- not
    just the no-skills path -- so there is no silent loophole where one path
    stays slow. Same score-equivalence contract."""
    suite = _as_en_suite()
    skills = [{"distilled_skill": {"entries": [
        {"source": str(c.prompt), "target": str(c.expected)}
        for c in suite.split("extraction")
    ]}}]
    single = _BatchMock(_knowledge(), preferred_batch_size=1)
    batched = _BatchMock(_knowledge(), preferred_batch_size=3)

    r1 = _harness(max_batch_size=1).run(single, suite, split="extraction", skills=skills)
    r2 = _harness(max_batch_size=3).run(batched, suite, split="extraction", skills=skills)

    assert batched.skills_batch_sizes, "the skills path must be batched"
    assert max(batched.skills_batch_sizes) <= 3
    assert single.skills_batch_sizes == []

    def signature(res):
        return [(c.case_id, c.actual, round(c.score, 6)) for c in res.case_results]

    assert signature(r1) == signature(r2)


def test_batched_exception_is_wrapped_as_typed_error():
    """A batched call that raises (e.g. CUDA OOM) is wrapped in
    BatchedInferenceError carrying the module id, the attempted batch size and
    the original exception. NO silent fallback to single-case -- that would
    hide the OOM and change outputs."""
    suite = _as_en_suite()
    module = _BatchMock(_knowledge(), preferred_batch_size=4, fail_batch=True)
    with pytest.raises(BatchedInferenceError) as exc_info:
        _harness(max_batch_size=4).run(module, suite, split="extraction")
    err = exc_info.value
    assert err.module_id == "batch-mock"
    assert err.batch_size == 4
    assert isinstance(err.original, RuntimeError)
    assert "simulated CUDA OOM" in str(err.original)


def test_batch_size_one_does_not_wrap_exceptions():
    """At batch=1 the single-case infer path preserves pre-A2 behaviour: a raw
    exception from infer propagates AS-IS, not wrapped in BatchedInferenceError
    (wrapping is a batched-path-only concern)."""
    suite = _as_en_suite()

    class _Boom(_BatchMock):
        def infer(self, capability, prompt):
            raise ValueError("non-batched failure")

    module = _Boom(_knowledge(), preferred_batch_size=1)
    with pytest.raises(ValueError, match="non-batched failure"):
        _harness(max_batch_size=1).run(module, suite, split="extraction")


def test_wrong_output_count_raises_typed_mismatch_error():
    """A batched call returning the wrong number of outputs (a correctness bug)
    raises InferenceCountMismatchError -- the harness does NOT zip/truncate,
    which would silently misalign outputs to cases."""
    suite = _as_en_suite()
    module = _BatchMock(_knowledge(), preferred_batch_size=4, short_batch=True)
    with pytest.raises(InferenceCountMismatchError) as exc_info:
        _harness(max_batch_size=4).run(module, suite, split="extraction")
    err = exc_info.value
    assert err.expected > err.got, "short_batch returns one fewer than asked"


def test_module_without_preferred_batch_size_defaults_to_one():
    """A module that does not declare preferred_batch_size (e.g. a non-ABC
    adapter that forgot the knob) is treated as 1 via getattr's default -- the
    harness never forces batching on a module that did not opt in."""
    suite = _as_en_suite()
    module = _BatchMock(_knowledge(), preferred_batch_size=8)
    del module.preferred_batch_size  # simulate an adapter that forgot the knob
    _harness(max_batch_size=8).run(module, suite, split="extraction")
    assert module.batch_sizes == [], "absent preferred_batch_size defaults to 1"


def test_non_divisible_case_count_short_final_chunk_is_score_equivalent():
    """Adversarial audit 2026-08-17 (attack B): the two score-equivalence tests
    above both use case counts that divide evenly by the chosen batch size (12
    / 4 -> [4,4,4]; 12 / 3 -> [3,3,3,3]), so the short FINAL chunk path is
    never exercised. The Assamese fixture has 12 extraction cases, which
    divides evenly by 2, 3, 4 and 6 -- so every batch size the existing tests
    pick produces no tail. A future regression that dropped or padded the final
    chunk would pass all of them. This test uses batch 5 over 12 cases ->
    [5, 5, 2] and asserts (a) the chunk sizes are exactly [5, 5, 2] and
    (b) the result is still score-equivalent to the single-case path."""
    suite = _as_en_suite()
    n_cases = len(suite.split("extraction"))
    assert n_cases == 12, "fixture drift: this test assumes the 12-case Assamese suite"

    single = _BatchMock(_knowledge(), preferred_batch_size=1)
    batched = _BatchMock(_knowledge(), preferred_batch_size=5)

    r1 = _harness(max_batch_size=1).run(single, suite, split="extraction")
    r2 = _harness(max_batch_size=5).run(batched, suite, split="extraction")

    # [5, 5, 2] -- the tail is exactly the remainder, never dropped/padded.
    assert batched.batch_sizes == [5, 5, 2], (
        "non-divisible batch must produce a short final chunk, got {}".format(
            batched.batch_sizes
        )
    )
    assert single.batch_sizes == []

    def signature(res):
        return [(c.case_id, c.actual, round(c.similarity, 6), round(c.score, 6))
                for c in res.case_results]

    assert signature(r1) == signature(r2), (
        "the short final chunk must be scored identically to the single-case path"
    )


def test_order_preservation_is_a_module_obligation_not_harness_verified():
    """Adversarial audit 2026-08-17 (attack A): the harness enforces COUNT
    (wrong count -> InferenceCountMismatchError) but it CANNOT enforce ORDER --
    output i must answer prompts[i], yet without ground-truth labels (which
    would defeat the held-out split) there is nothing for the harness to check
    against. So ORDER preservation is a load-bearing MODULE obligation
    (see ModuleAdapter.infer_batch). This test PINS that boundary rather than
    hiding it: a module that returns same-count outputs in the WRONG order is
    NOT caught by the harness, and the corrupted result is produced silently.
    That is the honest, documented limit -- the alternative (pretending to
    verify order) would be the real loophole. A real batched connector that
    reorders internally (length-sorted batching, hash-bucketed continuous
    batching, unsorted async.gather) MUST un-sort before returning."""
    suite = _as_en_suite()
    knowledge = _knowledge()

    class _Reordering(_BatchMock):
        """Returns the right COUNT but in REVERSED order -- a faithful harness
        cannot tell the outputs are attached to the wrong cases without
        labels, and labels would give the held-out answers away."""

        def infer_batch(self, capability, prompts):
            self.batch_sizes.append(len(prompts))
            outs = [self._answer(p) for p in prompts]
            return list(reversed(outs))  # same count, wrong order

    module = _Reordering(knowledge, preferred_batch_size=4)
    # The harness does NOT raise: there is no count error, and it has no labels
    # to detect the order swap. This is the pinned boundary.
    result = _harness(max_batch_size=4).run(module, suite, split="extraction")
    assert module.batch_sizes, "the batched path was used"
    assert len(result.case_results) == len(suite.split("extraction")), (
        "count is preserved (the only thing the harness can check)"
    )
    # The result IS corrupted: each case_id now holds a different case's answer.
    # The harness cannot know -- that is the documented module obligation.
    correct = {c.case_id: knowledge[str(c.prompt)] for c in suite.split("extraction")}
    misaligned = [
        c.case_id for c in result.case_results if c.actual != correct[c.case_id]
    ]
    assert misaligned, (
        "an order-swapping module MUST corrupt the result (else it did not "
        "reorder); the harness not catching it is the pinned boundary"
    )