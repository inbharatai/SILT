"""Teacher-score cache (A1, audit 2026-08-17).

Gap negotiation measures the sender (teacher) on every suite's extraction
split. Under deterministic decoding that score is a pure function of (teacher,
suite, harness), so re-measuring it on a re-run is wasted GPU time. A1 caches
it, keyed by (teacher fingerprint, suite fingerprint, harness fingerprint),
read-through on the SENDER path only. The receiver is never cached.

These tests pin the honesty contract, not just the happy path:

  * a repeat sender run is served from cache (no second harness.run);
  * the receiver is ALWAYS fresh, even with a cache attached;
  * a suite whose prompts OR reference answers change is a miss (no stale
    serve);
  * a different teacher is a miss;
  * a different harness (similarity backend) is a miss;
  * no cache attached == byte-identical to the pre-A1 behaviour;
  * the cache does not change the gap results; and
  * disk backing survives a fresh cache instance.
"""

from __future__ import annotations

import copy

import pytest

from asea.benchmarks.cache import TeacherScoreCache, suite_fingerprint
from asea.benchmarks.harness import BenchmarkHarness
from asea.core.errors import CacheCorruptionError
from asea.core.gap import GapEngine
from asea.core.plugins import default_registry
from asea.core.protocol import CapabilityKey, Domain, Modality
from asea.evaluator.similarity import ExactMatch, LexicalSimilarity
from asea.modules.mock.zoo import make_generic_receiver, make_generic_sender, text_cap


class _Counting:
    """Wraps a module to count infer() calls; delegates everything else
    (manifest, module_id, fingerprint, is_mock, ...) via __getattr__ so the
    harness and GapEngine see a faithful proxy of the wrapped module."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.infer_calls = 0

    def __getattr__(self, name):
        # __getattr__ only fires for attrs NOT found normally; infer is found
        # below, so it is counted, everything else is delegated untouched.
        return getattr(self._inner, name)

    def infer(self, capability, prompt):
        self.infer_calls += 1
        return self._inner.infer(capability, prompt)

    def infer_with_skills(self, capability, prompt, skills):
        self.infer_calls += 1
        return self._inner.infer_with_skills(capability, prompt, skills)


def _harness(similarity=None):
    return BenchmarkHarness(plugins=default_registry(), similarity=similarity)


def _as_cap():
    return CapabilityKey(
        task_type="translate", modality=Modality.TEXT,
        domain=Domain.TRANSLATION, language="as->en",
    )


def _sender(module_id="generic-sender-mock"):
    # Knows some as->en extraction cases so measure() actually scores it.
    return make_generic_sender(
        module_id=module_id,
        capabilities=[text_cap("translate", "as->en")],
        knowledge={"translate/text/translation/as->en": {"ভাত": "rice", "পানী": "water"}},
    )


def _recv():
    return make_generic_receiver(capabilities=[_as_cap()])


def test_no_cache_means_fresh_sender_every_time(as_en_suite):
    """Pre-A1 behaviour: with no cache, every measure() re-runs the sender."""
    sender = _Counting(_sender())
    engine = GapEngine(harness=_harness())  # teacher_cache=None
    engine.measure(sender, _recv(), [as_en_suite])
    first = sender.infer_calls
    assert first > 0
    engine.measure(sender, _recv(), [as_en_suite])
    assert sender.infer_calls == 2 * first, "without a cache the sender runs again"


def test_cache_short_circuits_repeat_sender_runs(as_en_suite):
    """A second measure() with the same teacher+suite+harness is a cache hit:
    the sender's infer is NOT called again."""
    cache = TeacherScoreCache()
    sender = _Counting(_sender())
    engine = GapEngine(harness=_harness(), teacher_cache=cache)
    engine.measure(sender, _recv(), [as_en_suite])
    after_first = sender.infer_calls
    assert after_first > 0
    assert len(cache) == 1
    engine.measure(sender, _recv(), [as_en_suite])
    assert sender.infer_calls == after_first, "cache hit must not re-run the sender"
    assert len(cache) == 1, "a hit must not create a second entry"


def test_receiver_is_never_cached(as_en_suite):
    """The receiver is the thing being improved -- it is ALWAYS measured
    fresh, cache or no cache. Counting the receiver (not the sender) proves
    the cache only sits on the sender path."""
    cache = TeacherScoreCache()
    sender = _sender()
    receiver = _Counting(make_generic_receiver(capabilities=[_as_cap()]))
    engine = GapEngine(harness=_harness(), teacher_cache=cache)
    engine.measure(sender, receiver, [as_en_suite])
    after_first = receiver.infer_calls
    assert after_first > 0
    engine.measure(sender, receiver, [as_en_suite])
    assert receiver.infer_calls == 2 * after_first, "receiver must run fresh every time"
    assert len(cache) == 1, "only the sender entry is cached"


def test_suite_content_change_invalidates(as_en_suite):
    """A suite whose reference answer changes is a different suite_fingerprint
    and therefore a cache miss -- a stale score is never served."""
    suite_a = as_en_suite
    suite_b = copy.deepcopy(as_en_suite)
    # Mutate one expected on the EXTRACTION split (the split the cache keys on).
    for c in suite_b.cases:
        if c.split == "extraction":
            c.expected = "CHANGED-REFERENCE"
            break
    assert suite_fingerprint(suite_a) != suite_fingerprint(suite_b)

    cache = TeacherScoreCache()
    sender = _Counting(_sender())
    engine = GapEngine(harness=_harness(), teacher_cache=cache)
    engine.measure(sender, _recv(), [suite_a])
    after_first = sender.infer_calls
    engine.measure(sender, _recv(), [suite_b])
    assert sender.infer_calls > after_first, "changed suite must be a cache miss"
    assert len(cache) == 2


def test_teacher_change_is_a_cache_miss(as_en_suite):
    """A different teacher (different module_id -> different fingerprint) is a
    cache miss even with the same suite and harness."""
    cache = TeacherScoreCache()
    sender_a = _Counting(_sender(module_id="teacher-a"))
    sender_b = _Counting(_sender(module_id="teacher-b"))
    engine = GapEngine(harness=_harness(), teacher_cache=cache)
    engine.measure(sender_a, _recv(), [as_en_suite])
    after_first = sender_b.infer_calls
    engine.measure(sender_b, _recv(), [as_en_suite])
    assert sender_b.infer_calls > after_first, "different teacher must be a miss"
    assert len(cache) == 2


def test_harness_change_invalidates(as_en_suite):
    """Swapping the similarity backend changes the harness fingerprint, so a
    score cached under lexical similarity is NOT served under exact-match
    similarity -- a score computed under one harness is never silently served
    under another."""
    cache = TeacherScoreCache()
    sender = _Counting(_sender())
    engine_lex = GapEngine(
        harness=_harness(similarity=LexicalSimilarity()), teacher_cache=cache
    )
    engine_exact = GapEngine(
        harness=_harness(similarity=ExactMatch()), teacher_cache=cache
    )
    engine_lex.measure(sender, _recv(), [as_en_suite])
    after_first = sender.infer_calls
    engine_exact.measure(sender, _recv(), [as_en_suite])
    assert sender.infer_calls > after_first, "different harness must be a miss"
    assert len(cache) == 2


def test_cache_does_not_change_gap_results(as_en_suite):
    """Caching is score-equivalent: the gaps measured with a cache equal the
    gaps measured without one (same sender_score, same receiver_score, same
    actionability). The cache must be an optimisation, not a behaviour change."""
    receiver = make_generic_receiver(capabilities=[_as_cap()])
    sender = _sender()

    uncached = GapEngine(harness=_harness()).measure(sender, receiver, [as_en_suite])
    cached_engine = GapEngine(harness=_harness(), teacher_cache=TeacherScoreCache())
    cached = cached_engine.measure(sender, receiver, [as_en_suite])
    # A second cached run forces a hit so the served-from-cache path is
    # included in the equivalence check, not just the miss-then-store path.
    cached_again = cached_engine.measure(sender, receiver, [as_en_suite])

    def signature(gaps):
        return [(g.capability.as_str(), round(g.sender_score, 6),
                 round(g.receiver_score, 6)) for g in gaps]

    assert signature(uncached) == signature(cached) == signature(cached_again)


def test_disk_backing_persists_across_instances(as_en_suite, tmp_path):
    """An on-disk cache survives a fresh TeacherScoreCache pointed at the same
    dir -- a hit is served from disk without re-running the sender."""
    dir1 = tmp_path / "teacher_cache"
    cache = TeacherScoreCache(backing_dir=dir1)
    sender = _Counting(_sender())
    engine = GapEngine(harness=_harness(), teacher_cache=cache)
    engine.measure(sender, _recv(), [as_en_suite])
    assert sender.infer_calls > 0
    calls_before = sender.infer_calls

    # A brand-new cache instance on the same dir must see the persisted entry.
    cache2 = TeacherScoreCache(backing_dir=dir1)
    engine2 = GapEngine(harness=_harness(), teacher_cache=cache2)
    engine2.measure(sender, _recv(), [as_en_suite])
    assert sender.infer_calls == calls_before, "disk-backed hit must not re-run sender"


def test_similarity_backend_reconfiguration_invalidates(as_en_suite):
    """F1 (audit 2026-08-17): the harness fingerprint covers the similarity
    backend's INSTANCE config, not just its class. Two LexicalSimilarity
    instances with different char_weight/token_weight produce different scores
    on the same inputs, so they MUST key differently -- a score cached under
    one weighting is never served under the other. (The class-swap case is
    already covered by test_harness_change_invalidates; this is the
    same-class/different-config case the green suite did NOT cover.)"""
    cache = TeacherScoreCache()
    sender = _Counting(_sender())
    engine_w1 = GapEngine(
        harness=_harness(similarity=LexicalSimilarity(char_weight=0.4, token_weight=0.6)),
        teacher_cache=cache,
    )
    engine_w2 = GapEngine(
        harness=_harness(similarity=LexicalSimilarity(char_weight=0.9, token_weight=0.1)),
        teacher_cache=cache,
    )
    engine_w1.measure(sender, _recv(), [as_en_suite])
    after_first = sender.infer_calls
    engine_w2.measure(sender, _recv(), [as_en_suite])
    assert sender.infer_calls > after_first, (
        "reconfiguring the similarity backend weights must be a cache miss"
    )
    assert len(cache) == 2


def test_corrupt_disk_entry_raises_typed_error_not_silent_miss(as_en_suite, tmp_path):
    """F2 (audit 2026-08-17): a clean miss is silent (returns None, caller
    re-runs), but a CORRUPT on-disk entry is a typed CacheCorruptionError -- the
    cache does not self-heal by silently discarding and recomputing, because
    that would hide a disk/serialisation problem. put() writes atomically so
    normal operation cannot produce such a file; this simulates external
    corruption (disk error / manual edit) and checks the read surfaces it."""
    cache_dir = tmp_path / "teacher_cache"
    cache = TeacherScoreCache(backing_dir=cache_dir)
    sender = _Counting(_sender())
    engine = GapEngine(harness=_harness(), teacher_cache=cache)
    engine.measure(sender, _recv(), [as_en_suite])
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    # Externally corrupt the entry (truncated / invalid JSON).
    files[0].write_text("{ this is not valid json ", encoding="utf-8")

    key = engine._sender_cache_key(sender, as_en_suite)
    fresh = TeacherScoreCache(backing_dir=cache_dir)  # empty _mem -> reads disk
    with pytest.raises(CacheCorruptionError):
        fresh.get(key)