"""Gap negotiation.

The adapter refuses to transfer anything until it can name a *measured*
deficiency. Manifest claims alone are not sufficient evidence: a module that
says it speaks Assamese may be wrong, and a module that does not claim Assamese
may still handle it adequately. So the engine intersects two sources:

  1. declared capability sets (cheap, from the handshake), and
  2. measured scores on the ``extraction`` split (expensive, authoritative).

Measured evidence wins. A "gap" with no headroom is dropped, which is what stops
the system from generating busywork packets that cannot improve anything.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from ..benchmarks.cache import TeacherScoreCache, suite_fingerprint
from ..benchmarks.harness import BenchmarkHarness, BenchmarkSuite
from ..core.interfaces import ModuleAdapter
from ..core.protocol import CapabilityManifest, Gap


class GapPolicy:
    """Thresholds controlling when a deficiency is worth acting on."""

    def __init__(
        self,
        receiver_ceiling: float = 0.85,
        min_headroom: float = 0.05,
    ) -> None:
        #: Receivers already scoring above this are treated as competent.
        self.receiver_ceiling = receiver_ceiling
        #: Sender must beat receiver by at least this margin.
        self.min_headroom = min_headroom


class GapEngine:
    def __init__(
        self,
        harness: Optional[BenchmarkHarness] = None,
        policy: Optional[GapPolicy] = None,
        teacher_cache: Optional[TeacherScoreCache] = None,
    ) -> None:
        self.harness = harness or BenchmarkHarness()
        self.policy = policy or GapPolicy()
        #: Teacher (sender) extraction-score cache (A1, audit 2026-08-17).
        #: When None, every sender run is fresh -- byte-identical to the
        #: pre-A1 behaviour, so the existing suite is unaffected. When set,
        #: the sender's extraction result is served read-through on a hit and
        #: stored on a miss. The receiver is NEVER cached -- see
        #: :meth:`_sender_result` and the honesty contract in
        #: :mod:`asea.benchmarks.cache`.
        self.teacher_cache = teacher_cache

    # -- declared-level diff ---------------------------------------------

    def declared_gaps(
        self, sender: CapabilityManifest, receiver: CapabilityManifest
    ) -> List[str]:
        """Capability keys the sender claims and the receiver does not."""
        return sorted(sender.capability_set() - receiver.capability_set())

    # -- measured diff ----------------------------------------------------

    def measure(
        self,
        sender: ModuleAdapter,
        receiver: ModuleAdapter,
        suites: List[BenchmarkSuite],
    ) -> List[Gap]:
        """Score both modules on the diagnostic split and emit real gaps."""
        gaps: List[Gap] = []
        for suite in suites:
            if not suite.split("extraction"):
                continue
            capability = suite.capability()

            receiver_result = self.harness.run(receiver, suite, split="extraction")
            sender_result = self._sender_result(sender, suite)

            gap = Gap(
                capability=capability,
                receiver_score=receiver_result.score,
                sender_score=sender_result.score,
                declared_only=False,
            )
            if self._is_actionable(gap):
                gaps.append(gap)
        return gaps

    def _is_actionable(self, gap: Gap) -> bool:
        if gap.receiver_score >= self.policy.receiver_ceiling:
            return False
        return gap.headroom >= self.policy.min_headroom

    # -- teacher cache (A1) ------------------------------------------------

    def _harness_fingerprint(self, suite: BenchmarkSuite) -> str:
        """Hash of everything about this engine's harness that affects a
        sender's extraction score: the similarity backend (via its
        :meth:`fingerprint`, which covers INSTANCE config -- e.g.
        :class:`LexicalSimilarity`'s weights -- not just the class) and the
        metric plugin registered for this suite's modality (via its
        :meth:`fingerprint`, or "none"). The suite itself is hashed separately
        (:func:`suite_fingerprint`); the teacher separately
        (:meth:`ModuleAdapter.fingerprint`). Including the harness here means
        swapping lexical for embedding similarity, RECONFIGURING a backend's
        weights, or swapping a metric plugin, invalidates the cache -- a score
        computed under one harness is never silently served under another
        (audit 2026-08-17, F1/F4)."""
        sim_fp = self.harness.similarity.fingerprint()
        metric_fp = "none"
        if self.harness.plugins is not None:
            plugin = self.harness.plugins.metric(suite.modality)
            if plugin is not None:
                metric_fp = plugin.fingerprint()
        raw = "|".join([sim_fp, metric_fp])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _sender_cache_key(self, sender: ModuleAdapter, suite: BenchmarkSuite) -> str:
        return "t={}|s={}|h={}".format(
            sender.fingerprint(), suite_fingerprint(suite), self._harness_fingerprint(suite)
        )

    def _sender_result(self, sender: ModuleAdapter, suite: BenchmarkSuite):
        """Run the sender on the extraction split, OR serve its cached result.

        Only the SENDER path is cached. The receiver is measured fresh in
        :meth:`measure` and is the thing the transfer is supposed to change --
        caching it would freeze the improvement we are trying to detect. A
        miss re-runs the harness (the honest, expensive path) and stores the
        result; a hit returns the prior ``SuiteResult`` unchanged. With no
        cache attached (``teacher_cache is None``) this is a direct
        ``harness.run`` -- byte-identical to the pre-A1 behaviour."""
        if self.teacher_cache is None:
            return self.harness.run(sender, suite, split="extraction")
        key = self._sender_cache_key(sender, suite)
        cached = self.teacher_cache.get(key)
        if cached is not None:
            return cached
        result = self.harness.run(sender, suite, split="extraction")
        self.teacher_cache.put(key, result)
        return result

    # -- combined ---------------------------------------------------------

    def negotiate_with_gaps(
        self,
        sender: ModuleAdapter,
        receiver: ModuleAdapter,
        suites: List[BenchmarkSuite],
    ) -> tuple:
        """Negotiate AND return the measured gaps in one pass.

        ``measure`` scores both modules on every suite's extraction split -- the
        expensive, authoritative call. Calling ``negotiate`` (which runs
        ``measure``) and then ``measure`` again double-bills that cost: for real
        models that is a second full sender+receiver evaluation across all
        suites for no reason, since the gaps are deterministic under
        ``do_sample=False``. This method measures once and returns both the
        loggable report dict and the ``Gap`` objects the pipeline iterates.
        """
        s_manifest, r_manifest = sender.manifest(), receiver.manifest()
        measured = self.measure(sender, receiver, suites)
        report = {
            "sender": s_manifest.module_id,
            "receiver": r_manifest.module_id,
            "declared_gaps": self.declared_gaps(s_manifest, r_manifest),
            "measured_gaps": [
                {
                    "capability": g.capability.as_str(),
                    "receiver_score": round(g.receiver_score, 4),
                    "sender_score": round(g.sender_score, 4),
                    "headroom": round(g.headroom, 4),
                }
                for g in measured
            ],
            "actionable": len(measured),
        }
        return report, measured

    def negotiate(
        self,
        sender: ModuleAdapter,
        receiver: ModuleAdapter,
        suites: List[BenchmarkSuite],
    ) -> Dict[str, object]:
        """Full negotiation result, suitable for logging verbatim.

        Thin wrapper over :meth:`negotiate_with_gaps`; callers that also need the
        ``Gap`` objects should call that instead to avoid a second measurement.
        """
        return self.negotiate_with_gaps(sender, receiver, suites)[0]
