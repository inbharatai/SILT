"""Capability Diff (B1a, audit 2026-08-17).

A *capability diff* answers a question the promotion gate cannot: "between two
checkpoints of this receiver's approved set -- snapshot A and snapshot B --
what actually moved, per capability, on held-out data?" The gate decides one
packet at a time; the diff quantifies the cumulative effect of a whole batch of
promotions (or a rollback, or a re-run) in one signed report.

It is the single patentable-novel feature the roadmap ranked "if you only pick
one", and it is built to be honest about exactly two things:

1. **It reuses the evaluator's scoring path, it does not reimplement it.** The
   per-capability score under a snapshot is ``harness.run(receiver, suite,
   split="heldout", skills=<receiver's approved set in that snapshot>)`` -- the
   *same* call :class:`~asea.evaluator.evaluator.Evaluator` makes at
   ``evaluator.py:131-134``. The only thing that varies between A and B is the
   approved skill set; the receiver, the harness, the similarity backend, the
   metric plugin and the held-out cases are identical across the two
   measurements. So any score delta is attributable to the skill-set delta, not
   to measurement drift. (Verified by ``test_diff_score_reuses_harness_scoring``
   -- the diff's ``score_a`` equals a direct ``harness.run`` with the same
   skills, to the byte.)

2. **The signature is local HMAC, and the report says so.** The report is
   HMAC-SHA256-signed with a key that lives at ``<workspace>/diff.key``, is
   generated on first use, and is NEVER uploaded (the signing key is host-local; patent pending
   India, provisional filed 2026-08-21 -- see INTEGRATION_PROMPT). The signature proves the report was
   not tampered with *after generation, to the holder of that same local key*.
   It is NOT a portable third-party attestation: anyone without the key cannot
   verify it, and possession of the key does not prove authorship to a stranger.
   That stronger property (portable asymmetric attestation) is B1b and is
   deliberately NOT built (out of scope of this release). The report carries an
   ``honesty_note`` stating this verbatim so no one mistakes a local HMAC for a
   notarised certificate.

HONESTY CONTRACT (binding):

  * The diff conditions the receiver on its ENTIRE approved set in each
    snapshot (every capability, not just the suite's) -- because that is what
    the receiver actually consumes (``MemoryStore.approved_skills(target)``
    returns all approved packets for a module). Conditioning on only the
    matching-capability packets would flatter the diff by hiding cross-capability
    bleed.
  * Verdict thresholds are the SAME as the gates: ``regression_tolerance``
    (default 0.02) for improved/regressed, ``max_control_movement`` (default
    0.05) for moved. A capability delta is flagged ``improved`` (delta >
    +tol), ``regressed`` (delta < -tol), and ``moved`` (|delta| >
    max_control_movement) INDEPENDENTLY -- a big improvement is both improved
    AND moved, and that is reported, not buried.
  * An EMPTY snapshot (token exists but its approved set was empty) is a
    legitimate delta of zero, NOT an error -- the receiver is measured under no
    skills (its native baseline). A MISSING snapshot (token does not exist, or
    escapes the snapshots directory) IS an error:
    :class:`~asea.core.errors.SnapshotNotFoundError`. The two must not be
    conflated; a missing snapshot silently treated as empty would fabricate a
    "no change" diff.
  * ``packets_added`` / ``packets_removed`` are by a 16-char content hash over
    ``{capability, distilled_skill}`` (the semantic payload that actually
    conditions the receiver), NOT by ``packet_id`` -- two packets with
    different ids but identical distilled content are the same skill, and
    re-running the pipeline (which regenerates uuids) must not fake an "added"
    entry. (This is a truncated hash of the redacted-skill payload, distinct
    from :meth:`SkillPacket.content_hash`, which also covers ``task_type`` and
    ``packet_type``; both are content-addressed, this one is keyed on exactly
    the fields the receiver consumes.)
  * Every failure mode is typed: missing snapshot, missing/corrupt signing key,
    tampered report. Nothing silently degrades to an empty diff, an unsigned
    report, or a ``valid=True`` verdict.
  * The signature covers a CANONICAL serialization (sorted keys, compact
    separators, floats pre-rounded to 6 decimals) of the report MINUS the
    ``signature`` field. Reproducible across runs and across Python versions --
    a re-serialisation that drifted a float's repr would otherwise break
    verification silently.
  * Nothing here weakens a gate. The diff is a read-only measurement over
    snapshots; it promotes nothing, writes nothing to the approved set, and
    touches the network not at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._signing import SIGNATURE_ALG, LocalSigner
from .benchmarks.harness import BenchmarkHarness, BenchmarkSuite
from .core.errors import (
    SignatureMismatchError,
    SigningKeyError,
    SnapshotNotFoundError,
)
from .core.interfaces import ModuleAdapter
from .memory.store import RollbackLayer

#: The honesty note carried verbatim in every report. Local HMAC, not portable.
HONESTY_NOTE = (
    "Local HMAC-SHA256 signature. Tamper-evident to the holder of the local "
    "signing key only; NOT a portable third-party attestation and NOT proof of "
    "authorship to anyone without the key. Patent pending (India, filed "
    "2026-08-21); local only -- never uploaded. Portable asymmetric "
    "attestation (B1b) is deliberately not built (out of scope of this release)."
)

#: Where the local signing key lives, relative to the workspace.
KEY_FILENAME = "diff.key"

#: Float precision for the signed canonical payload (avoids repr drift).
_FLOAT_PRECISION = 6


def _round(x: float) -> float:
    return round(float(x), _FLOAT_PRECISION)


@dataclass
class CapabilityDelta:
    """One capability's before/after under the two approved sets."""

    capability: str
    score_a: float  # receiver + snapshot A's approved set, held-out
    score_b: float  # receiver + snapshot B's approved set, held-out
    delta: float  # score_b - score_a
    improved: bool  # delta > +regression_tolerance
    regressed: bool  # delta < -regression_tolerance
    moved: bool  # |delta| > max_control_movement
    cases: int  # held-out case count (identical for a/b by construction)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "score_a": _round(self.score_a),
            "score_b": _round(self.score_b),
            "delta": _round(self.delta),
            "improved": self.improved,
            "regressed": self.regressed,
            "moved": self.moved,
            "cases": self.cases,
        }


@dataclass
class DiffReport:
    """A signed capability-diff report."""

    workspace: str
    receiver: str
    token_a: str
    token_b: str
    harness_fingerprint: str
    regression_tolerance: float
    max_control_movement: float
    deltas: List[CapabilityDelta]
    packets_added: List[str]  # content_hash, for this receiver
    packets_removed: List[str]
    summary: Dict[str, int]
    generated_at: str
    honesty_note: str
    key_fingerprint: str
    signature_alg: str
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": self.workspace,
            "receiver": self.receiver,
            "token_a": self.token_a,
            "token_b": self.token_b,
            "harness_fingerprint": self.harness_fingerprint,
            "regression_tolerance": _round(self.regression_tolerance),
            "max_control_movement": _round(self.max_control_movement),
            "deltas": [d.to_dict() for d in self.deltas],
            "packets_added": sorted(self.packets_added),
            "packets_removed": sorted(self.packets_removed),
            "summary": dict(self.summary),
            "generated_at": self.generated_at,
            "honesty_note": self.honesty_note,
            "key_fingerprint": self.key_fingerprint,
            "signature_alg": self.signature_alg,
            "signature": self.signature,
        }


class CapabilityDiffer:
    """Compute and sign capability diffs between two approved-set snapshots."""

    def __init__(
        self,
        harness: BenchmarkHarness,
        rollback: RollbackLayer,
        workspace: Path,
        regression_tolerance: float = 0.02,
        max_control_movement: float = 0.05,
    ) -> None:
        # Same defaults as the Evaluator so the diff's verdicts agree with what
        # the gates would say -- a diff that flagged "regressed" under a
        # different threshold than the gate would be internally inconsistent.
        self.harness = harness
        self.rollback = rollback
        self.workspace = Path(workspace)
        self.regression_tolerance = regression_tolerance
        self.max_control_movement = max_control_movement
        # Local HMAC signer with the diff's own key file -- a leaked diff key
        # cannot forge an erasure certificate (B3) and vice versa. The signer
        # implements the D1 (strict load, no mid-verify mint) and A1 (non-string
        # signature -> typed error) guarantees shared across signed reports.
        self._signer = LocalSigner(self.workspace, KEY_FILENAME)

    # -- public API -------------------------------------------------------

    def diff(
        self,
        receiver: ModuleAdapter,
        suites: List[BenchmarkSuite],
        token_a: str,
        token_b: str,
    ) -> DiffReport:
        """Measure the receiver under snapshot A vs snapshot B on every
        suite's held-out split and return a SIGNED report.

        Both snapshots are loaded via :meth:`RollbackLayer.snapshot_packets`,
        which applies the same path-escape guard as rollback. The receiver is
        conditioned on its ENTIRE approved set in each snapshot (all
        capabilities), matching what it actually consumes.
        """
        skills_a = self._receiver_skills(token_a, receiver)
        skills_b = self._receiver_skills(token_b, receiver)

        deltas: List[CapabilityDelta] = []
        for suite in suites:
            heldout = suite.split("heldout")
            if not heldout:
                # A suite with no held-out split cannot be diffed -- there is
                # nothing to measure. Skip it explicitly rather than emitting a
                # zero-case delta that would look like "measured and found no
                # change". Silence here would be a loophole; the skip is
                # visible because the capability is absent from the report.
                continue
            res_a = self.harness.run(receiver, suite, split="heldout", skills=skills_a)
            res_b = self.harness.run(receiver, suite, split="heldout", skills=skills_b)
            delta = res_b.score - res_a.score
            deltas.append(
                CapabilityDelta(
                    capability=suite.capability().as_str(),
                    score_a=res_a.score,
                    score_b=res_b.score,
                    delta=delta,
                    improved=delta > self.regression_tolerance,
                    regressed=delta < -self.regression_tolerance,
                    moved=abs(delta) > self.max_control_movement,
                    cases=len(res_a.case_results),
                )
            )

        added, removed = self._packet_delta(skills_a, skills_b)
        summary = {
            "improved": sum(1 for d in deltas if d.improved),
            "regressed": sum(1 for d in deltas if d.regressed),
            "moved": sum(1 for d in deltas if d.moved),
            "unchanged": sum(
                1 for d in deltas if not d.improved and not d.regressed and not d.moved
            ),
            "packets_added": len(added),
            "packets_removed": len(removed),
        }

        report = DiffReport(
            workspace=str(self.workspace),
            receiver=receiver.module_id,
            token_a=token_a,
            token_b=token_b,
            harness_fingerprint=self._harness_fingerprint(suites),
            regression_tolerance=self.regression_tolerance,
            max_control_movement=self.max_control_movement,
            deltas=deltas,
            packets_added=sorted(added),
            packets_removed=sorted(removed),
            summary=summary,
            generated_at=datetime.now(timezone.utc).isoformat(),
            honesty_note=HONESTY_NOTE,
            key_fingerprint=self._signer.key_fingerprint(),
            signature_alg=SIGNATURE_ALG,
        )
        report.signature = self._signer.sign(report.to_dict())
        return report

    def verify(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a report's HMAC against the local signing key.

        Returns ``{"valid": True, ...}`` on match; raises
        :class:`SignatureMismatchError` on a tampered report (including a
        missing or non-string signature field) and :class:`SigningKeyError` if
        the key is missing/unreadable. A missing key is NEVER a silent pass -- a
        report whose key has vanished cannot be verified. Signing is delegated
        to :class:`~asea._signing.LocalSigner`, which loads the key ONCE and
        strictly (no mid-verify mint) and treats a non-string signature as a
        typed mismatch (adversarial audit 2026-08-17, D1/A1).
        """
        result = self._signer.verify(report)  # raises typed errors
        # The signer's verdict is generic; surface the diff-specific identifiers
        # so a caller can confirm WHICH report verified.
        result["token_a"] = report.get("token_a")
        result["token_b"] = report.get("token_b")
        result["receiver"] = report.get("receiver")
        return result

    # -- internals --------------------------------------------------------

    def _receiver_skills(self, token: str, receiver: ModuleAdapter) -> List[Dict[str, Any]]:
        """The receiver's ENTIRE approved set in a snapshot, redacted.

        All capabilities, not just one -- the receiver consumes its whole
        approved set, so the diff must condition on the whole set to be honest
        about cross-capability bleed. An empty snapshot returns ``[]`` (native
        baseline); a missing token raises :class:`SnapshotNotFoundError`.
        """
        packets = self.rollback.snapshot_packets(token)  # raises on missing/escape
        return [
            p.redacted_for_receiver()
            for p in packets
            if p.target_module == receiver.module_id
        ]

    def _packet_delta(
        self, skills_a: List[Dict[str, Any]], skills_b: List[Dict[str, Any]]
    ) -> tuple:
        """(added, removed) content hashes for this receiver across A->B.

        By ``content_hash``-equivalent distilled payload, NOT packet_id: the
        pipeline regenerates uuids on every run, so an id-based delta would fake
        churn on a re-run that produced identical skills. The redacted skill
        carries ``distilled_skill`` and ``capability`` -- hash those.
        """
        def _skill_hash(s: Dict[str, Any]) -> str:
            payload = {"capability": s.get("capability"), "distilled_skill": s.get("distilled_skill")}
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
            return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

        set_a = {_skill_hash(s) for s in skills_a}
        set_b = {_skill_hash(s) for s in skills_b}
        added = set_b - set_a
        removed = set_a - set_b
        return sorted(added), sorted(removed)

    def _harness_fingerprint(self, suites: List[BenchmarkSuite]) -> str:
        """Identity of the scoring harness used for this diff, so a reader can
        tell two diffs apart that were measured under different similarity
        backends or metric plugins (which would change every score). Mirrors
        :meth:`asea.core.gap.GapEngine._harness_fingerprint` so the diff and the
        gap engine describe the harness the same way: the similarity backend's
        :meth:`~asea.core.interfaces.SimilarityBackend.fingerprint` (which
        covers INSTANCE config, e.g. LexicalSimilarity weights) plus the
        :meth:`~asea.core.interfaces.MetricPlugin.fingerprint` of every metric
        plugin registered for a modality present in the diffed suites. This is
        descriptive (it rides in the report, not a cache key), but it is
        honest -- two diffs under different backends get different fingerprints.
        """
        sim_fp = self.harness.similarity.fingerprint()
        metric_fps = ["none"]
        if self.harness.plugins is not None:
            modalities = sorted({s.modality for s in suites})
            plugin_fps = []
            for mod in modalities:
                plugin = self.harness.plugins.metric(mod)
                if plugin is not None:
                    plugin_fps.append(plugin.fingerprint())
            if plugin_fps:
                metric_fps = sorted(plugin_fps)
        raw = "|".join([sim_fp] + metric_fps)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # -- signing is delegated to LocalSigner (see _signing.py) ------------