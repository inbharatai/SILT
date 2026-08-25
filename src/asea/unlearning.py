"""Verified unlearning (B3, audit 2026-08-17).

A *verified-unlearning* certificate answers: "a skill packet that was approved
has now been rolled back -- is the capability it conferred actually GONE?" It
does this by measuring the receiver under three conditions on the held-out split:

    baseline       = receiver alone (no skills)
    with_skill     = receiver + the approved set that CONTAINED the packet
                     (snapshot ``token_before``)
    post_rollback  = receiver + the approved set AFTER the rollback
                     (snapshot ``token_after``)

and certifying two independent, honest things:

1. **adapter_removed** -- the packet's content is absent from the post-rollback
   approved set the receiver reads (a content-hash delta over the two
   snapshots, NOT a packet_id delta -- the pipeline regenerates uuids, so an
   id-based check would fake churn on a re-run with identical skills).
2. **capability_gone** -- the post-rollback held-out score reverted to the
   no-skill baseline within ``tolerance``: ``abs(post - baseline) <= tolerance``
   (TWO-SIDED -- a post-rollback score *far below* baseline is a regression
   introduced by the after-set, not a reversion, and must NOT certify as "gone";
   adversarial audit 2026-08-17, finding 1). The lift the skill conferred
   (``with_skill - baseline``) is gone.

``verified = adapter_removed AND capability_gone``. ``substantive`` additionally
requires the skill actually conferred a measurable lift (``lift > tolerance``)
-- a skill that taught nothing is "unlearned" only trivially, and that
distinction is reported, not buried.

HONESTY BOUNDARY (binding, the "no loopholes" core of this feature):

  * This certifies SKILL-LAYER unlearning: the packet is gone from the approved
    set the receiver reads, AND the measured capability reverted to baseline.
    It does NOT certify weight-level forgetting. SILT trains no weights -- it
    conditions a receiver on redacted skill payloads. A real receiver connector
    with its own internal state (cache, KV, fine-tuned weights) may RETAIN
    capability after the adapter is removed; that is explicitly OUT OF SCOPE and
    never claimed. If ``post_rollback`` stays above baseline after removal,
    ``capability_gone`` is False and the certificate says NOT verified -- the
    honest outcome, never a silent pass.
  * A receiver that already knew the capability via its OWN knowledge (baseline
    already high) shows ``skill_conferred_lift=False``; the cert is
    ``verified`` but ``substantive=False`` and says so. Calling that "forgetting"
    would be a lie; calling it "trivially verified, the skill added nothing" is
    the truth, and the ``substantive`` field makes a reader unable to miss it.
  * The signature is the same LOCAL HMAC as the capability diff (see
    :mod:`asea._signing`): tamper-evident to the local key holder, NOT a
    portable third-party attestation. It uses its OWN key file (``unlearn.key``)
    so a leaked diff key cannot forge an erasure certificate. The
    ``honesty_note`` states this verbatim.
  * Every failure mode is typed: a missing snapshot raises
    :class:`~asea.core.errors.SnapshotNotFoundError`; a suite with no held-out
    split raises :class:`~asea.core.errors.UnlearningError`; a tampered/missing
    key raises :class:`~asea.core.errors.SigningKeyError` /
    :class:`~asea.core.errors.SignatureMismatchError`. Nothing silently degrades
    to "verified" or to an unsigned certificate.
  * This weakens no gate. It is a read-only measurement over two snapshots; it
    promotes nothing, writes nothing to the approved set, and touches the
    network not at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._signing import SIGNATURE_ALG, LocalSigner
from .benchmarks.harness import BenchmarkHarness, BenchmarkSuite
from .core.errors import SnapshotNotFoundError, UnlearningError
from .core.interfaces import ModuleAdapter
from .memory.store import RollbackLayer

#: The honesty note carried verbatim in every certificate. Skill-layer, not
#: weight-level; local HMAC, not portable attestation.
HONESTY_NOTE = (
    "Verified SKILL-LAYER unlearning: the packet is absent from the approved set "
    "the receiver reads and the measured held-out capability reverted to the "
    "no-skill baseline within tolerance. This does NOT certify weight-level "
    "forgetting -- SILT trains no weights; a receiver connector with internal "
    "state may retain capability independently, which is out of scope and not "
    "claimed. Signature is local HMAC-SHA256 (tamper-evident to the local key "
    "holder only, NOT a portable third-party attestation). Patent pending "
    "(India, filed 2026-08-21); local only -- never uploaded."
)

#: Where the local signing key lives, relative to the workspace. Distinct from
#: the diff's ``diff.key`` so the two report types cannot cross-forge.
KEY_FILENAME = "unlearn.key"

#: Float precision for the signed canonical payload (avoids repr drift).
_FLOAT_PRECISION = 6


def _round(x: float) -> float:
    return round(float(x), _FLOAT_PRECISION)


@dataclass
class ErasureCertificate:
    """A signed verified-unlearning certificate for one capability."""

    workspace: str
    receiver: str
    token_before: str
    token_after: str
    capability: str
    baseline_score: float        # receiver alone, held-out
    with_skill_score: float      # receiver + before-set (packet present)
    post_rollback_score: float   # receiver + after-set (packet removed)
    lift: float                  # with_skill - baseline (what the skill added)
    residual: float              # post - baseline (what remains after rollback)
    tolerance: float
    cases: int                   # held-out case count
    packets_removed: List[str]   # 16-char content hashes gone from before->after
    packets_added: List[str]     # 16-char content hashes NEW in after vs before
    adapter_removed: bool        # the packet content is absent from the after-set
    skill_conferred_lift: bool   # lift > tolerance (the skill actually taught)
    capability_gone: bool        # residual <= tolerance (reverted to baseline)
    verified: bool               # adapter_removed AND capability_gone
    substantive: bool            # verified AND skill_conferred_lift
    generated_at: str
    honesty_note: str
    key_fingerprint: str
    signature_alg: str
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": self.workspace,
            "receiver": self.receiver,
            "token_before": self.token_before,
            "token_after": self.token_after,
            "capability": self.capability,
            "baseline_score": _round(self.baseline_score),
            "with_skill_score": _round(self.with_skill_score),
            "post_rollback_score": _round(self.post_rollback_score),
            "lift": _round(self.lift),
            "residual": _round(self.residual),
            "tolerance": _round(self.tolerance),
            "cases": self.cases,
            "packets_removed": sorted(self.packets_removed),
            "packets_added": sorted(self.packets_added),
            "adapter_removed": self.adapter_removed,
            "skill_conferred_lift": self.skill_conferred_lift,
            "capability_gone": self.capability_gone,
            "verified": self.verified,
            "substantive": self.substantive,
            "generated_at": self.generated_at,
            "honesty_note": self.honesty_note,
            "key_fingerprint": self.key_fingerprint,
            "signature_alg": self.signature_alg,
            "signature": self.signature,
        }


class UnlearningVerifier:
    """Measure and certify skill-layer unlearning between two approved-set
    snapshots for one capability (suite)."""

    def __init__(
        self,
        harness: BenchmarkHarness,
        rollback: RollbackLayer,
        workspace: Path,
        tolerance: float = 0.02,
    ) -> None:
        # Same default tolerance as the Evaluator's regression_tolerance so the
        # cert's "reverted to baseline" bar agrees with what the gate would call
        # a regression -- a cert that called 0.03 "gone" while the gate called
        # it a regression would be internally inconsistent.
        self.harness = harness
        self.rollback = rollback
        self.workspace = Path(workspace)
        self.tolerance = tolerance
        self._signer = LocalSigner(self.workspace, KEY_FILENAME)

    # -- public API -------------------------------------------------------

    def verify(
        self,
        receiver: ModuleAdapter,
        suite: BenchmarkSuite,
        token_before: str,
        token_after: str,
    ) -> ErasureCertificate:
        """Certify that the capability conferred by the approved set at
        ``token_before`` is gone at ``token_after`` (the post-rollback set).

        ``token_before`` is the snapshot that CONTAINED the packet(s) to
        unlearn; ``token_after`` is the snapshot after the rollback (the packet
        removed). Both are loaded via :meth:`RollbackLayer.snapshot_packets`
        (same path-escape guard as rollback). The receiver is conditioned on its
        ENTIRE approved set in each snapshot, matching what it consumes.
        """
        if not suite.split("heldout"):
            # A suite with no held-out split cannot be measured -- there is no
            # capability to verify gone. Raise rather than emit a zero-case
            # certificate that would look like "measured and found it gone".
            raise UnlearningError(
                "suite '{}' has no held-out split; cannot verify unlearning".format(
                    suite.suite_id
                )
            )

        skills_before = self._receiver_skills(token_before, receiver)
        skills_after = self._receiver_skills(token_after, receiver)

        removed, added = self._packet_delta(skills_before, skills_after)
        adapter_removed = len(removed) > 0

        baseline = self.harness.run(receiver, suite, split="heldout")
        with_skill = self.harness.run(
            receiver, suite, split="heldout", skills=skills_before
        )
        post = self.harness.run(
            receiver, suite, split="heldout", skills=skills_after
        )

        lift = with_skill.score - baseline.score
        residual = post.score - baseline.score
        skill_conferred_lift = lift > self.tolerance
        # TWO-SIDED reversion (adversarial audit 2026-08-17, finding 1): a
        # post-rollback score FAR BELOW baseline is a regression introduced by
        # the after-set (e.g. a harmful packet added on rollback), not a
        # reversion to baseline. The honesty note promises "reverted to baseline
        # within tolerance"; the math must be |post - baseline| <= tol to match.
        # A one-sided <= would certify that regression as "gone" -- an
        # overclaim. ``packets_added`` (above) surfaces the cause so a reviewer
        # can see the after-set changed content, not just that something left.
        capability_gone = abs(residual) <= self.tolerance
        verified = adapter_removed and capability_gone
        substantive = verified and skill_conferred_lift

        cert = ErasureCertificate(
            workspace=str(self.workspace),
            receiver=receiver.module_id,
            token_before=token_before,
            token_after=token_after,
            capability=suite.capability().as_str(),
            baseline_score=baseline.score,
            with_skill_score=with_skill.score,
            post_rollback_score=post.score,
            lift=lift,
            residual=residual,
            tolerance=self.tolerance,
            cases=len(baseline.case_results),
            packets_removed=sorted(removed),
            packets_added=sorted(added),
            adapter_removed=adapter_removed,
            skill_conferred_lift=skill_conferred_lift,
            capability_gone=capability_gone,
            verified=verified,
            substantive=substantive,
            generated_at=datetime.now(timezone.utc).isoformat(),
            honesty_note=HONESTY_NOTE,
            key_fingerprint=self._signer.key_fingerprint(),
            signature_alg=SIGNATURE_ALG,
        )
        cert.signature = self._signer.sign(cert.to_dict())
        return cert

    def verify_certificate(self, cert: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a certificate's HMAC against the local ``unlearn.key``.

        Returns ``{"valid": True, ...}`` on match; raises
        :class:`SignatureMismatchError` on a tampered cert (including a
        missing/non-string signature) and :class:`SigningKeyError` if the key is
        missing. A missing key is NEVER a silent pass. Delegates to
        :class:`~asea._signing.LocalSigner` (D1 strict-load, A1 non-string-sign
        guarantees).
        """
        result = self._signer.verify(cert)
        result["receiver"] = cert.get("receiver")
        result["capability"] = cert.get("capability")
        result["verified"] = cert.get("verified")
        result["packets_removed"] = cert.get("packets_removed", [])
        result["packets_added"] = cert.get("packets_added", [])
        return result

    # -- internals --------------------------------------------------------

    def _receiver_skills(self, token: str, receiver: ModuleAdapter) -> List[Dict[str, Any]]:
        """The receiver's ENTIRE approved set in a snapshot, redacted. All
        capabilities, matching what the receiver consumes. An empty snapshot
        returns ``[]`` (native baseline); a missing/escaping token raises
        :class:`SnapshotNotFoundError`."""
        packets = self.rollback.snapshot_packets(token)  # raises on missing/escape
        return [
            p.redacted_for_receiver()
            for p in packets
            if p.target_module == receiver.module_id
        ]

    @staticmethod
    def _packet_delta(
        skills_before: List[Dict[str, Any]],
        skills_after: List[Dict[str, Any]],
    ) -> tuple:
        """(removed, added) 16-char content hashes across before -> after.

        By content hash over ``{capability, distilled_skill}`` (the semantic
        payload that conditions the receiver), NOT ``packet_id`` -- the pipeline
        regenerates uuids, so an id-based delta would fake churn on a re-run that
        produced an identical skill. Mirrors the diff's ``packets_added`` /
        ``packets_removed`` so the two reports describe the same delta the same
        way. ``packets_added`` is surfaced (adversarial audit 2026-08-17,
        finding 2) so a rollback that ALSO adds a harmful packet cannot be
        misattributed to the removal alone -- a reviewer sees both halves of the
        delta.
        """
        def _skill_hash(s: Dict[str, Any]) -> str:
            payload = {
                "capability": s.get("capability"),
                "distilled_skill": s.get("distilled_skill"),
            }
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
            return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

        set_before = {_skill_hash(s) for s in skills_before}
        set_after = {_skill_hash(s) for s in skills_after}
        return sorted(set_before - set_after), sorted(set_after - set_before)