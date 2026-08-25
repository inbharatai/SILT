"""Verified unlearning (B3, audit 2026-08-17).

An erasure certificate certifies SKILL-LAYER unlearning: a skill packet rolled
back from the approved set is gone (adapter_removed) AND the held-out capability
it conferred reverted to the no-skill baseline (capability_gone). These tests pin
the honesty contract, not just the happy path:

  * substantive unlearning (skill taught something, then it is gone) verifies
    AND is flagged substantive;
  * a packet NOT actually removed does NOT verify (honest "not unlearned");
  * a skill that conferred NO lift (the receiver already knew it, or the skill
    was harmful) is "verified" only TRIVIALLY -- ``substantive=False`` makes that
    impossible to miss, so the cert never claims real forgetting of nothing;
  * removal is by CONTENT hash, not packet_id -- a re-run regenerating uuids on
    an identical skill is NOT fake removal;
  * the HONESTY BOUNDARY is stated verbatim: SKILL-LAYER, NOT weight-level
    forgetting; a receiver with internal state may retain capability (out of
    scope, not claimed);
  * the signature is local HMAC with its OWN key (``unlearn.key``): a fresh cert
    verifies; a tampered cert raises ``SignatureMismatchError``; a different key
    raises; a missing key raises ``SigningKeyError`` (NEVER a silent pass); a
    non-string signature is a typed mismatch;
  * missing snapshot -> ``SnapshotNotFoundError``; no held-out split ->
    ``UnlearningError``; the CLI ``unlearn-verify`` exits 2 on a missing key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asea.benchmarks.harness import BenchmarkHarness
from asea.core.errors import (
    SignatureMismatchError,
    SigningKeyError,
    SnapshotNotFoundError,
    UnlearningError,
)
from asea.core.plugins import default_registry
from asea.core.protocol import (
    CapabilityKey,
    Domain,
    Modality,
    OriginKind,
    PromotionStatus,
    Provenance,
    SkillPacket,
)
from asea.evaluator.similarity import LexicalSimilarity
from asea.memory.store import MemoryStore, RollbackLayer
from asea.modules.mock.zoo import make_generic_receiver
from asea.unlearning import KEY_FILENAME, SIGNATURE_ALG, UnlearningVerifier


_AS_CAP = CapabilityKey(
    task_type="translate", modality=Modality.TEXT,
    domain=Domain.TRANSLATION, language="as->en",
)


# -- helpers (mirror test_capability_diff.py) ---------------------------------


def _provenance() -> Provenance:
    return Provenance(
        origin_kind=OriginKind.CURATED_CORPUS,
        chain=["trusted-source"],
        synthetic_depth=0,
        is_mock=False,
        source_reference="unit-test",
    )


def _promoted_packet(pid, target, cap, distilled_skill, source="trusted-source"):
    return SkillPacket(
        packet_id=pid,
        task_type=cap.task_type,
        source_module=source,
        target_module=target,
        sender_capability=cap,
        modality=cap.modality,
        language=cap.language,
        domain=cap.domain,
        provenance=_provenance(),
        distilled_skill=distilled_skill,
        promotion_status=PromotionStatus.PROMOTED,
        rollback_token="snap-" + pid,
    )


def _skill_for_heldout(suite):
    """Exact-match entries mapping every held-out prompt to its reference."""
    return {"entries": [
        {"source": str(c.prompt), "target": str(c.expected)}
        for c in suite.split("heldout")
    ]}


def _anti_skill_for_heldout(suite):
    """Entries mapping every held-out prompt to a WRONG answer -> score DOWN."""
    return {"entries": [
        {"source": str(c.prompt), "target": "WRONG-ANSWER-DELIBERATELY"}
        for c in suite.split("heldout")
    ]}


def _write_snapshot(workspace: Path, token: str, packets):
    pdir = workspace / "memory" / "snapshots" / token
    pdir.mkdir(parents=True, exist_ok=True)
    for p in packets:
        (pdir / "{}.json".format(p.packet_id)).write_text(
            p.model_dump_json(indent=2), encoding="utf-8"
        )
    return pdir


def _verifier(workspace: Path) -> UnlearningVerifier:
    store = MemoryStore(workspace / "memory")
    return UnlearningVerifier(
        harness=BenchmarkHarness(plugins=default_registry(), similarity=LexicalSimilarity()),
        rollback=RollbackLayer(store),
        workspace=workspace,
    )


def _receiver(module_id="learner", knowledge=None):
    return make_generic_receiver(
        module_id=module_id,
        capabilities=[_AS_CAP],
        knowledge=knowledge or {},
        fallback="echo",
    )


# -- the substantive happy path ----------------------------------------------


def test_substantive_unlearning_verifies(as_en_suite, tmp_path):
    """A receiver that knew nothing is taught by a skill (lift > tolerance),
    the skill is then rolled back (after-set empty), and the capability reverts
    to baseline. adapter_removed AND capability_gone AND skill_conferred_lift ->
    verified AND substantive. The certificate signs and round-trips."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    skill = _skill_for_heldout(as_en_suite)
    pkt = _promoted_packet("p1", "learner", _AS_CAP, skill)
    _write_snapshot(tmp_path, "before", [pkt])   # packet present
    _write_snapshot(tmp_path, "after", [])        # rolled back -> empty

    cert = v.verify(receiver, as_en_suite, "before", "after")

    assert cert.adapter_removed is True
    assert cert.skill_conferred_lift is True, "the skill must have raised capability"
    assert cert.capability_gone is True
    assert cert.verified is True
    assert cert.substantive is True
    # The numbers tell the same story: lift ~1.0, residual ~0.
    assert cert.lift > v.tolerance
    assert cert.residual <= v.tolerance
    assert cert.with_skill_score > cert.baseline_score
    assert cert.post_rollback_score == pytest.approx(cert.baseline_score, abs=v.tolerance)
    # Signed and round-trips.
    assert cert.signature is not None
    assert v.verify_certificate(cert.to_dict())["valid"] is True


def test_certificate_carries_the_honesty_boundary_verbatim(as_en_suite, tmp_path):
    """The honesty_note must state the SKILL-LAYER boundary so no reader mistakes
    this for weight-level forgetting. Pinned by substring (the note is the
    contract; editing it to drop the boundary would silently re-scope the
    claim)."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(tmp_path, "after", [])
    cert = v.verify(receiver, as_en_suite, "before", "after")
    note = cert.honesty_note
    assert "SKILL-LAYER" in note
    assert "weight-level forgetting" in note
    assert "out of scope" in note
    # And the note rides inside the signed payload (cannot be edited post-sign).
    assert cert.to_dict()["honesty_note"] == note


# -- the honest "not unlearned" path -----------------------------------------


def test_not_verified_when_packet_was_not_removed(as_en_suite, tmp_path):
    """If the after-set STILL contains the packet (nothing was rolled back),
    adapter_removed is False and capability_gone is False (the capability
    persists because the skill is still there). verified is False -- the cert
    must NOT claim unlearning when nothing was removed."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    skill = _skill_for_heldout(as_en_suite)
    pkt = _promoted_packet("p1", "learner", _AS_CAP, skill)
    _write_snapshot(tmp_path, "before", [pkt])
    _write_snapshot(tmp_path, "after", [pkt])  # NOT removed

    cert = v.verify(receiver, as_en_suite, "before", "after")

    assert cert.adapter_removed is False
    assert cert.packets_removed == []
    # post == with_skill (same skill set) -> residual == lift > tol -> not gone
    assert cert.capability_gone is False
    assert cert.verified is False
    assert cert.substantive is False


def test_not_verified_when_before_equals_after(as_en_suite, tmp_path):
    """token_before == token_after is the degenerate 'compare a snapshot to
    itself' case: nothing removed, nothing changed, verified False. Not an
    error -- an honest zero-result certificate (mirrors the diff's empty-delta
    honesty)."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    pkt = _promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))
    _write_snapshot(tmp_path, "only", [pkt])

    cert = v.verify(receiver, as_en_suite, "only", "only")

    assert cert.adapter_removed is False
    assert cert.verified is False


# -- the trivially-verified boundary cases (substantive=False) ----------------


def test_trivially_verified_when_receiver_already_knew_it(as_en_suite, tmp_path):
    """HONESTY BOUNDARY: the receiver's OWN knowledge already solves held-out
    (baseline ~1.0). Adding a matching skill adds no measurable lift, so
    removing it changes nothing. The cert is ``verified`` (adapter_removed AND
    capability_gone both hold, vacuously) but ``substantive=False`` and
    ``skill_conferred_lift=False`` -- the cert must NOT claim real forgetting of
    something the skill never taught. This is the stateless-mock analogue of
    'capability retained independently of the adapter'."""
    cap = as_en_suite.capability()
    knowledge = {cap.as_str(): {str(c.prompt): str(c.expected) for c in as_en_suite.split("heldout")}}
    receiver = _receiver(knowledge=knowledge)
    v = _verifier(tmp_path)
    pkt = _promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))
    _write_snapshot(tmp_path, "before", [pkt])
    _write_snapshot(tmp_path, "after", [])

    cert = v.verify(receiver, as_en_suite, "before", "after")

    assert cert.adapter_removed is True
    assert cert.skill_conferred_lift is False, "the receiver already knew it -> no lift"
    assert cert.capability_gone is True  # vacuously: post == baseline
    assert cert.verified is True
    assert cert.substantive is False, "trivial unlearning must NOT be flagged substantive"
    assert cert.lift <= v.tolerance


def test_removing_a_harmful_skill_is_trivially_verified(as_en_suite, tmp_path):
    """A skill that HURTS is removed: capability reverts to baseline
    (capability_gone True), adapter_removed True, so ``verified`` -- but the
    skill conferred NEGATIVE lift, so ``skill_conferred_lift=False`` and
    ``substantive=False``. Removing a harmful adapter is honest 'verified
    unlearning' only in the trivial sense; the cert says so rather than claiming
    the receiver 'forgot' a capability it never had.

    The harmful skill here OVERRIDES correct receiver knowledge with wrong
    answers: the receiver already knows the as->en mapping (baseline ~1.0), and
    the anti-skill forces a wrong English target on every case, dropping the
    held-out score below baseline. (Against a no-knowledge echo receiver an
    English anti-skill would paradoxically score HIGHER than the Assamese echo
    baseline via language-preservation alone -- so the override-of-knowledge
    setup is what makes the skill genuinely harmful.)"""
    cap = as_en_suite.capability()
    knowledge = {cap.as_str(): {str(c.prompt): str(c.expected) for c in as_en_suite.split("heldout")}}
    receiver = _receiver(knowledge=knowledge)
    v = _verifier(tmp_path)
    anti = _anti_skill_for_heldout(as_en_suite)
    pkt = _promoted_packet("p1", "learner", _AS_CAP, anti)
    _write_snapshot(tmp_path, "before", [pkt])
    _write_snapshot(tmp_path, "after", [])

    cert = v.verify(receiver, as_en_suite, "before", "after")

    assert cert.adapter_removed is True
    assert cert.with_skill_score < cert.baseline_score, "anti-skill must lower the score vs known answers"
    assert cert.skill_conferred_lift is False, "negative lift is not a conferred lift"
    assert cert.capability_gone is True  # post (no skill) == baseline (knowledge)
    assert cert.verified is True
    assert cert.substantive is False


# -- content-hash delta, not packet_id ---------------------------------------


def test_removal_is_by_content_hash_not_packet_id(as_en_suite, tmp_path):
    """before has packet p1 with skill S; after has packet p2 (DIFFERENT id)
    with IDENTICAL skill S. The skill content is still present, so
    adapter_removed is False and verified is False -- a re-run that regenerates
    uuids on an identical skill is NOT fake removal (mirrors the diff's
    content-hash delta)."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    skill = _skill_for_heldout(as_en_suite)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, skill)])
    _write_snapshot(tmp_path, "after", [_promoted_packet("p2", "learner", _AS_CAP, skill)])

    cert = v.verify(receiver, as_en_suite, "before", "after")

    assert cert.packets_removed == []
    assert cert.packets_added == []
    assert cert.adapter_removed is False
    assert cert.verified is False


def test_post_below_baseline_is_not_capability_gone(
    as_en_suite, tmp_path,
):
    """Finding 1 (CONFIRMED) fix-pin + Finding 2 (packets_added surfaced). The
    reviewer's scenario: before={good P1}, after={harmful P2}, and the receiver
    already knows the answers (so baseline is high). P1 is removed (so
    adapter_removed=True) and P2 is a NEW harmful anti-skill that OVERRIDES the
    receiver's correct knowledge with wrong answers, driving post-rollback
    BELOW baseline. ``capability_gone`` is TWO-SIDED (``abs(post-baseline) <=
    tol``): a post BELOW baseline is a regression introduced by the after-set,
    NOT a reversion, so capability_gone is False and verified is False. Under
    the OLD one-sided ``post-baseline <= tol`` this would have certified
    ``capability_gone=True`` and ``verified=True`` -- overclaiming that P1 was
    "unlearned and the capability reverted" when really P2 regressed it. The
    cert also surfaces ``packets_added=[P2]`` so a reviewer sees the after-set
    changed content, not just that P1 left."""
    cap = as_en_suite.capability()
    knowledge = {cap.as_str(): {str(c.prompt): str(c.expected) for c in as_en_suite.split("heldout")}}
    receiver = _receiver(knowledge=knowledge)
    v = _verifier(tmp_path)
    good = _skill_for_heldout(as_en_suite)
    anti = _anti_skill_for_heldout(as_en_suite)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, good)])
    _write_snapshot(tmp_path, "after", [_promoted_packet("p2", "learner", _AS_CAP, anti)])

    cert = v.verify(receiver, as_en_suite, "before", "after")

    # P1 removed, P2 (distinct content) added -- both halves of the delta shown.
    assert cert.adapter_removed is True
    assert len(cert.packets_removed) == 1
    assert len(cert.packets_added) == 1
    assert cert.packets_removed != cert.packets_added
    # post is BELOW baseline (the anti-skill overrode correct knowledge).
    assert cert.post_rollback_score < cert.baseline_score
    # TWO-SIDED: a below-baseline post is NOT a reversion -> not gone -> not verified.
    assert cert.capability_gone is False
    assert cert.verified is False
    assert cert.substantive is False


def test_remaining_packet_providing_capability_blocks_verified(
    as_en_suite, tmp_path,
):
    """Finding 3 (PLAUSIBLE) coverage: before holds TWO distinct-content packets
    that both confer the capability (P1 = held-out entries; P2 = held-out entries
    PLUS an extra entry, so a different content hash but still solves held-out
    via exact lookup). after keeps P2 only. P1 is genuinely removed
    (adapter_removed=True, packets_removed=[P1]) BUT the capability is NOT gone
    -- P2 still provides it, so post stays high, residual > tolerance,
    capability_gone=False, verified=False. A cert that said "P1 unlearned,
    capability gone" here would be a lie: the capability persists via P2. This
    pins the multi-packet-before path no existing test covered."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    heldout_entries = _skill_for_heldout(as_en_suite)
    # P2: same held-out solving entries + one EXTRA entry -> distinct content hash
    # but still solves every held-out case via exact-match lookup.
    p2_skill = {"entries": heldout_entries["entries"] + [
        {"source": "EXTRA-NONHELDOUT-PROMPT", "target": "EXTRA-NONHELDOUT-ANSWER"}
    ]}
    _write_snapshot(tmp_path, "before", [
        _promoted_packet("p1", "learner", _AS_CAP, heldout_entries),
        _promoted_packet("p2", "learner", _AS_CAP, p2_skill),
    ])
    _write_snapshot(tmp_path, "after", [_promoted_packet("p2", "learner", _AS_CAP, p2_skill)])

    cert = v.verify(receiver, as_en_suite, "before", "after")

    assert cert.adapter_removed is True, "P1 was removed"
    assert len(cert.packets_removed) == 1, "only P1's content hash is gone"
    assert cert.packets_added == [], "P2 was already in before"
    # P2 retains the capability -> post stays high -> not reverted -> not verified.
    assert cert.capability_gone is False
    assert cert.verified is False


# -- typed errors ------------------------------------------------------------


def test_missing_snapshot_raises_typed(as_en_suite, tmp_path):
    """A token that does not exist raises SnapshotNotFoundError, NOT a silent
    empty-snapshot 'verified' (a missing snapshot treated as empty would
    fabricate 'the packet was removed')."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    with pytest.raises(SnapshotNotFoundError):
        v.verify(receiver, as_en_suite, "before", "no-such-token")


def test_path_escape_snapshot_raises_typed(as_en_suite, tmp_path):
    """A token containing '..' that escapes snapshots/ is rejected, not followed
    -- the same containment guard rollback uses."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    with pytest.raises(SnapshotNotFoundError):
        v.verify(receiver, as_en_suite, "before", "..")


def test_suite_with_no_heldout_raises_unlearning_error(tmp_path):
    """A suite with no held-out split cannot be measured -> UnlearningError, not
    a zero-case certificate that would look like 'measured and found it gone'."""
    from asea.benchmarks.harness import BenchmarkSuite
    from asea.core.protocol import Modality as M, Domain as D
    suite = BenchmarkSuite(
        suite_id="heldoutless",
        task_type="translate",
        modality=M.TEXT,
        domain=D.TRANSLATION,
        language="as->en",
        cases=[],  # no heldout (no cases at all)
    )
    receiver = _receiver()
    v = _verifier(tmp_path)
    _write_snapshot(tmp_path, "before", [])
    _write_snapshot(tmp_path, "after", [])
    with pytest.raises(UnlearningError, match="no held-out split"):
        v.verify(receiver, suite, "before", "after")


# -- signature integrity -----------------------------------------------------


def test_certificate_round_trips_and_tamper_is_detected(as_en_suite, tmp_path):
    """A fresh certificate verifies; editing any signed field after signing
    raises SignatureMismatchError (tamper-evident to the local key holder)."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(tmp_path, "after", [])
    cert = v.verify(receiver, as_en_suite, "before", "after").to_dict()
    assert v.verify_certificate(cert)["valid"] is True

    tampered = dict(cert)
    tampered["post_rollback_score"] = 0.999  # edit a signed field
    with pytest.raises(SignatureMismatchError):
        v.verify_certificate(tampered)

    tampered2 = dict(cert)
    tampered2["verified"] = True if not cert["verified"] else False
    with pytest.raises(SignatureMismatchError):
        v.verify_certificate(tampered2)


def test_certificate_signed_under_a_different_key_raises(as_en_suite, tmp_path):
    """A certificate signed under workspace A's ``unlearn.key`` does NOT verify
    under workspace B's (different) key -> SignatureMismatchError. Local HMAC is
    key-holder-scoped; this is the honest limit, not a loophole. B mints its own
    key first so the failure reaches the HMAC-comparison step (a missing key is
    a different, separately-tested error)."""
    receiver = _receiver()
    ws_a = tmp_path / "ws_a"; ws_a.mkdir()
    ws_b = tmp_path / "ws_b"; ws_b.mkdir()
    va = _verifier(ws_a)
    _write_snapshot(ws_a, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(ws_a, "after", [])
    signed = va.verify(receiver, as_en_suite, "before", "after").to_dict()

    import shutil
    shutil.copytree(ws_a / "memory", ws_b / "memory", dirs_exist_ok=True)
    vb = _verifier(ws_b)
    # Mint ws_b's OWN (different) key with a trivial verify, so ws_b reaches the
    # HMAC-comparison step with a real distinct key, not a missing-key error.
    _write_snapshot(ws_b, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(ws_b, "after", [])
    vb.verify(receiver, as_en_suite, "before", "after")
    assert (ws_b / KEY_FILENAME).exists()
    assert vb._signer.key_fingerprint() != signed["key_fingerprint"], "keys must differ"
    with pytest.raises(SignatureMismatchError):
        vb.verify_certificate(signed)


def test_missing_key_is_not_a_silent_pass(as_en_suite, tmp_path):
    """verify_certificate with no ``unlearn.key`` raises SigningKeyError -- NEVER
    silently returns valid (which would let any certificate 'verify' once the
    key is gone) and NEVER silently mints a fresh key (D1: no re-load with
    generate=True on the verify path)."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(tmp_path, "after", [])
    cert = v.verify(receiver, as_en_suite, "before", "after").to_dict()
    # Delete the key; verification must fail typed, and must NOT re-create it.
    (tmp_path / KEY_FILENAME).unlink()
    with pytest.raises(SigningKeyError):
        v.verify_certificate(cert)
    assert not (tmp_path / KEY_FILENAME).exists(), "D1: verify must not mint a key"


def test_non_string_signature_is_typed_mismatch(as_en_suite, tmp_path):
    """A certificate whose ``signature`` field is missing or non-string is a
    typed SignatureMismatchError, never a bare TypeError from compare_digest
    (adversarial audit 2026-08-17, A1)."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(tmp_path, "after", [])
    base = v.verify(receiver, as_en_suite, "before", "after").to_dict()

    no_sig = {k: val for k, val in base.items() if k != "signature"}
    with pytest.raises(SignatureMismatchError):
        v.verify_certificate(no_sig)

    non_str = dict(base, signature=None)
    with pytest.raises(SignatureMismatchError):
        v.verify_certificate(non_str)

    list_sig = dict(base, signature=["x"])
    with pytest.raises(SignatureMismatchError):
        v.verify_certificate(list_sig)


def test_signature_is_deterministic_across_recomputations(as_en_suite, tmp_path):
    """Re-signing the same certificate dict (same key) yields the identical HMAC
    -- float repr does not drift, because floats are pre-rounded before signing."""
    receiver = _receiver()
    v = _verifier(tmp_path)
    _write_snapshot(tmp_path, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(tmp_path, "after", [])
    cert = v.verify(receiver, as_en_suite, "before", "after")
    again = v._signer.sign(cert.to_dict())
    assert again == cert.signature


# -- CLI exit codes ----------------------------------------------------------


def test_cli_unlearn_verify_missing_key_exits_two(as_en_suite, tmp_path, capsys):
    """I1: ``asea unlearn-verify`` with no local signing key prints the reason
    and exits 2 (typed hard error) -- not a raw traceback at exit 1, and never
    exit 0 (silent pass). Mirrors the diff-verify CLI contract."""
    from asea.cli import main

    ws = tmp_path / "wscli"; ws.mkdir()
    report_path = ws / "cert.json"
    report_path.write_text(
        json.dumps({"signature": "x", "signature_alg": SIGNATURE_ALG}),
        encoding="utf-8",
    )
    assert not (ws / KEY_FILENAME).exists(), "precondition: no key"
    rc = main(["unlearn-verify", "--workspace", str(ws), "--report", str(report_path)])
    assert rc == 2, "missing signing key must exit 2, not 0 or 1"
    out = capsys.readouterr().out
    assert '"unverifiable": true' in out


def test_cli_unlearn_verify_tampered_exits_one(as_en_suite, tmp_path, capsys):
    """A tampered certificate exits 1 with ``valid: false`` -- not 0."""
    from asea.cli import main

    ws = tmp_path / "wscli"; ws.mkdir()
    receiver = _receiver()
    v = _verifier(ws)
    _write_snapshot(ws, "before", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(ws, "after", [])
    cert = v.verify(receiver, as_en_suite, "before", "after").to_dict()
    cert["post_rollback_score"] = 0.999
    report_path = ws / "cert.json"
    report_path.write_text(json.dumps(cert), encoding="utf-8")
    rc = main(["unlearn-verify", "--workspace", str(ws), "--report", str(report_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert '"valid": false' in out