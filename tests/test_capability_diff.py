"""Capability Diff (B1a, audit 2026-08-17).

A capability diff measures a receiver under two approved-set snapshots on
held-out data and emits a locally HMAC-signed report. These tests pin the
honesty contract, not just the happy path:

  * the diff REUSES the evaluator's scoring path (``harness.run`` with the
    receiver's snapshot approved set) -- its ``score_a`` equals a direct
    ``harness.run`` with the same skills, to the byte (no parallel scoring);
  * an IMPROVED capability is flagged improved; a REGRESSED one is flagged
    regressed; a control capability that MOVED (the A3 control-bleed bound) is
    flagged moved, not buried;
  * ``packets_added`` / ``packets_removed`` are by ``content_hash``, NOT
    ``packet_id`` -- a re-run that regenerates uuids but produces identical
    skills is NOT fake churn;
  * the receiver is conditioned on its ENTIRE approved set in a snapshot (all
    capabilities), so cross-capability bleed is measured, not hidden;
  * an EMPTY snapshot is a legitimate delta of zero (no error); a MISSING
    snapshot is a typed ``SnapshotNotFoundError`` (NOT silently empty);
  * the signature is local HMAC: a fresh report verifies; a tampered report
    raises ``SignatureMismatchError``; a report signed under a DIFFERENT key
    raises; a missing key raises ``SigningKeyError`` (NEVER a silent pass); the
    report carries an ``honesty_note`` stating it is NOT portable attestation;
  * the diff conditions only on the skill-set delta -- same receiver, harness
    and held-out cases across A and B, so the delta is attributable to the
    approved set alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asea.benchmarks.harness import BenchmarkCase, BenchmarkHarness, BenchmarkSuite
from asea.capability_diff import CapabilityDiffer, SIGNATURE_ALG
from asea.core.errors import SignatureMismatchError, SigningKeyError, SnapshotNotFoundError
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


_AS_CAP = CapabilityKey(
    task_type="translate", modality=Modality.TEXT,
    domain=Domain.TRANSLATION, language="as->en",
)


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
    """A skill whose exact-match entries map every held-out prompt to its
    reference answer. A receiver conditioned on it scores ~1.0 on held-out."""
    return {"entries": [
        {"source": str(c.prompt), "target": str(c.expected)}
        for c in suite.split("heldout")
    ]}


def _anti_skill_for_heldout(suite):
    """A skill whose entries map every held-out prompt to a WRONG answer, so
    conditioning on it drives the held-out score DOWN (a regression)."""
    return {"entries": [
        {"source": str(c.prompt), "target": "WRONG-ANSWER-DELIBERATELY"}
        for c in suite.split("heldout")
    ]}


def _synthetic_heldout_suite(language: str, n: int = 4) -> BenchmarkSuite:
    """A second suite WITH a held-out split and DISTINCT prompts, so the diff
    can measure two capabilities at once. The bundled hindi_english suite is
    regression-only (no held-out), so it cannot serve as a second measured
    capability; this synthetic one can. Prompts/answers are deliberately
    disjoint from the Assamese fixture so a skill targeting one cannot bleed
    into the other via exact-match lookup."""
    return BenchmarkSuite(
        suite_id="synth_{}".format(language),
        task_type="translate",
        modality=Modality.TEXT,
        domain=Domain.TRANSLATION,
        language=language,
        cases=[
            BenchmarkCase(
                case_id="{}-h-{}".format(language, i),
                prompt="PROMPT-{}-{}".format(language, i),
                expected="ANSWER-{}-{}".format(language, i),
                split="heldout",
            )
            for i in range(n)
        ],
    )


def _skill_for_suite_heldout(suite) -> dict:
    """Exact-match entries mapping a suite's held-out prompts to references."""
    return {"entries": [
        {"source": str(c.prompt), "target": str(c.expected)}
        for c in suite.split("heldout")
    ]}


def _write_snapshot(workspace: Path, token: str, packets, *, create: bool = True):
    """Write packets directly into a snapshot dir (the format
    RollbackLayer.snapshot() produces and snapshot_packets reads)."""
    pdir = workspace / "memory" / "snapshots" / token
    if create:
        pdir.mkdir(parents=True, exist_ok=True)
    for p in packets:
        (pdir / "{}.json".format(p.packet_id)).write_text(
            p.model_dump_json(indent=2), encoding="utf-8"
        )
    return pdir


def _differ(workspace: Path) -> CapabilityDiffer:
    store = MemoryStore(workspace / "memory")
    return CapabilityDiffer(
        harness=BenchmarkHarness(plugins=default_registry(), similarity=LexicalSimilarity()),
        rollback=RollbackLayer(store),
        workspace=workspace,
    )


def _receiver(module_id="learner"):
    # Empty knowledge + echo fallback: baseline holds ~0 similarity to English
    # reference text. Conditioning on a skill with exact entries -> lookup hits
    # -> ~1.0. So the skill-set delta is visible in the score.
    return make_generic_receiver(
        module_id=module_id,
        capabilities=[_AS_CAP],
        knowledge={},
        fallback="echo",
    )


# -- reuse of the evaluator's scoring path -----------------------------------


def test_diff_score_reuses_harness_scoring(as_en_suite, tmp_path):
    """The diff does NOT reimplement scoring. Its ``score_a`` is byte-identical
    to a direct ``harness.run(receiver, suite, heldout, skills=A_skills)`` --
    the same call the Evaluator makes -- so the delta is attributable to the
    skill-set, not to a parallel scoring implementation drifting."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    skill = {"distilled_skill": _skill_for_heldout(as_en_suite)}
    pkt = _promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))
    _write_snapshot(tmp_path, "A", [pkt])
    _write_snapshot(tmp_path, "B", [])  # empty B -> baseline

    report = differ.diff(receiver, [as_en_suite], "A", "B")
    # B is empty -> score_b is the receiver's baseline (no skills).
    # Recompute the SAME baseline directly from the harness the differ used.
    direct_baseline = differ.harness.run(receiver, as_en_suite, split="heldout", skills=[]).score
    assert report.deltas[0].score_b == pytest.approx(direct_baseline), (
        "score_b (empty snapshot) must equal a direct harness.run baseline"
    )
    # A holds the skill -> score_a is the conditioned score. Recompute directly.
    direct_conditioned = differ.harness.run(
        receiver, as_en_suite, split="heldout",
        skills=[pkt.redacted_for_receiver()],
    ).score
    assert report.deltas[0].score_a == pytest.approx(direct_conditioned), (
        "score_a must equal a direct harness.run with the same skills -- "
        "the diff reuses the evaluator's scoring path, it does not reimplement it"
    )


# -- verdicts: improved / regressed / moved ----------------------------------


def test_improved_capability_is_flagged_improved(as_en_suite, tmp_path):
    """A->B where B adds a skill that fixes held-out -> delta > 0 -> improved."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])  # baseline
    _write_snapshot(tmp_path, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    report = differ.diff(receiver, [as_en_suite], "A", "B")
    d = report.deltas[0]
    assert d.delta > 0
    assert d.improved is True
    assert d.regressed is False
    assert "improved" in report.summary and report.summary["improved"] >= 1


def test_regressed_capability_is_flagged_regressed(as_en_suite, tmp_path):
    """A->B where B's skill maps every held-out prompt to a WRONG answer ->
    delta < 0 -> regressed. The diff reports a regression the new approved set
    introduced, not just improvements."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    # A: a GOOD skill (high score). B: a BAD skill (drives score down).
    _write_snapshot(tmp_path, "A", [_promoted_packet("good", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(tmp_path, "B", [_promoted_packet("bad", "learner", _AS_CAP, _anti_skill_for_heldout(as_en_suite))])
    report = differ.diff(receiver, [as_en_suite], "A", "B")
    d = report.deltas[0]
    assert d.delta < 0, "B's wrong-answer skill must lower the held-out score vs A's good skill"
    assert d.regressed is True
    assert d.improved is False


def test_control_capability_movement_is_surfaced_not_hidden(as_en_suite, tmp_path):
    """The A3 control-movement bound, applied to the diff: a capability the new
    approved set was NOT targeting (as->en) stays unchanged, while the
    capability it DID target (a distinct synthetic one) jumps up and is flagged
    ``moved`` (|delta| > max_control_movement) AND ``improved`` -- reported
    per-capability, not buried as a single aggregate. The receiver is
    conditioned on its ENTIRE approved set, so the diff shows what moved where."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    xx_suite = _synthetic_heldout_suite("xx->en")
    xx_cap = CapabilityKey(
        task_type="translate", modality=Modality.TEXT,
        domain=Domain.TRANSLATION, language="xx->en",
    )
    # A: empty. B: an xx->en packet that fixes xx_suite's held-out split.
    # On the as->en suite the xx->en entries (PROMPT-xx-i) don't match the
    # Assamese held-out prompts -> no hit -> as->en unchanged (delta ~0).
    # On the xx->en suite the entries match -> xx->en jumps up -> moved + improved.
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [_promoted_packet("xx1", "learner", xx_cap, _skill_for_suite_heldout(xx_suite))])
    report = differ.diff(receiver, [as_en_suite, xx_suite], "A", "B")
    by_cap = {d.capability: d for d in report.deltas}
    as_d = by_cap["translate/text/translation/as->en"]
    xx_d = by_cap["translate/text/translation/xx->en"]
    # as->en: the new approved set targets xx->en, not as->en -> no bleed -> unchanged
    assert as_d.improved is False and as_d.regressed is False, (
        "an xx->en skill must not move the as->en capability"
    )
    assert abs(as_d.delta) < 1e-6, "as->en must be a true zero delta (no bleed)"
    # xx->en improved by a lot -> improved AND moved (movement either direction)
    assert xx_d.delta > 0
    assert xx_d.improved is True
    assert xx_d.moved is True, "a capability that improved >max_control_movement must be flagged moved"


# -- packets by content_hash, not packet_id ---------------------------------


def test_packets_delta_is_by_content_hash_not_packet_id(as_en_suite, tmp_path):
    """A re-run regenerates packet uuids but produces identical distilled
    content. The added/removed delta is by ``content_hash`` (semantic payload),
    so identical-content packets with different ids are NOT counted as churn.
    Two snapshots holding the SAME skill under different packet_ids -> zero
    added/removed, and (per the score test) zero delta."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    skill = _skill_for_heldout(as_en_suite)
    _write_snapshot(tmp_path, "A", [_promoted_packet("aaaa", "learner", _AS_CAP, skill)])
    _write_snapshot(tmp_path, "B", [_promoted_packet("bbbb", "learner", _AS_CAP, skill)])
    report = differ.diff(receiver, [as_en_suite], "A", "B")
    assert report.packets_added == [], "same content under a new id is NOT an addition"
    assert report.packets_removed == [], "same content under a new id is NOT a removal"
    d = report.deltas[0]
    assert abs(d.delta) < 1e-9, "identical approved sets -> zero delta"


def test_packets_added_removed_for_genuinely_different_skills(as_en_suite, tmp_path):
    """A genuinely new skill in B (different content) IS reported as added;
    a skill present in A but absent in B IS reported as removed."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [_promoted_packet("p_old", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p_new", "learner", _AS_CAP, _anti_skill_for_heldout(as_en_suite))])
    report = differ.diff(receiver, [as_en_suite], "A", "B")
    assert len(report.packets_added) == 1
    assert len(report.packets_removed) == 1
    assert report.packets_added != report.packets_removed, "different content -> different hashes"


# -- missing vs empty snapshot ----------------------------------------------


def test_missing_snapshot_raises_typed_error(as_en_suite, tmp_path):
    """A token that does not exist is a typed SnapshotNotFoundError, NOT a
    silent empty-diff (which would fabricate 'no change')."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    with pytest.raises(SnapshotNotFoundError):
        differ.diff(receiver, [as_en_suite], "A", "does-not-exist")


def test_path_escape_snapshot_token_raises(as_en_suite, tmp_path):
    """A token containing '..' that escapes the snapshots dir is rejected --
    the same containment guard rollback() uses. The diff must not read packets
    from outside snapshots/."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    with pytest.raises(SnapshotNotFoundError):
        differ.diff(receiver, [as_en_suite], "A", "..")


def test_empty_snapshot_is_a_legitimate_zero_delta_not_an_error(as_en_suite, tmp_path):
    """A snapshot token that EXISTS but holds no packets for the receiver is an
    empty approved set -> the receiver is measured under no skills (its native
    baseline). That is a legitimate delta (possibly zero), NOT an error."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [])
    report = differ.diff(receiver, [as_en_suite], "A", "B")  # must not raise
    assert abs(report.deltas[0].delta) < 1e-9


# -- signature: tamper-evident, local, honest --------------------------------


def test_fresh_report_verifies_and_carries_honesty_note(as_en_suite, tmp_path):
    """A freshly-signed report verifies against the local key, carries the
    local-HMAC honesty note (NOT portable attestation), and the signature_alg
    tag -- and those fields are themselves covered by the signature."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    report = differ.diff(receiver, [as_en_suite], "A", "B")
    d = report.to_dict()
    assert d["signature_alg"] == SIGNATURE_ALG
    assert "NOT a portable" in d["honesty_note"]
    assert "never uploaded" in d["honesty_note"].lower()
    result = differ.verify(d)
    assert result["valid"] is True
    assert result["key_fingerprint"] == d["key_fingerprint"]


def test_tampered_score_raises_signature_mismatch(as_en_suite, tmp_path):
    """Editing a score after signing breaks the HMAC -> SignatureMismatchError,
    not a bare valid=False (so a caller cannot mistake a forgery for a fresh
    unsigned report)."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    d = differ.diff(receiver, [as_en_suite], "A", "B").to_dict()
    d["deltas"][0]["score_b"] = 0.99  # tamper
    with pytest.raises(SignatureMismatchError):
        differ.verify(d)


def test_tampered_packets_added_raises(as_en_suite, tmp_path):
    """Adding a fake packet hash after signing breaks the signature."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    d = differ.diff(receiver, [as_en_suite], "A", "B").to_dict()
    d["packets_added"].append("fake-hash-not-signed")
    with pytest.raises(SignatureMismatchError):
        differ.verify(d)


def test_missing_signature_raises(as_en_suite, tmp_path):
    """A report with no signature field is not valid -- verify raises rather
    than treating absent-as-valid."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    d = differ.diff(receiver, [as_en_suite], "A", "B").to_dict()
    d["signature"] = None
    with pytest.raises(SignatureMismatchError):
        differ.verify(d)


def test_report_signed_under_a_different_key_raises(as_en_suite, tmp_path):
    """A report signed under one key does NOT verify under a different key --
    the key_fingerprint differs and the HMAC mismatches. (Local HMAC is
    key-holder-scoped; this is the honest limit, not a loophole.)"""
    receiver = _receiver()
    ws_a = tmp_path / "ws_a"
    ws_a.mkdir()
    differ_a = _differ(ws_a)
    _write_snapshot(ws_a, "A", [])
    _write_snapshot(ws_a, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    signed = differ_a.diff(receiver, [as_en_suite], "A", "B").to_dict()

    # A second workspace mints a DIFFERENT key (no shared diff.key).
    ws_b = tmp_path / "ws_b"
    ws_b.mkdir()
    differ_b = _differ(ws_b)
    # Copy the snapshots over so differ_b can read them, but NOT the key.
    import shutil
    shutil.copytree(ws_a / "memory", ws_b / "memory", dirs_exist_ok=True)
    assert not (ws_b / "diff.key").exists()
    # Mint ws_b's OWN (different) key by running a trivial diff there, so
    # verify() reaches the HMAC-comparison step with a real, distinct key
    # rather than failing earlier on a missing key (that is a different,
    # separately-tested error: SigningKeyError).
    differ_b.diff(receiver, [as_en_suite], "A", "B")
    assert (ws_b / "diff.key").exists()
    assert differ_b._signer.key_fingerprint() != signed["key_fingerprint"], "keys must differ"
    with pytest.raises(SignatureMismatchError):
        differ_b.verify(signed)


def test_missing_signing_key_is_not_a_silent_pass(as_en_suite, tmp_path):
    """verify() with no key file present raises SigningKeyError -- it NEVER
    silently returns valid (which would let any report 'verify' once the key
    is gone) and NEVER silently mints a fresh key to verify against. Also pins
    D1: verify() must not create diff.key as a side effect of the failed call
    (no re-load with generate=True on the verify path)."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    # Build a report (this mints the key), then DELETE the key and verify.
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    signed = differ.diff(receiver, [as_en_suite], "A", "B").to_dict()
    (tmp_path / "diff.key").unlink()
    with pytest.raises(SigningKeyError):
        differ.verify(signed)
    # D1: the failed verify must NOT have minted a fresh key as a side effect.
    assert not (tmp_path / "diff.key").exists(), (
        "verify() must never create diff.key (no generate=True on the verify path)"
    )


def test_signature_is_deterministic_across_recomputations(as_en_suite, tmp_path):
    """The canonical payload is reproducible: re-signing the same report dict
    (same key) yields the identical HMAC -- float repr does not drift between
    runs, because floats are pre-rounded before signing."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    r1 = differ.diff(receiver, [as_en_suite], "A", "B")
    # Re-sign the same dict (key already exists, so no regeneration) via the
    # shared signer -- the canonical payload is reproducible, so the HMAC is
    # byte-identical across recomputations.
    sig_again = differ._signer.sign(r1.to_dict())
    assert sig_again == r1.signature, "re-signing identical contents must be deterministic"


# -- only the skill-set varies across A and B --------------------------------


def test_delta_is_attributable_to_skill_set_only(as_en_suite, tmp_path):
    """Same receiver, same harness, same held-out cases across A and B: the
    ONLY thing that differs is the approved set. So a no-op diff (A==B) yields
    exactly zero delta AND zero added/removed -- nothing else wiggles."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    skill = _skill_for_heldout(as_en_suite)
    _write_snapshot(tmp_path, "A", [_promoted_packet("p1", "learner", _AS_CAP, skill)])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p2", "learner", _AS_CAP, skill)])
    report = differ.diff(receiver, [as_en_suite], "A", "B")
    assert report.packets_added == [] and report.packets_removed == []
    assert abs(report.deltas[0].delta) < 1e-9
    assert report.summary["improved"] == 0
    assert report.summary["regressed"] == 0
    assert report.summary["moved"] == 0


# -- integration: real RollbackLayer.snapshot round-trips through snapshot_packets


def test_snapshot_packets_reads_what_real_snapshot_wrote(as_en_suite, tmp_path):
    """snapshot_packets reads the exact packets a real RollbackLayer.snapshot()
    captured -- not a parallel format. Approve a PROMOTED packet through the
    store, snapshot, then confirm snapshot_packets returns it."""
    store = MemoryStore(tmp_path / "memory")
    rollback = RollbackLayer(store)
    pkt = _promoted_packet("real1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))
    store.approve(pkt)  # writes approved/real1.json (PROMOTED + rollback_token)
    token = rollback.snapshot(label="real")
    loaded = rollback.snapshot_packets(token)
    assert len(loaded) == 1
    assert loaded[0].packet_id == "real1"
    assert loaded[0].content_hash() == pkt.content_hash()


# -- adversarial-audit edge fixes (2026-08-17) ------------------------------


def test_file_token_not_a_directory_raises_typed_error(as_en_suite, tmp_path):
    """F1: a token that resolves to a FILE inside snapshots/ (a stray json
    someone dropped, or an operator typo) is neither the typed "missing" nor
    the honest "empty" case -- globbing a file would silently yield no packets
    (fabricating a zero delta) or raise untyped NotADirectoryError. It must
    raise SnapshotNotFoundError so the failure stays typed."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    # Drop a stray FILE (not a directory) into snapshots/.
    rogue = tmp_path / "memory" / "snapshots" / "rogue.json"
    rogue.write_text("{}", encoding="utf-8")
    with pytest.raises(SnapshotNotFoundError):
        differ.diff(receiver, [as_en_suite], "A", "rogue.json")


def test_non_string_signature_field_raises_typed_error(as_en_suite, tmp_path):
    """A1: a malformed/tampered report with a non-string ``signature`` (e.g. an
    int or list) raises SignatureMismatchError, NOT a bare TypeError from
    compare_digest. The typed-error promise covers malformed values, not just
    absent ones."""
    receiver = _receiver()
    differ = _differ(tmp_path)
    _write_snapshot(tmp_path, "A", [])
    _write_snapshot(tmp_path, "B", [_promoted_packet("p1", "learner", _AS_CAP, _skill_for_heldout(as_en_suite))])
    d = differ.diff(receiver, [as_en_suite], "A", "B").to_dict()
    for bad in (12345, ["a", "b"], {"k": "v"}, True):
        d["signature"] = bad
        with pytest.raises(SignatureMismatchError):
            differ.verify(d)


def test_cli_diff_verify_missing_key_exits_two(as_en_suite, tmp_path, capsys):
    """I1: ``asea diff-verify`` with no local signing key prints the reason and
    exits 2 (a hard, typed error) -- not a raw traceback at exit 1, and never
    exit 0 (which would be a silent pass). The docstring promises exit 2; the
    code must deliver it."""
    from asea.cli import main

    ws = tmp_path / "wscli"
    ws.mkdir()
    report_path = ws / "report.json"
    report_path.write_text(
        json.dumps({"signature": "x", "signature_alg": SIGNATURE_ALG}),
        encoding="utf-8",
    )
    assert not (ws / "diff.key").exists(), "precondition: no key"
    rc = main(["diff-verify", "--workspace", str(ws), "--report", str(report_path)])
    assert rc == 2, "missing signing key must exit 2 (typed hard error), not 0 or 1"
    out = capsys.readouterr().out
    assert '"unverifiable": true' in out, "the exit-2 path must flag the report unverifiable"