"""SPRT early-stop -- the asymmetric test itself (B2, audit 2026-08-17).

These tests pin the patentable-novel asymmetry at the API level: the test may
stop early to REJECT, but NEVER to PROMOTE. Integration with the evaluator
(that the evaluator actually short-circuits a real held-out run) is covered in
test_distillation_and_evaluation.py; this file is the pure statistical core.

Honesty contract pinned here:
  * a clearly-failing packet (all cases regress) is early-REJECTED after fewer
    than all cases, with the reject boundary at >=95% confidence (beta=0.05);
  * a clearly-good packet (no cases regress) NEVER stops early to promote --
    should_stop() stays False even after the LLR crosses the promote boundary;
  * the asymmetry is enforced in the API: should_stop() is True ONLY for REJECT;
    a caller looping on ``while not sprt.should_stop()`` cannot early-promote;
  * the test is pure/deterministic (same stream -> same verdict sequence);
  * crossing the promote boundary does NOT lock in a promote -- the test can
    still reach a reject afterwards (it never STOPPED on promote);
  * config validation refuses an inverted test (p0 <= p1) and out-of-range
    rates, with a typed SprtConfigError -- a misconfigured SPRT that silently
    swapped its hypotheses would early-reject every good packet;
  * the stop record documents everything a reader needs to tell a statistical
    early-stop from a full sweep.
"""

from __future__ import annotations

import math

import pytest

from asea.core.errors import SprtConfigError
from asea.sprt import CONTINUE, PROMOTE_ELIGIBLE, REJECT, SPRT, SprtConfig


# -- boundaries: 95% confidence --------------------------------------------


def test_reject_boundary_is_95pct_confidence():
    """log B = log(beta/(1-alpha)); with alpha=beta=0.05 that is
    log(0.05/0.95) ~= -2.944. The false-reject rate is bounded by beta=0.05,
    i.e. the early-reject carries >=95% confidence."""
    cfg = SprtConfig()
    assert cfg.log_B == pytest.approx(math.log(0.05 / 0.95))
    assert cfg.log_A == pytest.approx(math.log(0.95 / 0.05))


def test_reject_boundary_is_symmetric_to_promote_for_equal_errors():
    """With alpha == beta, the two boundaries are equidistant from 0 (|log A|
    == |log B|). The asymmetry is NOT in the boundaries -- they are symmetric
    here -- it is in which boundary is allowed to STOP."""
    cfg = SprtConfig()
    assert cfg.log_A == pytest.approx(-cfg.log_B)


# -- the asymmetry: early-REJECT yes, early-PROMOTE no ----------------------


def test_clearly_failing_packet_is_early_rejected():
    """An all-regressions stream: each case pushes LLR down by log(p1/p0). The
    test reaches REJECT after a SMALL number of cases (well before a full
    sweep), and should_stop() becomes True. With defaults, log(p1/p0) =
    log(0.2) ~= -1.609 and log B ~= -2.944, so 2 regressed cases suffice."""
    sprt = SPRT(SprtConfig())
    total = 50
    verdicts = []
    for _ in range(total):
        verdicts.append(sprt.update(regressed=True))
    assert sprt.should_stop() is True
    assert sprt.verdict() == REJECT
    assert sprt.cases_evaluated < total, "must stop before evaluating all cases"
    assert sprt.cases_evaluated == 2, "default config rejects a perfect-failure after 2 cases"
    # Once stopped, further updates are no-ops returning REJECT.
    assert sprt.update(regressed=False) == REJECT
    assert sprt.cases_evaluated == 2


def test_clearly_good_packet_never_early_promotes():
    """An all-NON-regressions stream: LLR rises, crosses the promote boundary
    (PROMOTE_ELIGIBLE), but should_stop() STAYS False -- the test never stops
    to promote. After all cases the consumer would have run the FULL sweep."""
    sprt = SPRT(SprtConfig())
    total = 50
    for _ in range(total):
        sprt.update(regressed=False)
    assert sprt.should_stop() is False, "a good packet must NEVER be early-promoted"
    assert sprt.verdict() == PROMOTE_ELIGIBLE
    assert sprt.cases_evaluated == total, "no early stop -> all cases evaluated"
    # The promote boundary WAS crossed (reported), but it was never a stop.
    rec = sprt.stop_record()
    assert rec["promote_eligible_seen"] is True


def test_asymmetry_enforced_in_should_stop_only_reject_stops():
    """The asymmetry lives in should_stop(): it is True ONLY for REJECT. Drive
    the test to PROMOTE_ELIGIBLE and assert should_stop() is False; drive it to
    REJECT and assert True. A caller looping on should_stop() cannot
    early-promote regardless of how it reads the verdict."""
    sprt = SPRT(SprtConfig())
    # Drive up to PROMOTE_ELIGIBLE (all non-regressions, ~6 cases to cross log_A).
    for _ in range(6):
        sprt.update(regressed=False)
    assert sprt.verdict() == PROMOTE_ELIGIBLE
    assert sprt.should_stop() is False, "PROMOTE_ELIGIBLE must NOT stop"

    sprt2 = SPRT(SprtConfig())
    for _ in range(2):
        sprt2.update(regressed=True)
    assert sprt2.verdict() == REJECT
    assert sprt2.should_stop() is True


def test_crossing_promote_boundary_does_not_lock_in_a_promote():
    """The test never STOPPED on promote, so a packet that looked good then
    started regressing can still reach a REJECT. This proves the asymmetry is
    'never stop on promote', not 'ignore the promote boundary then freeze'."""
    sprt = SPRT(SprtConfig())
    # Looks good: cross the promote boundary.
    for _ in range(6):
        sprt.update(regressed=False)
    assert sprt.verdict() == PROMOTE_ELIGIBLE
    assert sprt.should_stop() is False
    # Then it turns bad: feed enough regressions to drive LLR down past log_B.
    for _ in range(6):
        sprt.update(regressed=True)
    assert sprt.verdict() == REJECT, "a promote-eligible packet that turns bad can still be rejected"
    assert sprt.should_stop() is True


# -- purity / determinism ---------------------------------------------------


def test_sprt_is_pure_and_deterministic():
    """Same stream -> same LLR trail and verdict sequence. No model state, no
    randomness: the test is a pure function of (config, regression stream)."""
    stream = [True, False, True, True, False, False, True, True]
    s1 = SPRT(SprtConfig())
    s2 = SPRT(SprtConfig())
    v1, v2 = [], []
    for r in stream:
        v1.append(s1.update(regressed=r))
        v2.append(s2.update(regressed=r))
    assert v1 == v2
    assert s1.stop_record()["llr_trail"] == s2.stop_record()["llr_trail"]


def test_continue_verdict_before_any_boundary():
    """A short mixed stream that has not reached either boundary returns
    CONTINUE and should_stop() is False."""
    sprt = SPRT(SprtConfig())
    sprt.update(regressed=False)  # +0.588
    assert sprt.verdict() == CONTINUE
    assert sprt.should_stop() is False
    sprt.update(regressed=False)  # +1.176, still < log_A (2.944)
    assert sprt.verdict() == CONTINUE


# -- config validation ------------------------------------------------------


def test_inverted_config_rejected():
    """p0 <= p1 means H0 (unacceptable) regresses LESS than H1 (acceptable) --
    an inverted test that would early-reject GOOD packets. Refused with a typed
    SprtConfigError, not silently coerced."""
    with pytest.raises(SprtConfigError):
        SprtConfig(p0=0.1, p1=0.5)
    with pytest.raises(SprtConfigError):
        SprtConfig(p0=0.3, p1=0.3)  # equal -> not strictly greater


def test_out_of_range_rates_rejected():
    with pytest.raises(SprtConfigError):
        SprtConfig(p0=0.0, p1=0.0)  # p1 not > 0
    with pytest.raises(SprtConfigError):
        SprtConfig(p0=1.0, p1=0.1)  # p0 not < 1
    with pytest.raises(SprtConfigError):
        SprtConfig(alpha=0.0)
    with pytest.raises(SprtConfigError):
        SprtConfig(beta=1.5)


def test_stricter_beta_makes_reject_harder_to_reach():
    """A smaller beta (false-reject rate) -> a more negative log B -> more
    regressed cases required to early-reject. The confidence knob is honest:
    tightening beta makes early-reject more conservative (more evidence needed)."""
    loose = SPRT(SprtConfig(beta=0.05))
    strict = SPRT(SprtConfig(beta=0.01))  # log B = log(0.01/0.95) ~ -4.55
    assert SprtConfig(beta=0.01).log_B < SprtConfig(beta=0.05).log_B
    # Both fed all-regressions; the stricter test needs more cases to reject.
    n_loose = 0
    while not loose.should_stop() and n_loose < 50:
        loose.update(regressed=True); n_loose += 1
    n_strict = 0
    while not strict.should_stop() and n_strict < 50:
        strict.update(regressed=True); n_strict += 1
    assert n_strict > n_loose, "smaller beta -> more cases to early-reject"


# -- stop record ------------------------------------------------------------


def test_stop_record_documents_the_early_stop():
    """A reader uses the stop record to tell a statistical early-stop from a
    full sweep: it carries the config, the case index where it stopped, the
    regression count, the LLR, the verdict, and the boundary values."""
    sprt = SPRT(SprtConfig(p0=0.5, p1=0.1, alpha=0.05, beta=0.05))
    for _ in range(3):
        sprt.update(regressed=True)
    rec = sprt.stop_record()
    assert rec["verdict"] == REJECT
    assert rec["stopped_at"] == 2  # rejected at case 2 (see test above)
    assert rec["cases_evaluated"] == 2
    assert rec["regressions"] == 2
    assert "config" in rec and rec["config"]["p0"] == 0.5
    assert "llr_trail" in rec and len(rec["llr_trail"]) == 2