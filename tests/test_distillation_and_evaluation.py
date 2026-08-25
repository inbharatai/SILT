"""Distillation, metrics and the before/after evaluator."""

from __future__ import annotations

import pytest

from asea.benchmarks.harness import BenchmarkHarness, CaseResult, SuiteResult
from asea.core.errors import DistillationError, EvaluationError
from asea.core.plugins import default_registry
from asea.core.protocol import (
    Domain,
    EvaluationScores,
    Gap,
    LearningLevel,
    OriginKind,
    PacketType,
    PromotionStatus,
    Provenance,
)
from asea.distill.strategies import (
    CodeDistiller,
    StructuredDistiller,
    TextDistiller,
    TTSDistiller,
)
from asea.evaluator import metrics
from asea.evaluator.evaluator import Evaluator
from asea.promotion.gate import PromotionGate
from asea.evaluator.metrics_plugins import CodeMetric, TTSMetric
from asea.evaluator.similarity import ExactMatch, LexicalSimilarity, char_ratio, token_f1
from asea.extraction.extractors import TextExtractor
from asea.modules.mock.zoo import make_generic_receiver, make_generic_sender, text_cap
from asea.sprt import SPRT, SprtConfig


# -- similarity -------------------------------------------------------------


def test_lexical_similarity_is_labelled_non_semantic():
    assert LexicalSimilarity().is_semantic is False, (
        "the bundled backend must never claim to be semantic"
    )


def test_similarity_bounds():
    sim = LexicalSimilarity()
    assert sim.similarity("hello world", "hello world") == 1.0
    assert sim.similarity("", "") == 1.0
    assert 0.0 <= sim.similarity("abc", "xyz") <= 1.0


def test_token_f1_is_order_insensitive():
    """Documented weakness, asserted so nobody mistakes it for a feature."""
    assert token_f1("I eat rice", "rice eat I") == 1.0
    assert char_ratio("I eat rice", "rice eat I") < 1.0


def test_exact_match_backend():
    assert ExactMatch().similarity("Yes", " yes ") == 1.0
    assert ExactMatch().similarity("yes", "no") == 0.0


# -- language preservation --------------------------------------------------


@pytest.mark.parametrize(
    "text,language,expected",
    [
        ("ভাত পানী", "as", 1.0),
        ("rice water", "as", 0.0),
        ("rice water", "as->en", 1.0),
        ("मैं चावल", "hi", 1.0),
        ("", "as", 1.0),
        ("12345", "as", 1.0),
        ("hello", "klingon", 1.0),
    ],
)
def test_language_preservation(text, language, expected):
    assert metrics.language_preservation(text, language) == pytest.approx(expected)


def test_language_preservation_catches_partial_code_switching():
    score = metrics.language_preservation("my নাম Ram", "as->en")
    assert 0.0 < score < 1.0


# -- hallucination ----------------------------------------------------------


def test_hallucination_flags_absolutes():
    assert metrics.hallucination_risk("This is guaranteed to always cure it.") > 0.3


def test_hallucination_flags_vague_authority():
    assert metrics.hallucination_risk("Studies show this works.") > 0.3


def test_hallucination_low_for_grounded_output():
    assert metrics.hallucination_risk("I eat rice", "I eat rice") < 0.2


def test_hallucination_empty_output_is_max_risk():
    assert metrics.hallucination_risk("") == 1.0


# -- schema compliance ------------------------------------------------------


def test_schema_compliance_requires_populated_payload(capability, clean_provenance, packet_factory):
    empty = packet_factory(capability, clean_provenance,
                           packet_type=PacketType.GLOSSARY, distilled_skill={"entries": []})
    assert metrics.schema_compliance(empty) == 0.0

    good = packet_factory(capability, clean_provenance, packet_type=PacketType.GLOSSARY,
                          distilled_skill={"entries": [{"source": "a", "target": "b"}]})
    assert metrics.schema_compliance(good) == 1.0

    untyped = packet_factory(capability, clean_provenance)
    assert metrics.schema_compliance(untyped) == 0.0


# -- modality metrics -------------------------------------------------------


def test_tts_metric_is_strict_but_delimiter_tolerant():
    m = TTSMetric()
    assert m.score("kam", "/kam/") == 1.0
    assert m.score("kam", "[k a m]") == 1.0
    assert m.score("kam", "kum") == 0.0


def test_code_metric_ignores_whitespace_only():
    m = CodeMetric()
    assert m.score("if x is None:", "if  x   is None:") == 1.0
    assert 0.0 < m.score("if x is None:", "if x == None:") < 1.0


# -- distillation -----------------------------------------------------------


def _packets(pairs, domain=Domain.TRANSLATION):
    cap = text_cap("translate", "as->en", domain)
    gap = Gap(capability=cap, receiver_score=0.1, sender_score=0.9)
    sender = make_generic_sender(
        capabilities=[cap], knowledge={cap.as_str(): dict(pairs)}
    )
    receiver = make_generic_receiver(capabilities=[cap])
    return TextExtractor().extract(
        sender, receiver, gap,
        [{"case_id": "c{}".format(i), "prompt": p, "expected": e,
          "meta": {"human_verified": True}} for i, (p, e) in enumerate(pairs)],
    )


def test_distillation_drops_raw_sender_output():
    """The single most important invariant in the system."""
    out = TextDistiller().distill(_packets([("ভাত", "rice"), ("পানী", "water")]))
    assert len(out) == 1
    assert out[0].sender_output is None
    assert out[0].promotion_status == PromotionStatus.DISTILLED
    assert out[0].packet_type == PacketType.GLOSSARY


def test_distillation_compresses_many_into_one():
    packets = _packets([("ভাত", "rice"), ("পানী", "water"), ("ঘৰ", "house")])
    out = TextDistiller().distill(packets)
    assert len(out) == 1
    assert len(out[0].distilled_skill["entries"]) == 3
    assert out[0].notes["member_count"] == 3


def test_distillation_prefers_verified_reference_over_sender_output():
    """Sender says 'coffee', the corpus says 'tea'. The corpus wins."""
    cap = text_cap("translate", "as->en", Domain.TRANSLATION)
    gap = Gap(capability=cap, receiver_score=0.1, sender_score=0.9)
    sender = make_generic_sender(capabilities=[cap], knowledge={cap.as_str(): {"চাহ": "coffee"}})
    receiver = make_generic_receiver(capabilities=[cap])
    packets = TextExtractor().extract(
        sender, receiver, gap,
        [{"case_id": "c", "prompt": "চাহ", "expected": "tea", "meta": {"human_verified": True}}],
    )
    out = TextDistiller().distill(packets)
    assert out[0].distilled_skill["entries"][0]["target"] == "tea"


def test_distillation_deduplicates():
    out = TextDistiller().distill(_packets([("ভাত", "rice"), ("ভাত", "rice")]))
    assert len(out[0].distilled_skill["entries"]) == 1


def test_distiller_rejects_foreign_modality():
    with pytest.raises(DistillationError):
        TTSDistiller().distill(_packets([("ভাত", "rice")]))


def test_provenance_merge_takes_the_weakest_link():
    packets = _packets([("ভাত", "rice"), ("পানী", "water")])
    packets[0].provenance = Provenance(
        origin_kind=OriginKind.MODEL_GENERATED, chain=["m1"], synthetic_depth=2, is_mock=True
    )
    packets[1].provenance = Provenance(
        origin_kind=OriginKind.HUMAN_VERIFIED, chain=["h1"], synthetic_depth=0, is_mock=False
    )
    merged = TextDistiller().distill(packets)[0].provenance
    assert merged.origin_kind == OriginKind.MODEL_GENERATED
    assert merged.synthetic_depth == 2
    assert merged.is_mock is True
    assert merged.chain == ["m1", "h1"]


def test_tts_payload_declares_its_scope_limit():
    cap = text_cap("g2p", "as-ipa")  # modality overridden below
    packets = _packets([("ক", "k")])
    for p in packets:
        p.modality = TTSDistiller.modality
        p.notes["grapheme"] = p.notes["prompt"]
    payload = TTSDistiller().distill(packets)[0].distilled_skill
    assert payload["scope"] == "symbolic_g2p_only"
    assert "voice_timbre" in payload["not_transferable"]


def test_code_and_rule_payload_shapes():
    packets = _packets([("== None", "is None")])
    for p in packets:
        p.modality = CodeDistiller.modality
    code_payload = CodeDistiller().distill(packets)[0].distilled_skill
    assert code_payload["examples"][0]["fixed"] == "is None"

    packets2 = _packets([("chest pain", "seek emergency care")])
    for p in packets2:
        p.modality = StructuredDistiller.modality
    rule_payload = StructuredDistiller().distill(packets2)[0].distilled_skill
    assert rule_payload["rules"][0]["action"] == "seek emergency care"


# -- evaluation -------------------------------------------------------------


def _evaluate(suite, sender_split=("extraction",), receiver_knowledge=None):
    plugins = default_registry()
    harness = BenchmarkHarness(plugins=plugins)
    cap = suite.capability()
    sender = make_generic_sender(
        capabilities=[cap],
        knowledge={cap.as_str(): {
            str(c.prompt): c.expected for c in suite.cases if c.split in sender_split
        }},
    )
    receiver = make_generic_receiver(
        capabilities=[cap], knowledge=receiver_knowledge, fallback="echo"
    )
    gap = Gap(capability=cap, receiver_score=0.2, sender_score=0.9)
    packets = plugins.extractor(cap.modality).extract(
        sender, receiver, gap,
        [{"case_id": c.case_id, "prompt": c.prompt, "expected": c.expected,
          "meta": c.meta} for c in suite.split("extraction")],
    )
    distilled = plugins.distiller(cap.modality).distill(packets)
    return Evaluator(harness=harness), distilled[0], receiver


def test_evaluation_measures_real_heldout_improvement(as_en_suite):
    evaluator, packet, receiver = _evaluate(as_en_suite)
    report = evaluator.evaluate(packet, receiver, as_en_suite)
    assert report.improvement > 0.0
    assert packet.promotion_status == PromotionStatus.EVALUATED
    assert packet.evaluator_score is not None
    assert report.scores.baseline_score < report.scores.candidate_score


def test_evaluation_report_carries_the_proxy_caveat(as_en_suite):
    evaluator, packet, receiver = _evaluate(as_en_suite)
    report = evaluator.evaluate(packet, receiver, as_en_suite)
    assert "lexical similarity proxy" in report.to_dict()["caveat"]
    assert report.to_dict()["similarity_is_semantic"] is False


def test_evaluation_requires_distilled_payload(capability, clean_provenance,
                                               packet_factory, as_en_suite):
    packet = packet_factory(capability, clean_provenance)
    with pytest.raises(EvaluationError, match="no distilled_skill"):
        Evaluator().evaluate(packet, make_generic_receiver(), as_en_suite)


def test_regression_sweep_runs_on_control_suite(as_en_suite, hi_en_suite):
    hindi_knowledge = {
        hi_en_suite.capability().as_str(): {
            str(c.prompt): c.expected for c in hi_en_suite.cases
        }
    }
    evaluator, packet, receiver = _evaluate(
        as_en_suite, receiver_knowledge=hindi_knowledge
    )
    report = evaluator.evaluate(packet, receiver, as_en_suite, [hi_en_suite])
    assert len(report.regressions) == 1
    assert report.regressions[0]["suite_id"] == "hindi_english_v1"
    assert report.regressions[0]["regressed"] is False
    assert report.scores.regression_detected is False


def test_control_movement_rejects_end_to_end_through_real_evaluator(
    as_en_suite, hi_en_suite, capability, clean_provenance, packet_factory,
):
    """A3 Gate 1 end-to-end (audit 2026-08-17). The packet's distilled_skill
    BLEEDS into a non-target control suite. The receiver knows neither
    Assamese nor Hindi (fallback="echo" -> ~0 similarity to the English
    expected), so the baseline on BOTH the target and the control is ~0. The
    skill carries exact entries for the target (legitimate -- it is an as->en
    packet) AND exact entries for the hi->en control (the bleed). Conditioning
    on it therefore lifts the target (good) AND lifts the control (movement).

    ``no_regression`` does not fire -- the control IMPROVED, it did not drop.
    Before A3 this packet would have promoted: target improved, no regression,
    no case regressions. The new ``no_control_movement`` hard check sees
    |delta| on the control and REJECTS. This exercises the REAL
    Evaluator.evaluate -> PromotionGate.apply wiring -- the
    scores-construction test in test_promotion_gate.py cannot catch a
    regression in the evaluator's ``moved`` computation, because it bypasses
    the evaluator entirely. (Mirror of Gate 2's test_runner_rejects_on_control
    _movement in test_deep_apply.py.)"""
    # Bleeding skill: exact as->en target entries + exact hi->en control entries.
    target_entries = [
        {"source": str(c.prompt), "target": str(c.expected)}
        for c in as_en_suite.split("heldout")
    ]
    control_entries = [
        {"source": str(c.prompt), "target": str(c.expected)}
        for c in hi_en_suite.split("regression")
    ]
    packet = packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": target_entries + control_entries},
        safety_score=1.0,
        learning_level=LearningLevel.L3_SKILL_PACKET,
        promotion_status=PromotionStatus.DISTILLED,
    )
    receiver = make_generic_receiver(capabilities=[capability], fallback="echo")
    evaluator = Evaluator(harness=BenchmarkHarness(plugins=default_registry()))

    evaluator.evaluate(packet, receiver, as_en_suite, [hi_en_suite])

    decision = PromotionGate().apply(packet, rollback_token="snap-bleed")
    assert decision.status == PromotionStatus.REJECTED
    reason = packet.rejection_reason or ""
    assert "no_control_movement" in reason
    # It is movement, not a drop: the control improved, so no_regression passed.
    assert "no_regression" not in reason
    # And the evaluator genuinely computed the movement (not a constructed score).
    assert packet.scores.control_movement_detected is True
    assert packet.scores.regression_detected is False


def test_harness_refuses_unknown_split(as_en_suite):
    with pytest.raises(EvaluationError, match="unknown split"):
        as_en_suite.split("train")


def test_harness_refuses_empty_split(as_en_suite):
    with pytest.raises(EvaluationError, match="no cases in split"):
        BenchmarkHarness().run(make_generic_receiver(), as_en_suite, split="regression")


def test_split_discipline_holds_in_shipped_data(as_en_suite):
    """Extraction and heldout prompts must not overlap, or the A/B is meaningless."""
    extraction = {str(c.prompt) for c in as_en_suite.split("extraction")}
    heldout = {str(c.prompt) for c in as_en_suite.split("heldout")}
    assert extraction & heldout == set()


# -- SPRT early-stop (B2, audit 2026-08-17) ---------------------------------
#
# The SPRT is an ASYMMETRIC sequential test wired into the evaluator's candidate
# held-out run: early-REJECT at 95% confidence is allowed (stop and fail a
# clearly-failing packet after a handful of cases), early-PROMOTE is FORBIDDEN
# (a good packet runs the FULL held-out set, so the regular gate still sees every
# case). The asymmetry lives in SPRT.should_stop (True only on REJECT) and is
# enforced end-to-end here, not just in the unit tests in test_sprt.py.


def _heldout(as_en_suite):
    return list(as_en_suite.split("heldout"))


def _glossary_packet(capability, clean_provenance, packet_factory, entries):
    return packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": entries},
        safety_score=1.0,
        learning_level=LearningLevel.L3_SKILL_PACKET,
        promotion_status=PromotionStatus.DISTILLED,
    )


def test_sprt_early_rejects_a_failing_packet_end_to_end(
    as_en_suite, capability, clean_provenance, packet_factory,
):
    """B2 end-to-end REJECT path. The receiver ALREADY knows the correct
    as->en mapping (so its baseline held-out score is ~1.0); the packet's
    distilled_skill carries WRONG targets that OVERRIDE that knowledge
    (infer_with_skills checks the skill before the receiver's own table), so
    every candidate case regresses vs baseline. The SPRT hits its REJECT
    boundary at 95% confidence after 2 all-regress cases (defaults p0=0.5,
    p1=0.1: LLR = 2*log(0.2) = -3.22 <= log_B = -2.94), so the harness
    stops the candidate run PARTIAL -- 2 of 6 held-out cases scored -- and
    the evaluator stamps scores.sprt with a reject record. The promotion
    gate's new HARD ``no_statistical_early_reject`` check then REJECTS the
    packet on that record alone, regardless of the partial (optimistic)
    aggregate. This exercises the real Evaluator.evaluate -> harness
    stop_callback -> SPRT -> PromotionGate.apply wiring."""
    heldout = _heldout(as_en_suite)
    assert len(heldout) == 6, "fixture sanity: assamese_english_v1 has 6 held-out cases"

    # Receiver knows the RIGHT answers -> baseline ~1.0 on every held-out case.
    cap = as_en_suite.capability()
    receiver = make_generic_receiver(
        capabilities=[cap],
        knowledge={cap.as_str(): {str(c.prompt): str(c.expected) for c in heldout}},
        fallback="echo",
    )
    # Skill carries WRONG targets (clearly unrelated English) that override.
    wrong_entries = [
        {"source": str(c.prompt), "target": "QQZZXX{}".format(i)}
        for i, c in enumerate(heldout)
    ]
    packet = _glossary_packet(capability, clean_provenance, packet_factory, wrong_entries)

    evaluator = Evaluator(
        harness=BenchmarkHarness(plugins=default_registry()),
        sprt=SprtConfig(),
    )
    report = evaluator.evaluate(packet, receiver, as_en_suite)

    # The SPRT stopped the candidate run early -- partial, not full.
    assert report.scores.sprt is not None
    assert report.scores.sprt["verdict"] == "reject"
    assert len(report.candidate.case_results) < len(report.baseline.case_results)
    assert report.scores.case_count < len(heldout)
    assert report.scores.case_count >= 1

    # The gate fails the packet on the SPRT record (a HARD check).
    packet.promotion_status = PromotionStatus.EVALUATED
    decision = PromotionGate().apply(packet, rollback_token="snap-sprt")
    assert decision.status == PromotionStatus.REJECTED
    failed = {c["name"] for c in decision.to_dict()["checks"] if not c["passed"]}
    assert "no_statistical_early_reject" in failed


def test_sprt_never_early_promotes_a_good_packet(
    as_en_suite, capability, clean_provenance, packet_factory,
):
    """B2 asymmetry end-to-end: the PROMOTE side is forbidden. A GOOD packet
    (correct targets, receiver knows nothing -> every candidate case beats
    baseline) drives the SPRT's LLR up through the promote boundary (6
    non-regressions -> LLR = 6*log(1.8) = 3.53 >= log_A = 2.94, so
    promote_eligible is reached), but should_stop() stays False -- the SPRT
    only ever stops to REJECT -- so the harness runs the FULL held-out set.
    scores.sprt stays None (the evaluator only stamps a record on an actual
    early stop), case_count == 6, and the gate's no_statistical_early_reject
    check PASSES. A good packet is never short-circuited past the regular
    gate."""
    heldout = _heldout(as_en_suite)
    cap = as_en_suite.capability()
    receiver = make_generic_receiver(capabilities=[cap], fallback="echo")
    correct_entries = [
        {"source": str(c.prompt), "target": str(c.expected)} for c in heldout
    ]
    packet = _glossary_packet(capability, clean_provenance, packet_factory, correct_entries)

    evaluator = Evaluator(
        harness=BenchmarkHarness(plugins=default_registry()),
        sprt=SprtConfig(),
    )
    report = evaluator.evaluate(packet, receiver, as_en_suite)

    # No early stop -> no SPRT record, full run.
    assert report.scores.sprt is None
    assert report.scores.case_count == len(heldout)
    assert len(report.candidate.case_results) == len(heldout)

    packet.promotion_status = PromotionStatus.EVALUATED
    decision = PromotionGate().apply(packet, rollback_token="snap-sprt-good")
    sprt_check = next(
        c for c in decision.to_dict()["checks"] if c["name"] == "no_statistical_early_reject"
    )
    assert sprt_check["passed"] is True


def _pre_b2_score_cases(harness, cases, actuals, suite, with_skills):
    """The EXACT pre-B2 scoring loop: build all CaseResults in suite order from
    a prebuilt ``actuals`` list, using the same metrics in the same order as the
    old two-phase ``run()``. This is the byte-identical anchor -- it does not
    interleave inference and scoring, so if the B2 interleave changed any
    observable field this reference still reflects pre-B2 and the comparison
    fails (adversarial audit 2026-08-17, finding 2)."""
    metric_plugin = harness.plugins.metric(suite.modality) if harness.plugins else None
    results = []
    for case, actual in zip(cases, actuals):
        sim = harness.similarity.similarity(str(case.expected), str(actual))
        if metric_plugin is not None:
            task = metric_plugin.score(case.expected, actual, None)
        else:
            task = sim
        lang = metrics.language_preservation(actual, suite.language)
        halluc = metrics.hallucination_risk(actual, case.expected)
        score = metrics.aggregate(1.0, sim, task, lang, halluc)
        results.append(
            CaseResult(
                case_id=case.case_id, prompt=case.prompt, expected=case.expected,
                actual=actual, similarity=sim, task_success=task,
                language_preservation=lang, hallucination_risk=halluc, score=score,
            )
        )
    return results


def _pre_b2_two_phase_single(harness, module, suite, skills):
    """Pre-B2 single-case two-phase ``run()``: collect ALL actuals first, then
    score in a separate loop. Byte-identical reference for the default path."""
    cases = suite.split("heldout")
    cap = suite.capability()
    actuals = []
    for case in cases:
        if skills:
            actuals.append(module.infer_with_skills(cap, case.prompt, skills))
        else:
            actuals.append(module.infer(cap, case.prompt))
    results = _pre_b2_score_cases(harness, cases, actuals, suite, bool(skills))
    return SuiteResult(
        suite_id=suite.suite_id, module_id=module.module_id, split="heldout",
        with_skills=bool(skills), case_results=results,
        similarity_is_semantic=harness.similarity.is_semantic,
    )


def _pre_b2_two_phase_batched(harness, module, suite, skills, batch):
    """Pre-B2 batched two-phase ``run()``: collect ALL actuals in chunks first,
    then score in a separate loop. Byte-identical reference for the batched
    interleave path (adversarial audit 2026-08-17, finding 3 coverage)."""
    cases = suite.split("heldout")
    cap = suite.capability()
    actuals = []
    for start in range(0, len(cases), batch):
        chunk = cases[start:start + batch]
        prompts = [c.prompt for c in chunk]
        if skills:
            outs = module.infer_with_skills_batch(cap, prompts, skills)
        else:
            outs = module.infer_batch(cap, prompts)
        actuals.extend(outs)
    results = _pre_b2_score_cases(harness, cases, actuals, suite, bool(skills))
    return SuiteResult(
        suite_id=suite.suite_id, module_id=module.module_id, split="heldout",
        with_skills=bool(skills), case_results=results,
        similarity_is_semantic=harness.similarity.is_semantic,
    )


class _BatchReceiver:
    """A minimal batch-capable receiver for the byte-identical anchor. Knows a
    {prompt: answer} table (deterministic, score-equivalent to single-case) and
    implements the batch APIs by looping internally in PROMPT ORDER and
    un-sorting nothing -- the honest order-preserving obligation. The generic
    mock receiver has no ``infer_batch`` (effective batch stays 1), so this
    standalone module is what exercises the batched interleave path."""

    is_mock = True
    consumes_skills = True

    def __init__(self, knowledge, preferred_batch_size):
        from asea.modules.mock.base import lookup_in_skills
        self._lookup = lookup_in_skills
        self.module_id = "batch-ref"
        self.display_name = "Batch Ref"
        self._knowledge = knowledge
        self.preferred_batch_size = preferred_batch_size

    def _answer(self, prompt, skills):
        if skills:
            hit = self._lookup(skills, prompt)
            if hit is not None:
                return hit
        return self._knowledge.get(str(prompt), prompt)  # echo fallback

    def infer(self, cap, prompt):
        return self._answer(prompt, None)

    def infer_with_skills(self, cap, prompt, skills):
        return self._answer(prompt, skills)

    def infer_batch(self, cap, prompts):
        return [self._answer(p, None) for p in prompts]

    def infer_with_skills_batch(self, cap, prompts, skills):
        return [self._answer(p, skills) for p in prompts]


def test_sprt_enabled_on_a_good_packet_is_byte_identical_to_pre_b2(
    as_en_suite, capability, clean_provenance, packet_factory,
):
    """B2 byte-identical guarantee, anchored to a PRE-B2 two-phase reference
    (adversarial audit 2026-08-17, finding 2): the previous form compared two
    NEW-code paths (SPRT-off vs SPRT-on), so a refactor change shared by both
    would have passed while the byte-identical-to-pre-B2 contract broke. This
    version rebuilds the old two-phase ``run()`` inline (collect all actuals,
    THEN score) and asserts the refactored harness output EQUALS that reference
    -- for the candidate AND the baseline, with SPRT off AND on -- and that the
    evaluator SCORES are equal too (finding 3). A good packet never early-stops,
    so SPRT-on must be byte-identical to SPRT-off must be byte-identical to
    pre-B2."""
    heldout = _heldout(as_en_suite)
    cap = as_en_suite.capability()
    receiver = make_generic_receiver(capabilities=[cap], fallback="echo")
    correct_entries = [
        {"source": str(c.prompt), "target": str(c.expected)} for c in heldout
    ]
    harness = BenchmarkHarness(plugins=default_registry())

    # The pre-B2 reference, computed directly (no interleave, no SPRT). The
    # skills MUST be the redacted payload the evaluator actually feeds the
    # receiver (``packet.redacted_for_receiver()``), not the raw entries --
    # lookup_in_skills dispatches on packet_type / the distilled_skill wrapper,
    # so raw ``{"entries": ...}`` would miss and echo, making the reference
    # wrong rather than a faithful pre-B2 replay.
    ref_packet = _glossary_packet(capability, clean_provenance, packet_factory, correct_entries)
    skills = [ref_packet.redacted_for_receiver()]
    ref_candidate = _pre_b2_two_phase_single(harness, receiver, as_en_suite, skills)
    ref_baseline = _pre_b2_two_phase_single(harness, receiver, as_en_suite, None)

    ev_off = Evaluator(harness=harness)                       # SPRT disabled (default)
    ev_on = Evaluator(harness=harness, sprt=SprtConfig())     # SPRT enabled

    packet_off = _glossary_packet(capability, clean_provenance, packet_factory, correct_entries)
    packet_on = _glossary_packet(capability, clean_provenance, packet_factory, correct_entries)

    report_off = ev_off.evaluate(packet_off, receiver, as_en_suite)
    report_on = ev_on.evaluate(packet_on, receiver, as_en_suite)

    # Refactored harness == pre-B2 two-phase reference (the real anchor).
    assert report_off.candidate.model_dump() == ref_candidate.model_dump()
    assert report_off.baseline.model_dump() == ref_baseline.model_dump()
    # SPRT-on == SPRT-off == pre-B2; scores equal too.
    assert report_on.candidate.model_dump() == ref_candidate.model_dump()
    assert report_on.baseline.model_dump() == ref_baseline.model_dump()
    assert report_off.scores.model_dump() == report_on.scores.model_dump()
    assert report_off.scores.sprt is None
    assert report_on.scores.sprt is None
    assert report_off.scores.case_count == report_on.scores.case_count == len(heldout)


def test_harness_batched_interleave_is_byte_identical_to_pre_b2_two_phase(
    as_en_suite, capability, clean_provenance, packet_factory,
):
    """B2 byte-identical for the BATCHED interleave path (adversarial audit
    2026-08-17, finding 3): with ``stop_callback=None`` the batched path (infer
    a chunk, then score each case in the chunk, repeat) must equal the pre-B2
    two-phase batched reference (infer ALL chunks, THEN score all cases). The
    default-path anchor above only covers effective batch=1; this covers the
    chunked interleave with a non-divisible case count (6 cases / batch 4 ->
    [4, 2]). Uses a real SPRT-disabled harness run (no callback) so it also
    proves the batched path is byte-identical to pre-B2 in the non-SPRT case the
    evaluator's regression sweep could hit if a real connector declares a batch
    size."""
    heldout = _heldout(as_en_suite)
    cap = as_en_suite.capability()
    knowledge = {str(c.prompt): str(c.expected) for c in heldout}
    batch = 4  # 6 cases -> [4, 2], exercises a short final chunk
    module = _BatchReceiver(knowledge=knowledge, preferred_batch_size=batch)
    harness = BenchmarkHarness(plugins=default_registry(), max_batch_size=batch)

    # Redacted payload (same wrapper the evaluator uses) so lookup_in_skills
    # resolves -- a raw {"entries": ...} would miss and echo, making the anchor
    # trivially-and-meaninglessly equal rather than a faithful byte-identical
    # check.
    ref_packet = _glossary_packet(
        capability, clean_provenance, packet_factory,
        [{"source": str(c.prompt), "target": str(c.expected)} for c in heldout],
    )
    skills = [ref_packet.redacted_for_receiver()]

    # No stop_callback -> byte-identical-to-pre-B2 must hold on the batched path.
    got_candidate = harness.run(module, as_en_suite, split="heldout", skills=skills)
    got_baseline = harness.run(module, as_en_suite, split="heldout", skills=None)
    ref_candidate = _pre_b2_two_phase_batched(harness, module, as_en_suite, skills, batch)
    ref_baseline = _pre_b2_two_phase_batched(harness, module, as_en_suite, None, batch)

    assert got_candidate.model_dump() == ref_candidate.model_dump()
    assert got_baseline.model_dump() == ref_baseline.model_dump()


def test_gate_rejects_on_sprt_record_alone(capability, clean_provenance, packet_factory):
    """Gate isolation (mirror of the test_distilled_payload_present pattern):
    a packet that passes EVERY other check but carries an SPRT early-reject
    record is blocked by ``no_statistical_early_reject`` ALONE. This pins the
    hard check at the gate without depending on the evaluator wiring, so a
    regression that stops the evaluator from stamping scores.sprt cannot
    silently disable the gate check too."""
    scores = EvaluationScores(
        schema_compliance=1.0,
        semantic_similarity=0.9,
        task_success=0.9,
        language_preservation=1.0,
        hallucination_risk=0.1,
        aggregate=0.9,
        baseline_score=0.5,
        candidate_score=0.6,   # improvement >= min_improvement
        regression_detected=False,
        control_movement_detected=False,
        case_count=10,
        case_regression_count=0,
        # The discriminating field: an SPRT early-reject is on record.
        sprt={
            "config": {"p0": 0.5, "p1": 0.1, "alpha": 0.05, "beta": 0.05},
            "cases_evaluated": 2,
            "regressions": 2,
            "llr": -3.2189,
            "verdict": "reject",
            "stopped_at": 2,
            "promote_eligible_seen": False,
            "llr_trail": [-1.6094, -3.2189],
        },
    )
    packet = packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "a", "target": "b"}]},
        safety_score=1.0,
        learning_level=LearningLevel.L3_SKILL_PACKET,
        evaluator_score=0.9,
        promotion_status=PromotionStatus.EVALUATED,
    )
    packet.scores = scores

    decision = PromotionGate().apply(packet, rollback_token="snap-sprt-only")
    assert decision.status == PromotionStatus.REJECTED
    failed = {c["name"] for c in decision.to_dict()["checks"] if not c["passed"]}
    # The SPRT record is the SOLE failure -- every other check passes.
    assert failed == {"no_statistical_early_reject"}
    assert "no_statistical_early_reject" in (packet.rejection_reason or "")


@pytest.mark.parametrize("malformed", [
    {},                                            # empty dict: pydantic-valid, no keys
    {"verdict": "reject", "cases_evaluated": 2},   # missing llr
    {"verdict": "reject", "cases_evaluated": 2, "llr": None},  # llr is None
    {"verdict": "reject", "cases_evaluated": 2, "llr": "not-a-number"},  # bad llr
])
def test_gate_fail_closes_on_a_malformed_sprt_record(
    malformed, capability, clean_provenance, packet_factory,
):
    """Finding 1 (CONFIRMED) fix-pin: the ``no_statistical_early_reject`` DECISION
    is fail-closed for ANY non-None ``scores.sprt`` -- including a malformed or
    partial dict that could arrive from a corrupt on-disk packet. The DETAIL
    string must format defensively so ``decide()`` never raises out of
    ``apply()`` before the packet is stamped REJECTED (a fail-closed check that
    crashes is fail-open in practice: the packet is left in EVALUATED with no
    rejection_reason). For every malformed shape: no exception, status REJECTED,
    the packet is stamped, and the check is the recorded failure."""
    scores = EvaluationScores(
        schema_compliance=1.0, semantic_similarity=0.9, task_success=0.9,
        language_preservation=1.0, hallucination_risk=0.1, aggregate=0.9,
        baseline_score=0.5, candidate_score=0.6,
        regression_detected=False, control_movement_detected=False,
        case_count=10, case_regression_count=0, sprt=malformed,
    )
    packet = packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "a", "target": "b"}]},
        safety_score=1.0, learning_level=LearningLevel.L3_SKILL_PACKET,
        evaluator_score=0.9, promotion_status=PromotionStatus.EVALUATED,
    )
    packet.scores = scores

    # Must NOT raise -- fail-closed, not fail-by-crashing.
    decision = PromotionGate().apply(packet, rollback_token="snap-malformed")
    assert decision.status == PromotionStatus.REJECTED
    assert packet.promotion_status == PromotionStatus.REJECTED
    failed = {c["name"] for c in decision.to_dict()["checks"] if not c["passed"]}
    assert "no_statistical_early_reject" in failed
    assert "no_statistical_early_reject" in (packet.rejection_reason or "")
