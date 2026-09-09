# Capability catalog — implemented, optional and experimental

This is the operational catalog for SILT source commit
[`b583ba0d7de077d4e8594b08f0f77cc29655459e`](https://github.com/inbharatai/SILT/tree/b583ba0d7de077d4e8594b08f0f77cc29655459e).
Runtime/source and test citations below are pinned to that commit, not to moving
`main`. The catalog combines **18 legacy entries (L01–L18) and 25 activation
entries (C01–C25)**; **C01 is the same core packet path as L01** and is indexed
there rather than duplicated. C25 records unsupported/unproven outcomes, not a
completed capability. No runtime, policy, metric or feature default is changed.

[README overview](../README.md#capability-status-and-activation) ·
[Core packets](#core-packet-transfer) · [Optional adaptation](#optional-adaptation-and-diagnostics) ·
[Experimental setup](#experimental-entry-points) · [Composition/deployment](#composition-and-deployment) ·
[Compiler](#structural-compiler) · [Specialist](#source-derived-specialist) ·
[Evaluation/resources](#evaluation-observability-and-governance) · [Evidence](#recorded-evidence-and-measurement-units) ·
[Remaining gaps](#remaining-gaps)

## Status and evidence legend

| Term | Meaning — not an implication of the next state |
|---|---|
| **Implemented / source_code_verified** | A source path and inspected contract exist; this does **not** mean installed locally, dependencies provisioned, enabled, executed or quality-validated. |
| **Enabled** | Applicable configuration and runtime have been selected. L3 is default *when explicitly invoked*; experimental Studio is off without the exact startup flag. |
| **Executed** | A command/run occurred in a stated environment. A script, fixture, import, help command, finite loss or completed generation is not a pretrained quality success. |
| **Admitted** | The named subsystem accepted its own evidence/policy. Packet `PROMOTED`, Compose admitted evaluation, compiler seq2seq certificate and recovery `artifact_admitted` are not interchangeable. |
| **Activated** | An explicit operator action revalidated eligible evidence and selected a deployment pointer. Import/build/evaluate/plan/select do not do this automatically; a pointer is not a model server. |
| **Evidence kind** | Source/test definitions establish implementation scope; deterministic mocks and random native fixtures test mechanics; recorded pretrained runs are model/task/runtime-specific. Historical counts and outcomes are not rerun by this documentation change. |

`RECONSTRUCTED_UNVALIDATED`, compiler `UNADMITTED`, voice `pending_human`, and
specialist `BUILT_UNCERTIFIED` are intentional boundaries. An engineering-valid
export is not a model-quality certificate. Hash chains, local HMAC, runner
telemetry and independently authenticated evidence are different things.

## Core packet transfer

<a id="l01"></a> <a id="c01"></a>
### L01 / C01 — L3 inspectable skill packets

**Mechanism:** Pipeline.run defaults to L3; SkillPacket validates terminal-state metadata and redacted_for_receiver omits sender_output. Approved payloads are rendered as context by real connectors. Candidate evaluation intentionally exposes candidate context before permanent admission.

**Default / enablement:** Implemented; primary path when an operator calls run. No background self-learning or automatic model discovery implied.

**Invoke / knobs:** asea run --config `<json>` --workspace `<dir>`; Pipeline.run(adapter_id, suites, requested_level=LearningLevel.L3_SKILL_PACKET); render_skills/build_messages.

**Evidence kind:** Static source tracing; existing redaction/end-to-end tests; historical real-model reports with positive and negative results.

**Limits:** Conditioning is connector-dependent and not guaranteed retention/generalization. No evidence every model can absorb every skill. L3 changes no learner weights and copies no teacher weights; experimental specialist reconstruction separately uses source weights.

**Pinned source / tests / recorded reports:** [src/asea/core/pipeline.py:164-294](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/pipeline.py#L164-L294) · [src/asea/core/protocol.py:321-431](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/protocol.py#L321-L431) · [src/asea/modules/real/prompting.py:90-148](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/modules/real/prompting.py#L90-L148) · [tests/test_end_to_end.py:77](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_end_to_end.py#L77) · [docs/real_run_findings.md:18-35](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/real_run_findings.md#L18-L35) · [docs/real_run_findings.md:136-184](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/real_run_findings.md#L136-L184) · [docs/real_run_findings.md:201-255](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/real_run_findings.md#L201-L255) · [src/asea/specialist/recovery.py:643-664](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L643-L664) · [src/asea/specialist/recovery.py:1081-1108](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L1081-L1108)

<a id="l02"></a>
### L02 — Measured gaps and relevance filtering

**Mechanism:** GapEngine compares extraction split scores (receiver ceiling .85; minimum headroom .05). RelevanceFilter rejects duplicate prompts, low reference similarity, receiver competence and near-identical answers. Pipeline shares the harness similarity object with relevance.

**Default / enablement:** Active in normal Pipeline.run. Library/config-built default similarity is lexical; Studio TransferRequest defaults to embedding. Teacher cache is optional, off unless injected.

**Invoke / knobs:** Pipeline(gap_engine=GapEngine(policy=GapPolicy(...)), relevance=RelevanceFilter(policy=RelevancePolicy(...)), similarity=...); RelevancePolicy defaults: correctness .75, receiver ceiling .85, min_delta .05. Studio similarity=embedding|lexical and optional relevance_floor.

**Evidence kind:** Source trace and existing extraction/conformance tests; the historical report documents a lexical/embedding inconsistency fix and medical sample improvement +.2697 after relevance-floor calibration .75→.35. These are iterative small-suite results, not clinical efficacy or unchanged-default outcomes.

**Limits:** The filter estimates correctness from a chosen similarity metric, not truth. Reference-free probes skip the reference correctness floor. Bad/missing references and metric mismatch can allow unhelpful signals or reject useful ones.

**Pinned source / tests / recorded reports:** [src/asea/core/gap.py:26-97](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/gap.py#L26-L97) · [src/asea/filters/relevance.py:26-114](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/filters/relevance.py#L26-L114) · [src/asea/core/pipeline.py:124-131](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/pipeline.py#L124-L131) · [src/asea/core/pipeline.py:183-248](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/pipeline.py#L183-L248) · [src/asea/studio/server.py:162-173](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/server.py#L162-L173) · [docs/real_run_findings.md:119-184](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/real_run_findings.md#L119-L184) · [docs/real_run_findings.md:273-310](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/real_run_findings.md#L273-L310) · [tests/test_conformance.py:248-270](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_conformance.py#L248-L270)

<a id="l03"></a>
### L03 — Capability-specific distillation

**Mechanism:** Four BaseDistiller subclasses group by capability, cap entries at 200, deduplicate and emit a new packet without sender_output. taught_value prefers a provided reference, falling back to sender output. Distillation here is largely structured packaging, not a neural training algorithm.

**Default / enablement:** Active after relevance and safety on the normal packet path.

**Invoke / knobs:** PluginRegistry.register_distiller(...); BaseDistiller.max_entries; TextDistiller, TTSDistiller, CodeDistiller, StructuredDistiller.build_payload.

**Evidence kind:** Source trace; existing distillation tests; historical real glossary/G2P/structured packet reports.

**Limits:** Absence of raw sender_output field is not a semantic guarantee that teacher text is transformed: taught_value may copy the same output into distilled_skill. CodeDistiller sets verified_by_tests from truthiness of test_command, without executing it. TTS payload explicitly excludes voice timbre and acoustic prosody.

Legacy code similarity is not the later function-IO oracle.

**Pinned source / tests / recorded reports:** [src/asea/distill/strategies.py:33-123](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/distill/strategies.py#L33-L123) · [src/asea/distill/strategies.py:129-253](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/distill/strategies.py#L129-L253) · [tests/test_distillation_and_evaluation.py:155-161](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_distillation_and_evaluation.py#L155-L161) · [docs/SILT_TTS_G2P_TEST.md:302-328](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SILT_TTS_G2P_TEST.md#L302-L328)

<a id="l04"></a>
### L04 — Safety tripwires

**Mechanism:** SafetyFilter.apply scores extracted packets; patterns detect some credentials, PII, injection markers, self-harm and medical shapes. Threshold and categorical blocking lead to explicit rejection. Gate 1 checks safety score; Gate 2 inherits the minimum available source-packet score.

**Default / enablement:** Active in Pipeline.run; default min_safety_score .7, block_pii=True, max_hallucination_risk .5.

**Invoke / knobs:** Pipeline(safety=SafetyFilter(SafetyPolicy(...))); PromotionPolicy.min_safety_score; no general safety model installed by default.

**Evidence kind:** Static source, existing adversarial injection tests and safety tests; no new safety evaluation.

**Limits:** A rule tripwire, not clinical validation or a general classifier. SafetyFilter.score reads prompt and sender_output before distillation; it is not a comprehensive post-distillation or trained-output safety audit. Gate 2 inherits score rather than proving newly trained behavior safe.

**Pinned source / tests / recorded reports:** [src/asea/filters/safety.py:1-11](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/filters/safety.py#L1-L11) · [src/asea/filters/safety.py:86-95](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/filters/safety.py#L86-L95) · [src/asea/filters/safety.py:114-170](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/filters/safety.py#L114-L170) · [src/asea/core/pipeline.py:250-268](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/pipeline.py#L250-L268) · [src/asea/deepapply/dataset.py:103-123](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/dataset.py#L103-L123) · [src/asea/deepapply/gate2.py:151-162](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/gate2.py#L151-L162) · [tests/test_adversarial.py:72-105](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_adversarial.py#L72-L105)

<a id="l05"></a>
### L05 — Held-out A/B and regression/control checks

**Mechanism:** Evaluator.evaluate fully scores baseline heldout, candidate heldout, and available regression splits. It records aggregate gain, per-case regression counts across all paired cases, regression >.02 and absolute control movement >.05. Pipeline evaluates one candidate packet at a time.

**Default / enablement:** Active in normal pipeline; default per-case regression ratio ceiling 1.0 (does not prevent individual losses). Missing regression splits are skipped.

**Invoke / knobs:** Evaluator(harness=..., regression_tolerance=.02, max_control_movement=.05); PromotionPolicy(max_case_regression_ratio=...); supplied BenchmarkSuites define coverage.

**Evidence kind:** Existing split/regression tests; historical reports: lexical NLLB→Qwen rejected .5901→.5795; embedding follow-up .5943→.6475; cross-family SmolLM rejected; historical G2P .6023→.7359 on six illustrative heldout cases.

**Limits:** Benchmarks do not establish teacher pretraining non-exposure. “Heldout” means not used for this extraction workflow, not secret/unseen to every model. A single-packet test is not automatically a test of all previously approved packets combined. Legacy CodeMetric is textual, not functional execution.

**Pinned source / tests / recorded reports:** [src/asea/evaluator/evaluator.py:96-286](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/evaluator/evaluator.py#L96-L286) · [src/asea/promotion/gate.py:31-49](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/promotion/gate.py#L31-L49) · [src/asea/promotion/gate.py:195-208](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/promotion/gate.py#L195-L208) · [tests/test_distillation_and_evaluation.py:262-368](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_distillation_and_evaluation.py#L262-L368) · [docs/real_run_findings.md:136-184](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/real_run_findings.md#L136-L184) · [docs/real_run_findings.md:201-255](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/real_run_findings.md#L201-L255) · [docs/SILT_TTS_G2P_TEST.md:302-328](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SILT_TTS_G2P_TEST.md#L302-L328)

<a id="l06"></a>
### L06 — Double Gate / trainer-independent admission

**Mechanism:** PromotionGate.decide and DeepApplyGate.decide take all emitted checks with all(c.passed). No backend branch in Gate 2. Intake requires PROMOTED packets; runner trains, evaluates adapter, snapshots and decides. Gate2 adds finite-loss and artifact-size sanity checks.

**Default / enablement:** Gate 1 active on packet run; Gate 2 only in separately invoked deep-apply. Standard policy emits 15 checks low-risk / 16 high-risk (conditional mock/rollback checks can be omitted).

**Invoke / knobs:** PromotionPolicy and DeepApplyPolicy; DeepApplyConfig.build_policy passes threshold overrides, strict_no_mock and min_trainable_params. DeepApplyRunner.run; gates expose apply/decide/enforce.

**Evidence kind:** Source trace and existing backend-invariance/intake tests. Historical real deep-apply report is a real optimizer + Gate 2 rejection, not a naturally completed real Gate1→Gate2 quality win: it seeds a PROMOTED packet.

**Limits:** Trainer-independent means outcome criteria are not relaxed by backend selection, not distrust of every metadata field or hostile plugins. Thresholds are configurable; hard is descriptive metadata, not a separate enforcement engine. A missing bypass parameter does not make permissive policy impossible. Standard backend has no parity requirement.

**Pinned source / tests / recorded reports:** [src/asea/promotion/gate.py:117-365](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/promotion/gate.py#L117-L365) · [src/asea/deepapply/gate2.py:89-344](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/gate2.py#L89-L344) · [src/asea/deepapply/runner.py:105-189](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/runner.py#L105-L189) · [src/asea/deepapply/runner.py:326-420](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/runner.py#L326-L420) · [tests/test_streamed_backend.py:489-509](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_streamed_backend.py#L489-L509) · [tests/test_deep_apply.py:251-267](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_deep_apply.py#L251-L267) · [tests/test_deep_apply.py:334-482](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_deep_apply.py#L334-L482) · [docs/deep_apply_real_run_findings.md:31-39](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/deep_apply_real_run_findings.md#L31-L39) · [docs/deep_apply_real_run_findings.md:118-144](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/deep_apply_real_run_findings.md#L118-L144)

<a id="l07"></a>
### L07 — Mandatory high-risk named approval

**Mechanism:** Domain medical/legal/finance maps to HIGH; Gate1 and Gate2 append a non-configurable approval check using bool(human_approver). If every other check passes and approval does not, status is PENDING_HUMAN; any other failure gives REJECTED. An approver supplied during initial run can permit immediate promotion. approve_pending reruns gate predicates on stored evidence, not fresh inference.

**Default / enablement:** Active for correctly declared high-risk domains in normal gates; not mandatory for low-risk transfers. No separate authenticated-human proof.

**Invoke / knobs:** asea run ... --approver `<name>`; asea approve --workspace `<dir>` --packet `<id>` --approver `<name>`; Pipeline.approve_pending(packet_id, approver); POST /api/transfers/{job_id}/approve with packet_id/approver; corresponding DeepApplyRunner.approve_pending.

**Evidence kind:** Existing tests prove high-risk/no-policy-disable and approval-does-not-waive-other-checks. Adversarial tests explicitly document domain-mislabel bypass. Code trace shows Studio approver is a string (min length 3), not identity authentication.

**Limits:** No human-presence, credential, professional-license or authentic review verification. Local Studio legacy endpoint has no auth dependency; CORS is not authentication. Trusted domain/suite author required. PENDING is conditional, not “regardless of scores”; rejection takes priority.

**Pinned source / tests / recorded reports:** [src/asea/promotion/gate.py:343-363](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/promotion/gate.py#L343-L363) · [src/asea/deepapply/gate2.py:291-314](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/gate2.py#L291-L314) · [src/asea/core/pipeline.py:164-170](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/pipeline.py#L164-L170) · [src/asea/core/pipeline.py:371-394](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/pipeline.py#L371-L394) · [src/asea/cli.py:304-318](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/cli.py#L304-L318) · [src/asea/studio/server.py:134-145](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/server.py#L134-L145) · [src/asea/studio/server.py:162-178](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/server.py#L162-L178) · [src/asea/studio/server.py:706-712](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/server.py#L706-L712) · [tests/test_promotion_gate.py:173-213](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_promotion_gate.py#L173-L213) · [tests/test_adversarial.py:186-213](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_adversarial.py#L186-L213)

## Optional adaptation and diagnostics

<a id="l08"></a>
### L08 — Optional packet-derived deep-apply LoRA

**Mechanism:** StandardTrainerBackend uses HF CausalLM/PEFT, AdamW and backward/step, saves adapter_model. Runner binds provenance, dataset hash, scores, rollback and adapter stores. Packet-derived adapters remain separate from base weights.

**Default / enablement:** Optional [deep] dependencies and explicit API/Studio action; backend standard by default, rank8/alpha16, lr1e-4, 16 steps capped64, seed0, CPU ceiling1.5B.

**Invoke / knobs:** pip install -e ".[deep]"; DeepApplyRunner.run(receiver, packet_ids, DeepApplyConfig(...), target_suite, regression_suites=...); deep_apply/from_pipeline; Studio Deep-apply tab. No deep-apply subcommand in legacy asea CLI parser.

**Evidence kind:** Source and existing admission/rollback test doubles; the checked-in historical CPU SmolLM2-135M standard-backend report records three real optimizer steps, 230,400 trainable parameters, loss 7.4639573, evaluator .27, gain 0 and REJECTED. The PROMOTED intake was seeded: this is optimizer/refusal mechanism evidence, not a natural Gate 1→2 quality win.

**Limits:** No universal quality gains. Source-intake mock rejection is default-configurable for standard path (strict_no_mock=False accepted by existing test); streamed/zeroforge add unconditional clean-dataset check. Several config knobs are backend-specific: standard ignores load_in_4bit/compute_device. “Never merged” valid for packet-derived deepapply, not all SILT: specialist recover has native_merged branch.

**Pinned source / tests / recorded reports:** [src/asea/deepapply/trainer.py:310-449](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/trainer.py#L310-L449) · [src/asea/deepapply/runner.py:105-189](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/runner.py#L105-L189) · [src/asea/deepapply/runner.py:295-676](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/runner.py#L295-L676) · [src/asea/deepapply/runner.py:684-724](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/runner.py#L684-L724) · [src/asea/deepapply/dataset.py:66-125](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/dataset.py#L66-L125) · [src/asea/cli.py:292-393](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/cli.py#L292-L393) · [src/asea/specialist/recovery.py:646](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L646) · [src/asea/specialist/recovery.py:663-664](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L663-L664) · [src/asea/specialist/recovery.py:1081-1108](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L1081-L1108) · [tests/test_deep_apply.py:264-267](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_deep_apply.py#L264-L267) · [tests/test_deep_apply.py:988-1028](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_deep_apply.py#L988-L1028) · [docs/deep_apply_real_run_findings.md:9-19](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/deep_apply_real_run_findings.md#L9-L19) · [docs/deep_apply_real_run_findings.md:60-86](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/deep_apply_real_run_findings.md#L60-L86) · [docs/deep_apply_real_run_findings.md:118-144](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/deep_apply_real_run_findings.md#L118-L144)

<a id="l09"></a>
### L09 — SiltStream layer-streamed LoRA

**Mechanism:** Registered streamed backend is SiltStreamBackend, not retired CUDA-only StreamedTrainerBackend. Uses HFDiskBank/HFStreamer, PEFT, pre-training forward-parity check, checkpointed backprop and saved adapter. Failure blocks before Gate2. CPU for small models and CUDA paths exist.

**Default / enablement:** Optional explicit backend="streamed"; disk bank default. CPU ceiling 1.5B estimate; requested unavailable CUDA and CPU 4bit fail. Real path always constructs HFDiskBank despite storage_tier label knob.

**Invoke / knobs:** DeepApplyConfig(backend="streamed", storage_tier="disk", compute_device="cpu"|"cuda", load_in_4bit=False|True); SiltStreamBackend.run_parity_check / train. CUDA 4-bit loading additionally requires a compatible `bitsandbytes` installation; neither `[deep]` nor `[localmodels]` installs it.

**Evidence kind:** Current source plus random/toy forward/loss/gradient parity tests; an opt-in real SmolLM2-135M test exists. No matching retained Qwen7B/RTX5050 streamed execution receipt is included here. The older CPU standard report describes retired streamed limitations, not the current backend.

**Limits:** HF train checks one sample pre-training forward logits, not bitwise identity of learned weights/gradients/training trajectory. Full resident model is loaded and resident parity computed before streaming, so all-hardware/any-large-model claims unsupported. Hardware savings and GPU results need exact retained artifacts. Soup attribution remains in NOTICE.

**Pinned source / tests / recorded reports:** [src/asea/deepapply/backends/streamed.py:304-441](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/backends/streamed.py#L304-L441) · [src/asea/deepapply/backends/streamed.py:570-871](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/backends/streamed.py#L570-L871) · [src/asea/deepapply/trainer.py:665-692](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/trainer.py#L665-L692) · [tests/test_streamed_backend.py:288-296](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_streamed_backend.py#L288-L296) · [tests/test_streamed_backend.py:399-461](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_streamed_backend.py#L399-L461) · [tests/test_streamed_backend.py:658-732](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_streamed_backend.py#L658-L732) · [docs/deep_apply_real_run_findings.md:15-16](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/deep_apply_real_run_findings.md#L15-L16) · [docs/deep_apply_real_run_findings.md:131-139](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/deep_apply_real_run_findings.md#L131-L139) · [NOTICE:6-27](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/NOTICE#L6-L27)

<a id="l10"></a>
### L10 — ZeroForge forward-only optimization

**Mechanism:** ZeroForgeBackend injects LoRA wrappers, disk-banks HF CausalLM layers, uses shared sampled forward parity, then train_zeroforge under torch.no_grad. Each direction takes plus/minus perturbation losses; parameters are restored and updated. Report records backward_passes=0.

**Default / enablement:** Optional explicit backend="zeroforge"; not default. Auto-selects CUDA if available, otherwise CPU with size ceiling. CUDA-only optional 4bit; multiple forwards per update.

**Invoke / knobs:** DeepApplyConfig(backend="zeroforge", learning_rate=..., max_steps=...); low-level ZeroForgeBackend.train config exposes zeroforge_eps=.001 and zeroforge_n_directions=4 (not forwarded fields in DeepApplyConfig.to_train_dict). Optional CUDA 4-bit loading additionally requires compatible `bitsandbytes`; it is not installed by `[deep]` or `[localmodels]`.

**Evidence kind:** [Backend registration test](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_streamed_backend.py#L269-L284) covers discovery, not optimization quality. Source confirms forward-only updates and zero backward passes. Opt-in GPU/CPU scripts are scaffolding; a retained Qwen7B/RTX5050 ZeroForge execution receipt is not included in the repository. No general performance or quality gain follows.

**Limits:** No GPU required is not unique to this mode: standard small-model CPU backprop exists. “Where backprop physically cannot go (no GPU)” is false. Forward-only still needs RAM, resident loading/parity and many forward passes. Stochastic benefit and performance need model/task evidence. compute_device is not consumed in ZeroForge train.

**Pinned source / tests / recorded reports:** [src/asea/deepapply/backends/zeroforge.py:170-421](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/backends/zeroforge.py#L170-L421) · [src/asea/deepapply/backends/siltstream_vendor/zeroforge.py:75-146](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/backends/siltstream_vendor/zeroforge.py#L75-L146) · [src/asea/deepapply/runner.py:149-167](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/runner.py#L149-L167) · [src/asea/deepapply/trainer.py:334-418](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/trainer.py#L334-L418)

<a id="l11"></a>
### L11 — SiltSpring per-state/per-skill compression

**Mechanism:** Toy SpringModel/CompressionCertifier implement certificates, budget choice, serving refusal and LoRA staleness. Real HF certify_hf_states measures resident reference and quantized bank states; Studio executes this reporting function, not toy state-selection/deployment wrapper. Quantized layer storage expands per layer during computation.

**Default / enablement:** Optional explicit certify/API/Studio action, not automatic hardware adaptation. Library tolerance=.02; Studio SpringRequest and SpringJob tolerance=.05. Default real job loads full model fp32 CPU / fp16 CUDA.

**Invoke / knobs:** CompressionCertifier.certify(suites,tolerance), choose_state(budget_bytes,required_skills), serve(state,skill); certify_hf_states(...); POST /api/spring {module_id,suite_ids,levels:[int8,int4,int2],tolerance,device,max_len}. Legacy asea CLI has no spring subcommand.

**Evidence kind:** Toy/random-model certification and staleness tests plus opt-in real-HF probe scripts. Raw Qwen0.5B all-certified / Qwen1.5B int2-revoked receipts are not included in this repository; their outcomes are not asserted here.

**Limits:** CRITICAL: suites_from_benchmark iterates suite.cases without filtering heldout, and Studio passes full suites; cannot claim enforced heldout-only certification. Score is next-token loss on tokenized prompt text, not task-output correctness. CompressionCertifier.admit_skill is explicitly simulated parameter perturbation. Real HF results do not carry LoRA fingerprint, serve() enforcement or automatic store integration. Budget uses packed decoder-layer bytes, not total process/peak/device memory. Full resident loading/reference precedes streaming.

**Current real HF workflow:** full resident load/reference → quantized decoder-layer bank → streamed next-token prompt-loss report. `suites_from_benchmark` consumes all cases and Studio supplies full suites; it does **not** enforce heldout-only certification. Toy choose/serve/fingerprint/logging semantics must not be inferred for the real report. See [hardware boundary](#hardware-and-receipt-boundary).

**Pinned source / tests / recorded reports:** [src/asea/spring/certifier.py:46-79](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/spring/certifier.py#L46-L79) · [src/asea/spring/certifier.py:119-305](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/spring/certifier.py#L119-L305) · [src/asea/deepapply/backends/siltstream_vendor/spring.py:177-267](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/backends/siltstream_vendor/spring.py#L177-L267) · [src/asea/deepapply/backends/siltstream_vendor/hf_real.py:312-374](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/backends/siltstream_vendor/hf_real.py#L312-L374) · [src/asea/studio/spring_jobs.py:198-315](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/spring_jobs.py#L198-L315) · [src/asea/studio/server.py:260-271](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/server.py#L260-L271) · [tests/test_siltspring_certification.py:85-93](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_siltspring_certification.py#L85-L93) · [tests/test_siltspring_certification.py:101-273](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_siltspring_certification.py#L101-L273) · [scripts/real_siltspring_1p5b.py:28-81](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/scripts/real_siltspring_1p5b.py#L28-L81)

<a id="l12"></a>
### L12 — Asymmetric SPRT early rejection

**Mechanism:** SPRT.update maintains log-likelihood ratio; should_stop only on REJECT. Evaluator fully scores baseline then invokes stop callback on candidate. Any non-None scores.sprt fails Gate1 no_statistical_early_reject, including malformed records.

**Default / enablement:** Implemented but OFF by default: Evaluator.sprt=None; no default CLI/config wiring enabling it.

**Invoke / knobs:** Pipeline(evaluator=Evaluator(harness=..., sprt=SprtConfig(p0=.5,p1=.1,alpha=.05,beta=.05))); tests construct this API explicitly.

**Evidence kind:** Static implementation and existing early-reject/no-early-promote/end-to-end/malformed-record tests.

**Limits:** 95% is nominal configured sequential-test error framing under Bernoulli/i.i.d. hypotheses, not a universal posterior confidence that capability failed. Case correlation/order and hypothesis misspecification limit interpretation. Passing full suite still needs all ordinary gate checks.

**Pinned source / tests / recorded reports:** [src/asea/sprt.py:70-196](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/sprt.py#L70-L196) · [src/asea/evaluator/evaluator.py:96-127](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/evaluator/evaluator.py#L96-L127) · [src/asea/evaluator/evaluator.py:145-178](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/evaluator/evaluator.py#L145-L178) · [src/asea/evaluator/evaluator.py:223-231](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/evaluator/evaluator.py#L223-L231) · [src/asea/promotion/gate.py:227-267](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/promotion/gate.py#L227-L267) · [tests/test_sprt.py:77-115](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_sprt.py#L77-L115) · [tests/test_distillation_and_evaluation.py:396-487](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_distillation_and_evaluation.py#L396-L487) · [tests/test_distillation_and_evaluation.py:599-649](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_distillation_and_evaluation.py#L599-L649) · [tests/test_distillation_and_evaluation.py:692-782](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_distillation_and_evaluation.py#L692-L782)

<a id="l13"></a>
### L13 — Signed Capability Diff

**Mechanism:** CapabilityDiffer.diff runs heldout A/B on each supplied suite, reports deltas and independent improved/regressed/moved flags, content-hash added/removed packets, harness fingerprint and HMAC-SHA256 report. Missing-heldout suites are omitted.

**Default / enablement:** Implemented optional diagnostic; not run automatically on transfer. Local key diff.key is generated on signing, not verification.

**Invoke / knobs:** asea diff --config C --workspace W --token-a A --token-b B [--out R]; asea diff-verify --workspace W --report R; CapabilityDiffer.diff/verify.

**Evidence kind:** Existing mock/harness snapshot and tamper/key tests; static HMAC implementation. No real-model diff outcome independently established here.

**Limits:** Holder-only attestation: anyone with key can sign; verification does not establish factual correctness, third-party authorship or legal proof. Only supplied suites/receiver/runtime and managed skill snapshots measured; model weights are not differenced.

**Pinned source / tests / recorded reports:** [src/asea/capability_diff.py:189-372](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/capability_diff.py#L189-L372) · [src/asea/_signing.py:52-169](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/_signing.py#L52-L169) · [src/asea/cli.py:349-368](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/cli.py#L349-L368) · [tests/test_capability_diff.py:175-294](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_capability_diff.py#L175-L294) · [tests/test_capability_diff.py:346-455](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_capability_diff.py#L346-L455) · [tests/test_capability_diff.py:528-544](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_capability_diff.py#L528-L544)

<a id="l14"></a>
### L14 — Verified skill-layer unlearning

**Mechanism:** UnlearningVerifier.verify compares before/after content hashes and three heldout runs. verified requires some content removed AND abs(post-baseline)<=tolerance; substantive additionally requires prior lift>tolerance. Certificate locally signed under separate unlearn.key. It verifies supplied snapshots; does not perform rollback itself.

**Default / enablement:** Implemented optional diagnostic, not automatic removal or background forgetting.

**Invoke / knobs:** First explicit rollback/snapshot operations; asea unlearn --config C --workspace W --suite S --token-before A --token-after B [--out R]; asea unlearn-verify --workspace W --report R; UnlearningVerifier.verify.

**Evidence kind:** Existing mock-based substantive/trivial/no-removal/two-sided-regression and HMAC tests; source review.

**Limits:** Not weights/LoRA erasure, data deletion from snapshots/audit, causal guarantee that removed packet was sole source, or forgetting inside stateful connectors. verified can be trivial; substantive is the meaningful lift condition. HMAC validity is separate from behavioral verified flag.

**Pinned source / tests / recorded reports:** [src/asea/unlearning.py:161-326](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/unlearning.py#L161-L326) · [src/asea/cli.py:370-391](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/cli.py#L370-L391) · [tests/test_unlearning.py:140-190](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_unlearning.py#L140-L190) · [tests/test_unlearning.py:231-292](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_unlearning.py#L231-L292) · [tests/test_unlearning.py:314-391](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_unlearning.py#L314-L391) · [tests/test_unlearning.py:438-561](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_unlearning.py#L438-L561)

## Provenance, state, plugins and exports

<a id="l15"></a>
### L15 — Provenance, synthetic-depth and mock containment

**Mechanism:** Extractor constructs provenance; distiller merges chains, maximum depth and any-mock taint; Gate1/2 consume these declarations. Registry checks consistency of module and manifest mock flags. Standard policy strict_no_mock=True and max_synthetic_depth=2.

**Default / enablement:** Active checks in normal path. Declarations come from trusted modules/suite authors; strict_no_mock is configurable. Demo A-D explicitly disable strict mock check.

**Invoke / knobs:** PromotionPolicy(strict_no_mock=True,max_synthetic_depth=2); config promotion_policy; Provenance fields and registered ModuleAdapter.manifest; module is_mock and probe meta.human_verified.

**Evidence kind:** Static propagation and existing adversarial tests explicitly pin consistent liar and human_verified falsification as accepted risks.

**Limits:** Not independent origin verification or empirical prevention of model collapse. Trusted-author declarations can be false consistently. No-self-lineage prevents declared loops, not all semantic feedback loops or autonomous learning.

**Pinned source / tests / recorded reports:** [src/asea/distill/strategies.py:63-88](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/distill/strategies.py#L63-L88) · [src/asea/promotion/gate.py:270-300](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/promotion/gate.py#L270-L300) · [src/asea/promotion/gate.py:330-341](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/promotion/gate.py#L330-L341) · [src/asea/config.py:64-72](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/config.py#L64-L72) · [src/asea/config.py:112-146](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/config.py#L112-L146) · [tests/test_registry_and_handshake.py:58-68](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_registry_and_handshake.py#L58-L68) · [tests/test_adversarial.py:374-482](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_adversarial.py#L374-L482)

<a id="l16"></a>
### L16 — Audit chain, storage isolation and rollback

**Mechanism:** Pipeline creates AuditLog and MemoryStore; snapshot before gate. AuditLog appends SHA256 chained JSON with flush/fsync and per-instance threading lock. RollbackLayer stages and validates all approved packet files and restores whole approved directory with recovery guards. Adapter runner has separate adapter snapshots/rollback.

**Default / enablement:** Active within pipeline; rollback explicit. Audit verification is available, not a cryptographically external anchor on every write. Storage tests cover atomicity/path/transaction mechanics.

**Invoke / knobs:** asea audit --workspace W; asea rollback --workspace W --token T; Pipeline.rollback_to; RollbackLayer.snapshot/rollback; DeepApplyRunner.rollback_adapter.

**Evidence kind:** Static source; existing hash-tamper/storage/rollback tests and historical reports with intact chain. No fresh crash or concurrency test.

**Limits:** “Per-skill rollback token” identifies snapshot timing, but packet rollback restores the WHOLE approved set, potentially undoing later unrelated skills. Unkeyed local chain detects internal edits unless rewritten; trailing truncation/rewrite cannot be ruled out without external head anchor. Lock is per AuditLog instance, not process-wide or multi-process. Review approval path logs human_decision, not every low-level transition; evidence not prevention.

Rollback is a whole approved-set restoration and can undo later unrelated skills; unlearning verification itself performs no deletion. Audit HMAC reporting is not the same as the unkeyed audit chain.

**Pinned source / tests / recorded reports:** [src/asea/audit/logger.py:28-136](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/audit/logger.py#L28-L136) · [src/asea/core/pipeline.py:316-350](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/pipeline.py#L316-L350) · [src/asea/core/pipeline.py:371-399](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/pipeline.py#L371-L399) · [src/asea/memory/store.py:193-318](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/memory/store.py#L193-L318) · [src/asea/memory/store.py:379-473](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/memory/store.py#L379-L473) · [src/asea/deepapply/runner.py:651-676](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/runner.py#L651-L676) · [tests/test_memory_audit_rollback.py:32-87](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_memory_audit_rollback.py#L32-L87) · [tests/test_memory_audit_rollback.py:100-213](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_memory_audit_rollback.py#L100-L213) · [tests/test_storage_hardening.py:121](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_storage_hardening.py#L121)

<a id="l17"></a>
### L17 — Plugin model/modality seams and connectors

**Mechanism:** PluginRegistry defaults text/TTS/code/structured extractors/distillers, text/TTS/code metrics (fallback for missing metric). Pipeline resolves by declared modality. Handshake requires some shared modality and negotiates learning level. Real connectors include HF causal, HF seq2seq, Ollama and embedding, while generic no-preset config builds MockModule.

**Default / enablement:** Bundled plugins active; real connectors require configured model/dependencies/service. New modalities and real acoustic bridges require explicit plugins/connectors, not automatic conversion.

**Invoke / knobs:** PluginRegistry.register_extractor/register_distiller/register_metric; CONNECTOR_FACTORIES registry; build_module preset+args; asea modalities; implement ModuleAdapter.infer_with_skills; Handshake.open.

**Evidence kind:** Existing conformance tests use deterministic/mock modality pairs and an added OCR plugin. Historical real G2P and text/structured runs are narrow outcomes. Real acoustic/composition runtime is separate experimental system, not evidence legacy L3 can transfer audio skills.

**Limits:** No universal arbitrary model/modality bridge. ASR→TTS lacks shared modality and is refused. TTS legacy plugin transfers grapheme-to-phoneme strings, not waveform, timbre or prosody. Recognized architecture names/paths indicate discoverability, not validated compatibility across families.

**Pinned source / tests / recorded reports:** [src/asea/core/plugins.py:18-96](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/plugins.py#L18-L96) · [src/asea/core/handshake.py:73-129](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/core/handshake.py#L73-L129) · [src/asea/config.py:34-72](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/config.py#L34-L72) · [src/asea/config.py:112-146](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/config.py#L112-L146) · [tests/test_conformance.py:48-67](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_conformance.py#L48-L67) · [tests/test_conformance.py:152-245](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_conformance.py#L152-L245) · [src/asea/distill/strategies.py:159-192](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/distill/strategies.py#L159-L192) · [docs/SILT_TTS_G2P_TEST.md:317-328](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SILT_TTS_G2P_TEST.md#L317-L328) · [architecture.md:42-55](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/architecture.md#L42-L55)

<a id="l18"></a>
### L18 — L4/L5 dataset and job-spec export

**Mechanism:** export_artifact_bundle/export_dataset select promoted data, reject mock by default, write JSONL/manifests and optional L4/L5 job spec. build_job_spec marks NOT_EXECUTED; policy-derived eval_gate when supplied, explicitly illustrative fallback otherwise. Standard Gate1 admits only applicable L0-L3; separate deepapply handles L4.

**Default / enablement:** Optional explicit export; no training on export and L5 remains export-only in legacy packet pipeline.

**Invoke / knobs:** asea export --workspace W [--name N] [--base-model MODEL] [--include-mock] [--no-bundle]; build_job_spec(...,level=L4_PEFT_CANDIDATE|L5_DISTILL_DATASET,policy=...).

**Evidence kind:** Static export code; existing promoted-only/mock-refusal/bundle tests. These prove packaging, not a trained adapter or trained full model.

**Limits:** Bundle is not “the trained model”; it needs original receiver/runtime unless a separately documented source-derived serving bundle. Job spec is a recipe, not executed optimization or quality admission. “L4/L5 through any code path” must be limited to standard PromotionGate, not SILT as a whole.

**Pinned source / tests / recorded reports:** [src/asea/distill/export.py:158-193](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/distill/export.py#L158-L193) · [src/asea/distill/export.py:292-433](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/distill/export.py#L292-L433) · [src/asea/distill/export.py:483-592](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/distill/export.py#L483-L592) · [src/asea/cli.py:330-347](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/cli.py#L330-L347) · [src/asea/promotion/gate.py:315-328](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/promotion/gate.py#L315-L328) · [src/asea/deepapply/gate2.py:263-275](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/deepapply/gate2.py#L263-L275) · [tests/test_export.py:31-65](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_export.py#L31-L65) · [tests/test_export.py:159-191](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_export.py#L159-L191)

## Experimental entry points

<a id="c02"></a>
### C02 — Experimental Studio mount, security and CLI job bridge

**Default / enablement:** flag-gated; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** Before fresh server import/start set SILT_ENABLE_EXPERIMENTAL=1 exactly. Use loopback /experimental and X-SILT-Experimental-Token. Choose an external SILT_EXPERIMENTAL_ROOT.

**Mechanism:** Server conditionally imports/mounts router; router independently checks strict flag, local hostname and same origin. Lazy root/worker, fixed command allowlist, bounded queue/logs, named confirmation for deployment changes.

**Limits:** Unset, 0 and true do not enable it. Public landing origin is rejected by experimental routes even though legacy CORS may allow it. Merge/restart without exact flag cannot enable it. Tokens/operator names are local workflow controls, not independent human authentication.

**Evidence kind:** Fresh-process GET-only setup checks observed unset/0/true → 404, exact 1 → workbench 200; API without token → 403 and with token → schema 200; public Host/Origin → 403. Legacy OpenAPI remained identical, ML libraries were not imported, and no active pointer was created. These are setup boundaries, not model functionality.

**Pinned source:** [src/asea/studio/server.py:147–154](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/server.py#L147-L154) · [src/asea/studio/experimental.py:47–51](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/experimental.py#L47-L51) · [src/asea/studio/experimental.py:89–104](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/experimental.py#L89-L104) · [src/asea/studio/experimental.py:1063–1097](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/experimental.py#L1063-L1097)

**Test definitions (not new executions):** [tests/test_integration_contracts.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_integration_contracts.py) · [tests/test_experimental_studio.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_experimental_studio.py) · [tests/test_specialist_studio.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_studio.py)

**Guides / recorded evidence:** [docs/EXPERIMENTAL_STUDIO.md:13–54](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_STUDIO.md#L13-L54) · [docs/SPECIALIST_STUDIO.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SPECIALIST_STUDIO.md)

<a id="c03"></a>
### C03 — Public CLI namespaces, extras and config contract

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** Installed scripts: silt-compose/compile/artifacts/validate/execution/specialist; equivalent python -m asea.<namespace>. Core asea is separate. CLI does not require Studio flag.

**Mechanism:** Lightweight imports/help and explicit local path interfaces. core dependency Pydantic; studio extra FastAPI/Uvicorn; localmodels/deep are separately installed optional ML runtime sets. compose_text_example.json is schema-ready; compiler_switch_example.json is descriptive, not an executable --config.

**Limits:** No auto-install or auto-download from Compose/Compiler/Specialist. silt-artifacts download is a distinct explicit network operation, not a hidden loader fallback. Examples do not establish that model paths exist. localmodels alone does not include PEFT required for factor recovery/serving.

**Evidence kind:** The read-only review exercised all six CLI --help entry points successfully without operational jobs or model loads. Inspected tests cover command/contracts, not proof that a user has installed local model dependencies.

Acquisition validates a pinned 40-hex revision, expected metadata hashes and a bounded inert-file inventory; download is a distinct explicit network operation. No silent pickle/remote-code or model-download fallback is supplied by Compose/compiler/specialist. Asset integrity is not model runnable-ness, trusted authorship or a license grant. See [acquisition source](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/artifacts/__main__.py) and [tests](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_model_acquisition.py).

**Pinned source:** [pyproject.toml:23–78](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/pyproject.toml#L23-L78) · [src/asea/compose/__main__.py:19–89](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/__main__.py#L19-L89) · [src/asea/specialist/__main__.py:31–85](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/__main__.py#L31-L85) · [src/asea/artifacts/__main__.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/artifacts/__main__.py) · [configs/compose_text_example.json](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/configs/compose_text_example.json) · [configs/compiler_switch_example.json:1–35](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/configs/compiler_switch_example.json#L1-L35)

**Test definitions (not new executions):** [tests/test_composition_v1.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_composition_v1.py) · [tests/test_model_acquisition.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_model_acquisition.py) · [tests/test_specialist_workflow.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_workflow.py)

**Guides / recorded evidence:** [docs/COMPOSITION_V1.md:45–67](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/COMPOSITION_V1.md#L45-L67)

## Composition and deployment

<a id="c04"></a>
### C04 — Typed linear local text composition

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** python -m asea.compose --workspace OUT run --spec SPEC --input TEXT; provision local regular safetensors/tokenizer assets.

**Mechanism:** Closed 1–16-node single-input chain, causal or seq2seq text adapters, greedy generation, no silent context truncation, sequential model release, explicit dtype/cache/output-format contract.

**Limits:** Not an arbitrary DAG with joins, remote plugins, latent transfer or automatic graph synthesis. Float32 is default; later schema permits explicit float16/bfloat16 for supported adapters. Fixture backend is Python-test opt-in and cannot production-admit.

**Evidence kind:** Recorded earlier public-command Qwen BF16 six-task diagnostic passes are narrow mechanics evidence; current 16-task source scored 12, not universal coding ability.

**Pinned source:** [src/asea/compose/schema.py:19–49](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/schema.py#L19-L49) · [src/asea/compose/schema.py:67–159](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/schema.py#L67-L159) · [src/asea/compose/runtime.py:241–326](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/runtime.py#L241-L326)

**Test definitions (not new executions):** [tests/test_composition_v1.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_composition_v1.py) · [tests/test_generation_observability.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_generation_observability.py)

**Guides / recorded evidence:** [docs/COMPOSITION_V1.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/COMPOSITION_V1.md) · [docs/GENERATION_OBSERVABILITY.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/GENERATION_OBSERVABILITY.md)

<a id="c05"></a>
### C05 — Whisper ASR and speech-to-text composition bridge

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** hf_asr/whisper node; explicit PCM WAV input; installed local model; CPU float32.

**Mechanism:** WhisperProcessor plus WhisperForConditionalGeneration, bounded <=30-second input, optional SciPy resampling, text output can feed text/TTS chain.

**Limits:** No arbitrary codecs, streaming conversation, Indian-English/accent robustness or general speech certificate. WER is not accuracy.

**Evidence kind:** Earlier synthetic-utterance bridge executed. Later 60 real FLEURS en_us cases: 408 edits/1432 reference words =28.49% raw corpus WER; 57 met threshold, 3 failed, overall rejected. Tests are not those model measurements.

**Pinned source:** [src/asea/compose/schema.py:110–142](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/schema.py#L110-L142) · [src/asea/compose/runtime.py:131–155](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/runtime.py#L131-L155) · [src/asea/compose/runtime.py:367–380](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/runtime.py#L367-L380)

**Test definitions (not new executions):** [tests/test_composition_v1.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_composition_v1.py) · [tests/test_speech_pilot.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_speech_pilot.py) · [tests/test_generation_observability.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_generation_observability.py)

**Guides / recorded evidence:** [docs/HARDENING_PASS2.md:48–54](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/HARDENING_PASS2.md#L48-L54) · [docs/EXPERIMENTAL_HANDOFF.md:16–41](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_HANDOFF.md#L16-L41)

<a id="c06"></a>
### C06 — Vits/MMS text-to-waveform and audio-output bridge

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** hf_tts/vits node; CPU float32; local tokenizer/model; text <=1024 characters/512 tokens.

**Mechanism:** Vits forward emits bounded finite PCM WAV; text and ASR intermediates can feed it.

**Limits:** Audio-output composition quality admission is explicitly blocked. Waveform validity is not intelligibility, pronunciation, MOS or human consent. Recorded MMS English weights are CC-BY-NC-4.0, not commercially authorized.

**Evidence kind:** Recorded MMS TTS -> Whisper -> SmolLM2 -> MMS TTS chain on one generated test utterance, ~7.89s and ~1144MiB in that run.

**Pinned source:** [src/asea/compose/runtime.py:381–416](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/runtime.py#L381-L416) · [src/asea/certification/__init__.py:587–589](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/__init__.py#L587-L589)

**Test definitions (not new executions):** [tests/test_generation_observability.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_generation_observability.py) · [tests/test_validation_governance.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_validation_governance.py)

**Guides / recorded evidence:** [docs/EXPERIMENTAL_HANDOFF.md:21–39](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_HANDOFF.md#L21-L39) · [docs/HARDENING_PASS2.md:48–54](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/HARDENING_PASS2.md#L48-L54)

<a id="c07"></a>
### C07 — Matched vision-language inference and text/TTS bridge

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** hf_vision/smolvlm, bounded PNG/JPEG/WebP plus optional question; local complete Idefics3 model+processor.

**Mechanism:** Idefics3Processor and Idefics3ForConditionalGeneration; image->text ports can feed text/TTS. Image bytes and prompt bound to input digest; Transformers 4.51.3 required.

**Limits:** Not a standalone vision encoder automatically attached to an arbitrary LLM; no general screenshot/OCR/grounding guarantee. Latent cross-modal bridge remains absent. The later schema/runtime, not stale initial v1 prose, is authoritative.

**Evidence kind:** Actual SmolVLM-256M answered Red on one synthetic rectangle; later 12 correlated SILT UI screenshots are exploratory, not 12 independent products or broad vision certification.

**Pinned source:** [src/asea/compose/schema.py:19–49](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/schema.py#L19-L49) · [src/asea/compose/schema.py:110–142](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/schema.py#L110-L142) · [src/asea/compose/runtime.py:327–366](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/runtime.py#L327-L366)

**Test definitions (not new executions):** [tests/test_vision_runtime.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_vision_runtime.py) · [tests/test_integration_contracts.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_integration_contracts.py)

**Guides / recorded evidence:** [docs/VISION_LOCAL.md:1–33](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/VISION_LOCAL.md#L1-L33) · [docs/EXPERIMENTAL_HANDOFF.md:21–25](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_HANDOFF.md#L21-L25) · [docs/HARDENING_PASS2.md:56–58](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/HARDENING_PASS2.md#L56-L58)

<a id="c08"></a>
### C08 — Composition evaluation/admission is distinct from runs

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** compose evaluate --spec SPEC --suite SUITE; explicit target/control reference suite, metrics, thresholds, current model/runtime/input hashes and measured resource bounds.

**Mechanism:** Only every-case passing real nonfixture text-output evaluation becomes admitted. Full suite preflight, model/input rechecks, policy/implementation/worker evidence binding.

**Limits:** A successful run, exit 0, tensor shape, fixture pass or waveform output is not admission. Scope is exercised metrics, not universal quality or external attestation.

**Evidence kind:** Gate is implemented; observed model pilot rejections remain rejections.

**Pinned source:** [src/asea/certification/__init__.py:544–674](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/__init__.py#L544-L674)

**Test definitions (not new executions):** [tests/test_composition_v1.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_composition_v1.py) · [tests/test_certification_hardening.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_certification_hardening.py) · [tests/test_observability_integration.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_observability_integration.py)

**Guides / recorded evidence:** [docs/COMPOSITION_V1.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/COMPOSITION_V1.md) · [docs/FUNCTION_ORACLE.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/FUNCTION_ORACLE.md)

<a id="c09"></a>
### C09 — Manual experimental activation and rollback

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** compose activate --evaluation ADMITTED_ID [--approver NAME]; Studio additionally requires named operator and exact ACTIVATE/ROLLBACK confirmation.

**Mechanism:** Revalidate admitted records, hashes/provenance; atomically select experimental active.json pointer. Rollback only to known revalidated deployment. High-risk requires a nonempty named approver.

**Limits:** No autoactivation after run/evaluation/import/selection/build/merge. Pointer is not a model server and does not change legacy active state. Exact source nuance: approve_high_risk boolean is accepted but the backend enforces approver, not both; Studio sends both only when approval checkbox is checked.

**Evidence kind:** Source and existing negative tests inspected; no activate/rollback request sent. Fresh GET proof left all active pointers absent.

**Pinned source:** [src/asea/certification/__init__.py:880–916](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/__init__.py#L880-L916) · [src/asea/compose/__main__.py:40–43](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/__main__.py#L40-L43) · [src/asea/studio/experimental.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/experimental.py)

**Test definitions (not new executions):** [tests/test_certification_hardening.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_certification_hardening.py) · [tests/test_experimental_studio.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_experimental_studio.py)

**Guides / recorded evidence:** [docs/EXPERIMENTAL_STUDIO.md:128–139](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_STUDIO.md#L128-L139)

<a id="c10"></a>
### C10 — Schema planning and measured bundle selection

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** compose plan --spec SPEC; compose select with supplied evaluation IDs, suite hash and measured budget limits.

**Mechanism:** Plan returns draft/admitted=false/activated=false. Select ranks supplied, suite-matching admitted bundles by measured RSS then latency with optional disk/graph constraints.

**Limits:** No automatic model discovery, optimal architecture search, quality inference from I/O type, or activation. Missing suite hash returns no_feasible_bundle. CLI plan constructs workspace despite schema-only plan function; preview is explicitly workspace-free.

**Evidence kind:** Implemented deterministic local selection, not demonstrated arbitrary-B model fitting.

**Pinned source:** [src/asea/certification/__init__.py:919–977](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/__init__.py#L919-L977) · [src/asea/compose/__main__.py:66–89](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/__main__.py#L66-L89)

**Test definitions (not new executions):** [tests/test_composition_v1.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_composition_v1.py) · [tests/test_integration_contracts.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_integration_contracts.py)

**Guides / recorded evidence:** [docs/EXPERIMENTAL_STUDIO.md:92–119](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_STUDIO.md#L92-L119)

<a id="c11"></a>
### C11 — Dependency export and relocated bundle import

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** compose export explicit deployment; --include-dependencies for model bytes; python -m asea.artifacts import-bundle with new output/spec paths.

**Mechanism:** Safe streamed ZIP export; strict bounded import validates dependency inventories and rebinds model paths. Imported spec explicitly UNADMITTED, evidence_valid=false, admitted=false, activated=false.

**Limits:** Default dependency-excluding export is not self-contained. Prior machine/workspace admission cannot travel as trusted local admission. Same-host relocation evidence is not clean-machine/Windows certification. No automatic import/reevaluation/activation.

**Evidence kind:** Existing dependency-inclusive Qwen diagnostic bundle was exported/imported and run while old source location unavailable on same host/runtime. Initial COMPOSITION_V1 claim import is absent is stale.

**Pinned source:** [src/asea/certification/__init__.py:980–983](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/__init__.py#L980-L983) · [src/asea/artifacts/bundle.py:242–306](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/artifacts/bundle.py#L242-L306) · [src/asea/artifacts/__main__.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/artifacts/__main__.py)

**Test definitions (not new executions):** [tests/test_bundle_import.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_bundle_import.py) · [tests/test_export.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_export.py)

**Guides / recorded evidence:** [docs/BUNDLE_IMPORT.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/BUNDLE_IMPORT.md) · [docs/EXPERIMENTAL_HANDOFF.md:24–25](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_HANDOFF.md#L24-L25)

## Structural compiler

<a id="c12"></a>
### C12 — Native Switch expert structural compiler

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** compiler prune --model LOCAL_SWITCH --output NEW --keep-experts N --calibration FILE; inspect/evaluate/infer separate.

**Mechanism:** Actual Switch ModuleDict pruning plus smaller router rows and native config update, safe standalone safetensors export, calibration routing/dispatch measurements; source remains unchanged.

**Limits:** Switch seq2seq only, not a generic MoE/GLM/frontier compiler or evidence of coding skill. Routing usage is not causal expert importance. Basic commands remain UNADMITTED. --repair-training fails unsupported.

**Evidence kind:** Real 619339008-param Switch ->392809728 keep4; four diagnostic literal cases source/candidate both 0/4, insufficient suite certification rejected. No retention claim from equal zero.

**Pinned source:** [src/asea/compiler/core.py:403–465](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compiler/core.py#L403-L465) · [src/asea/compiler/core.py:471–548](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compiler/core.py#L471-L548) · [src/asea/compiler/__main__.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compiler/__main__.py)

**Test definitions (not new executions):** [tests/test_compiler_v1.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_compiler_v1.py)

**Guides / recorded evidence:** [docs/COMPILER_V1.md:1–25](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/COMPILER_V1.md#L1-L25) · [docs/EXPERIMENTAL_HANDOFF.md:19–20](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_HANDOFF.md#L19-L20)

<a id="c13"></a>
### C13 — Separate narrow compiler certificate gate

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** compiler certify reruns its own evaluator on explicit distinct source/candidate/suite and writes new certificate output.

**Mechanism:** Source lineage/precision, target/control statistical floors, paired-loss checks, structural size reduction, actual resources and fresh inventory checks. Successful status CERTIFIED/ADMITTED_SEQ2SEQ_TEXT_ONLY possible under policy.

**Limits:** Not imported evidence JSON, coding certificate, automatic promotion or activation. BF16/research lineage cannot silently enter v1 certification. No successful real compiled-model quality certificate established by inspected release evidence.

**Evidence kind:** Unit tests verify both acceptance and refusal mechanics; they are not pretrained model-quality evidence.

**Pinned source:** [src/asea/compiler/core.py:872–942](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compiler/core.py#L872-L942)

**Test definitions (not new executions):** [tests/test_compiler_admission.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_compiler_admission.py) · [tests/test_compiler_diagnostics.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_compiler_diagnostics.py)

**Guides / recorded evidence:** [docs/COMPILER_V1.md:18–25](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/COMPILER_V1.md#L18-L25)

<a id="c14"></a>
### C14 — Compiler diagnostics, identity roundtrip and research REAP

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** compiler diagnose/roundtrip; explicit research precision/router/scorer options.

**Mechanism:** Fresh-worker native-vs-wrapper generation/token/logit diagnostics, full-expert identity export/reload, EOS/sentinel spans, numerical observations; probability-mass or dispatched-token REAP selection with coverage guard.

**Limits:** AUDIT_ONLY/UNADMITTED. Identity serialization is not pruning; matched logits not quality. Missing expert coverage is not zero importance. Research BF16/router preservation excluded from v1 certification.

**Evidence kind:** Recorded four-debug controls matched wrapper/native and identity reload; keep7 preserved four outputs but source/candidate only 2/4 spans; REAP refused undercovered expert.

**Pinned source:** [src/asea/compiler/diagnostics.py:632–739](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compiler/diagnostics.py#L632-L739) · [src/asea/compiler/core.py:403–465](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compiler/core.py#L403-L465)

**Test definitions (not new executions):** [tests/test_compiler_diagnostics.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_compiler_diagnostics.py)

**Guides / recorded evidence:** [docs/HARDENING_PASS2.md:23–36](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/HARDENING_PASS2.md#L23-L36) · [docs/COMPILER_DIAGNOSTICS.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/COMPILER_DIAGNOSTICS.md)

## Source-derived specialist

<a id="c15"></a>
### C15 — Source-derived coherent Qwen2 MLP reconstruction

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** specialist reconstruct --source-dir LOCAL --output-dir NEW --calibration-path TRAIN; family qwen2, method activation/magnitude/uniform, default retention .75.

**Mechanism:** Physically slices same selected channels across gate/up rows and down columns in native Qwen2; retains learned attention/backbone. Activation score mean absolute native down-input times output-column norm. Width rounded to multiple of64.

**Limits:** Not packet transfer, a mask, arbitrary architecture reconstruction, or demonstrated retained quality. .75 means MLP channel retention, not 75% total parameters. Native Qwen2/SwiGLU guard; output RECONSTRUCTED_UNVALIDATED.

**Evidence kind:** Real Qwen2.5-Coder-0.5B source494032768 ->415586176 reconstructed base; not tiny random-fixture parameter arithmetic.

**Pinned source:** [src/asea/specialist/reconstruction.py:199–231](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/reconstruction.py#L199-L231) · [src/asea/specialist/reconstruction.py:649–701](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/reconstruction.py#L649-L701) · [src/asea/specialist/__main__.py:34–40](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/__main__.py#L34-L40)

**Test definitions (not new executions):** [tests/test_specialist_reconstruction.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_reconstruction.py)

**Guides / recorded evidence:** [docs/SPECIALIST_RECONSTRUCTION.md:63–89](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SPECIALIST_RECONSTRUCTION.md#L63-L89) · [docs/EXPERIMENTAL_RELEASE_2026_09.md:17–30](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_RELEASE_2026_09.md#L17-L30)

<a id="c16"></a>
### C16 — Source-derived Switch-to-dense T5 reconstruction

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** specialist reconstruct native Switch source; fixed top1 expert per sparse layer; explicit calibration/method and local output.

**Mechanism:** Constructs genuine native dense T5, preserving shared dense/backbone structure and selecting one source expert per sparse FFN; not a Switch wrapper.

**Limits:** Not compiler keep-N transformation; retention option does not control expert count here. No coding-quality conclusion from synthetic spans; not arbitrary T5/Llama source reconstruction.

**Evidence kind:** Recorded real Switch619339008 ->denseT5 base222903552 +factors1622016=224525568 after32 real recovery steps; construction evidence only.

**Pinned source:** [src/asea/specialist/reconstruction.py:702–785](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/reconstruction.py#L702-L785) · [src/asea/specialist/reconstruction.py:806–810](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/reconstruction.py#L806-L810)

**Test definitions (not new executions):** [tests/test_specialist_reconstruction.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_reconstruction.py) · [tests/test_specialist_recovery.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_recovery.py)

**Guides / recorded evidence:** [docs/SPECIALIST_RECONSTRUCTION.md:91–99](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SPECIALIST_RECONSTRUCTION.md#L91-L99) · [docs/EXPERIMENTAL_RELEASE_2026_09.md:32–32](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_RELEASE_2026_09.md#L32)

<a id="c17"></a>
### C17 — Real teacher-guided LoRA recovery, separate from compiler repair

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** specialist recover with explicit teacher/student/train/dev paths; lora_kd default, supervised_lora alternative; cached teacher default, resident explicit.

**Mechanism:** Actual optimization with native LoRA; full-vocabulary response-token teacher KL and supervised CE; frozen base and teacher, trained factor hashes; cached teacher freed before student training.

**Limits:** Not core packet deep-apply, and does not make compiler --repair-training supported. Lower token loss or nonzero optimizer steps is not functional recovery or quality certification.

**Evidence kind:** Qwen64 steps;42 training/8 dev tasks;1,413,120 factors; CE .91756688->.70981678; KL .13315734->.13044521 over411 dev response tokens. Final aggregate remained8/16.

**Pinned source:** [src/asea/specialist/recovery.py:53–65](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L53-L65) · [src/asea/specialist/recovery.py:503–538](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L503-L538) · [src/asea/specialist/recovery.py:643–652](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L643-L652) · [src/asea/specialist/__main__.py:41–59](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/__main__.py#L41-L59)

**Test definitions (not new executions):** [tests/test_specialist_recovery.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_recovery.py)

**Guides / recorded evidence:** [docs/SPECIALIST_RECOVERY.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SPECIALIST_RECOVERY.md) · [docs/EXPERIMENTAL_RELEASE_2026_09.md:17–32](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_RELEASE_2026_09.md#L17-L32)

<a id="c18"></a>
### C18 — Two explicit recovery export representations and strict loaders

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** Recovery CLI default --export-mode native_merged; --export-mode factor_preserving must be chosen explicitly. Serving factor bundle via registered load_standalone / specialist infer.

**Mechanism:** native_merged checks changed native matrices plus dtype-bounded merge/reload parity, then saves native HF safetensors. factor_preserving saves complete base+adapter+tokenizer, exact native/factor metadata keys/shapes/dtypes/values and matching runtime; exact full dev-forward/two-token generation parity.

**Limits:** No silent fallback, relaxed tolerances, teacher/cache dependency at serve, arbitrary plugin loader, GGUF export or factor-bundle HF-root equivalence. Loading base/ alone drops learned factors. artifact_admitted means engineering export, not model-quality admission.

**Evidence kind:** Real BF16 native merge failed and is preserved; recorded recovered artifact explicitly factor-preserving. Qwen416999296 unique total parameters and853192780 deployment bytes; standalone smoke parity is bounded, not global equivalence.

**Pinned source:** [src/asea/specialist/recovery.py:992–1079](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L992-L1079) · [src/asea/specialist/recovery.py:1080–1159](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/recovery.py#L1080-L1159) · [src/asea/specialist/standalone.py:24–33](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/standalone.py#L24-L33) · [src/asea/specialist/standalone.py:137–284](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/standalone.py#L137-L284)

**Test definitions (not new executions):** [tests/test_specialist_recovery.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_recovery.py) · [tests/test_specialist_workflow.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_workflow.py)

**Guides / recorded evidence:** [docs/EXPERIMENTAL_RELEASE_2026_09.md:34–42](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_RELEASE_2026_09.md#L34-L42)

<a id="c19"></a>
### C19 — Five-stage specialist build and once-only final workflow

**Default / enablement:** experimental; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** specialist build --recipe FILE --workspace NEW; separately finalize --study --suite --output, with frozen implementation/models/data and consumed marker.

**Mechanism:** Source-validation -> reconstruction -> reconstructed-validation -> recovery -> recovered-validation, records/freeze; final consumption durably precedes final suite hash/load. Backend finalize evaluates three core arms; recorded six-arm release comparison has separate controls.

**Limits:** BUILT_UNCERTIFIED/engineering_complete/candidate_frozen are not certificate or quality success. Certificate remains false. Consumed final must never be retried, retuned or reused as new evidence.

**Evidence kind:** Recorded V5 Studio five-stage completion BUILT_UNCERTIFIED, engineering_complete=true, candidate_frozen=true, certificate=false, quality_pass=false. Audit did not call build/finalize.

**Pinned source:** [src/asea/specialist/workflow.py:583–619](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/workflow.py#L583-L619) · [src/asea/specialist/workflow.py:623–733](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/workflow.py#L623-L733) · [src/asea/specialist/workflow.py:742–805](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/workflow.py#L742-L805)

**Test definitions (not new executions):** [tests/test_specialist_workflow.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_workflow.py) · [tests/test_specialist_studio.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_studio.py)

**Guides / recorded evidence:** [docs/SPECIALIST_WORKFLOW.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SPECIALIST_WORKFLOW.md) · [docs/EXPERIMENTAL_RELEASE_2026_09.md:9–13](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_RELEASE_2026_09.md#L9-L13)

<a id="c20"></a>
### C20 — Native Llama-family evaluation-only baseline support

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** specialist infer/evaluate --model strict dense native Llama checkpoint; explicit local generation settings.

**Mechanism:** Evaluation-only LlamaForCausalLM config, dimensions, local asset, native meta key/shape/tied alias and resource checks; keeps reconstruct/recover family guards separate.

**Limits:** Not Llama reconstruction/recovery or arbitrary-family support. Baseline SmolLM2 is in the Llama native evaluation family; cannot claim every Llama-size model fits available hardware.

**Evidence kind:** Tiny random Llama fixtures test loader mechanics; real off-the-shelf SmolLM2-360M-Instruct final arm scored8/16. These are separate evidence classes.

**Pinned source:** [src/asea/specialist/evaluation.py:72–145](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/evaluation.py#L72-L145) · [src/asea/specialist/workflow.py:646–648](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/workflow.py#L646-L648) · [src/asea/specialist/__main__.py:37–37](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/__main__.py#L37)

**Test definitions (not new executions):** [tests/test_specialist_workflow.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_workflow.py)

**Guides / recorded evidence:** [docs/SPECIALIST_WORKFLOW.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/SPECIALIST_WORKFLOW.md) · [docs/EXPERIMENTAL_RELEASE_2026_09.md:50–63](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_RELEASE_2026_09.md#L50-L63)

## Evaluation, observability and governance

<a id="c21"></a>
### C21 — External function-IO oracle and bounded observability

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** Explicit function_io suite; supported Linux candidate sandbox; trace-policy none/digest/value; composition or specialist validation invocation.

**Mechanism:** Trusted host holds expected outputs and compares bounded data-only returns outside candidate interpreter/chroot. Generation/output contract, EOS/completion, preprocessing and oracle outcomes are distinct recorded facts.

**Limits:** Retained functional_code mode shares unittest interpreter and is weaker, not automatically converted to external-oracle evidence. No universal algorithm identity/safety proof; digest traces do not contain unavailable raw values. A model-free validate still executes candidate code and was NOT run here.

**Evidence kind:** Historical16/16 Boolean grades means all16 tasks graded, NOT16 passes. Published source12/16/recovered8/16. Recovered14 host oracle invocations reflects preprocessing failures already scored, not missing tasks.

**Named observability controls:** `compose preview --spec SPEC --output-format raw_python|fenced_python` proposes an output contract without writing it or constructing a Compose workspace. Generation observability separates requested/forwarded/resolved settings, token caps and EOS/completion from preprocessing/executability and oracle grades. Compose trace retention defaults to `none`; specialist `evaluate`/`validate` default to `digest`, and the staged specialist workflow explicitly uses `digest`. `digest` is not anonymization; `value` retains potentially sensitive/wrong returns. Studio requires separate creation/storage consent, viewer reveal (`X-SILT-Reveal-Sensitive: true`) and sensitive-export confirmation; none implies another. Preview may create outer Studio job metadata. The displayed evidence projection is not a portable signed original or authenticity validation.

**Additional source/tests:** [generation runtime](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/runtime.py) · [oracle](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/function_oracle.py) · [Studio projection/consent](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/studio/experimental.py) · [generation tests](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_generation_observability.py) · [oracle tests](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_oracle_observability.py) · [Studio tests](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_observability_studio.py). **Guides:** [generation](GENERATION_OBSERVABILITY.md), [oracle](ORACLE_OBSERVABILITY.md), [workflow](OBSERVABILITY_WORKFLOW.md), [Studio privacy](OBSERVABILITY_STUDIO.md).

**Pinned source:** [src/asea/certification/function_oracle.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/function_oracle.py) · [src/asea/certification/sandbox.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/sandbox.py) · [src/asea/certification/__init__.py:544–638](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/certification/__init__.py#L544-L638) · [src/asea/specialist/evaluation.py:515–607](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/evaluation.py#L515-L607)

**Test definitions (not new executions):** [tests/test_function_oracle.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_function_oracle.py) · [tests/test_code_sandbox.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_code_sandbox.py) · [tests/test_oracle_observability.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_oracle_observability.py)

**Guides / recorded evidence:** [docs/FUNCTION_ORACLE.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/FUNCTION_ORACLE.md) · [docs/HARDENING_PASS2.md:9–13](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/HARDENING_PASS2.md#L9-L13)

<a id="c22"></a>
### C22 — Read-only resource probe and explicit Linux worker controls

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** execution probe is read-only; execution run --profile process_as supervises fixed SILT module only. Compose evaluate defaults observe_only, protected profile explicit.

**Mechanism:** Kernel RLIMIT_AS/CPU before trusted model imports, supervised deadlines, bounded output, parent-death/process-group handling and post-bootstrap RSS observations. Probe honestly separates capability from enforcement.

**Limits:** RLIMIT_AS is per-process virtual address space, not RSS/aggregate-memory/cgroup enforcement. cgroup_v2 and Windows execution backends unavailable/fail closed. No GPU validation or arbitrary-B-on4GB guarantee; estimates cannot rule out OOM.

**Evidence kind:** Recorded model target Linux x86_64 CPU/Python3.9, roughly4.18GiB and2 CPUs, no NVIDIA GPU. Largest exercised source619M; size ceilings are refusal guards, not full model/device support declarations.

**Pinned source:** [src/asea/execution/controls.py:31–81](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/execution/controls.py#L31-L81) · [src/asea/execution/controls.py:106–154](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/execution/controls.py#L106-L154) · [src/asea/compose/__main__.py:33–39](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/__main__.py#L33-L39) · [src/asea/specialist/workflow.py:618–619](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/workflow.py#L618-L619)

**Test definitions (not new executions):** [tests/test_execution_limits.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_execution_limits.py) · [tests/test_pass2_integration.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_pass2_integration.py)

**Guides / recorded evidence:** [docs/RESOURCE_CONTROLS.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/RESOURCE_CONTROLS.md) · [docs/HARDENING_PASS2.md:15–21](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/HARDENING_PASS2.md#L15-L21) · [docs/EXPERIMENTAL_HANDOFF.md:41–41](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_HANDOFF.md#L41)

<a id="c23"></a>
### C23 — Validation registry and governed evaluation

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** validation register/freeze/begin/finish; governed-evaluate requires explicit development/final purpose, new compose workspace and process_as controls.

**Mechanism:** Separate disjoint registry, suite/candidate/policy binding, reservation consumption before generation, append-only history, wrapper observes public Compose gate. finish accepts unverified result input, never issues certificate.

**Limits:** Owner-readable files are not sealed from same OS owner; local HMAC/hashes are not independent custodian or third-party attestation. Failed final cannot be renamed/reused as untouched data.

**Evidence kind:** Earlier adapted public coding pilot9/24 final, rejected, distinct from later specialist16-task final. Not official HumanEval/MBPP or contamination-free population score.

**Pinned source:** [src/asea/validation/__main__.py:11–58](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/validation/__main__.py#L11-L58) · [src/asea/validation/governed.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/validation/governed.py) · [src/asea/validation/__init__.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/validation/__init__.py)

**Test definitions (not new executions):** [tests/test_governed_evaluation.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_governed_evaluation.py) · [tests/test_validation_governance.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_validation_governance.py)

**Guides / recorded evidence:** [docs/GOVERNED_EVALUATION.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/GOVERNED_EVALUATION.md) · [docs/HARDENING_PASS2.md:38–46](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/HARDENING_PASS2.md#L38-L46)

<a id="c24"></a>
### C24 — Waveform evidence and pending-human listening packets

**Default / enablement:** explicit invocation; separate opt-in, not enabled or run by installing/merging the source.

**Invoke / knobs:** validation voice-check / export-listen-batch / check-review; reads explicit local files; optional exclusive evidence output.

**Mechanism:** Measures PCM structure/duration/levels/clipping/silence and licensing declarations. Produces blank listening fields; completeness checker can validate schema but never authenticates human ratings.

**Limits:** status=pending_human; quality_admission=false; audio_output_admission=false. Authenticated listening collection/approval and MOS are unavailable. Declared commercial purpose does not grant rights.

**Evidence kind:** Implemented evidence preparation, not implemented perceptual certification; no fake listener entries or agent approvals created.

**Pinned source:** [src/asea/validation/voice.py:31–90](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/validation/voice.py#L31-L90) · [src/asea/validation/voice.py:93–160](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/validation/voice.py#L93-L160) · [src/asea/validation/__main__.py:42–58](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/validation/__main__.py#L42-L58)

**Test definitions (not new executions):** [tests/test_validation_governance.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_validation_governance.py) · [tests/test_pass2_studio.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_pass2_studio.py)

**Guides / recorded evidence:** [docs/VALIDATION_GOVERNANCE.md](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/VALIDATION_GOVERNANCE.md) · [docs/HARDENING_PASS2.md:54–54](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/HARDENING_PASS2.md#L54)

## Remaining gaps

<a id="c25"></a>
### C25 — Universal learned cross-modal transfer/frontier-small-device claims

**Default / enablement:** unverified; no delivered broad mechanism.

**Invoke / knobs:** No supported invocation exists for these broad outcomes; narrower existing mechanisms above remain explicit/experimental.

**Mechanism:** Typed media bridges, matched vision-language model and two source-derived model-family transformations are implemented. Their boundaries/refusal paths are real.

**Limits:** Arbitrary latent ASR/vision/TTS-to-LLM capability transplantation, universal teacher-level retention, broad coding/safety quality, GLM/frontier arbitrary model reconstruction, native Windows/GPU certification and arbitrary billion-parameter model on4GB are not delivered or proven.

**Evidence kind:** Treat as future research requirements or absent broad features, not active functionality inferred from a diagram, model-class import, mocked test, or merge.

**Pinned source:** [src/asea/compose/schema.py:110–142](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/compose/schema.py#L110-L142) · [src/asea/specialist/reconstruction.py:199–231](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/reconstruction.py#L199-L231) · [src/asea/specialist/workflow.py:646–648](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/specialist/workflow.py#L646-L648) · [src/asea/execution/controls.py:31–81](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/src/asea/execution/controls.py#L31-L81)

**Test definitions (not new executions):** [tests/test_specialist_reconstruction.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_specialist_reconstruction.py) · [tests/test_vision_runtime.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_vision_runtime.py) · [tests/test_execution_limits.py](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/tests/test_execution_limits.py)

**Guides / recorded evidence:** [README.md:7–7](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/README.md#L7) · [docs/index.html:134–153](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/index.html#L134-L153) · [docs/EXPERIMENTAL_RELEASE_2026_09.md:95–97](https://github.com/inbharatai/SILT/blob/b583ba0d7de077d4e8594b08f0f77cc29655459e/docs/EXPERIMENTAL_RELEASE_2026_09.md#L95-L97)

## Recorded evidence and measurement units

The [frozen release report](EXPERIMENTAL_RELEASE_2026_09.md) binds its model
results to a local V5 experiment; these are not fresh executions of the catalog.

| Quantity | Recorded scope |
|---|---|
| Qwen source → reconstructed base → recovered stored total | **494,032,768 → 415,586,176 + 1,413,120 factors = 416,999,296** unique stored parameters; tied aliases counted once. MLP retention .75 is not .75 of all model parameters. |
| Qwen recovery | **64** real optimizer steps, 42 training / 8 development tasks. Development response-token CE .91756688→.70981678 and KL .13315734→.13044521 over 411 tokens; not functional accuracy. |
| Qwen source / deployment footprint | **999,604,531 / 853,192,780 bytes** on disk. These do not measure RAM, VRAM, active parameters, FLOPs, latency or energy. |
| Switch dense reconstruction/recovery | **619,339,008 → 222,903,552 base + 1,622,016 factors = 224,525,568** stored parameters; **32** real steps. Synthetic-span construction evidence only. |
| Experimental model-run environment | Recorded Linux x86_64, Python 3.9, CPU, roughly 4.18 GiB / 2 CPUs; largest exercised source about 619M. Not native Windows/GPU validation or arbitrary-B-on-4GB support. |

### All six frozen final arms — 16 tasks each

| Arm | Correct / total |
|---|---:|
| Original Qwen source | **12/16** |
| Activation reconstruction, unrepaired | **8/16** |
| Activation reconstruction, recovered | **8/16** |
| Random MLP, retained backbone, same 64-step LoRA budget | **0/16** |
| Uniform channels, unrepaired | **0/16** |
| Off-the-shelf SmolLM2-360M-Instruct | **8/16** |

All 96 generations completed and each task received a Boolean grade; **16/16
graded does not mean 16/16 correct**. Preprocessing failures remain scored
failures, not omitted tasks (14 recovered host-oracle calls is not a smaller
denominator). Recovered vs source: eight both-pass, four both-fail, four
source-only and zero recovered-only. Random MLP retains a pretrained backbone
and a limited repair budget; uniform has no recovery; SmolLM2 has different
architecture/pretraining. These controls do not isolate universal superiority.

Relative to unrepaired reconstruction, recovery produced **two wins and two
losses**, leaving 8/16 unchanged. Archived analysis associates the wins with
function-name/API binding and executability repair, not acquisition of new
algorithms; neither was a token-cap/EOS rescue. Among four source-only passes,
three involved program-semantics errors and one termination-policy failure.
See [the read-only analysis summary](EXPERIMENTAL_RELEASE_2026_09.md#what-the-later-read-only-case-analysis-adds).
No raw traces need be republished to establish these distinctions. Scores remain
unchanged; the final set is consumed. Future work needs fresh governed data,
not a rerun, task-hardcoded patch or retuning against that final set.

The selected recovered export is **factor-preserving**, requiring the registered
SILT loader, complete base/factors/tokenizer and matching runtime. The BF16
native-merge failure is retained. Bounded exact dev-forward/two-token generation
reload parity is not universal numerical equivalence. Teacher-independent
serving removes the teacher runtime requirement, not the receiver runtime or
hardware requirement. The source-code package is not a public model download.

Other reports are distinct cohorts: the earlier governed coding pilot was
9/24 and rejected; 60 real FLEURS en_us ASR cases yielded 408 edits / 1,432 words
= 28.49% raw corpus WER (57 within threshold, three failed, overall rejected).
A synthetic-utterance TTS/ASR chain and 12 correlated UI screenshots are narrow
mechanism/exploratory evidence, not broad voice/vision certification. Historical
L3 G2P is symbolic IPA text, includes a cloud-tagged GLM Ollama teacher, and does
not validate voice synthesis or local-GPU inference.

## Hardware and receipt boundary

- The [retained deep-apply report](deep_apply_real_run_findings.md) documents
  **CPU SmolLM2-135M standard training**, seeded promoted input and Gate 2
  rejection. Its old streamed limitation is historical, not current CPU support.
  [GPU probe code](../scripts/real_streamed_4bit_gpu.py) is not a Qwen7B/RTX5050
  streamed or ZeroForge execution receipt.
- [Spring probe code](../scripts/real_siltspring_1p5b.py) has no saved Qwen0.5B /
  Qwen1.5B state-loss stdout in this repository. A hand-built Studio test report
  is a fixture, not a captured run. Unsupported citation means receipts are not
  included here, **not** proof those runs never occurred.
- Current HF streamed training loads a full resident model and checks sampled
  pre-training forward logits. Current Spring first loads/evaluates the full
  resident reference; decoder banks are built afterwards. Full model residency,
  embeddings/head, host RAM, disk/restore banks, activations and runtime overhead
  all matter. There is **no current 7B-on-8GB end-to-end guarantee**.
- A recorded RTX5050 hardware inventory with CPU-only Torch/CUDA unavailable is
  hardware metadata, not a GPU execution receipt. Prepared `NOT_EXECUTED` jobs,
  schema-only training logs, declared budgets and recognized architecture paths
  cannot be upgraded to model/device execution.
- Restore stronger claims only with an exact model revision, backend/device/dtype,
  command and retained output binding steps/suites/tolerance, finite loss/parity,
  admission/certification result and a named measured memory metric. Do not
  substitute Soup's hardware numbers for SILT measurements; preserve [NOTICE](../NOTICE).

## Verification scope

- **Frozen V5 prepublication:** 2,039 passed / eight expected skips / 75 warnings,
  documented in the [release report](EXPERIMENTAL_RELEASE_2026_09.md#frozen-source-verification-what-was-actually-recorded).
- **Later publication-tree local offline check:** 2,044 passed / eight skipped /
  75 warnings, Linux CPU Python 3.9.25, torch 2.6.0+cpu, Transformers 4.51.3,
  PEFT .15.2. This is the recorded local check after five added regression cases,
  on a tree identical to public `b583ba0`; it is not a new run for this edit.
  The eight skips were six opt-in real-model tests plus pending cgroup/Windows
  enforcement templates. Tiny random native-model fixtures are not pretrained
  quality experiments. The local report's raw logs are not published here.
- **GitHub CI:** [run 34330505969](https://github.com/inbharatai/SILT/actions/runs/34330505969)
  recorded success at `b583ba0` for [Python 3.9](https://github.com/inbharatai/SILT/actions/runs/34330505969/job/102397722382),
  [3.11](https://github.com/inbharatai/SILT/actions/runs/34330505969/job/102397722476),
  [3.12](https://github.com/inbharatai/SILT/actions/runs/34330505969/job/102397722463)
  and [sanity](https://github.com/inbharatai/SILT/actions/runs/34330505969/job/102397722112).
  CI installs `.[dev,studio]`, not the local full-ML environment; per-job counts
  were not available to the anonymous review and are not invented.
- Earlier 420/421, 746 and other counts retain their own historical scopes.
  None certifies model quality, every optional dependency, every platform or
  current documentation edits. This content-only update runs no model/full tests.

## Historical documentation and genuine limits

Initial v1/Phase 6 labels in older guides are history, not authoritative absence
claims: matched vision, safe import/rebind, narrow compiler certification and
source-derived reconstruction/recovery are implemented in the pinned source.
Conversely, do not present genuinely absent work as finished: arbitrary latent
cross-modal capability transplantation, authenticated human listening/voice
approval, perceptual audio-output admission, delegated-cgroup aggregate-memory
and native Windows execution/security backends, frontier arbitrary-model
reconstruction, clean-machine universal portability, teacher-level final quality
and competitive advantage remain unimplemented or unproven as specified above.

Legal context is not execution evidence: [PATENT.md](../PATENT.md),
[LICENSE](../LICENSE) and [NOTICE](../NOTICE) are preserved. Established LoRA,
SPSA and teacher-guided methods are not claimed here as novel; no coverage of
new reconstruction/recovery by the earlier provisional is asserted. Earlier
legal or architecture wording does not override the current technical boundaries
in this catalog; legal-scope decisions belong to the owner/counsel.
