# SILT — Risk report

Ordered by how likely each risk is to actually cause harm, not by how dramatic it
sounds. Each entry states what the code does about it and — more importantly —
what the code does **not** do about it.

---

## 1. Hallucination laundering — HIGHEST RISK, PARTIALLY MITIGATED

**The failure.** A sender produces a confident, fluent, wrong answer. It passes
the filters. It is distilled into a packet with a provenance chain, a confidence
score and an evaluator score. It is now an *institutional artefact* that looks
more trustworthy than the raw output it came from. Downstream consumers trust the
metadata rather than the content.

**Mitigation in code.** The relevance filter requires the sender to *match a
reference*, not merely to be confident (`sender_correctness_floor`, default
0.75). Self-reported confidence is a tiebreaker only, never a gate. Distillation
prefers a human-verified reference over the sender's own output when both exist.

**What is NOT mitigated.** The heuristic in `metrics.hallucination_risk` catches
absolutes, vague appeals to authority, and output that shares few tokens with a
reference. It will **not** catch a plausible, well-hedged, vocabulary-overlapping
falsehood. That requires a real verifier or a human. Where no reference exists at
all, the sender-correctness check cannot fire and the packet rides on heuristics
alone. **Do not run reference-free extraction in any domain that matters.**

---

## 2. Model collapse from recursive synthetic data — MITIGATED STRUCTURALLY

**The failure.** Receiver learns from sender; receiver later becomes a sender;
its output feeds another learner. Each generation drifts further from real data
and variance collapses.

**Mitigation.** `Provenance.synthetic_depth` counts generations from
human-verified or curated data and the gate rejects packets above the ceiling
(default 2). `no_self_transfer` rejects a packet whose lineage contains the
receiver itself. `AdapterRegistry.bind` and `Handshake.open` both refuse
sender == receiver. Because L3 changes no weights, a bad promotion is undone by
deleting a JSON file.

**Residual.** The depth counter is only as good as the honesty of the modules
reporting it. A connector that lies about provenance defeats it.

---

## 3. Unsafe medical / legal / financial output — HARD-GATED

**The failure.** A triage rule that is subtly wrong reaches a user who acts on it.

**Mitigation.** `HIGH_RISK_DOMAINS = {medical, legal, finance}` cannot reach
`PROMOTED` without a named human approver recorded in the audit log. This is not
a configuration option: a maximally permissive policy still parks the packet in
`PENDING_HUMAN` (asserted by test). The safety filter additionally hard-blocks
dosage-shaped strings, diagnostic certainty, discouragement from seeking care,
self-harm content, credentials and PII-shaped strings; and penalises medical
guidance that contains no escalation advice.

**What is NOT mitigated.** The filter is a rule-based tripwire, not a clinical
reviewer. It will pass a responsibly-phrased but clinically wrong rule. Human
approval is doing the real work here, and a human approving without domain
expertise is a rubber stamp that adds an audit trail to a bad decision.

> **The sample medical data in `data/benchmarks/medical_triage.json` has not been
> reviewed by any clinician. It is not medical advice and must never be used for
> triage.**

---

## 4. Benchmark self-deception — MITIGATED BY DESIGN, EASY TO BREAK

**The failure.** Extraction and evaluation draw from the same distribution, so
improvement is measured on what was just memorised. Every run looks like a
success and nothing is learned.

**Mitigation.** The harness enforces three disjoint splits and raises on
cross-reads. The shipped Assamese suite extracts at word level and evaluates at
sentence level, so a packet must compose rather than recall.
`test_split_discipline_holds_in_shipped_data` asserts zero prompt overlap.

**Residual.** Nothing prevents a *user* from authoring a suite whose held-out
split duplicates its extraction split. The discipline is enforced mechanically
within a suite, not against a careless dataset author.

---

## 5. Weak language evaluation — ACKNOWLEDGED, NOT SOLVED

**The failure.** Lexical metrics on a few dozen Assamese pairs are noise dressed
as measurement. Word-order errors, register errors and dialect mismatches are
invisible to them.

**Mitigation.** `LexicalSimilarity.is_semantic` returns `False` and that flag is
carried into every evaluation report alongside an explicit caveat string. The
`language_preservation` metric does catch the most common real failure — silently
answering in the wrong script — and is Unicode-range aware for
Bengali-Assamese, Devanagari and Meitei Mayek.

**What is NOT mitigated.** Token-F1 is order-insensitive; it scores
"I rice eat" against "I eat rice" at 1.0. This is asserted in
`test_token_f1_is_order_insensitive` so it is a known property rather than a
lurking bug, and it means the demo numbers **overstate** the quality of
compositional glossary output for an SOV→SVO language pair.

**Required before any real claim:** an embedding backend (LaBSE / IndicSBERT) and
native-speaker review. The `SimilarityBackend` interface exists for exactly this.

---

## 6. Mock data escaping into real learning — MITIGATED

**The failure.** Placeholder data ends up in an approved store or a training set,
where it looks legitimate.

**Mitigation.** `is_mock` propagates from module → provenance → packet. The
default policy's `no_mock_provenance` check rejects such packets, and
`export_dataset` refuses them unless `include_mock=True` is passed explicitly.
The demo scripts disable the gate check to be able to run at all, and say so
loudly at the top of `examples/_common.py`; a test asserts the **default** still
protects.

---

## 7. Unexecuted code fixes scored as correct — ACKNOWLEDGED

`CodeMetric` compares text. It does not run tests. A "fix" that is textually
close to the reference but semantically broken scores well. Running
model-authored code in-process would be unsafe, so the sandbox runner is
deliberately absent rather than half-built. Any real coding deployment must
score on test exit status.

---

## 8. Audit tampering — TAMPER-EVIDENT, NOT TAMPER-PROOF

The hash chain detects modification or deletion and reports the first broken
index. Anyone with write access to the file can recompute the entire chain. This
is evidence, not prevention. Real deployments should ship entries to append-only
storage.

---

## 9. Retrieval doing the work instead of the knowledge — CONTROLLED

The mock receiver uses exact match, longest-condition rule triggering, fragment
splicing and token composition — all deterministic and inspectable. A fuzzy
embedding retriever would raise scores while making it impossible to tell whether
a gain came from transferred knowledge or from a lucky nearest neighbour. If you
swap in embedding retrieval, **re-measure everything**.

---

## 10. Over-claiming — THE RISK THAT DAMAGES CREDIBILITY

This system does not create AGI, does not copy models, does not train anything,
and does not make one model a replica of another. It moves inspectable knowledge
artefacts between modules under an evaluation gate. The README, the feasibility
review and the packet payloads all state their own limits, and the final report
distinguishes what runs from what is mocked.

---

## Safety disclaimers

- **Not medical, legal or financial advice.** The medical sample data is
  illustrative and clinically unreviewed.
- **Sample language data is unreviewed.** The Assamese, Hindi and phonetic data
  were written for pipeline testing and require native-speaker and phonetician
  review before any use.
- **All model behaviour in this build is mocked.** No conclusion about Qwen,
  Gemma or any AI4Bharat model can be drawn from these numbers.
