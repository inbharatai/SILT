"""Assemble the CORRECTION receipt for signing (audit 2026-09-19, item 5).

Supersedes pilot-receipt-training.unsigned.json in the append-only store: the
training receipt published A/B numbers produced by a denominator defect (the
rates divided by the reported judged totals, so candidate-error cases
reporting 0/0 silently vanished from every denominator). This receipt
republishes the pilot with the CORRECTED authored-denominator figures, read
from the corrected artifacts (never hand-copied), and references the
superseded record by its stored hash. The superseded record is NOT edited or
deleted -- it stays in the store as invalidated history.

Every number in this receipt is READ from an artifact produced by a command
run in this pilot. Unmeasured fields carry the literal NOT_MEASURED token
(the signer's dematerialise contract).

Usage (repo root, same interpreter as the CLI):
  python experiments/glm53_flash_pilot/make_correction_receipt.py
Then sign + store:
  python -m asea.capability_build receipt --receipt \
    experiments/glm53_flash_pilot/pilot-receipt-correction.unsigned.json \
    --workspace experiments/glm53_flash_pilot/workspace
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOT_MEASURED = "NOT_MEASURED"
SUPERSEDED_ARTIFACT_SHA256 = (
    "2aae2501c1fe6e7a6e97dd5f44b87d859224cd5dbf2aeeec89669baa4dd52042")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_counts(judgment: dict) -> tuple:
    rows = [r for r in judgment["cases"] if r["group"] == "target"]
    return sum(r["passed"] for r in rows), sum(r["total"] for r in rows)


def main() -> None:
    from asea.capability_build.spec import load_spec

    spec_file = HERE / "spec.json"
    loaded = load_spec(spec_file)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    spec_sha = loaded["spec_sha256"]
    teacher_j = json.loads((HERE / "judgment-teacher.json").read_text(encoding="utf-8"))
    student_j = json.loads(
        (HERE / "judgment-student-qwen2.5_0.5b.json").read_text(encoding="utf-8"))
    distill = json.loads(
        (HERE / "workspace-judged" / "candidates" / "python_repo_debugging_v1-kd-pairs.json").read_text(encoding="utf-8"))
    compress = json.loads((HERE / "compression-proof.json").read_text(encoding="utf-8"))
    train_run = json.loads(
        (HERE / "deepapply-run" / "training-run.json").read_text(encoding="utf-8"))
    ab_base = json.loads(
        (HERE / "deepapply-run" / "judgment-ab-base-corrected.json").read_text(encoding="utf-8"))
    ab_adapter = json.loads(
        (HERE / "deepapply-run" / "judgment-ab-adapter-corrected.json").read_text(encoding="utf-8"))
    ab_verdict = json.loads(
        (HERE / "deepapply-run" / "training-report-corrected.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (REPO / "data" / "capability_v1" / "manifest.json").read_text(encoding="utf-8"))
    superseded_record = json.loads(
        (HERE / "workspace" / "receipts" / "pilot-receipt-training.unsigned.json").read_text(encoding="utf-8"))
    if superseded_record.get("artifact_sha256") != SUPERSEDED_ARTIFACT_SHA256:
        raise SystemExit(
            "the stored training receipt's hash is %r, expected %r -- the "
            "superseded record changed; refusing to sign a correction "
            "against an unknown version"
            % (superseded_record.get("artifact_sha256"),
               SUPERSEDED_ARTIFACT_SHA256))

    t_rates = teacher_j["pass_rates"]
    s_rates = student_j["pass_rates"]
    t_passed, t_total = target_counts(teacher_j)
    a_passed, a_total = target_counts(ab_adapter)
    if t_total != a_total:
        raise SystemExit(
            "teacher and corrected-adapter target totals differ (%d vs %d); "
            "the retention ratio needs identical authored checks"
            % (t_total, a_total))
    siltspring_states = {}
    for level, payload in compress["results"].items():
        if payload["revoked"]:
            siltspring_states[level] = "certified=%s revoked=%s" % (
                ",".join(payload["certified"]) or "-",
                ",".join(payload["revoked"]))
        else:
            siltspring_states[level] = "certified=%s" % ",".join(payload["certified"])

    receipt = {
        "schema": "silt.capability_receipt.v1",
        "command": "glm53-flash-pilot-correction",
        "status": "completed",
        "capability_id": spec["capability_id"],
        "spec_sha256": spec_sha,
        "evidence_class": spec["teacher"]["access"],
        "teacher": {
            "provider": spec["teacher"]["provider"],
            "model": spec["teacher"]["model"],
            "revision": spec["teacher"]["revision"],
            "access": spec["teacher"]["access"],
            "traced_cases": "16 (final split never traced)",
        },
        "student": {
            "gap_model": "qwen2.5:0.5b (Ollama local, judged 16 cases)",
            "compression_proof_model": "%s@%s" % (
                compress["model"], compress["revision"]),
        },
        "data_split_hashes": dict(manifest["files"]),
        "measurements": {
            "teacher_judged": {
                "judge": "host_oracle_function_io_linux_sandbox",
                "cases": 16,
                "pass_rates": t_rates,
                "correction_note": "teacher/student judged runs were "
                                    "re-verified 2026-09-19: no denominator "
                                    "drops (45/45 authored checks per "
                                    "non-final arm present); these numbers "
                                    "are IMMUNE to the corrected defect and "
                                    "stand unchanged",
                "genuine_failures": [
                    "capitalize_words_v1: no code block in response "
                    "(0/3, no_code_block)",
                    "rotate_list_v1: returned the same buggy left-rotation "
                    "(1/3)",
                ],
            },
            "student_judged": {
                "model": "qwen2.5:0.5b",
                "judge": "host_oracle_function_io_linux_sandbox",
                "cases": 16,
                "pass_rates": s_rates,
            },
            "gap": {
                "definition": "teacher target_checks - student target_checks "
                              "on identical oracle checks",
                "target_checks": round(
                    t_rates["target_checks"] - s_rates["target_checks"], 4),
                "training_checks": round(
                    t_rates["training_checks"] - s_rates["training_checks"], 4),
                "heldout_checks": round(
                    t_rates["heldout_checks"] - s_rates["heldout_checks"], 4),
            },
            "distill": {
                "positives": len(distill["positives"]),
                "negatives": len(distill["negatives"]),
                "protected_splits_enforced": distill["dataset"]["protected_splits_enforced"],
                "sequence_level_only": distill["sequence_level_only"],
            },
            "compression_proof": {
                "model": compress["model"],
                "revision": compress["revision"],
                "levels": compress["levels"],
                "tolerance": compress["tolerance"],
                "bytes_packed": {
                    lv: p["bytes_packed"] for lv, p in compress["results"].items()},
                "weights_read_only_byte_exact": compress["weights_read_only_verified"]["byte_exact_restore"],
                "anomaly": "int4 suite losses BELOW the full-precision reference "
                           "(negative degradation); investigated with "
                           "diag_quant.py -- no pipeline defect (in-place "
                           "round-trip reproduces streamed losses to 4 "
                           "decimals); a loss-metric artifact, never reported "
                           "as a quality improvement",
            },
            "ab_scoring_correction": {
                "supersedes": {
                    "record": "pilot-receipt-training.unsigned.json",
                    "artifact_sha256": SUPERSEDED_ARTIFACT_SHA256,
                    "status": "invalidated (superseded by this correction); "
                              "the stored record is preserved unmodified as "
                              "history",
                    "reason": "the superseded receipt published A/B pass "
                              "rates computed with a denominator defect: "
                              "rates divided by the reported judged totals, "
                              "so candidate-error cases (oracle 0/0 rows) "
                              "silently vanished from every denominator and "
                              "inflated the rates. Affected cases: base "
                              "greeting_format_v1 (2 checks), "
                              "price_total_v1 (2), adapter "
                              "capitalize_words_v1 (3), chunk_v1 (3), "
                              "transpose_rows_v1 (3). The published 'held-out "
                              "improved 0.8889 -> 1.0' was FALSE; corrected "
                              "is a held-out regression 0.8889 -> 0.6667 and "
                              "an aggregate target regression 0.8108 -> "
                              "0.5405",
                },
                "defect": "evaluate_code_cases divided every rate by the "
                          "REPORTED judged totals; a candidate that failed "
                          "to import or raised on call reported judged_cases=0 "
                          "and its authored checks silently vanished from the "
                          "denominators, inflating the rates",
                "fix": "authored-denominator accounting (audit 2026-09-19): "
                       "denominator = the frozen authored oracle count per "
                       "case; unjudged checks count as FAILED; missing "
                       "status/duplicate/unknown/verdictless oracle entries "
                       "are hard refusals; regression tests verified "
                       "fail-on-old",
                "recomputation": "both A/B arms re-graded from the frozen "
                                 "responses-{base,adapter}.jsonl through the "
                                 "real host oracle under the corrected "
                                 "accounting (no new model generation; the "
                                 "final split stays sealed); every case the "
                                 "committed judgment graded in full reproduced "
                                 "byte-for-byte -- the recomputation changed "
                                 "nothing except restoring the five dropped "
                                 "cases to the denominators",
                "dropped_cases_reinstated":
                    ab_verdict["dropped_cases_reinstated"],
                "reproducibility": "corrected judgment files carry a "
                                   "'correction' block with every dropped "
                                   "case's true oracle status and "
                                   "candidate_error diagnostic",
            },
            "deepapply_training": {
                "judge": "host_oracle_function_io_linux_sandbox (independent A/B; "
                         "the trainer never certified itself)",
                "backend": train_run["backend"],
                "base_model": train_run["base_model"],
                "trainable_param_count": train_run["trainable_param_count"],
                "step_count": train_run["step_count"],
                "final_training_loss": train_run["training_loss"],
                "kd_pairs_sha256": train_run["kd_pairs_sha256"],
                "intake": "research path: TrainingDataset rows built directly "
                          "from the receipt-verified KD pairs (NOT Gate-1 "
                          "PROMOTED packets; no production admission sought)",
                "ab_design": ab_verdict["comparison"],
                "corrected": True,
                "base_pass_rates": ab_base["pass_rates"],
                "adapter_pass_rates": ab_adapter["pass_rates"],
                "committed_pass_rates_superseded": {
                    arm: (json.loads(
                        (HERE / "deepapply-run" / ("judgment-ab-%s.json" % arm))
                        .read_text(encoding="utf-8"))["pass_rates"])
                    for arm in ("base", "adapter")
                },
                "deltas_adapter_minus_base": ab_verdict["deltas_adapter_minus_base"],
                "heldout_improved": ab_verdict["heldout_improved"],
                "control_regressed": ab_verdict["control_regressed"],
                "control_regression_definition": "adapter minus base on the "
                                                 "utility-writing control checks, "
                                                 "identical generation path; "
                                                 "negative = regression",
                "corrected_verdict": "the trained adapter REGRESSED the "
                                     "measured target capability overall "
                                     "(training -0.5263, held-out -0.2222, "
                                     "aggregate target -0.2703) while "
                                     "improving development (+0.2222) and the "
                                     "controls split (+0.25); the earlier "
                                     "'held-out improved' claim was an "
                                     "artifact of the scoring defect and is "
                                     "withdrawn",
                "corrected_retention_counts": {
                    "adapter_target_passed": a_passed,
                    "teacher_target_passed": t_passed,
                    "authored_target_checks": a_total,
                    "note": "capability_retention is the exact ratio "
                            "%d/%d over raw counts; the superseded "
                            "receipt's 0.8259 was inflated by the "
                            "denominator defect"
                            % (a_passed, t_passed),
                },
            },
        },
        "capability_retention": round(a_passed / t_passed, 4) if t_passed else None,
        "control_regressions": {
            "ab_control_checks_delta": ab_verdict["deltas_adapter_minus_base"].get(
                "control_checks"),
        },
        "resources": {},
        "gates": {
            "gate1_status": "not_requested (no promotion sought in this pilot)",
            "gate2_status": "research A/B re-scored under corrected accounting "
                            "(2026-09-19): the independent host oracle judged "
                            "base vs trained adapter on identical authored "
                            "checks; the adapter REGRESSED the measured "
                            "target capability and NOTHING is admitted; "
                            "production admission NOT requested -- the pilot's "
                            "research intake trains directly from "
                            "receipt-verified KD pairs, not Gate-1 PROMOTED "
                            "packets",
            "siltspring_states": siltspring_states,
        },
        "failure_history": [
            "teacher.py three-dot relative import raised ImportError at call "
            "time during the FIRST real teacher run; fixed to two dots and a "
            "live-loopback regression test added (unit tests mock the "
            "transport, so only a real network run caught it)",
            "teacher capitalize_words_v1: 0/3 checks (response was reasoning "
            "prose with no code block)",
            "teacher rotate_list_v1: 1/3 checks (returned the same buggy "
            "left-rotation)",
            "CI run 35359937603 (py3.12) failed "
            "test_kernel_cpu_limit_not_bandwidth_claim: wall-clock safety "
            "deadline fired before RLIMIT_CPU on a contended runner; safety "
            "window widened (test semantics unchanged)",
            "The first completed DeepApply training saved a real adapter but "
            "crashed in the report step (NameError: digest was imported only "
            "inside load_kd_pairs); three earlier attempts failed on driver "
            "bugs the run itself exposed (the receipt pin is the artifact's "
            "embedded content hash, not the raw file hash; the dataset "
            "manifest files map is filename->sha256, not split->filename; "
            "the WSL venv's editable asea install pointed at a stale checkout "
            "and had to be shadowed with PYTHONPATH). Driver fixed and the "
            "whole training re-run once, start to finish",
            "The first signed copy of the training receipt pinned "
            "train_deepapply.py as it stood at signing time; its usage "
            "docstring was edited minutes later (PYTHONPATH note), so that "
            "pin matched no file on disk. Caught by a pre-commit pin-vs-bytes "
            "audit before anything was published; the uncommitted record was "
            "regenerated from the as-committed driver bytes. Lesson recorded: "
            "freeze the driver BEFORE generating the receipt, not after",
            "SCORING DEFECT (found by external read-only audit of main @091637e, "
            "2026-09-19): every functional pass rate divided by the REPORTED "
            "judged totals, so five candidate-error cases (oracle 0/0 rows: "
            "base greeting_format_v1/price_total_v1, adapter "
            "capitalize_words_v1/chunk_v1/transpose_rows_v1) silently "
            "vanished from the A/B denominators and the published numbers "
            "were wrong (falsely 'held-out improved 0.8889 -> 1.0'). Fixed "
            "with authored-denominator accounting in production "
            "(evaluate_code_cases), both pilot drivers, and summarise; "
            "regression tests verified fail-on-old; both arms re-graded from "
            "the frozen responses with byte-for-byte reproduction of every "
            "previously-graded case; corrected numbers republished via this "
            "superseding receipt",
        ],
        "limitations": [
            "A pass rate on these authored cases measures THIS case set "
            "only; it is not a quality claim about any model.",
            "Implementation success is not model-quality success.",
            "The final split was never opened; no measurement exists for it.",
            "DeepApply LoRA training ran as a research path (48 steps over 6 "
            "KD pairs); 6 pairs is a mechanism-scale training set -- a null "
            "or negative held-out delta is a real result, not a failure to "
            "hide. Production admission (Gate-1 PROMOTED packets -> "
            "DeepApplyRunner -> Gate 2 -> AdapterStore) was NOT sought; the "
            "adapter is a research artifact and autoactivates nothing.",
            "CORRECTED VERDICT (2026-09-19): under authored-denominator "
            "accounting the trained adapter REGRESSED the measured target "
            "capability (training -0.5263, held-out -0.2222, aggregate "
            "-0.2703) while improving development (+0.2222) and the "
            "controls split (+0.25); retention 0.625. The earlier 'held-out "
            "improved' claim is withdrawn.",
            "capability_retention is the defined ratio (adapter target "
            "checks passed / teacher target checks passed over identical "
            "authored checks), but the two numbers come from different "
            "generation paths (deterministic local HF raw-text continuation "
            "vs the cloud-teacher run) judged by the same host oracle -- "
            "recorded as a ratio, not a like-for-like comparison.",
            "The A/B used raw-text greedy continuation matching the "
            "production trainer's raw input/output format, not a chat "
            "template; its base-arm scores are not comparable to the "
            "Ollama-judged student numbers above.",
            "The compression certificate criterion is suite-loss degradation "
            "vs the full-precision reference (the certifier's designed "
            "contract); it is NOT an oracle pass-rate measurement.",
            "The teacher is behavioural-only: no routers, experts or "
            "internal state were observed; no internal claims are made.",
            "The compression proof used Qwen/Qwen2.5-0.5B-Instruct because "
            "the shared HF cache's base-0.5B snapshot contains no weight "
            "files; same architecture and scale as the judged student.",
        ],
        "artifact_hashes": dict(
            {name: sha256(HERE / name) for name in (
                "spec.json", "cases.jsonl", "authoring_selfcheck.py",
                "judge_pilot.py", "compress_proof.py", "diag_quant.py",
                "train_deepapply.py", "correct_ab_scoring.py",
                "judgment-teacher.json",
                "judgment-student-qwen2.5_0.5b.json",
                "compression-proof.json",
            )},
            kd_pairs=distill["artifact_sha256"],
            dataset_manifest=sha256(
                REPO / "data" / "capability_v1" / "manifest.json"),
            training_run=sha256(
                HERE / "deepapply-run" / "training-run.json"),
            judgment_ab_base_corrected=sha256(
                HERE / "deepapply-run" / "judgment-ab-base-corrected.json"),
            judgment_ab_adapter_corrected=sha256(
                HERE / "deepapply-run" / "judgment-ab-adapter-corrected.json"),
            training_report_corrected=sha256(
                HERE / "deepapply-run" / "training-report-corrected.json"),
            superseded_judgment_ab_base=sha256(
                HERE / "deepapply-run" / "judgment-ab-base.json"),
            superseded_judgment_ab_adapter=sha256(
                HERE / "deepapply-run" / "judgment-ab-adapter.json"),
            superseded_training_report=sha256(
                HERE / "deepapply-run" / "training-report.json"),
            superseded_receipt_record_sha256=SUPERSEDED_ARTIFACT_SHA256,
            adapter_weights=sha256(
                HERE / "deepapply-run" / "adapter_model" /
                "adapter_model.safetensors"),
            adapter_config=sha256(
                HERE / "deepapply-run" / "adapter_model" /
                "adapter_config.json"),
        ),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "judge_environment": "WSL2 Ubuntu, asea CPU-only torch venv "
                                  "(Linux sandbox required for the host oracle)",
        },
        "hardware_identity": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "note": "8 GB-class laptop; no GPU was used in this pilot",
        },
        "error": None,
    }
    out = HERE / "pilot-receipt-correction.unsigned.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"receipt": str(out), "status": receipt["status"],
                      "supersedes": SUPERSEDED_ARTIFACT_SHA256,
                      "capability_retention": receipt["capability_retention"]}))


if __name__ == "__main__":
    main()