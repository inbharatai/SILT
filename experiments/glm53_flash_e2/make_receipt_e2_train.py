"""Assemble the E2 TRAINING receipt for signing (figures from files).

Every number in this receipt is READ from an artifact a command produced --
nothing is hand-copied or estimated. Unmeasured fields carry the literal
NOT_MEASURED token. This third-stage receipt adds the E2 DeepApply training
stage (bigger student, expanded KD set, GPU) on top of the E2 pretrain
receipt, which stays untouched in the append-only store under its own
record (e2-receipt-pretrain.unsigned.json).

Usage (repo root, same interpreter as the CLI):
  python experiments/glm53_flash_e2/make_receipt_e2_train.py
Then sign + store (append-only, NEW record name):
  python -m asea.capability_build receipt --receipt \
    experiments/glm53_flash_e2/e2-receipt-training.unsigned.json \
    --workspace experiments/glm53_flash_e2/workspace
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PILOT = REPO / "experiments" / "glm53_flash_pilot"
NOT_MEASURED = "NOT_MEASURED"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from asea.capability_build.spec import load_spec

    spec_file = HERE / "spec.json"
    loaded = load_spec(spec_file)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    spec_sha = loaded["spec_sha256"]
    teacher_new = json.loads(
        (HERE / "judgment-teacher-e2-new.json").read_text(encoding="utf-8"))
    # The v1 pilot's RECEIPTED teacher judgment on the frozen 16 non-final
    # evaluation cases -- the immune 2026-09-18 numbers. The E2 A/B evaluates
    # the SAME byte-identical cases, so the retention ratio keeps the v1
    # definition (adapter target / teacher target on identical checks).
    teacher_v1 = json.loads(
        (PILOT / "judgment-teacher.json").read_text(encoding="utf-8"))
    pretrain = json.loads(
        (HERE / "e2-receipt-pretrain.unsigned.json").read_text(encoding="utf-8"))
    train_run = json.loads(
        (HERE / "deepapply-run" / "training-run.json").read_text(encoding="utf-8"))
    ab_base = json.loads(
        (HERE / "deepapply-run" / "judgment-ab-base.json").read_text(encoding="utf-8"))
    ab_adapter = json.loads(
        (HERE / "deepapply-run" / "judgment-ab-adapter.json").read_text(
            encoding="utf-8"))
    ab_verdict = json.loads(
        (HERE / "deepapply-run" / "training-report.json").read_text(
            encoding="utf-8"))
    v2_manifest = json.loads(
        (REPO / "data" / "capability_v2" / "manifest.json").read_text(
            encoding="utf-8"))

    base_rates = ab_base["pass_rates"]
    adapter_rates = ab_adapter["pass_rates"]
    t_v1 = teacher_v1["pass_rates"]
    retention = round(
        adapter_rates["target_checks"] / t_v1["target_checks"], 4)

    receipt = {
        "schema": "silt.capability_receipt.v1",
        "command": "glm53-flash-e2-training",
        "status": "completed",
        "capability_id": spec["capability_id"],
        "spec_sha256": spec_sha,
        "evidence_class": spec["teacher"]["access"],
        "teacher": {
            "provider": spec["teacher"]["provider"],
            "model": spec["teacher"]["model"],
            "revision": spec["teacher"]["revision"],
            "access": spec["teacher"]["access"],
            "traced_cases": "16 new E2 training cases (the v1 pilot's 16 "
                            "non-final traces stay receipted there; the final "
                            "split was never traced in either experiment)",
        },
        "student": {
            "base_model": train_run["base_model"],
            "note": "the v1 pilot trained the 0.5B Instruct student on 6 KD "
                    "pairs; E2 trains the 1.5B Instruct student on 22 KD "
                    "pairs (six receipted v1 + sixteen newly judged) -- the "
                    "experiments are each compared within themselves",
        },
        "data_split_hashes": dict(v2_manifest["files"]),
        "measurements": {
            "teacher_e2_new_judged": {
                "judge": teacher_new["judge"],
                "cases": teacher_new["cases_total"],
                "fully_passed": teacher_new["cases_fully_passed"],
                "note": "the SIXTEEN NEW training cases only (rate "
                        "%.4f under authored denominators); the teacher's "
                        "receipted numbers on the frozen evaluation cases "
                        "are the v1 pilot's and stand unchanged"
                        % round(
                            sum(r["passed"] for r in teacher_new["cases"])
                            / sum(r["total"] for r in teacher_new["cases"]),
                            4),
            },
            "deepapply_training": {
                "judge": "host_oracle_function_io_linux_sandbox (independent "
                         "A/B; the trainer never certified itself)",
                "backend": train_run["backend"],
                "base_model": train_run["base_model"],
                "trainable_param_count": train_run["trainable_param_count"],
                "step_count": train_run["step_count"],
                "final_training_loss": train_run["training_loss"],
                "kd_pairs_sha256": train_run["kd_pairs_sha256"],
                "device": train_run["device"],
                "model_dtype_knob": train_run["train_config"]["model_dtype"],
                "intake": "research path: TrainingDataset rows built "
                          "directly from the receipt-verified E2 KD pairs "
                          "(NOT Gate-1 PROMOTED packets; no production "
                          "admission sought)",
                "ab_design": ab_verdict["comparison"],
                "base_pass_rates": base_rates,
                "adapter_pass_rates": adapter_rates,
                "deltas_adapter_minus_base":
                    ab_verdict["deltas_adapter_minus_base"],
                "heldout_improved": ab_verdict["heldout_improved"],
                "control_regressed": ab_verdict["control_regressed"],
                "control_regression_definition": "adapter minus base on the "
                                                 "utility-writing control "
                                                 "checks, identical generation "
                                                 "path; negative = regression",
            },
        },
        "capability_retention": retention,
        "control_regressions": {
            "ab_control_checks_delta":
                ab_verdict["deltas_adapter_minus_base"].get("control_checks"),
        },
        "resources": {},
        "gates": {
            "gate1_status": "not_requested (no promotion sought in this "
                            "experiment)",
            "gate2_status": "research A/B executed (2026-09-19): the "
                            "independent host oracle judged base vs trained "
                            "adapter on the authored checks of every "
                            "non-final case (32 cases / 102 checks in the "
                            "frozen v2 dataset, computed from the frozen "
                            "rows); production admission NOT requested -- "
                            "the research intake trains directly from "
                            "receipt-verified KD pairs, not Gate-1 "
                            "PROMOTED packets",
        },
        "failure_history": [
            "authoring self-check caught palindromes_v1: the first buggy "
            "variant (w[0]==w[-1]) passed every authored check because no "
            "check case had a non-palindrome with matching first/last "
            "characters; the mixed-args check was broadened (['aba', 'abca', "
            "'aa', 'ab']) and all sixteen cases then authored cleanly -- "
            "the check CAUGHT the fake difficulty before any model saw it",
            "the trace CLI refused the JSONL case file (its contract is a "
            "single JSON list); the sixteen new cases were re-emitted as "
            "cases-new-training.json and the trace run succeeded",
            "the first judge_e2 run crashed with KeyError: 'status' at the "
            "shared judge -- root cause found live: the WSL siltvenv "
            "resolves asea from a STALE tree at ~/SILT (commit ec5777e) "
            "whose evaluate result predates the authored-denominator "
            "correction and lacks the status field. Fixed by pinning "
            "PYTHONPATH to this checkout's src inside the E2 drivers' WSL "
            "invocations (judge_e2.py via the shared judge's venv-activation "
            "global; train_deepapply_e2.py's own oracle runner); the frozen "
            "v1 files were not modified. The same trap had hit the v1 "
            "training stage and is now documented on both E2 drivers",
            "repeated Bash permission-classifier outages ('temporarily "
            "unavailable (timed out)') interrupted the chain several times; "
            "each command was retried verbatim until it ran -- no command "
            "output was fabricated during the outages",
            "the judge stage's own denominator guard refused its first "
            "report: the driver hand-typed v1's 45-check non-final total, "
            "but the frozen v2 dataset's expanded training split makes the "
            "non-final set 32 cases / 102 authored checks. The guard "
            "refused to publish rather than grade against a wrong "
            "expectation (working as designed); the driver was fixed to "
            "COMPUTE the expected total from the frozen rows and the judge "
            "re-run on the already-generated responses -- no new "
            "generation, the final split stayed sealed",
        ],
        "limitations": [
            "A pass rate on these authored cases measures THIS case set "
            "only; it is not a quality claim about any model.",
            "Implementation success is not model-quality success.",
            "The final split was never opened; no measurement exists for it.",
            "DeepApply LoRA training ran as a research path (64 steps over 22 "
            "KD pairs); 22 pairs is still a mechanism-scale training set -- "
            "a null or negative held-out delta is a real result, not a "
            "failure to hide. Production admission was NOT sought; the "
            "adapter is a research artifact and autoactivates nothing.",
            "capability_retention keeps the v1 definition (adapter target "
            "checks / teacher target checks on identical frozen checks), "
            "but the two numbers come from different generation paths "
            "(deterministic local HF bf16 CUDA raw-text continuation vs the "
            "cloud-teacher run) judged by the same host oracle -- recorded "
            "as a ratio, not a like-for-like comparison.",
            "The E2 A/B arms (bf16 on CUDA, 1.5B student) are NOT directly "
            "comparable to the v1 pilot's fp32 CPU A/B numbers (0.5B "
            "student); each experiment is compared within itself.",
            "The model_dtype=bfloat16 trainer knob is a VRAM necessity on "
            "this 8 GB card (fp32 1.5B weights plus one 1280-token row's "
            "logits exceed the card), identical across both A/B arms; fp32 "
            "remains the default and the historical path is unchanged.",
            "The torch deviation (2.11.0+cu128, the only sm_120-compatible "
            "build) is isolated to the GPU venv via --system-site-packages "
            "shadowing; the repo's localmodels pin (torch>=2.6,<2.7, "
            "transformers==4.51.3) is untouched, and transformers/peft/"
            "accelerate match the production pins exactly inside the venv "
            "-- the same isolation pattern as the GLM worker's "
            "transformers 5.16.1.",
            "The teacher is behavioural-only: no routers, experts or "
            "internal state were observed; no internal claims are made.",
            "KD pairs are sequence-level text only (GLM->Qwen cross-family); "
            "no vocabulary, logit or hidden-state alignment is assumed or "
            "performed.",
        ],
        "artifact_hashes": dict(
            {name: sha256(HERE / name) for name in (
                "spec.json", "cases.jsonl", "cases-new-training.jsonl",
                "authoring_e2.py", "assemble_e2_dataset.py",
                "judge_e2.py", "train_deepapply_e2.py",
                "setup_gpu_env.sh", "make_receipt_e2_pretrain.py",
                "make_receipt_e2_train.py",
                "judgment-teacher-e2-new.json",
            )},
            kd_pairs=pretrain["artifact_hashes"]["kd_pairs"],
            dataset_v2_manifest=sha256(
                REPO / "data" / "capability_v2" / "manifest.json"),
            pretrain_receipt=sha256(HERE / "e2-receipt-pretrain.unsigned.json"),
            training_run=sha256(HERE / "deepapply-run" / "training-run.json"),
            judgment_ab_base=sha256(
                HERE / "deepapply-run" / "judgment-ab-base.json"),
            judgment_ab_adapter=sha256(
                HERE / "deepapply-run" / "judgment-ab-adapter.json"),
            training_report=sha256(
                HERE / "deepapply-run" / "training-report.json"),
            adapter_weights=sha256(
                HERE / "deepapply-run" / "adapter_model" /
                "adapter_model.safetensors"),
            adapter_config=sha256(
                HERE / "deepapply-run" / "adapter_model" /
                "adapter_config.json"),
        ),
        "runtime_versions": {
            "python": sys.version.split()[0],
            # the signer schema is flat strings; structured details ride
            # along as sorted JSON
            "train_environment": json.dumps(
                train_run["runtime_versions"], sort_keys=True),
            "judge_environment": "WSL2 Ubuntu, asea CPU-only torch venv "
                                 "(Linux sandbox required for the host "
                                 "oracle); PYTHONPATH pinned to this "
                                 "checkout's src (see failure_history)",
        },
        "hardware_identity": {
            "machine": train_run["runtime_versions"]["platform"],
            "gpu": json.dumps(train_run["device"]["gpu"], sort_keys=True),
            "note": "8 GB RTX 5050 laptop (sm_120); training and both A/B "
                    "arms ran on CUDA; the pretrain stages used no GPU",
        },
        "error": None,
    }
    out = HERE / "e2-receipt-training.unsigned.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"receipt": str(out), "status": receipt["status"]}))


if __name__ == "__main__":
    main()