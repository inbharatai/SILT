"""GLM-5.3-Flash pilot DeepApply training stage (REAL run, 2026-09-19).

The pilot's distill stage produced 6 sequence-level KD positives distilled
from REAL judged teacher traces (glm-5.3-flash:cloud, host-oracle judged in
the Linux sandbox) plus a DeepApply hand-off descriptor. This driver is that
hand-off EXECUTED as far as this machine can honestly take it:

  train -- verify the KD artifact (hash pinned to the signed pilot receipt,
      spec fingerprint, training-split-only) and run the REAL production
      trainer (StandardTrainerBackend: peft LoRA on
      Qwen/Qwen2.5-0.5B-Instruct, CPU) with real per-step loss telemetry.
      The adapter is saved; nothing is admitted or activated.

  ab    -- generate responses for every non-final case twice through the
      SAME deterministic HF path (greedy, raw-text continuation matching
      the production trainer's raw "input\\noutput" training format): once
      for the plain base model, once with the trained adapter attached.
      The final split is NEVER generated.

  judge -- extract candidates with the same strict rule as the teacher and
      student judgments (last fenced ```python block; whole def-response;
      else no_code_block) and grade BOTH arms through the REAL host oracle
      + Linux code sandbox (``python -m asea.capability_build evaluate``).
      The trainer never certifies itself: this is an independent A/B.

Production-admission boundary (binding): the production intake
(``build_training_dataset`` over Gate-1 PROMOTED packets via
``DeepApplyRunner.from_pipeline``) is NOT what produced these rows -- these
rows come from the pilot's judged KD pairs, and the training-set manifest
records that. Gate 1 was not sought for this material (no pipeline run was
made) and Gate 2 admission was not requested. No packet was promoted, no
adapter was admitted, nothing auto-activates. This stage measures what the
mechanism does; it grants no production status.

Usage (WSL, repo root, siltvenv -- export PYTHONPATH first; the venv's editable
asea install points at a STALE ~/SILT checkout, which shadowed the max_length
knob and cost one run to diagnose):
  export PYTHONPATH=/mnt/c/Users/reetu/Desktop/SILT/src
  python experiments/glm53_flash_pilot/train_deepapply.py train
  python experiments/glm53_flash_pilot/train_deepapply.py ab
  python experiments/glm53_flash_pilot/train_deepapply.py judge
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAP = "python_repo_debugging_v1"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 1024

os.environ.setdefault("HF_HOME", "/mnt/c/Users/reetu/.cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"

# The strict extraction rule is shared with the teacher/student judgments --
# imported, not re-implemented, so the A/B cannot drift from the rule the
# baseline numbers were produced under.
sys.path.insert(0, str(HERE))
from judge_pilot import extract_candidate, summarise  # noqa: E402

KD_PATH = HERE / "workspace-judged" / "candidates" / ("%s-kd-pairs.json" % CAP)
RECEIPT_PATH = HERE / "pilot-receipt.unsigned.json"
RUN_DIR = HERE / "deepapply-run"
ADAPTER_DIR = RUN_DIR / "adapter_model"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases() -> dict:
    cases = {}
    for line in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            cases[record["sample_id"]] = record
    return cases


def load_kd_pairs() -> dict:
    """Load the KD artifact after verifying it against the signed receipt.

    Two independent checks: (1) the artifact store's own fail-closed read
    (``CapabilityStore.get`` recomputes the embedded content hash and refuses
    a tampered or foreign artifact), and (2) the artifact the store accepted
    must be the one the signed pilot receipt pinned."""
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    pinned = receipt["artifact_hashes"]["kd_pairs"]
    from asea.artifacts import digest
    from asea.capability_build.store import CapabilityStore

    store = CapabilityStore(HERE / "workspace-judged")
    kd = store.get("candidates", "%s-kd-pairs" % CAP)
    if digest(kd) != pinned:
        raise SystemExit(
            "KD artifact hash mismatch: receipt pins %s, store read gives %s -- "
            "refusing to train on material that is not the receipted "
            "artifact" % (pinned, digest(kd))
        )

    # Spec identity: the pairs must belong to this pilot's frozen spec.
    from asea.capability_build.spec import load_spec

    spec_sha = load_spec(HERE / "spec.json")["spec_sha256"]
    if kd["dataset"]["spec_fingerprint"] != spec_sha:
        raise SystemExit(
            "KD pairs spec fingerprint %s does not match the pilot spec %s"
            % (kd["dataset"]["spec_fingerprint"], spec_sha)
        )

    # Training-split-only: every positive must be an approved training row.
    manifest = json.loads(
        (REPO / "data" / "capability_v1" / "manifest.json").read_text(encoding="utf-8"))
    training_file = REPO / "data" / "capability_v1" / "training.jsonl"
    if sha256(training_file) != manifest["files"]["training.jsonl"]:
        raise SystemExit(
            "training.jsonl hash does not match the frozen manifest -- the "
            "dataset changed after the freeze"
        )
    training_ids = {
        json.loads(line)["sample_id"]
        for line in training_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for pos in kd["positives"]:
        if pos["sample_id"] not in training_ids:
            raise SystemExit(
                "KD positive %s is not in the training split -- protected "
                "material may never become training data" % pos["sample_id"]
            )
    return kd


def _pilot_receiver():
    """Build the minimal real receiver the production trainer reads (model id
    + capabilities). The StandardTrainerBackend loads the real weights itself;
    this object is the honest identity card (never a mock: the real base model
    is trained). Inference is not provided -- the A/B generation stage loads
    the model its own way."""
    from asea.core.interfaces import ModuleAdapter
    from asea.core.protocol import (CapabilityKey, CapabilityManifest, Domain,
                                    LearningLevel, Modality)

    class _Receiver(ModuleAdapter):
        is_mock = False

        def __init__(self):
            super().__init__(MODEL_ID, "Qwen2.5-0.5B-Instruct (pilot receiver)")

        def manifest(self):
            return CapabilityManifest(
                module_id=MODEL_ID,
                display_name="Qwen2.5-0.5B-Instruct (pilot receiver)",
                roles=["receiver"],
                capabilities=[CapabilityKey(
                    task_type="python_repo_debugging",
                    modality=Modality.CODE,
                    domain=Domain.SOFTWARE,
                    language="python",
                )],
                max_learning_level=LearningLevel.L4_PEFT_CANDIDATE,
                is_mock=False,
                version=MODEL_ID,
            )

        def infer(self, capability, prompt):
            raise NotImplementedError(
                "the pilot receiver trains; generation happens in the ab "
                "stage through the deterministic HF path"
            )

    return _Receiver()


def do_train() -> None:
    kd = load_kd_pairs()
    from asea.artifacts import digest
    from asea.deepapply.dataset import TrainingDataset
    from asea.deepapply.runner import DeepApplyConfig
    from asea.deepapply.trainer import StandardTrainerBackend

    rows = [
        {
            "input": pos["prompt"],
            "output": pos["response"],
            "sample_id": pos["sample_id"],
            "source": "kd_pairs",
        }
        for pos in kd["positives"]
    ]
    dataset_hash = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str)
        .encode("utf-8")).hexdigest()
    manifest = {
        "row_count": len(rows),
        "packet_count": 0,
        "source_packet_ids": ["%s-kd-pairs (pilot research artifact, NOT "
                              "Gate-1 packets)" % CAP],
        "source_packet_ids_sorted": ["%s-kd-pairs (pilot research artifact, NOT "
                                     "Gate-1 packets)" % CAP],
        "synthetic_depth_max": 0,
        "source_domains": ["software"],
        "contains_mock": False,
        "min_safety_score": None,
        "dataset_hash": dataset_hash,
        "provenance": "sequence-level KD positives distilled from REAL judged "
                      "teacher traces (see make_receipt.py measurements); "
                      "production admission still requires the Gate-1 "
                      "PROMOTED-packet intake via DeepApplyRunner.from_pipeline",
    }
    dataset = TrainingDataset(rows, manifest)

    # The honest small-model-CPU path: production defaults everywhere except
    # the two documented knobs -- max_length raised past the 256 default so
    # the real teacher responses (284-1172 tokens) are not truncated
    # mid-response, and 48 steps (8 epochs over 6 rows, under the 64 cap).
    config = DeepApplyConfig(
        backend="standard",
        lora_rank=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        learning_rate=1e-4,
        max_steps=48,
        max_steps_cap=64,
        epochs=8,
        seed=0,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=1280,
    )
    steps = []

    def on_step(event):
        if event.get("phase") == "train_step":
            steps.append({k: event.get(k) for k in
                          ("step", "max_steps", "loss", "diverged")})

    backend = StandardTrainerBackend()
    receiver = _pilot_receiver()
    if not backend.supports(receiver):
        raise SystemExit("StandardTrainerBackend refuses this receiver/environment")
    train_cfg = dict(config.to_train_dict())
    train_cfg["_on_step"] = on_step

    started = time.monotonic()
    artifact = backend.train(receiver, dataset, train_cfg, RUN_DIR)
    wall_s = round(time.monotonic() - started, 1)

    report = {
        "run": "deepapply training stage (REAL)",
        "stage": "train",
        "capability_id": CAP,
        "backend": backend.name,
        "backend_version": backend.version,
        "base_model": artifact.model_id,
        "adapter_path": str(artifact.adapter_path),
        "trainable_param_count": artifact.trainable_param_count,
        "training_loss": artifact.training_loss,
        "lora_config": artifact.lora_config,
        "train_config": {k: v for k, v in config.to_train_dict().items()},
        "steps_recorded": steps,
        "step_count": len(steps),
        "wall_seconds": wall_s,
        "dataset": manifest,
        "kd_pairs_sha256": digest(kd),
        "device": "cpu (no CUDA in this environment)",
        "honesty": [
            "The production trainer (StandardTrainerBackend) ran for real; "
            "every loss above is a measured training loss.",
            "Training success is not capability success; the independent "
            "oracle A/B (judge stage) is the only accepted evidence here.",
            "These rows are pilot KD pairs, not Gate-1 PROMOTED packets; no "
            "production admission was sought or granted.",
            "Nothing auto-activates; the adapter is a research artifact.",
        ],
    }
    out = RUN_DIR / "training-run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "adapter": report["adapter_path"],
        "trainable_params": report["trainable_param_count"],
        "final_loss": report["training_loss"],
        "steps": report["step_count"],
        "wall_seconds": wall_s,
        "report": str(out),
    }))


def _generate_arm(model, tokenizer, cases: dict, out_path: Path) -> None:
    """Greedy raw-text continuation for every non-final case; checkpointed."""
    import torch

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])
    handle = out_path.open("a", encoding="utf-8")
    for sid in sorted(cases):
        case = cases[sid]
        if case["split"] == "final":
            continue  # never opened
        if sid in done:
            continue
        inputs = tokenizer([case["prompt"]], return_tensors="pt")
        started = time.monotonic()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs.input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        record = {
            "sample_id": sid,
            "split": case["split"],
            "group": case["group"],
            "response": text,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
            "new_tokens": int(new_tokens.shape[-1]),
        }
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        print("generated %s (%d tokens, %.1fs)" % (
            sid, record["new_tokens"], record["latency_ms"] / 1000.0),
            file=sys.stderr)


def do_ab() -> None:
    cases = load_cases()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    base.eval()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print("arm: base", file=sys.stderr)
    _generate_arm(base, tokenizer, cases, RUN_DIR / "responses-base.jsonl")

    print("arm: adapter", file=sys.stderr)
    adapted = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    adapted.eval()
    _generate_arm(adapted, tokenizer, cases, RUN_DIR / "responses-adapter.jsonl")
    print(json.dumps({
        "base_responses": str(RUN_DIR / "responses-base.jsonl"),
        "adapter_responses": str(RUN_DIR / "responses-adapter.jsonl"),
        "note": "raw-text continuation, greedy, matching the production "
                "trainer's raw input/output format; final split never "
                "generated",
    }))


def _run_oracle(candidate_path: Path, oracle_path: Path) -> dict:
    """Invoke the real evaluate CLI directly (this driver runs INSIDE the
    Linux sandbox environment already; no wsl.exe wrapper needed)."""
    proc = subprocess.run(
        [sys.executable, "-m", "asea.capability_build", "evaluate",
         "--cases", str(oracle_path), "--source", str(candidate_path)],
        cwd=str(REPO), capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "oracle run FAILED for %s (exit %d)\nstdout: %s\nstderr: %s"
            % (candidate_path.stem, proc.returncode, proc.stdout[-2000:],
               proc.stderr[-2000:])
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _judge_arm(arm: str, cases: dict) -> list:
    responses = RUN_DIR / ("responses-%s.jsonl" % arm)
    rows = []
    judged_dir = RUN_DIR / "judging"
    candidates = judged_dir / "candidates"
    oracles = judged_dir / "oracle"
    candidates.mkdir(parents=True, exist_ok=True)
    oracles.mkdir(parents=True, exist_ok=True)
    for line in responses.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sid = record["sample_id"]
        case = cases[sid]
        source, how = extract_candidate(record["response"])
        if source is None:
            rows.append({
                "sample_id": sid, "split": case["split"], "group": case["group"],
                "passed": 0, "total": len(case["oracle"]),
                "per_check": {}, "extraction": how,
                "candidate_sha256": None,
                "oracle_status": "no_candidate",
                "latency_ms": record["latency_ms"],
                "new_tokens": record["new_tokens"],
            })
            continue
        candidate_path = candidates / ("%s-%s.py" % (sid, arm))
        candidate_path.write_text(source, encoding="utf-8")
        oracle_payload = [
            {
                "sample_id": sid,
                "group": case["group"],
                "prompt": case["prompt"],
                "id": check["id"],
                "function": check["function"],
                "args": check["args"],
                "kwargs": check.get("kwargs", {}),
                "expected": check["expected"],
            }
            for check in case["oracle"]
        ]
        oracle_path = oracles / ("%s-%s.json" % (sid, arm))
        oracle_path.write_text(json.dumps(oracle_payload, indent=1), encoding="utf-8")
        payload = _run_oracle(candidate_path, oracle_path)
        result = payload["result"]
        if result.get("blocked"):
            raise SystemExit(
                "oracle BLOCKED for %s (%s): %r -- refusing to grade locally"
                % (sid, arm, result)
            )
        authored = len(case["oracle"])
        if int(result.get("authored_cases", -1)) != authored:
            raise SystemExit(
                "oracle accounting mismatch for %s (%s): reported %s authored "
                "checks, the frozen case defines %d -- refusing the report"
                % (sid, arm, result.get("authored_cases"), authored)
            )
        per_check = {
            c["id"]: c.get("matched")
            for c in (result.get("oracle") or {}).get("cases", [])
        }
        diagnostic = (result.get("oracle") or {}).get("diagnostic") or {}
        rows.append({
            "sample_id": sid, "split": case["split"], "group": case["group"],
            # AUTHORED denominator (2026-09-19 correction): checks the oracle
            # could not judge (candidate import/call error, timeout, resource
            # kill) count as FAILED, never dropped. The previous release
            # divided by the reported judged count and silently erased
            # candidate-error cases from every rate -- the defect the
            # published A/B numbers were corrected for.
            "passed": int(result.get("passed_cases", 0)),
            "total": authored,
            "judged_cases": int(result.get("judged_cases", 0)),
            "unjudged_checks": int(result.get("unjudged_cases", 0)),
            "per_check": per_check, "extraction": how,
            "candidate_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "oracle_status": result["status"],
            "diagnostic_event": diagnostic.get("event"),
            "candidate_error": diagnostic.get("candidate_error"),
            "latency_ms": record["latency_ms"],
            "new_tokens": record["new_tokens"],
        })
        print("judged %s [%s]: %d/%d checks (%s)" % (
            sid, arm, rows[-1]["passed"], rows[-1]["total"], how), file=sys.stderr)
    return rows


def do_judge() -> None:
    cases = load_cases()
    reports = {}
    for arm, name in (("base", "student Qwen/Qwen2.5-0.5B-Instruct, no adapter (A/B base arm)"),
                      ("adapter", "student Qwen/Qwen2.5-0.5B-Instruct + trained LoRA adapter (A/B adapter arm)")):
        rows = _judge_arm(arm, cases)
        summary = summarise(rows, name)
        out = RUN_DIR / ("judgment-ab-%s.json" % arm)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        reports[arm] = summary
        print(json.dumps({"arm": arm, "pass_rates": summary["pass_rates"]}))

    # The frozen non-final dataset defines exactly 45 authored checks per
    # arm (16 cases). A completed report whose check total differs is
    # refused, not published (audit 2026-09-19).
    for arm, summary in reports.items():
        arm_total = sum(r["total"] for r in summary["cases"])
        if arm_total != 45:
            raise SystemExit(
                "A/B %s arm reports %d authored checks; the frozen non-final "
                "dataset defines 45 -- refusing the report" % (arm, arm_total)
            )

    base_rates = reports["base"]["pass_rates"]
    adapter_rates = reports["adapter"]["pass_rates"]
    deltas = {
        key: round(adapter_rates[key] - base_rates[key], 4)
        for key in ("target_checks", "control_checks", "training_checks",
                    "development_checks", "heldout_checks", "controls_split_checks")
        if base_rates.get(key) is not None and adapter_rates.get(key) is not None
    }
    heldout_delta = deltas.get("heldout_checks")
    control_delta = deltas.get("control_checks")
    verdict = {
        "run": "deepapply training + independent oracle A/B (REAL)",
        "capability_id": CAP,
        "base_model": MODEL_ID,
        "comparison": "identical deterministic HF generation path, identical "
                      "strict extraction, identical host oracle; the ONLY "
                      "difference between arms is the trained adapter",
        "base_pass_rates": base_rates,
        "adapter_pass_rates": adapter_rates,
        "deltas_adapter_minus_base": deltas,
        "heldout_improved": (heldout_delta is not None and heldout_delta > 0),
        "control_regressed": (control_delta is not None and control_delta < 0),
        "honesty": [
            "A pass rate on these authored cases measures THIS case set only; "
            "it is not a quality claim about the model.",
            "6 KD pairs is a mechanism-scale training set; a null or negative "
            "held-out delta is a real result, not a failure to hide.",
            "The trainer never certified itself: every number here comes from "
            "the independent host oracle in the Linux sandbox.",
            "The final split was never generated or judged.",
            "Production admission (Gate 1 PROMOTED packets -> DeepApplyRunner "
            "-> Gate 2 -> AdapterStore) was NOT sought; this adapter is a "
            "research artifact and autoactivates nothing.",
        ],
    }
    out = RUN_DIR / "training-report.json"
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"report": str(out), "deltas": deltas}))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "train":
        do_train()
    elif mode == "ab":
        do_ab()
    elif mode == "judge":
        do_judge()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()