"""E2 DeepApply training stage (REAL run, 2026-09-19): expanded KD set on a
bigger student, trained on GPU.

Same production machinery and same honesty discipline as the v1 pilot
(experiments/glm53_flash_pilot/train_deepapply.py), with exactly three
recorded differences:

  1. STUDENT: Qwen/Qwen2.5-1.5B-Instruct (the v1 pilot used the 0.5B
     Instruct student) -- the bigger cached receiver for the expanded KD
     set.
  2. TRAINING SET: the E2 KD pairs distilled from the six receipted v1
     training traces PLUS the sixteen newly authored E2 training cases
     (all host-oracle judged; identical distill command, identical
     leakage discipline).
  3. HARDWARE: the RTX 5050 (sm_120, 8 GB) through the same
     StandardTrainerBackend, which auto-selects CUDA when available. The
     GPU-forced environment deviations are recorded honestly in the
     training report: torch 2.11.0+cu128 (the localmodels pin
     torch>=2.6,<2.7 has no sm_120-compatible build; Blackwell needs
     cu128) and a NEW documented trainer knob ``model_dtype="bfloat16"``
     (fp32 1.5B weights plus the logits tensor alone exceed 8 GB VRAM;
     bf16 fits with headroom). transformers==4.51.3, peft==0.15.2 and
     accelerate==1.10.1 match the production pins exactly. The torch
     deviation is the same pattern as the GLM worker's isolated
     transformers 5.16.1: isolated to this run's environment, recorded,
     never a repo pin change.

  train -- verify the KD artifact (hash pinned to the E2 receipt, spec
      fingerprint, training-split-only against the frozen data/capability_v2
      manifest) and run the REAL production trainer with per-step loss
      telemetry on CUDA. The adapter is saved; nothing is admitted or
      activated.

  ab    -- generate responses for every non-final case (the frozen v1
      evaluation material, byte-identical inside data/capability_v2) twice
      through the SAME deterministic HF path (greedy, raw-text
      continuation, SAME bfloat16 dtype on both arms): plain base, then
      trained adapter. The final split is NEVER generated.

  judge -- extract candidates with the shared strict rule (imported from
      the v1 judge) and grade BOTH arms through the REAL host oracle +
      Linux code sandbox (WSL). The trainer never certifies itself.
      AUTHORED denominators throughout (2026-09-19 correction): the
      expected per-arm total is COMPUTED from the frozen non-final rows
      (32 cases / 102 authored checks in the v2 dataset), never hand-typed.

Production-admission boundary (binding): rows come from the E2 judged KD
pairs, not Gate-1 PROMOTED packets; Gate 1 was not sought, Gate 2
admission was not requested, nothing auto-activates.

Usage (Windows, repo root, GPU venv):
  python experiments/glm53_flash_e2/train_deepapply_e2.py train
  python experiments/glm53_flash_e2/train_deepapply_e2.py ab
  python experiments/glm53_flash_e2/train_deepapply_e2.py judge
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAP = "python_repo_debugging_v1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 1024
MODEL_DTYPE = "bfloat16"

os.environ.setdefault(
    "HF_HOME", r"C:\Users\reetu\.cache\huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"

# The strict extraction rule and the shared summary discipline are IMPORTED
# from the v1 judge, not re-implemented, so the E2 A/B cannot drift from the
# rule the v1 baseline numbers were produced under.
sys.path.insert(0, str(REPO / "experiments" / "glm53_flash_pilot"))
from judge_pilot import extract_candidate, summarise, wsl_path  # noqa: E402

KD_PATH = HERE / "workspace-judged" / "candidates" / ("%s-kd-pairs.json" % CAP)
RECEIPT_PATH = HERE / "e2-receipt-pretrain.unsigned.json"
RUN_DIR = HERE / "deepapply-run"
ADAPTER_DIR = RUN_DIR / "adapter_model"
V2_DIR = REPO / "data" / "capability_v2"
# PYTHONPATH pins the WSL oracle to THIS checkout's src: the siltvenv
# otherwise resolves a stale asea tree at ~/SILT (commit ec5777e) whose
# evaluate result predates the authored-denominator correction. Same
# isolation pattern as the intervention loop and judge_e2.py.
WSL_VENV = ("source ~/siltvenv/bin/activate && "
            "export PYTHONPATH=/mnt/c/Users/reetu/Desktop/SILT/src")
WSL_REPO = "/mnt/c/Users/reetu/Desktop/SILT"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases() -> dict:
    """The A/B case set: every NON-FINAL split of the frozen v2 dataset.
    The final split is never read into memory in this driver at all."""
    cases = {}
    for split in ("training", "development", "heldout", "controls"):
        for line in (V2_DIR / ("%s.jsonl" % split)).read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                cases[record["sample_id"]] = record
    return cases


def load_kd_pairs() -> dict:
    """Load the E2 KD artifact after verifying it against the pretrain
    receipt (which pins its embedded content hash), the spec fingerprint,
    and the training-split-only rule against the frozen v2 manifest."""
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    pinned = receipt["artifact_hashes"]["kd_pairs"]
    from asea.artifacts import digest
    from asea.capability_build.store import CapabilityStore

    store = CapabilityStore(HERE / "workspace-judged")
    kd = store.get("candidates", "%s-kd-pairs" % CAP)
    if digest(kd) != pinned:
        raise SystemExit(
            "KD artifact hash mismatch: receipt pins %s, store read gives %s"
            % (pinned, digest(kd)))

    from asea.capability_build.spec import load_spec

    spec_sha = load_spec(HERE / "spec.json")["spec_sha256"]
    if kd["dataset"]["spec_fingerprint"] != spec_sha:
        raise SystemExit(
            "KD pairs spec fingerprint %s does not match the E2 spec %s"
            % (kd["dataset"]["spec_fingerprint"], spec_sha))

    manifest = json.loads(
        (V2_DIR / "manifest.json").read_text(encoding="utf-8"))
    training_file = V2_DIR / "training.jsonl"
    if sha256(training_file) != manifest["files"]["training.jsonl"]:
        raise SystemExit(
            "training.jsonl hash does not match the frozen v2 manifest -- the "
            "dataset changed after the freeze")
    training_ids = {
        json.loads(line)["sample_id"]
        for line in training_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for pos in kd["positives"]:
        if pos["sample_id"] not in training_ids:
            raise SystemExit(
                "KD positive %s is not in the v2 training split -- protected "
                "material may never become training data" % pos["sample_id"])
    return kd


def _e2_receiver():
    """The honest identity card for the 1.5B receiver (never a mock: the
    real base model is trained; inference is not provided -- generation
    happens in the ab stage through the deterministic HF path)."""
    from asea.core.interfaces import ModuleAdapter
    from asea.core.protocol import (CapabilityKey, CapabilityManifest, Domain,
                                    LearningLevel, Modality)

    class _Receiver(ModuleAdapter):
        is_mock = False

        def __init__(self):
            super().__init__(MODEL_ID, "Qwen2.5-1.5B-Instruct (E2 receiver)")

        def manifest(self):
            return CapabilityManifest(
                module_id=MODEL_ID,
                display_name="Qwen2.5-1.5B-Instruct (E2 receiver)",
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
                "the E2 receiver trains; generation happens in the ab stage "
                "through the deterministic HF path")

    return _Receiver()


def do_train() -> None:
    kd = load_kd_pairs()
    import torch  # noqa: F401  (presence proves the deep extras)

    from asea.deepapply.dataset import TrainingDataset
    from asea.deepapply.runner import DeepApplyConfig
    from asea.deepapply.trainer import StandardTrainerBackend

    rows = [
        {
            "input": pos["prompt"],
            "output": pos["response"],
            "sample_id": pos["sample_id"],
            "source": "kd_pairs_e2",
        }
        for pos in kd["positives"]
    ]
    dataset_hash = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str)
        .encode("utf-8")).hexdigest()
    manifest = {
        "row_count": len(rows),
        "packet_count": 0,
        "source_packet_ids": ["%s-kd-pairs E2 (pilot research artifact, NOT "
                              "Gate-1 packets)" % CAP],
        "source_packet_ids_sorted": ["%s-kd-pairs E2 (pilot research "
                                     "artifact, NOT Gate-1 packets)" % CAP],
        "synthetic_depth_max": 0,
        "source_domains": ["software"],
        "contains_mock": False,
        "min_safety_score": None,
        "dataset_hash": dataset_hash,
        "provenance": "sequence-level KD positives distilled from REAL judged "
                      "teacher traces (six receipted v1 training pairs plus "
                      "the sixteen E2 training cases, all host-oracle judged); "
                      "production admission still requires the Gate-1 "
                      "PROMOTED-packet intake via DeepApplyRunner.from_pipeline",
    }
    dataset = TrainingDataset(rows, manifest)

    # Production defaults everywhere except the three documented knobs:
    # max_length (real teacher responses are long), 64 steps (the cap), and
    # model_dtype bf16 (fp32 1.5B weights + logits exceed the 8 GB card).
    config = DeepApplyConfig(
        backend="standard",
        lora_rank=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        learning_rate=1e-4,
        max_steps=64,
        max_steps_cap=64,
        epochs=8,
        seed=0,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=1280,
        model_dtype=MODEL_DTYPE,
    )
    steps = []

    def on_step(event):
        if event.get("phase") == "train_step":
            steps.append({k: event.get(k) for k in
                          ("step", "max_steps", "loss", "diverged")})

    backend = StandardTrainerBackend()
    receiver = _e2_receiver()
    if not backend.supports(receiver):
        raise SystemExit("StandardTrainerBackend refuses this receiver/environment")
    train_cfg = dict(config.to_train_dict())
    train_cfg["_on_step"] = on_step

    import torch

    started = time.monotonic()
    artifact = backend.train(receiver, dataset, train_cfg, RUN_DIR)
    wall_s = round(time.monotonic() - started, 1)

    gpu = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": "sm_%d%d" % torch.cuda.get_device_capability(0),
            "total_vram_mib": round(props.total_memory / (1024 * 1024)),
        }

    import transformers
    import peft

    report = {
        "run": "E2 deepapply training stage (REAL, GPU)",
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
        "kd_pairs_sha256": None,  # filled below (digest of the loaded kd)
        "device": {
            "cuda_available": bool(torch.cuda.is_available()),
            "selected": "cuda" if torch.cuda.is_available() else "cpu",
            "model_dtype": MODEL_DTYPE,
            "gpu": gpu,
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "env_note": "ISOLATED GPU venv on top of --system-site-packages: "
                        "torch 2.11.0+cu128 inherited from the system Python "
                        "(the localmodels pin torch>=2.6,<2.7 has no "
                        "sm_120-compatible build; Blackwell requires cu128); "
                        "transformers/peft/accelerate pinned to the "
                        "production versions inside the venv",
        },
        "honesty": [
            "The production trainer (StandardTrainerBackend) ran for real on "
            "CUDA; every loss above is a measured training loss.",
            "Training success is not capability success; the independent "
            "oracle A/B (judge stage) is the only accepted evidence here.",
            "These rows are pilot KD pairs, not Gate-1 PROMOTED packets; no "
            "production admission was sought or granted.",
            "Nothing auto-activates; the adapter is a research artifact.",
            "The bfloat16 weight dtype (new documented trainer knob) is a "
            "VRAM necessity on this 8 GB card, identical across both A/B "
            "arms, and recorded here; fp32 was the v1 pilot's path.",
        ],
    }
    from asea.artifacts import digest

    report["kd_pairs_sha256"] = digest(kd)
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
        "device": report["device"],
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
        if sid in done:
            continue
        inputs = tokenizer([case["prompt"]], return_tensors="pt").to("cuda")
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
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16}.get(MODEL_DTYPE, torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # BOTH arms load under the SAME dtype on the SAME device: the only
    # difference between arms is the trained adapter.
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype).to("cuda")
    base.eval()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print("arm: base", file=sys.stderr)
    _generate_arm(base, tokenizer, cases, RUN_DIR / "responses-base.jsonl")

    print("arm: adapter", file=sys.stderr)
    adapted = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    adapted.eval()
    _generate_arm(adapted, tokenizer, cases,
                  RUN_DIR / "responses-adapter.jsonl")
    print(json.dumps({
        "base_responses": str(RUN_DIR / "responses-base.jsonl"),
        "adapter_responses": str(RUN_DIR / "responses-adapter.jsonl"),
        "note": "raw-text continuation, greedy, bfloat16 on CUDA for BOTH "
                "arms, matching the production trainer's raw input/output "
                "format; final split never generated",
    }))


def _run_oracle(candidate_path: Path, oracle_path: Path) -> dict:
    """Invoke the real evaluate CLI inside WSL (Linux sandbox; this driver
    runs on Windows)."""
    cmd = (
        "%s && cd %s && python -m asea.capability_build evaluate "
        "--cases %s --source %s"
        % (WSL_VENV, WSL_REPO, wsl_path(oracle_path),
           wsl_path(candidate_path))
    )
    proc = subprocess.run(
        ["wsl.exe", "bash", "-c", cmd], capture_output=True, text=True,
        timeout=300,
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
                "judged_cases": 0, "unjudged_checks": len(case["oracle"]),
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
            # kill) count as FAILED, never dropped.
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
    for arm, name in (("base", "student Qwen/Qwen2.5-1.5B-Instruct, no adapter (E2 A/B base arm)"),
                      ("adapter", "student Qwen/Qwen2.5-1.5B-Instruct + trained LoRA adapter (E2 A/B adapter arm)")):
        rows = _judge_arm(arm, cases)
        summary = summarise(rows, name)
        out = RUN_DIR / ("judgment-ab-%s.json" % arm)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        reports[arm] = summary
        print(json.dumps({"arm": arm, "pass_rates": summary["pass_rates"]}))

    # Denominator guard: the graded total per arm must equal the AUTHORED
    # oracle total of exactly the non-final rows the judge read from the
    # frozen v2 files -- COMPUTED, never hand-typed (the first run of this
    # driver hand-typed v1's 45-check total and the guard refused its own
    # report, which is the guard working: the v2 training split expanded
    # to 22 cases, so the non-final set is 32 cases / 102 authored checks).
    expected_total = sum(len(c["oracle"]) for c in cases.values())
    for arm, summary in reports.items():
        arm_total = sum(r["total"] for r in summary["cases"])
        if arm_total != expected_total:
            raise SystemExit(
                "A/B %s arm reports %d authored checks; the frozen non-final "
                "dataset defines %d -- refusing the report"
                % (arm, arm_total, expected_total)
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
        "run": "E2 deepapply training + independent oracle A/B (REAL, GPU)",
        "capability_id": CAP,
        "base_model": MODEL_ID,
        "kd_set": "six receipted v1 training pairs + E2 training cases "
                  "(host-oracle judged, same distill discipline)",
        "comparison": "identical deterministic HF generation path (bfloat16 "
                      "on CUDA, greedy), identical strict extraction, "
                      "identical host oracle; the ONLY difference between "
                      "arms is the trained adapter",
        "base_pass_rates": base_rates,
        "adapter_pass_rates": adapter_rates,
        "deltas_adapter_minus_base": deltas,
        "heldout_improved": (heldout_delta is not None and heldout_delta > 0),
        "control_regressed": (control_delta is not None and control_delta < 0),
        "honesty": [
            "A pass rate on these authored cases measures THIS case set "
            "only; it is not a quality claim about the model.",
            "The mechanism-scale caveat stands: the KD set is small; a null "
            "or negative held-out delta is a real result, not a failure to "
            "hide.",
            "The trainer never certified itself: every number here comes "
            "from the independent host oracle in the Linux sandbox.",
            "The final split was never generated or judged.",
            "Production admission (Gate 1 PROMOTED packets -> "
            "DeepApplyRunner -> Gate 2 -> AdapterStore) was NOT sought; this "
            "adapter is a research artifact and autoactivates nothing.",
            "The E2 A/B arms are NOT directly comparable to the v1 pilot's "
            "fp32 CPU A/B numbers: different student (1.5B vs 0.5B), "
            "different dtype and device, same frozen evaluation cases and "
            "the same oracle; each experiment is compared within itself.",
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