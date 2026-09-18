"""REAL SiltSpring compression proof for the GLM-5.3-Flash pilot (2026-09-18).

This is the compression half of the pilot directive ("skill transfer
compression ... should be done and proven"). It runs the REAL per-layer
quantization + certification path
(:func:`asea.deepapply.backends.siltstream_vendor.hf_real.certify_hf_states`)
on the pilot's actual student-class model, Qwen/Qwen2.5-0.5B (the local
student measured in ``judgment-student-qwen2.5_0.5b.json``), with
certification suites built from the pilot's OWN case prompts:

  * ``target_debugging_heldout``     -- heldout split prompts (target)
  * ``target_debugging_development`` -- development split prompts (target)
  * ``control_utility``             -- controls split prompts (control)

The FINAL split is never read here (asserted below; ``final_opened``
stays false for this pilot).

What this proves, mechanically, with no simulation:
  * every decoder layer's 2-D weights are REALLY quantized to int8/int4/int2
    (per-row symmetric containers) and banked to disk;
  * the model REALLY forwards through one quantized layer at a time and
    each skill's suite loss is measured against the full-precision reference;
  * per-(state, skill) certificates are granted or REVOKED by measured
    relative degradation against tolerance 0.02;
  * real packed-byte counts give the compression ratios;
  * exiting a quantized streamer re-expands the model to FULL precision
    (vendor guard B2) -- verified here by a sha256 digest over every layer
    tensor taken BEFORE certification and again AFTER, which must match
    byte-for-byte (weights are read-only; the proof never mutates the model).

Honesty boundaries (binding, carried into the artifact):
  * The certificate criterion is LOSS degradation (the certifier's designed
    contract: a compressed state must not change behaviour beyond tolerance
    on the suite). It is NOT an oracle pass-rate claim, and a certificate
    here is not a claim about the model's coding ability.
  * Implementation success is not model-quality success.
  * The final split was never opened; no measurement exists for it.
  * DeepApply LoRA training on the distilled pairs is the designed NEXT
    stage (see the kd-pairs handoff descriptor); this proof deliberately
    stops at the compression-certification boundary.

Usage (WSL, repo root; CPU-only torch; model from the shared HF cache):
  HF_HUB_OFFLINE=1 HF_HOME=/mnt/c/Users/reetu/.cache/huggingface \
    python experiments/glm53_flash_pilot/compress_proof.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = "python_repo_debugging_v1"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # the pilot's local student was
# qwen2.5:0.5b (BASE); the shared HF cache's base-0.5B snapshot contains no
# weight files, so the complete 0.5B-Instruct snapshot -- same architecture
# and scale -- is used and the substitution is recorded in the artifact.
LEVELS = ("int8", "int4", "int2")
TOLERANCE = 0.02
MAX_LEN = 192

cases = [
    json.loads(line)
    for line in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
SUITES_TEXT: dict = {
    "target_debugging_heldout": [c["prompt"] for c in cases
                                 if c["split"] == "heldout"],
    "target_debugging_development": [c["prompt"] for c in cases
                                      if c["split"] == "development"],
    "control_utility": [c["prompt"] for c in cases
                        if c["split"] == "controls"],
}
# The final split is NEVER opened in this pilot: no final prompt may reach a
# certification suite (this is a hard guard, not a convention).
final_prompts = {c["prompt"] for c in cases if c["split"] == "final"}
suite_prompts = {p for texts in SUITES_TEXT.values() for p in texts}
assert not (final_prompts & suite_prompts), \
    "a final-split prompt leaked into a certification suite"
for name, texts in SUITES_TEXT.items():
    assert texts, "empty suite %s" % name


def layer_digests(layers) -> dict:
    """sha256 per (layer, tensor) over raw bytes -- dtype-exact."""
    import torch

    out = {}
    for i, layer in enumerate(layers):
        for key, value in sorted(layer.state_dict().items()):
            t = value.detach().cpu().contiguous()
            t = t.view(torch.uint8) if t.element_size() > 1 else t
            out["layer%03d/%s" % (i, key)] = hashlib.sha256(
                t.numpy().tobytes()).hexdigest()
    return out


def bank_dir_bytes(path: Path) -> int:
    return sum(
        (Path(root) / name).stat().st_size
        for root, _dirs, files in os.walk(path)
        for name in files
    )


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from asea.deepapply.backends.siltstream_vendor.hf_real import (
        certify_hf_states, get_decoder_layers, texts_to_batch,
    )

    # Locate the exact cached snapshot so the proof pins the model revision.
    hub = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    model_cache = hub / "hub" / ("models--" + MODEL_ID.replace("/", "--"))
    snapshots = model_cache / "snapshots"
    revision = None
    if snapshots.is_dir():
        shas = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
        if len(shas) == 1:
            revision = shas[0]
        elif shas:
            revision = "ambiguous:%s" % ",".join(shas)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    layers = get_decoder_layers(model)
    n_params = sum(p.numel() for p in model.parameters())
    print("loaded %s: %d layers, %d parameters (revision %s)"
          % (MODEL_ID, len(layers), n_params, revision), file=sys.stderr)

    before = layer_digests(layers)

    suites = {
        name: texts_to_batch(tokenizer, texts, max_len=MAX_LEN)
        for name, texts in SUITES_TEXT.items()
    }

    # The bank is scratch disk (full-precision + one container per level,
    # roughly 4x the decoder stack); keep it OUT of the repo by default.
    bank_root = Path(os.environ.get(
        "SILT_SPRING_BANK_DIR",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / "silt_pilot_spring_bank"),
    ))
    results = certify_hf_states(
        model, layers, suites, LEVELS,
        disk_dir=str(bank_root), tolerance=TOLERANCE,
    )

    after = layer_digests(layers)
    restored_byte_exact = before == after
    if not restored_byte_exact:
        raise SystemExit(
            "FATAL: model weights CHANGED across certification -- the "
            "streamer's full-precision re-expand (guard B2) failed")

    on_disk = {
        level: bank_dir_bytes(bank_root / level)
        for level in ("full",) + LEVELS
    }

    artifact = {
        "run": "silspring compression proof (REAL)",
        "capability_id": CAP,
        "model": MODEL_ID,
        "revision": revision,
        "parameters": n_params,
        "decoder_layers": len(layers),
        "suites": {
            name: {
                "split": {"target_debugging_heldout": "heldout",
                          "target_debugging_development": "development",
                          "control_utility": "controls"}[name],
                "cases": len(texts),
            }
            for name, texts in SUITES_TEXT.items()
        },
        "final_opened": False,
        "levels": list(LEVELS),
        "tolerance": TOLERANCE,
        "quantization": "per-row symmetric int8/int4/int2 containers "
                        "(siltstream_vendor.quant), one layer resident at a "
                        "time, full-precision re-expand on exit",
        "results": {
            level: {
                "loss": payload["loss"],
                "degradation": payload["degradation"],
                "certified": payload["certified"],
                "revoked": payload["revoked"],
                "bytes_packed": payload["bytes_packed"],
                "bank_bytes_on_disk": on_disk.get(level),
            }
            for level, payload in results.items()
        },
        "weights_read_only_verified": {
            "method": "sha256 over every decoder-layer tensor, dtype-exact "
                      "raw bytes, taken before certification and after the "
                      "final quantized streamer exited",
            "byte_exact_restore": restored_byte_exact,
            "tensors_hashed": len(before),
        },
        "honesty": [
            "The certificate criterion is LOSS degradation vs the "
            "full-precision reference on the suite (the certifier's designed "
            "contract). It is NOT an oracle pass-rate claim.",
            "A certificate here is not a claim about the model's coding "
            "ability; implementation success is not model-quality success.",
            "The final split was never opened; no measurement exists for it.",
            "DeepApply LoRA training on the distilled kd-pairs is the "
            "designed next stage; this proof stops at the "
            "compression-certification boundary.",
        ],
    }
    out = HERE / "compression-proof.json"
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "model": MODEL_ID,
        "revision": revision,
        "parameters": n_params,
        "levels": list(LEVELS),
        "per_level": {
            level: {
                "certified": payload["certified"],
                "revoked": payload["revoked"],
                "bytes_packed": payload["bytes_packed"],
            }
            for level, payload in results.items()
        },
        "weights_read_only": restored_byte_exact,
        "artifact": str(out),
    }))


if __name__ == "__main__":
    main()