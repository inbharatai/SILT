"""Empirical probe: can a 4-bit (Params4bit) HF layer survive the disk-streamer's
free-to-empty -> reload cycle? This determines whether Qwen2.5-7B 4-bit can go
through the streamed backend's parity mechanism at all.

Honest probe: prints exactly what happens, no assumptions.
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "HuggingFaceTB/SmolLM2-135M"  # tiny, fast; 4-bit mechanics are model-size-invariant


def main():
    dev = "cuda"
    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"loading {MODEL} in 4-bit on {dev} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=qc).to(dev)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    ids = tok("hello world", return_tensors="pt").input_ids.to(dev)

    with torch.no_grad():
        ref = model(ids).logits.detach().clone()
    print("ref forward ok, logits shape", tuple(ref.shape), "finite", bool(torch.isfinite(ref).all()))

    layer = model.model.layers[0]
    # inspect a 4-bit param
    for n, p in layer.named_parameters():
        if "q_proj" in n:
            print("param", n, "type", type(p).__name__, "dtype", p.dtype, "is Params4bit", type(p).__name__ == "Params4bit")
            break

    # 1) capture state_dict (what the bank does)
    state = {k: v.detach().clone() for k, v in layer.state_dict().items() if "lora_" not in k}
    print("banked", len(state), "tensors; types:", sorted({type(v).__name__ for v in state.values()}))

    # 2) free to empty (what _free_layer does)
    freed = 0
    for n, p in layer.named_parameters():
        if "lora_" in n:
            continue
        p.data = torch.empty(0, dtype=p.dtype)
        freed += 1
    print("freed", freed, "params to empty(0)")

    # 3a) try the naive restore the current _load_layer uses
    try:
        for n, p in layer.named_parameters():
            if "lora_" in n:
                continue
            if n in state:
                p.data = state[n].to(p.dtype)
        with torch.no_grad():
            got = model(ids).logits.detach().clone()
        diff = float((got - ref).abs().max().item())
        print("NAIVE restore (param.data = state.to(dtype)): forward max|diff| =", diff,
              "MATCH" if diff == 0.0 else "MISMATCH")
    except Exception as exc:
        print("NAIVE restore FAILED:", type(exc).__name__, str(exc)[:200])

    # 3b) try load_state_dict restore (frees again first)
    for n, p in layer.named_parameters():
        if "lora_" not in n:
            p.data = torch.empty(0, dtype=p.dtype)
    try:
        missing, unexpected = layer.load_state_dict(state, strict=False), []
        print("load_state_dict missing=", list(missing.missing_keys)[:3], "unexpected=", list(missing.unexpected_keys)[:3])
        with torch.no_grad():
            got = model(ids).logits.detach().clone()
        diff = float((got - ref).abs().max().item())
        print("LOAD_STATE_DICT restore: forward max|diff| =", diff,
              "MATCH" if diff == 0.0 else "MISMATCH")
    except Exception as exc:
        print("LOAD_STATE_DICT restore FAILED:", type(exc).__name__, str(exc)[:200])

    print("DONE")


if __name__ == "__main__":
    main()