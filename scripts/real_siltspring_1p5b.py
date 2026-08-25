"""Real-HF SiltSpring certification on Qwen2.5-1.5B-Instruct.

Exercises ``siltstream_vendor.hf_real.certify_hf_states`` on REAL weights (not
the toy SpringModel the unit tests use): loads the model fp16 on the GPU,
evaluates the full-precision reference loss on two short suites, then banks
each decoder layer to disk at int8 / int4 / int2 and streams the dequantized
forward per layer, reporting per-state degradation vs the reference and the
certified / revoked skill sets.

Why 1.5B, not the 7B used for deep-apply: certification needs the FULL-PRECISION
reference model resident for the reference loss, and 7B fp16 = 14 GB > 8 GB on
this RTX 5050 -> NOT VERIFIABLE HERE. 1.5B fp16 ~= 3 GB fits. This honestly
verifies the SiltSpring real-HF function on real weights at the largest size
this card can hold a full-precision reference for.

Local only; patent pending (India). The script publishes nothing.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL = os.environ.get("ASEA_SPRING_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def main():
    if not os.environ.get("ASEA_RUN_REAL"):
        print("set ASEA_RUN_REAL=1 to run (loads {} fp16 on GPU)".format(MODEL))
        sys.exit(2)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from asea.deepapply.backends.siltstream_vendor.hf_real import (
        certify_hf_states,
        get_decoder_layers,
    )

    torch.cuda.reset_peak_memory_stats()
    print("loading {} fp16 on cuda ...".format(MODEL), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    model.eval()
    layers = get_decoder_layers(model)

    # Two short "skill" suites -- distinct text so certification can revoke one
    # and keep the other if a quant level hurts one more than the other.
    suites = {}
    for name, texts in {
        "en_common": ["The quick brown fox jumps over the lazy dog.",
                      "A model learns by minimizing a loss over data."],
        "code-ish": ["def add(a, b): return a + b",
                     "for i in range(n): total += i"],
    }.items():
        enc = tok(texts, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=48)
        suites[name] = enc["input_ids"].to("cuda")

    tmp = Path(tempfile.mkdtemp(prefix="silt_spring_"))
    levels = ["int8", "int4", "int2"]
    print("certifying states {} vs full-precision reference ...".format(levels), flush=True)
    results = certify_hf_states(model, layers, suites, levels, str(tmp), tolerance=0.05)

    print("=" * 64)
    print("SiltSpring real-HF certification ({})".format(MODEL))
    print("=" * 64)
    print("cuda available    : {} ({})".format(
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
    print("decoder layers    : {}".format(len(layers)))
    print("vram peak GB       : {:.2f}".format(torch.cuda.max_memory_allocated() / 1e9))
    ref = results["full"]["loss"]
    print("reference (full)   : loss = {}".format(
        {k: round(v, 4) for k, v in ref.items()}))
    for lv in ["full"] + levels:
        r = results[lv]
        deg = r["degradation"]
        print("-" * 64)
        print("state {:<6} bytes_packed={}".format(lv, r["bytes_packed"]))
        print("   loss        : {}".format(
            {k: round(float(v), 4) for k, v in r["loss"].items()}))
        print("   degradation : {}".format(
            {k: "{:+.4f}".format(float(v)) for k, v in deg.items()}))
        print("   certified   : {}".format(r["certified"]))
        print("   revoked     : {}".format(r["revoked"]))
    print("=" * 64)
    print("This verifies the SiltSpring real-HF certify path on real weights.")
    print("On 7B: NOT VERIFIABLE HERE (full-precision reference 14 GB > 8 GB).")
    print("=" * 64)


if __name__ == "__main__":
    main()