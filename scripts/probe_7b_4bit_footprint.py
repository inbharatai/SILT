"""One-shot: load Qwen2.5-7B-Instruct in 4-bit nf4 on the RTX 5050 and report
the honest resident footprint (allocated + reserved). Tells us whether two
co-resident 4-bit copies (Gate 2 A/B) can fit an 8 GB card, or whether the
evaluator must run A/B sequentially with an unload between.

Local only; patent pending (India). The script publishes nothing.
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = os.environ.get("ASEA_MODEL", "Qwen/Qwen2.5-7B-Instruct")

torch.cuda.reset_peak_memory_stats()
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print("loading {} in 4-bit nf4 ...".format(MODEL), flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb).to("cuda")
model.eval()

alloc = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.max_memory_reserved() / 1e9
peak = torch.cuda.max_memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9

# Tie check: does lm_head share embed_tokens?
tie = getattr(model.config, "tie_word_embeddings", None)
n_params = sum(p.numel() for p in model.parameters())
print("=" * 60)
print("Qwen2.5-7B 4-bit nf4 footprint (one copy, GPU resident)")
print("=" * 60)
print("model_id              : {}".format(MODEL))
print("total params          : {:,}".format(n_params))
print("tie_word_embeddings   : {}".format(tie))
print("memory_allocated GB   : {:.2f}".format(alloc))
print("max_memory_allocated GB: {:.2f}".format(peak))
print("max_memory_reserved GB: {:.2f}".format(reserved))
print("GPU total GB          : {:.2f}".format(total))
print("headroom for 2nd copy : {:.2f} GB  (need ~{:.2f})".format(
    total - alloc, alloc))
print("=" * 60)
if (total - alloc) >= alloc:
    print("VERDICT: two co-resident 4-bit copies FIT {:.2f}+{:.2f}={:.2f} < {:.2f}".format(
        alloc, alloc, 2 * alloc, total))
else:
    print("VERDICT: two co-resident 4-bit copies DO NOT FIT {:.2f}+{:.2f}={:.2f} > {:.2f}".format(
        alloc, alloc, 2 * alloc, total))
    print("        -> evaluator must run A/B sequentially with unload between")
print("=" * 60)