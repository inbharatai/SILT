"""Diagnostic: why did int4 show LOWER loss than full in compression-proof.json?

Isolates three layers of the pipeline on the same model + suite:
  A. pure quantize->dequantize round-trip error on real layer weights
     (no streamer, no bank): relative Frobenius error per level;
  B. in-place replacement of all 2-D layer weights with their round-tripped
     versions (no streamer): suite loss per level;
  C. the bank+streamer path exactly as certify_hf_states uses it: suite loss.

Run (WSL):
  HF_HUB_OFFLINE=1 HF_HOME=/mnt/c/Users/reetu/.cache/huggingface \
    python experiments/glm53_flash_pilot/diag_quant.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

cases = [json.loads(l) for l in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
texts = [c["prompt"] for c in cases if c["split"] == "heldout"]
assert len(texts) == 3


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from asea.deepapply.backends.siltstream_vendor.hf_real import (
        HFDiskBank, HFStreamer, get_decoder_layers, suite_loss, texts_to_batch,
    )
    from asea.deepapply.backends.siltstream_vendor.quant import (
        dequantize_state, quantize_state,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    layers = get_decoder_layers(model)
    ids = texts_to_batch(tok, texts, max_len=192)

    # -- A. round-trip weight error (no model involved) --------------------
    print("== A. quantize->dequantize round-trip weight error", file=sys.stderr)
    for level in ("int8", "int4", "int2"):
        num = den = 0.0
        n_zeroed = n_elems = 0
        for layer in layers:
            for name, w in layer.state_dict().items():
                if w.dim() != 2:
                    continue
                wc = w.detach().cpu().float()
                q = quantize_state({name: w}, level)
                back = dequantize_state(q)[name].float()
                num += (back - wc).pow(2).sum().item()
                den += wc.pow(2).sum().item()
                n_zeroed += int((back == 0).sum().item())
                n_elems += back.numel()
        print("  %s: rel_fro_err=%.4f zeroed=%.3f" % (
            level, (num / max(den, 1e-12)) ** 0.5, n_zeroed / max(n_elems, 1)),
            file=sys.stderr)

    # -- B. in-place round-tripped weights, no streamer ----------------------
    print("== B. in-place round-tripped weights (no streamer)", file=sys.stderr)
    originals = [ {n: p.detach().clone() for n, p in layer.named_parameters()}
                  for layer in layers ]
    with torch.no_grad():
        ref = suite_loss(model, ids)
        print("  full: %.4f" % ref, file=sys.stderr)
        for level in ("int8", "int4", "int2"):
            for layer in layers:
                state = {n: p.detach().cpu().clone() for n, p in layer.named_parameters()}
                q = quantize_state(state, level)
                back = dequantize_state(q)
                for n, p in layer.named_parameters():
                    if n in back:
                        p.data.copy_(back[n].to(p.dtype))
            ls = suite_loss(model, ids)
            print("  %s: loss=%.4f (rel deg %.4f)" % (
                level, ls, (ls - ref) / max(abs(ref), 1e-12)), file=sys.stderr)
            for layer, orig in zip(layers, originals):
                for n, p in layer.named_parameters():
                    p.data.copy_(orig[n])
    # verify restore
    for layer, orig in zip(layers, originals):
        for n, p in layer.named_parameters():
            assert torch.equal(p.detach(), orig[n]), "B restore failed"

    # -- C. bank + streamer path, exactly as certification uses it -----------
    print("== C. bank + streamer (as certify_hf_states)", file=sys.stderr)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        full_bank = HFDiskBank(layers, td + "/full", level="full")
        for level in ("int8", "int4", "int2"):
            bank = HFDiskBank(layers, td + "/" + level, level=level)
            with HFStreamer(model, bank, restore_bank=full_bank, device="cpu"):
                ls = suite_loss(model, ids)
            print("  %s: loss=%.4f (rel deg %.4f)" % (
                level, ls, (ls - ref) / max(abs(ref), 1e-12)), file=sys.stderr)
            # layer 0 weight AFTER a streamer exited: must equal original
            w0 = layers[0].named_parameters()
            for n, p in list(w0)[:1]:
                same = torch.equal(p.detach(), originals[0][n])
                print("  %s: layer0 %s restored=%s" % (level, n, same),
                      file=sys.stderr)


if __name__ == "__main__":
    main()