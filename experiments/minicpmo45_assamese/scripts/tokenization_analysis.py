"""§4 — Assamese tokenization expansion analysis (CPU-feasible, real).

Answers the prerequisite question: can MiniCPM-o's backbone tokenizer represent
Assamese efficiently, or does it blow up tokens/char (which would make any Assamese
training expensive and hurt fidelity)?

MiniCPM-o 2.6's documented backbone tokenizer is the Qwen2.5 BPE family. We load
Qwen2.5-0.5B's tokenizer (small download, tokenizer files only — NO model weights)
as the faithful proxy for the backbone BPE, and measure tokens-per-character on a
real parallel Assamese / English / Hindi sentence set. Result is written to
../tokenization.json. NLLB-200's tokenizer (cached, covers asm_Beng) is used as a
multilingual reference baseline for context.

Honest labelling: this is the Qwen2.5 backbone tokenizer family. Re-verify the
exact `tokenizer.json` from the MiniCPM-o 4.5 checkpoint on the GPU box (it adds
audio/vision special tokens, but the Assamese text BPE behaviour is what matters
here and is governed by the Qwen2.5 vocab).

  PYTHONPATH=src python experiments/minicpmo45_assamese/scripts/tokenization_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "minicpmo45_assamese"
sys.path.insert(0, str(REPO / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# Real parallel Assamese / English / Hindi sentences (common, non-copyright
# factual/calendrical phrases; no creative content). Used ONLY for tokenization
# measurement, never for training.
SENTENCES = [
    ("আজি বুধবাৰ আৰু বতৰ বৰষুণৰ", "Today is Wednesday and the weather is rainy", "आज बुधवार है और मौसम बारिशी है"),
    ("মই গুৱাহাটীলৈ যাওঁ", "I am going to Guwahati", "मैं गुवाहाटी जा रहा हूँ"),
    ("তেওঁৰ নাম ৰাম আৰু তেওঁ ডিব্ৰুগড়ত থাকে", "His name is Ram and he lives in Dibrugarh", "उनका नाम राम है और वे डिब्रूगढ़ में रहते हैं"),
    ("এই কিতাপৰ দাম দুশ টকা", "This book costs two hundred rupees", "इस किताब की कीमत दो सौ रुपये है"),
    ("মই পানী খাওঁ", "I drink water", "मैं पानी पीता हूँ"),
    ("সোমাল একেটা প্ৰশ্ন কৰিছিল", "Somal asked the same question", "सोमल ने वही सवाल पूछा था"),
    ("বিহাৰে স্কুলৰ বাবে গাড়ী লওক", "Tomorrow take the bus for school", "कल स्कूल के लिए बस लो"),
    ("জনপ্ৰতিনিধি সভাৰ বৈঠক শুক্ৰবাৰে", "The assembly meeting is on Friday", "विधान सभा की बैठक शुक्रवार को है"),
]


def measure(tokenizer, sentences):
    rows = []
    for as_text, en_text, hi_text in sentences:
        as_ids = tokenizer(as_text)["input_ids"]
        en_ids = tokenizer(en_text)["input_ids"]
        hi_ids = tokenizer(hi_text)["input_ids"]
        rows.append({
            "as": {"chars": len(as_text), "tokens": len(as_ids),
                   "tokens_per_char": round(len(as_ids) / max(1, len(as_text)), 3),
                   "text": as_text},
            "en": {"chars": len(en_text), "tokens": len(en_ids),
                   "tokens_per_char": round(len(en_ids) / max(1, len(en_text)), 3),
                   "text": en_text},
            "hi": {"chars": len(hi_text), "tokens": len(hi_ids),
                   "tokens_per_char": round(len(hi_ids) / max(1, len(hi_text)), 3),
                   "text": hi_text},
            "as_over_en": round((len(as_ids) / max(1, len(as_text)))
                                 / (len(en_ids) / max(1, len(en_text))), 3),
            "as_over_hi": round((len(as_ids) / max(1, len(as_text)))
                                 / (len(hi_ids) / max(1, len(hi_text))), 3),
        })
    def avg(key, sub):
        return round(sum(r[sub][key] for r in rows) / len(rows), 3)
    summary = {
        "as_tokens_per_char_avg": avg("tokens_per_char", "as"),
        "en_tokens_per_char_avg": avg("tokens_per_char", "en"),
        "hi_tokens_per_char_avg": avg("tokens_per_char", "hi"),
        "as_over_en_avg": round(sum(r["as_over_en"] for r in rows) / len(rows), 3),
        "as_over_hi_avg": round(sum(r["as_over_hi"] for r in rows) / len(rows), 3),
    }
    return rows, summary


def main():
    out = {"captured_at": "2026-08-13", "sentences": len(SENTENCES),
           "note": "Qwen2.5 BPE = MiniCPM-o 2.6 backbone tokenizer family. "
                   "Re-verify exact tokenizer.json from the MiniCPM-o 4.5 checkpoint on GPU box."}

    # Primary: Qwen2.5 backbone tokenizer (small download, tokenizer files only).
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        out["primary_tokenizer"] = "Qwen/Qwen2.5-0.5B (Qwen2.5 BPE = MiniCPM-o backbone family)"
        rows, summary = measure(tok, SENTENCES)
        out["primary"] = {"rows": rows, "summary": summary}
        print("primary Qwen2.5:", summary)
    except Exception as exc:
        out["primary_tokenizer"] = "FAILED: {}".format(exc)
        print("primary Qwen2.5 tokenizer unavailable:", exc)

    # Reference: NLLB-200 (cached, multilingual, covers asm_Beng).
    try:
        from transformers import AutoTokenizer
        nllb = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                             src_lang="asm_Beng")
        # NLLB needs tgt_lang set to encode; use eng_Latn for the EN rows.
        def nllb_measure(sentences):
            rows = []
            for as_text, en_text, hi_text in sentences:
                as_ids = nllb(as_text, src_lang="asm_Beng")["input_ids"]
                en_ids = nllb(en_text, src_lang="eng_Latn")["input_ids"]
                hi_ids = nllb(hi_text, src_lang="hin_Deva")["input_ids"]
                rows.append({
                    "as_tokens_per_char": round(len(as_ids)/max(1,len(as_text)),3),
                    "en_tokens_per_char": round(len(en_ids)/max(1,len(en_text)),3),
                    "hi_tokens_per_char": round(len(hi_ids)/max(1,len(hi_text)),3),
                })
            def avg(k): return round(sum(r[k] for r in rows)/len(rows),3)
            return {"as_tokens_per_char_avg": avg("as_tokens_per_char"),
                    "en_tokens_per_char_avg": avg("en_tokens_per_char"),
                    "hi_tokens_per_char_avg": avg("hi_tokens_per_char")}
        out["reference_nllb"] = nllb_measure(SENTENCES)
        print("reference NLLB:", out["reference_nllb"])
    except Exception as exc:
        out["reference_nllb"] = "FAILED: {}".format(exc)
        print("reference NLLB unavailable:", exc)

    Path(EXP / "tokenization.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote experiments/minicpmo45_assamese/tokenization.json")


if __name__ == "__main__":
    main()