# Error analysis (real, from the SILT-proxy run on this CPU machine)

**Framing:** this is the SILT-on-Assamese per-case analysis, NOT a MiniCPM-o
analysis. The receiver is qwen3.5:latest (G2P) and qwen2.5:1.5b-instruct (text),
real weights via ollama. MiniCPM-o per-case analysis is PENDING a GPU box.

## G2P replay — `tts_pronunciation_as` held-out (6 cases), exact reproduction of
`docs/SILT_TTS_G2P_TEST.md` §5.6

`baseline 0.6023 → candidate 0.7359, improvement +0.1336` (PROMOTED).

| case | expected | baseline out | candidate out | base | cand | delta | regressed? |
|---|---|---|---|---|---|---|---|
| **w_xat** | `xat` | `sat` | **`xat`** | 0.523 | **1.000** | **+0.477** | no ← learned স→/x/ |
| **w_kitap** | `kitap` | `kitaap` | **`kitap`** | 0.600 | **1.000** | **+0.400** | no ← exact |
| w_nam | `nam` | `naam` | `namx` | 0.554 | 0.557 | +0.003 | no |
| w_bhal | `bʱal` | `baal` | `bhal` | 0.628 | 0.625 | −0.003 | yes |
| w_mat | `mat` | `mat̪o` | `mɑt` | 0.667 | 0.646 | −0.021 | yes |
| w_kam | `kam` | `kaam` | `kʰam` | 0.642 | 0.587 | −0.055 | yes |

### What the errors show, plainly

- **The two wins carry the run.** Before the SILT-approved `grapheme→phoneme`
  table (including the `স → x` entry the receiver did not know), qwen3.5 wrote
  `সাত` as `sat` (wrong — it used /s/, the Bengali value) and `কিতাপ` as
  `kitaap` (wrong long vowel). After conditioning on the table it wrote `xat`
  (exact) and `kitap` (exact): the receiver grabbed the taught skill and applied
  it to words it had never seen. `task_success` rose 0.0 → 0.333 (2/6 exact).
- **Three cases drifted slightly negative** (w_bhal −0.003, w_mat −0.021,
  w_kam −0.055): reaching for the table, the candidate sometimes re-spelled a
  vowel or consonant (`bhal` vs `bʱal`, `mɑt` vs `mat̪o`, `kʰam` vs `kaam`). The
  gate accepted this because the net lift is real (+0.1336), `no_regression`
  passed, and `case_regression_limit` (3/6 regressed, ratio 0.50 ≤ 1.00) stayed
  in bounds. This is the gate doing its job, not a rubber stamp.
- **w_nam** is essentially flat (+0.003): the candidate wrote `namx` (added a
  spurious `x`, the newly-learned স→/x/ over-applied to a word with no স).

### The single most Assamese-distinctive item

`স → /x/` (the velar fricative) is the grapheme that separates Assamese from
Bengali (where স → /s/). The receiver's baseline error on `w_xat` (`sat` instead
of `xat`) is exactly the Bengali-influence error; the SILT-approved table fixed
it to `xat` (exact, +0.477). This is the clearest "grab-and-learn" signal in the
run and the reason the transfer was worth doing.

## Text transfer — `assamese_english` (as->en): no errors, because no transfer

The fresh `nllb-teacher → qwen2.5:1.5b-instruct` run produced **no extracted
pairs, no evaluations, no per-case errors** — `extracted=0, actionable=0,
measured_gaps=[]`. The gate measured the as->en gap on the 12-item extraction
split and found no actionable headroom (the receiver already translates the
single words well enough that NLLB has nothing to add on them). So there is
nothing to analyse per-case: SILT correctly refused to act. Recorded verbatim in
`metrics.json` → `text_transfer.report`.

### What this implies for the MiniCPM-o A/B/C comparison (hypothesis, not a result)

The text run shows SILT only transfers when a measured gap exists. For MiniCPM-o
the gaps will be measured against the *untouched MiniCPM-o baseline* (Arm A) on
IndicVoices-scale held-out sets, where the gaps are expected to be large (an
omni model with no Assamese speech training). The A/B/C comparison will then
test whether the SILT-gated subset (C) trains at least as well as the raw set
(B) at equal token budget — **reported truthfully, even if B wins.**

## MiniCPM-o per-case analysis: PENDING (GPU box)

Populated by `stt_eval.py` / `tts_eval.py` / `sts_eval.py` on a GPU box, scored
by **independent** ASR + native-speaker MOS (never self-eval). No MiniCPM-o
per-case number appears in this experiment until those run.