# Real-recording English ASR pilot (FLEURS, 60 sentences)

## Status and scope

**Acquired and offline-validated; no model inference, no generated labels, no ASR quality result.**
60 distinct FLEURS sentence IDs, one real utterance each: **48 target / 12 control**.
This is a small `en_us` **test** pilot, not full FLEURS and not an Indian-English benchmark.
The recordings total **625.12 seconds** (4.32–19.8 seconds each). All are mono 16 kHz.
Sentence identity is the grouping unit; **speaker independence is not established**. No speaker identities/names are extracted, and no gender is inferred or used in selection. The archived original TSV retains its upstream columns unchanged, including upstream gender labels; the selector ignores that column.

## Official source, immutable revision and license

Only `google/fleurs` is used for dataset metadata/audio. The public HF API resolved the historical script-bearing revision:

`4683b04af03d2d9549064c7d72060a9a94bb6046`

The current head was inspected (`70bb2e84b976b7e960aa89f1c648e09c59f894dd`), but **no current-head dataset bytes are mixed into this pilot**. The historical revision was deliberately chosen because it includes the inspectable original loader; current head no longer includes `fleurs.py`. The loader is preserved as inert text, never executed.

Exact source URLs:

- API: https://huggingface.co/api/datasets/google/fleurs/revision/4683b04af03d2d9549064c7d72060a9a94bb6046
- Card: https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/README.md
- Loader: https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/fleurs.py
- Metadata: https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/en_us/test.tsv
- Archive: https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/en_us/audio/test.tar.gz
- License: https://creativecommons.org/licenses/by/4.0/
- Legal code: https://creativecommons.org/licenses/by/4.0/legalcode.txt

Both the pinned API and card declare **CC BY 4.0** before audio acquisition. `upstream-README.md`, `CC-BY-4.0.txt`, upstream metadata and API evidence are preserved under `data/speech-pilot-v2/`.
The legalcode endpoint returned HTTP 403 in this environment (both `.txt` and `.en.txt` attempted). A previously preserved legalcode copy in the project was used **only for license text**, verified against SHA256 `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`. This is not an alternate dataset or audio source. The pinned dataset card itself supplied the licensing declaration. `provenance.json` records this exception explicitly.

Attribution: Google FLEURS; Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna (2022), *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, https://arxiv.org/abs/2205.12446. Retain this attribution, the upstream card and CC BY 4.0 notice when redistributing. No endorsement is implied.

## Frozen selection before outputs

Seed: `silt-speech-pilot-v2-fleurs-en-test-before-inference-1`.

1. Parse the official test TSV. Eligible rows have numeric sentence ID and numeric `.wav` basename, nonblank transcription, and `0 < num_samples <= 320000` (20 seconds at 16 kHz).
2. Sort eligible utterances by SHA256 of `seed|utterance|sentence_id|filename` (filename tie-break). Take the first for each sentence ID.
3. Sort the resulting sentence population by SHA256 of `seed|sentence|sentence_id` (ID tie-break). Select the first 60.
4. Assign first 48 to target and final 12 to control, once, before any audio outputs/inference. These are disjoint sentence subsets of the same English test population, not a separate demographic population.

The eligible pool was **637 utterances / 347 distinct sentence IDs**. `selection.json` was written before downloading the archive. Selection SHA256 (canonical JSON rows):

`4c749163be603e74ad1afdfe833014e7194cfbea86e352be1b82699821785295`

References are the dataset's original **`transcription` column (TSV index 3)**, not model-generated labels. The upstream `raw_transcription` (index 2) is preserved separately; it is not silently substituted for the reference. No outcome-based filtering is performed. Source silence, bad format or sample mismatch causes a stop, not replacement by a different sentence.

## Bounded archive handling and audio transformation

Preflight declared archive size: **289,851,356 bytes**, below the strict decimal **1,000,000,000-byte** cap. Disk free before download: **20,434,493,440 bytes**, above the 2,000,000,000-byte requirement. Actual downloaded archive bytes matched the declaration exactly.

- Archive LFS SHA256: `d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356`
- TSV SHA256: `74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca`
- TSV Git blob OID: `bcd7e3e4ecb8ad2a4d9754fc6be14dd6d3364b54`
- **TSV is not LFS at this revision**: metadata LFS digest is explicitly `null`, not invented. Its Git blob ID and SHA256 are verified; the archive has a verified LFS digest.

The download is streamed to a bounded temporary archive outside source, hashed and checked before parsing. No `tar.extractall` is used. The streaming tar reader checks traversal/absolute paths, rejects links/devices, limits expanded member bytes to 1 GB and members to 10,000, and writes only exact `test/<selected numeric filename>.wav` regular files. Each selected WAV is bounded to 2 MB. Destination and files are exclusively created, never overwritten. **648 members / 408,863,126 expanded member bytes** were scanned; only 60 original recordings were retained. The compressed corpus archive was deleted after processing.

Actual audio is outside the code tree:

`/agent/workspace/silt-pass2-data/fleurs-en/`

- `originals/<upstream numeric filename>.wav`: **60 original IEEE float32 WAVs**, 40,011,160 bytes total.
- `fleurs_en_<sentence_id>.wav`: **60 runtime PCM16 WAVs**, 20,006,480 bytes total.

Conversion uses Python stdlib: sample-wise `round(x*32768)` (ties-to-even), saturate to `[-32768,32767]`, canonical mono 16 kHz PCM16 RIFF. No resampling, trimming, silence synthesis, gain normalization, denoising, or text changes. All source samples were finite; zero source samples were outside [-1,1]. The converter rejects all-zero PCM output. Per-file original/output SHA256, sample counts, byte counts, duration, source encoding and transformation are in `manifest.json`.

Ordered output-digest-list SHA256:
`9a4b4c2cb9bfe59e20f60a4fd912efa19698a1a9a62856440af3734c8e28e00b`

## Evaluation contract: narrow, not a quality guarantee

`evaluation-suite.json` validates against the existing `EvaluationSuite`:

- `policy_version: composition-admission-v2`
- `claims: [reference_text]`
- `metric: word_error_rate`, per-case `threshold: 0.5`
- Absolute PCM16 `input_file` paths, upstream transcription references.

The **0.5 threshold is the existing hard admission scope**, not a claim of acceptable ASR quality or a guarantee. Keep target/control outcomes distinct and apply control guardrails per case. Do not claim progress by permuting/repeating the same sentences or counting reruns as new population evidence. Freeze models, decoding, text normalization and budgets consistently across candidates; report all 60 intended cases.

**All-errors-count framing:** runtime errors, timeouts, missing/empty hypotheses and malformed outputs are failures, not silently omitted successful-case-only denominators. For a missing transcript, an explicitly documented empty hypothesis gives deletions equal to the reference word count; retain the separate operational failure flag as well. Later parent evaluation must report micro/corpus WER as `sum(S+D+I) / sum(reference words)` over the fixed intended cases (and separately over target/control), not average per-utterance WER. WER can exceed 1 from insertions; do not clamp it. Do not label the suite's mean threshold pass rate as corpus WER. No such evaluation was run here.

## Reproduce / check, without inference

From the source root with its existing Python environment:

```sh
PYTHONPATH=src /agent/workspace/silt-venv/bin/python scripts/prepare_speech_pilot.py --check
PYTHONPATH=src /agent/workspace/silt-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_speech_pilot.py
```

Result: **15 tests passed**; offline `--check` passed for all 60 originals/PCM16 files, hashes, regenerated selection, references and suite schema. Tests use only tiny explicitly synthetic fixtures; these are not dataset audio. No model tool is imported or invoked.

A fresh acquisition refuses existing directories, including already-shipped metadata. To acquire again from the official pinned source while keeping supplied evidence intact, choose a new metadata destination and a new external audio destination:

```sh
PYTHONPATH=src /agent/workspace/silt-venv/bin/python scripts/prepare_speech_pilot.py --fetch \
  --output /tmp/fleurs-en-refetch-metadata \
  --audio-dir /agent/workspace/silt-pass2-data/fleurs-en-refetch
PYTHONPATH=src /agent/workspace/silt-venv/bin/python scripts/prepare_speech_pilot.py --check \
  --output /tmp/fleurs-en-refetch-metadata
```

The newly generated suite uses the new audio paths. The original selection/transcripts/hashes remain the same. Network credentials, HF SDK, dataset-loader execution and model inference are unnecessary.

Stop conditions: archive declared size >1 GB, disk <2 GB, missing license declaration, wrong revision, blocked archive, size/digest mismatch, unsafe/oversized archive member, duplicate/missing selection, bad/nonfinite/silent audio, or mismatched sample count. Archive-phase failure records `failure.json` including actual bytes transferred and partial PCM count, retains partial selected files for inspection, removes the temporary archive, and **does not substitute another source**. Existing directories must be inspected rather than overwritten. Preflight failures raise before an audio download. The only encountered blockage was the separately documented legalcode HTTP 403; official pinned dataset/audio acquisition succeeded.

**Packaging:** `data/speech-pilot-v2/**` is metadata/text evidence only. Keep `/agent/workspace/silt-pass2-data/fleurs-en/**` outside the source/code ZIP. Do not add it with an overly broad workspace archive command. Users can retrieve identical licensed recordings using the script. No Git or unrelated root modifications are part of this task.
