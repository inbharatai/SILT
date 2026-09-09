# Local matched vision-language runtime (experimental)

This path uses **one matched vision-language checkpoint**, not a standalone image encoder
substituted for a language model. The intended small local checkpoint is
`HuggingFaceTB/SmolVLM-256M-Instruct`, whose Transformers architecture is
`Idefics3ForConditionalGeneration`, with its matching `Idefics3Processor` loaded through
`AutoProcessor`. A local directory must already contain complete **safetensors** weights,
model configuration, tokenizer and processor assets. The runtime does not download models.

## Dependency and execution contract

- CPU Torch >=2.6; **Transformers exactly 4.51.3 for vision**; Pillow; NumPy;
  Accelerate for `low_cpu_mem_usage=True`; Safetensors. Existing text/audio adapters retain
  their Transformers 4.51.x contract. SciPy is needed only for audio resampling.
- `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`; offline and
  telemetry-disable environment flags. No custom model code, network image URLs, pipelines,
  pickle checkpoints, standalone encoder fallback, or model downloads.
- All model adapters use **float32 on CPU**, not a global bf16/fp16 default, and
  `low_cpu_mem_usage=True` to avoid the normal duplicate loading allocation. Start CPU text
  trials with <=135M-parameter models; this is operational guidance, not an enforced
  text parameter-count cap. Vision is the separate 256M matched model target.
- Vision explicitly requests the slow/Pillow processor (`use_fast=False`), so torchvision
  is not required. Native vision-class imports were verified with torchvision absent.
- At most two Torch intra-op threads (schema limit). Image graphs use the smaller of the
  configured RSS budget and 4096 MiB; default is 3800 MiB. This is **cooperative measurement**,
  not an OS-enforced 4 GiB memory sandbox. Processor allocation and generation can transiently
  exceed a budget before a check; wall-clock enforcement is also cooperative. Use an external
  process/container hard limit when strict isolation is required. These numbers do not
  establish feasibility or model quality on any machine.
- Unix process lifetime high-water RSS is recorded. Where `resource.getrusage` is unavailable
  (including Windows), runtime import works but execution is explicitly **BLOCKED**;
  `process_peak_rss_mb` is null and `measurement_status` is `unavailable`, never fabricated 0.
  This does not claim every other workspace/filesystem operation is Windows-portable.

## Input and graph shape

Parent-owned schema/CLI integration must admit:

```json
{
  "schema_version": 1,
  "name": "local-smolvlm",
  "input_type": "image",
  "nodes": [{
    "id": "describe",
    "input": "$input",
    "component": {
      "schema_version": 1,
      "kind": "hf_vision",
      "task": "smolvlm",
      "model_path": "/absolute/local/SmolVLM-256M-Instruct",
      "risk": "medium",
      "provenance": [{
        "source": "HuggingFaceTB/SmolVLM-256M-Instruct; record your actual local revision",
        "risk": "medium",
        "license": "verify the downloaded revision's license"
      }]
    },
    "max_new_tokens": 64
  }],
  "output_node": "describe"
}
```

`hf_vision` has `image -> text` ports. It can feed an existing text or TTS node in the
single-input chain. Runtime API:

```python
run_spec(workspace, spec, input_file="/absolute/local/photo.png")
run_spec(workspace, spec, text="What objects are visible?", input_file="/absolute/local/photo.png")
```

The corresponding CLI `run` arguments are `--input-file /absolute/local/photo.png` plus
optional `--input 'What objects are visible?'` (CLI is maintained separately). Omitted prompt
means exactly **`Describe the image.`**; an explicitly supplied empty prompt stays empty.
`prompt_prefix` is prepended inside the vision adapter and bound in graph/config identity.
Both user prompt and image sha256/size/dimensions/path are bound in `input_digest`.
Changing either image bytes or prompt changes that digest. Input bytes are rechecked against
the recorded digest before decoding for inference, then the exact in-memory pixels are used.

Only single-frame PNG, JPEG or WebP rasters are accepted. SVG/SVGZ and other formats are
rejected. Inputs must fit `limits.max_input_bytes` (default 16 MiB), **4,000,000 pixels**,
and **4096 pixels per side**. Pillow verifies the encoded image before pixel loading;
decompression warnings, invalid/truncated files and animated images are blocked. Only RGB
pixels survive the copy into the processor. EXIF/PNG metadata is not converted to prompt
text and EXIF orientation is not applied. Naturally visible text remains image content,
not permission to take an external action.

The processor chat message is one user message with `[{"type":"image"},
{"type":"text","text":prompt}]`. Its chat template is applied with generation prompt enabled;
the processor then receives rendered text and the actual PIL image. Both `input_ids` and
`pixel_values` are required. Native Idefics3 generates deterministically (`do_sample=False`,
`num_beams=1`, fixed seed, `use_cache=False`) up to the node's `max_new_tokens` (schema 1–512).
Only generated tokens after the input prefix are decoded. The adapter rejects combined
image/text tokens plus requested output beyond min(native context, 4096); it does not silently
truncate input. Deterministic options are not a promise of bit-identical outputs across all
Torch versions/CPU implementations.

## Existing text/audio behavior and loading integrity

Text models with a tokenizer chat template now use one user message plus an assistant
generation prompt. The rendered text is tokenized without adding duplicate special tokens.
Tokenizers without a chat template retain the plain-text path. Audio still requires a WAV
`input_file` and no text prompt. Text-only graphs still require text and no `input_file`.

`_load_model` blocks unexpected, mismatched, errored or missing tensors, including random
initialization caused by an incomplete checkpoint. The only missing-key exceptions are a
small **native-class alias whitelist** for a tied output embedding:
Whisper `proj_out.weight -> model.decoder.embed_tokens.weight`, native Idefics3
`lm_head.weight -> model.text_model.embed_tokens.weight`, and native Llama
`lm_head.weight -> model.embed_tokens.weight`. Each exception additionally requires the
native class/module, declared tied embeddings, identical non-meta Parameter objects,
a non-missing source, and that source tensor's presence in the local safetensors checkpoint.
An alias name or a tie between two randomly initialized parameters is not sufficient.
Vits has no missing-key exception. This supplements Transformers' own native tied-weight
loading handling; it does not whitelist arbitrary ignored-key regular expressions.

## Identity and parent integration requirements

Runtime semantics are explicitly versioned as `composition-cpu-v2-chat-smolvlm`.
Candidates/runs record `runtime_version` and `config_hash`. Current formulas are:

```
config_hash = digest({"spec": spec_json, "runtime_version": RUNTIME_VERSION})
graph_hash = digest({"spec": spec_json, "artifacts": artifacts,
                     "runtime_version": RUNTIME_VERSION})
```

This intentionally changes graph identity for old experimental graphs. **Certification's
independent graph recomputation must adopt this formula and check the runtime version**;
otherwise fresh candidates will be rejected there. Historical evidence must not be relabeled
as evidence for the changed prompt/runtime behavior. Do not merely omit the version from
verification to make old evidence pass.

Schema changes (not made by this runtime patch): add `hf_vision` to `Kind`, `smolvlm` to task
literals and kind/task matching, `hf_vision: ("image", "text")` to `PORTS`, and `image` to
composition input types. Evaluation currently requires exactly one of `input`/`input_file`;
custom image prompts need **both**. Let an evaluation case carry both fields, then enforce
text/audio/image-specific combinations against its spec during evaluation. With the old
exactly-one rule, image cases can only use the default prompt via `input_file`.

**Artifact validator review required:** the registered directory must cover and hash all
processor dependencies (`preprocessor_config.json`, `processor_config.json`, any image
processor config, chat template assets and tokenizer files). A closed file/config allowlist
may reject these today; do not weaken traversal, symlink, safetensors or remote-code guards
to make the processor load. Inspect every permitted dependency/reference in the hashed root.
The runtime only checks model_type and native processor identity; a strict importer is still
responsible for the full registered artifact boundary and authentic local model provenance.

## Evidence limits

`tests/test_vision_runtime.py` contains clearly labeled offline UNIT mocks, image validation,
resource failure, input/runtime digest and generation/loading-kwargs tests. Tiny locally
created random Whisper/Vits checkpoints verify native safetensors loading only, **not**
pretrained ASR/TTS quality. No pretrained SmolVLM checkpoint was downloaded or run for this
patch. There is no real-image benchmark score, latency claim, admission or certification.
Logs live under `/agent/workspace/silt-evidence/lens-*`.

Text exact/similarity metrics can evaluate an image model's textual answer against a fixed
reference, but similarity is not visual grounding or image understanding certification.
Use independently defined image/question/reference cases and held-out controls; record actual
pretrained revision, processor inventory, image bytes, prompt, runtime version and measured
resources. Avoid asserting that a passing fake or synthetic loading test proves real model
performance or safe behavior.
