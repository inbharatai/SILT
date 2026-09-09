"""Sequential CPU-only execution of closed local adapters. Optional ML imports are lazy."""
import contextlib
import gc
import hashlib
import io
import math
import json
import os
from pathlib import Path
import sys
import signal
import threading
import time
import warnings
import wave

try:
    import resource
except ImportError:  # Windows: importable, but execution must fail closed without RSS.
    resource = None

from asea.artifacts import Blocked, Workspace, atomic_json, digest, file_hash, safe_file, safe_path
from .schema import CompositionSpec, PORTS, dump_spec
from .generation_contract import (TRACE_SCHEMA, preflight_node, render_prompt, cache_kwargs,
                                  new_trace, observe, observe_call, observe_tokens, observe_media, observe_prompt,
                                  capture_traces, append_trace, failure_metadata, exception_is, primary_error)


RUNTIME_VERSION = "composition-cpu-v2-chat-smolvlm"
DEFAULT_IMAGE_PROMPT = "Describe the image."
MAX_IMAGE_PIXELS = 4_000_000
MAX_IMAGE_DIMENSION = 4096


def _prepare(workspace, spec, allow_fixtures=False):
    if any(n.component.kind == "fixture_text" for n in spec.nodes) and not allow_fixtures:
        raise Blocked("fixture adapters are unit-test-only and disabled in public CLI")
    for node in spec.nodes:
        try:
            preflight_node(node)
        except ValueError as exc:
            raise Blocked(str(exc)) from exc
    artifacts = {node.id: workspace.register(node.component) for node in spec.nodes}
    config_hash = digest({"spec": dump_spec(spec), "runtime_version": RUNTIME_VERSION})
    graph_hash = digest({"spec": dump_spec(spec), "artifacts": artifacts,
                         "runtime_version": RUNTIME_VERSION})
    identifier = "candidate-" + graph_hash
    candidate = {"schema_version": 1, "id": identifier, "status": "candidate", "graph_hash": graph_hash,
                 "spec": dump_spec(spec), "artifacts": artifacts,
                 "runtime_version": RUNTIME_VERSION, "config_hash": config_hash,
                 "risk": spec.effective_risk,
                 "fixture_only": any(n.component.kind == "fixture_text" for n in spec.nodes)}
    workspace.write_record("candidates", identifier, candidate)
    return candidate


def inspect_spec(workspace, spec):
    with workspace.writer():
        candidate = _prepare(workspace, spec)
        return {**candidate, "order": [n.id for n in spec.topological()],
                "output_type": PORTS[next(n for n in spec.nodes if n.id == spec.output_node).component.kind][1]}


def _audio_info(path, limits):
    path = safe_file(path)
    if path.stat().st_size > limits.max_input_bytes:
        raise Blocked("audio file exceeds byte limit")
    with wave.open(str(path), "rb") as stream:
        channels, width, rate, frames = stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()
        if channels not in (1, 2) or width != 2 or not 8000 <= rate <= 96000:
            raise Blocked("audio input must be 16-bit PCM WAV, mono/stereo, 8-96 kHz")
        if not frames or frames / rate > limits.max_audio_seconds:
            raise Blocked("audio duration outside configured limits")
    return {"path": str(path), "sample_rate": rate, "duration_seconds": frames / rate, **file_hash(path)}


def _image_payload(path, limits):
    """Verify a bounded raster snapshot; metadata is never interpreted as a prompt."""
    path = safe_file(path)
    if path.suffix.lower() in {".svg", ".svgz"}:
        raise Blocked("SVG image input is forbidden")
    if path.stat().st_size > limits.max_input_bytes:
        raise Blocked("image file exceeds byte limit")
    with path.open("rb") as stream:
        payload = stream.read(limits.max_input_bytes + 1)
    if not payload or len(payload) > limits.max_input_bytes:
        raise Blocked("image file empty or exceeds byte limit")
    try:
        from PIL import Image
    except ImportError as exc:
        raise Blocked("image input requires Pillow") from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise Blocked("image input must be PNG, JPEG or WebP (no SVG)")
                if (min(width, height) < 1 or max(width, height) > MAX_IMAGE_DIMENSION
                        or width * height > MAX_IMAGE_PIXELS):
                    raise Blocked("image exceeds dimension/pixel limits (4096 per side, 4M pixels)")
                if getattr(image, "n_frames", 1) != 1:
                    raise Blocked("animated/multiframe images are not supported")
                image.verify()
    except Blocked:
        raise
    except Exception as exc:
        raise Blocked("invalid or unsafe raster image") from exc
    return payload, {"path": str(path), "width": width, "height": height,
                     "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def _image_info(path, limits):
    return _image_payload(path, limits)[1]


def _load_image(value, limits):
    payload, info = _image_payload(value["path"], limits)
    from PIL import Image
    if any(value.get(key) != info[key] for key in ("sha256", "size", "width", "height")):
        raise Blocked("image changed after input digest was recorded")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            # Copy only decoded RGB pixels: no EXIF orientation or metadata instructions.
            rgb = image.convert("RGB")
            return Image.frombytes("RGB", rgb.size, rgb.tobytes())
    except Exception as exc:
        raise Blocked("image pixel decoding failed") from exc


def _load_audio(path, limits, trace=None):
    info = _audio_info(path, limits)
    import numpy as np
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        pcm_bytes = stream.readframes(stream.getnframes())
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        def audio_facts(evidence):
            sample_count = len(pcm_bytes) // (stream.getsampwidth() * channels)
            observe_media(evidence, input_audio_sample_count=sample_count,
                          input_audio_sample_rate=stream.getframerate(), input_audio_channels=channels,
                          input_audio_duration_seconds=sample_count / stream.getframerate(),
                          decoded_pcm_sha256=hashlib.sha256(pcm_bytes).hexdigest(), decoded_pcm_bytes=len(pcm_bytes),
                          source_file_binding=info)
        observe(trace, audio_facts)
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    if info["sample_rate"] != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        divisor = gcd(info["sample_rate"], 16000)
        samples = resample_poly(samples, 16000 // divisor, info["sample_rate"] // divisor)
    return samples


def _native_tied_aliases(model, path, missing):
    """Only documented native aliases backed by a loaded checkpoint tensor may pass."""
    aliases = {
        ("transformers.models.whisper.modeling_whisper", "WhisperForConditionalGeneration"):
            {"proj_out.weight": "model.decoder.embed_tokens.weight"},
        ("transformers.models.idefics3.modeling_idefics3", "Idefics3ForConditionalGeneration"):
            {"lm_head.weight": "model.text_model.embed_tokens.weight"},
        ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"):
            {"lm_head.weight": "model.embed_tokens.weight"},
    }.get((type(model).__module__, type(model).__name__), {})
    if not missing or not set(missing).issubset(aliases) or not getattr(model.config, "tie_word_embeddings", False):
        return False
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for alias in missing:
        source = aliases[alias]
        if (source in missing or alias not in parameters or source not in parameters
                or parameters[alias] is not parameters[source] or parameters[source].is_meta):
            return False
        # Being tied to another newly initialized parameter is NOT sufficient.
        try:
            from safetensors import safe_open
            checkpoint = Path(path) / "model.safetensors"
            if not checkpoint.exists():
                index = safe_file(Path(path) / "model.safetensors.index.json")
                if index.stat().st_size > 2 * 1024 * 1024:
                    return False
                relative = json.loads(index.read_text())["weight_map"][source]
                if Path(relative).is_absolute() or ".." in Path(relative).parts or "\\" in relative:
                    return False
                checkpoint = Path(path) / relative
            with safe_open(str(safe_file(checkpoint)), framework="pt", device="cpu") as stored:
                if source not in stored.keys():
                    return False
        except (ImportError, OSError, ValueError, KeyError):
            return False
    return True


def _load_model(model_class, path, weights):
    model, loading = model_class.from_pretrained(path, output_loading_info=True, **weights)
    missing = loading.get("missing_keys", [])
    bad = any(loading.get(key) for key in ("unexpected_keys", "mismatched_keys", "error_msgs"))
    if bad or (missing and not _native_tied_aliases(model, path, missing)):
        del model
        gc.collect()
        raise Blocked("checkpoint does not exactly match the selected model architecture; random/unmatched weights refused")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise Blocked("checkpoint left unmaterialized model parameters")
    return model.cpu().eval()


def _adapter(node, value, output_path, limits, allow_fixtures):
    trace = None
    try:
        trace = new_trace(node, value)
    except BaseException:
        pass  # Nonstandard test doubles may not expose metadata; never affect generation.
    try:
        try:
            preflight_node(node)
        except ValueError as exc:
            raise Blocked(str(exc)) from exc
        result = _adapter_impl(node, value, output_path, limits, allow_fixtures, trace)
        if trace is not None:
            trace.update(status="succeeded", stage="complete")
        return result
    except BaseException as exc:
        if trace is not None:
            try:
                # Establish terminal evidence before collecting optional metadata.
                trace["status"] = "failed"
                trace["status"] = ("cancelled" if exception_is(exc, (KeyboardInterrupt, SystemExit))
                                   else "blocked" if exception_is(exc, (Blocked,)) else "failed")
                trace["failure"] = failure_metadata(exc)
            except BaseException:
                pass
        raise
    finally:
        if trace is not None:
            try:
                append_trace(trace)
            except BaseException:
                pass


def _adapter_impl(node, value, output_path, limits, allow_fixtures, trace):
    if trace is not None:
        trace["stage"] = "loading"
    if node.component.kind == "fixture_text":
        if not allow_fixtures:
            raise Blocked("fixtures disabled")
        config = json.loads(safe_file(Path(node.component.model_path) / "fixture.json").read_text())
        if set(config) != {"prefix"} or not isinstance(config["prefix"], str):
            raise Blocked("invalid unit fixture")
        return {"type": "text", "text": config["prefix"] + value["text"]}
    config_path = safe_file(Path(node.component.model_path) / "config.json")
    if config_path.stat().st_size > 2 * 1024 * 1024:
        raise Blocked("model config exceeds 2 MiB")
    model_config = json.loads(config_path.read_text())
    # Hash only small known configuration files; no weights/tensors copied.
    def config_evidence(evidence):
        files = {}
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "preprocessor_config.json", "processor_config.json"):
            source = config_path.parent / name
            if source.exists() and source.stat().st_size <= 2 * 1024 * 1024:
                files[name] = file_hash(safe_file(source))
        evidence["config_files"] = files
        evidence["config_metadata_sha256"] = digest(files)
    observe(trace, config_evidence)
    if node.component.kind == "hf_vision" and node.component.task != "smolvlm":
        raise Blocked("vision adapter supports only the matched smolvlm task")
    expected_type = {"hf_asr": "whisper", "hf_tts": "vits", "hf_vision": "idefics3"}.get(node.component.kind)
    if expected_type and model_config.get("model_type") != expected_type:
        raise Blocked("checkpoint model_type does not match selected adapter")
    if node.component.kind == "hf_text" and bool(model_config.get("is_encoder_decoder", False)) != (node.component.task == "seq2seq"):
        raise Blocked("checkpoint encoder/decoder configuration does not match text task")
    # Prevent online resolution, telemetry and implicit model code execution.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch
        import transformers
        import numpy as np
    except ImportError as exc:
        raise Blocked("optional local inference dependencies missing: install CPU torch>=2.6, transformers 4.51.x (and scipy for resampling)") from exc
    # No GPU, remote code, pickle weights, pipelines, subprocesses or dynamic plugin lookup.
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 6):
        raise Blocked("torch >=2.6 required")
    if not transformers.__version__.startswith("4.51."):
        raise Blocked("validated adapter contract requires transformers 4.51.x")
    if node.component.kind == "hf_vision" and transformers.__version__ != "4.51.3":
        raise Blocked("SmolVLM adapter requires transformers 4.51.3")
    torch.set_num_threads(limits.threads)
    transformers.utils.logging.set_verbosity_error()
    local = {"local_files_only": True, "trust_remote_code": False}
    if node.component.kind in {"hf_asr", "hf_tts"} and node.dtype != "float32":
        raise Blocked("speech adapters require float32; other precisions are not validated")
    weights = {**local, "use_safetensors": True, "torch_dtype": getattr(torch, node.dtype),
               "low_cpu_mem_usage": True}
    path = str(safe_path(node.component.model_path))
    model = tokenizer = processor = inputs = generated = None
    try:
        with torch.inference_mode():
            if node.component.kind == "hf_text":
                tokenizer = transformers.AutoTokenizer.from_pretrained(path, **local)
                model_class = transformers.AutoModelForCausalLM if node.component.task == "causal" else transformers.AutoModelForSeq2SeqLM
                model = _load_model(model_class, path, weights)
                context = min(2048, getattr(model.config, "max_position_embeddings", 2048), getattr(tokenizer, "model_max_length", 2048))
                input_limit = max(1, context - node.max_new_tokens) if node.component.task == "causal" else context
                prompt = render_prompt(node, value["text"])
                templated = bool(getattr(tokenizer, "chat_template", None))
                if templated:
                    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                           tokenize=False, add_generation_prompt=True)
                observe(trace, observe_prompt, prompt, templated)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=False,
                                   **({"add_special_tokens": False} if templated else {}))
                if inputs["input_ids"].shape[-1] > input_limit:
                    raise Blocked("tokenized input exceeds model/context budget; input is not silently truncated")
                kwargs = {"max_new_tokens": node.max_new_tokens, "do_sample": False, "num_beams": 1, **cache_kwargs(node)}
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    kwargs["pad_token_id"] = tokenizer.eos_token_id
                observe(trace, observe_call, model, inputs, kwargs, node)
                generated = model.generate(**inputs, **kwargs)
                observe(trace, observe_tokens, generated, model, node, inputs["input_ids"].shape[-1])
                tokens = generated[0, inputs["input_ids"].shape[-1]:] if node.component.task == "causal" else generated[0]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                return {"type": "text", "text": text}
            if node.component.kind == "hf_vision":
                image = _load_image(value, limits)
                try:
                    processor = transformers.AutoProcessor.from_pretrained(path, use_fast=False, **local)
                    if type(processor) is not transformers.Idefics3Processor:
                        raise Blocked("SmolVLM requires the native matched Idefics3 processor")
                    prompt = node.prompt_prefix + value.get("text", DEFAULT_IMAGE_PROMPT)
                    if len(prompt) > limits.max_input_chars:
                        raise Blocked("vision prompt including prefix exceeds input character limit")
                    messages = [{"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": prompt}]}]
                    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    observe(trace, observe_prompt, rendered, True)
                    # PIL pixels, not an image URL or arbitrary processor-supplied chat media.
                    inputs = processor(text=rendered, images=[image], return_tensors="pt",
                                       add_special_tokens=False)
                    if "pixel_values" not in inputs or "input_ids" not in inputs:
                        raise Blocked("matched vision processor did not produce image and text tensors")
                    model = _load_model(transformers.Idefics3ForConditionalGeneration, path, weights)
                    context = min(4096, getattr(model.config.text_config, "max_position_embeddings", 4096))
                    if inputs["input_ids"].shape[-1] + node.max_new_tokens > context:
                        raise Blocked("vision image/prompt tokens exceed context budget; no silent truncation")
                    if _rss_mb() > min(limits.max_peak_rss_mb, 4096.0):
                        raise Blocked("measured vision memory budget exceeded after loading")
                    torch.manual_seed(0)
                    kwargs = {"max_new_tokens": node.max_new_tokens, "do_sample": False, "num_beams": 1, **cache_kwargs(node)}
                    observe(trace, observe_call, model, inputs, kwargs, node)
                    def vision_facts(evidence):
                        facts = {"image_width": image.width, "image_height": image.height}
                        mask = inputs.get("pixel_attention_mask")
                        if mask is not None and len(mask.shape) == 4:
                            facts["observed_nonempty_image_tile_count"] = int(mask.bool().any(dim=-1).any(dim=-1).sum().item())
                        observe_media(evidence, **facts)
                    observe(trace, vision_facts)
                    generated = model.generate(**inputs, **kwargs)
                    observe(trace, observe_tokens, generated, model, node, inputs["input_ids"].shape[-1])
                    tokens = generated[:, inputs["input_ids"].shape[-1]:]
                    return {"type": "text", "text": processor.batch_decode(tokens, skip_special_tokens=True)[0]}
                finally:
                    image.close()
            if node.component.kind == "hf_asr":
                processor = transformers.WhisperProcessor.from_pretrained(path, **local)
                model = _load_model(transformers.WhisperForConditionalGeneration, path, weights)
                samples = _load_audio(value["path"], limits, trace)
                if len(samples) > 30 * 16000:
                    raise Blocked("Whisper v1 supports at most 30 seconds (no silent truncation)")
                observe(trace, observe_media, processed_audio_sample_count=len(samples), processed_audio_sample_rate=16000,
                        processed_audio_duration_seconds=len(samples) / 16000)
                inputs = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                kwargs = {"max_new_tokens": node.max_new_tokens, "do_sample": False, "num_beams": 1, **cache_kwargs(node)}
                observe(trace, observe_call, model, inputs, kwargs, node)
                generated = model.generate(**inputs, **kwargs)
                observe(trace, observe_tokens, generated, model, node)
                return {"type": "text", "text": processor.batch_decode(generated, skip_special_tokens=True)[0]}
            if node.component.kind == "hf_tts":
                tokenizer = transformers.AutoTokenizer.from_pretrained(path, **local)
                model = _load_model(transformers.VitsModel, path, weights)
                text = node.prompt_prefix + value["text"]
                if len(text) > 1024:
                    raise Blocked("Vits text limited to 1024 characters")
                observe(trace, observe_prompt, text, False)
                inputs = tokenizer(text, return_tensors="pt")
                if inputs["input_ids"].shape[-1] > 512:
                    raise Blocked("Vits input limited to 512 tokens")
                torch.manual_seed(0)
                observe(trace, observe_call, model, inputs, {}, node)
                def tts_call(evidence):
                    evidence["forwarded"].pop("generate_kwargs", None)
                    evidence["forwarded"]["model_call_kwargs"] = {}
                    evidence["call_metadata_sha256"] = digest({"forwarded": evidence["forwarded"], "resolved": evidence["resolved"]})
                observe(trace, tts_call)
                generated = model(**inputs).waveform.squeeze().cpu().numpy()
                rate = int(model.config.sampling_rate)
                def waveform_facts(evidence):
                    observe_media(evidence, output_audio_sample_count=len(generated), output_audio_sample_rate=rate,
                                  output_audio_duration_seconds=len(generated) / rate)
                observe(trace, waveform_facts)
                if generated.ndim != 1 or not len(generated) or not np.isfinite(generated).all() or len(generated) / rate > limits.max_audio_seconds:
                    raise Blocked("TTS output invalid or exceeds duration cap")
                pcm = (np.clip(generated, -1, 1) * 32767).astype("<i2")
                with wave.open(str(output_path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(rate)
                    stream.writeframes(pcm.tobytes())
                return {"type": "audio", **_audio_info(output_path, limits)}
            raise Blocked("unsupported node kind")
    finally:
        del model, tokenizer, processor, inputs, generated
        gc.collect()


def _rss_mb():
    if resource is None:
        raise Blocked("resource measurement unavailable on this platform; RSS budget cannot be verified")
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        measured = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (AttributeError, OSError, ValueError, TypeError) as exc:
        raise Blocked("resource measurement unavailable; RSS budget cannot be verified") from exc
    if not math.isfinite(measured) or measured <= 0:
        raise Blocked("resource measurement invalid; RSS budget cannot be verified")
    return measured


@contextlib.contextmanager
def _managed_run_signals():
    """Main-thread CLI cancellation unwinds normally; SIGKILL needs disk recovery.

    Worker-thread embedding does not change process-global signal handlers.
    This is cooperative cancellation, not a hard inference time/memory limit.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}
    def cancel(signum, frame):
        if signum == signal.SIGINT:
            raise KeyboardInterrupt("run cancelled by SIGINT")
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, cancel)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _run(workspace, spec, text=None, input_file=None, allow_fixtures=False):
    with _managed_run_signals():
        return _execute_run(workspace, spec, text, input_file, allow_fixtures)


def _execute_run(workspace, spec, text=None, input_file=None, allow_fixtures=False):
    candidate = _prepare(workspace, spec, allow_fixtures)
    run_id = workspace.new_id("run")
    started = time.monotonic()
    record = {"schema_version": 1, "id": run_id, "candidate_id": candidate["id"], "graph_hash": candidate["graph_hash"],
              "status": "running", "fixture_only": candidate["fixture_only"], "node_results": [],
              "runtime_version": RUNTIME_VERSION, "config_hash": candidate["config_hash"],
              "generation_trace_schema": TRACE_SCHEMA, "generation_trace_v1": []}
    rss_limit = min(spec.limits.max_peak_rss_mb, 4096.0) if spec.input_type == "image" else spec.limits.max_peak_rss_mb
    output_dir = safe_path(workspace.root / "outputs" / run_id)
    pending = output_dir / "pending.json"
    try:
        output_dir.mkdir()
        # An isolated pending marker survives hard kills without an optimistic
        # signed running record. Recovery emits separate immutable evidence.
        atomic_json(pending, record)
        _rss_mb()  # Fail closed before loading any adapter when measurement is unavailable.
        if spec.input_type == "text":
            if input_file is not None or text is None or len(text) > spec.limits.max_input_chars:
                raise Blocked("text graph requires bounded --input, not --input-file")
            value = {"type": "text", "text": text}
        elif spec.input_type == "image":
            prompt = DEFAULT_IMAGE_PROMPT if text is None else text
            if input_file is None or len(prompt) > spec.limits.max_input_chars:
                raise Blocked("image graph requires --input-file and an optional bounded --input prompt")
            value = {"type": "image", "text": prompt, **_image_info(input_file, spec.limits)}
        elif spec.input_type == "audio":
            if input_file is None or text not in (None, ""):
                raise Blocked("audio graph requires --input-file and no text input")
            value = {"type": "audio", **_audio_info(input_file, spec.limits)}
        else:
            raise Blocked("unsupported graph input type")
        record["input_digest"] = digest(value)
        for node in spec.topological():
            if time.monotonic() - started > spec.limits.max_seconds or _rss_mb() > rss_limit:
                raise Blocked("measured run budget exceeded before next node")
            workspace.verify_artifact(candidate["artifacts"][node.id])
            tick = time.monotonic()
            # Libraries sometimes print during loading. JSON stdout remains clean.
            with contextlib.redirect_stdout(sys.stderr), capture_traces(record["generation_trace_v1"]):
                value = _adapter(node, value, output_dir / (node.id + ".wav"), spec.limits, allow_fixtures)
            workspace.verify_artifact(candidate["artifacts"][node.id])
            if value["type"] == "text" and len(value["text"]) > spec.limits.max_output_chars:
                raise Blocked("output exceeds text limit")
            record["node_results"].append({"node": node.id, "artifact_id": candidate["artifacts"][node.id],
                                            "seconds": time.monotonic() - tick, "output_digest": digest(value)})
        elapsed, peak = time.monotonic() - started, _rss_mb()
        if elapsed > spec.limits.max_seconds or peak > rss_limit:
            raise Blocked("measured run budget exceeded")
        record.update(status="succeeded", output=value)
    except BaseException as exc:
        status = ("cancelled" if exception_is(exc, (KeyboardInterrupt, SystemExit))
                  else "blocked" if exception_is(exc, (Blocked,)) else "failed")
        error, unavailable = primary_error(exc, Blocked)
        record.update(status=status, error=error)
        if unavailable:
            record["error_metadata"] = failure_metadata(exc)
            record["error_metadata"]["message_availability"] = "unavailable_unsafe_formatting"
        record.pop("output", None)
        raise
    finally:
        measurement_error = None
        try:
            peak = _rss_mb()
        except Blocked as exc:
            peak, measurement_error = None, str(exc)
            if record["status"] != "cancelled":
                record.update(status="blocked", error=measurement_error)
            record.pop("output", None)
        record["resources"] = {"wall_seconds": time.monotonic() - started, "process_peak_rss_mb": peak,
                               "threads": spec.limits.threads, "device": "cpu",
                               "measurement": "process lifetime high-water RSS; cooperative wall budget",
                               "measurement_status": "unavailable" if measurement_error else "measured"}
        if measurement_error:
            record["resources"]["measurement_error"] = measurement_error
        workspace.write_record("runs", run_id, record)
        workspace.audit("run", run_id=run_id, candidate_id=candidate["id"], status=record["status"])
        if safe_path(pending).exists():
            pending.unlink()
            fd = os.open(str(output_dir), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        if measurement_error and sys.exc_info()[0] is None:
            raise Blocked(measurement_error)
    return record


def run_spec(workspace, spec, text=None, input_file=None, allow_fixtures=False):
    with workspace.writer():
        return _run(workspace, spec, text, input_file, allow_fixtures)
