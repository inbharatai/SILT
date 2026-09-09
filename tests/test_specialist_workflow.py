"""Local orchestration contracts and tiny RANDOM native-model mechanics only."""
import json
from pathlib import Path
import sys

import pytest

from asea.artifacts import file_hash, model_inventory
from asea.specialist import workflow as w
from asea.specialist import evaluation as e
from asea.specialist.__main__ import main, parser


def put(path, value):
    path.write_text(json.dumps(value))
    return path


def suite_at(path):
    return put(path, {"schema_version": 1, "name": "tiny-original-orchard-harbor-mechanics",
        "reference_source": "original tiny fixtures; no coding quality claim", "claims": ["coding"],
        "cases": [{"id": identifier, "group": group, "input": identifier + " prompt", "reference": "original fixture",
                   "metric": "function_io", "threshold": 1.0, "output_format": "raw_python",
                   "function_cases": [{"id": "one", "function": "f", "args": [1], "kwargs": {}, "expected": 1}]}
                  for identifier, group in (("orchard", "target"), ("harbor", "control"))]})


def recipe_at(root, **updates):
    values = {"source_path": "teacher", "training": "train.json", "calibration": "calibration.json",
              "validation_data": "validation.json", "validation_suite": "validation-suite.json"}
    values.update(updates)
    return put(root / "input-recipe.json", values)


def test_strict_recipe_defaults_and_cli_api_arguments(tmp_path):
    recipe = recipe_at(tmp_path)
    config = w.recipe_config(recipe)
    assert config["max_length"] == config["max_new_tokens"] == 256
    assert config["recovery"]["steps"] == 64 and config["recovery"]["rank"] == 8
    assert config["recovery"]["learning_rate"] == 0.0001 and config["recovery"]["kd_weight"] == 0.7
    assert config["reconstruction"]["retention"] == 0.75
    assert config["dtype"] == "bfloat16"
    assert config["recovery"]["export_mode"] == "native_merged"
    args = parser().parse_args(["recover", "--teacher-dir", "t", "--student-dir", "s", "--output-dir", "o",
        "--training-path", "train", "--validation-path", "dev", "--steps", "3", "--rank", "2",
        "--learning-rate", "0.01", "--kd-weight", "0", "--method", "supervised_lora", "--max-samples", "9",
        "--max-length", "256", "--dtype", "float32", "--seed", "9"])
    assert args.steps == 3 and args.max_samples == 9 and args.max_length == 256
    for bad in ({"final_suite": "final.json"}, {"recovery": {"unknown": 1}}, {"seed": True},
                {"timeout_seconds": 3601}, {"max_new_tokens": 385}, {"encoding": "compose-prefix"},
                {"recovery": {"export_mode": "auto"}}, {"receipt_mode": "compact"},
                {"reconstruction": {"receipt_mode": "compact"}}, {"recovery": {"receipt_mode": True}}):
        with pytest.raises(ValueError):
            w.recipe_config(recipe_at(tmp_path, **bad))


def test_exclusive_reports_and_no_nonfinite(tmp_path):
    path = tmp_path / "out.json"
    w.write_json(path, {"status": "REJECTED"})
    with pytest.raises(FileExistsError):
        w.write_json(path, {"status": "completed"})
    with pytest.raises(ValueError):
        w.write_json(tmp_path / "bad.json", {"score": float("nan")})
    assert not (tmp_path / "bad.json").exists()


def test_real_data_preflight_never_opens_final(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "data/specialist-v1"
    manifest = w._bounded_json(root / "manifest.json")
    files = {k: {"sha256": v} for k, v in manifest["environment"]["tokenizer_files_sha256"].items()}
    config = {k: str(root / n) for k, n in {"training": "train.json", "calibration": "calibration.json",
        "validation_data": "validation.json", "validation_suite": "validation-suite.json",
        "data_manifest": "manifest.json", "selection_lock": "selection-lock.json"}.items()}
    original = w.file_hash
    def guarded(path):
        assert not Path(path).name.startswith("final")
        return original(path)
    monkeypatch.setattr(w, "file_hash", guarded)
    original_read = w._bounded_json
    def guarded_read(path):
        assert not Path(path).name.startswith("final")
        return original_read(path)
    monkeypatch.setattr(w, "_bounded_json", guarded_read)
    result = w.data_preflight(config, files)
    assert result["counts"] == {"train": 42, "validation": 8, "final": 16}
    assert result["validation_tasks"] == 8 and not result["final_opened"]
    wrong = dict(config, training=str(root / "final.json"))
    with pytest.raises(ValueError, match="path/hash|quarantined"):
        w.data_preflight(wrong, files)
    with pytest.raises(ValueError, match="designated metadata"):
        w.data_preflight(dict(config, selection_lock=str(root / "final.json")), files)


def test_failed_truncated_missing_generation_count_as_tasks(tmp_path, monkeypatch):
    suite = suite_at(tmp_path / "suite.json")
    def never(*a, **k):
        raise AssertionError("truncated output must not be executed")
    import asea.certification.function_oracle as oracle
    monkeypatch.setattr(oracle, "evaluate_functions", never)
    result = e.validate([{"id": "orchard", "status": "completed", "text": "def f(x): return x",
                          "generation": {"truncated": True, "stop_reason": "max_new_tokens"}}], suite)
    assert result["tasks_total"] == result["tasks_failed"] == 2
    assert result["tasks_passed"] == 0 and result["certificate"] is False
    assert all(r["failure_stage"] for r in result["cases"])


def test_host_verdict_not_return_values_and_digest_default(tmp_path, monkeypatch):
    suite = suite_at(tmp_path / "suite.json")
    import asea.certification.function_oracle as oracle
    observed = []
    def blocked(source, cases, trace_policy):
        observed.append(trace_policy)
        return {"status": "BLOCKED", "supported": False, "passed": True,
                "return_trace": {"returns": [{"value": 1}]}}
    monkeypatch.setattr(oracle, "evaluate_functions", blocked)
    rows = [{"id": name, "status": "completed", "text": "def f(x): return x",
             "generation": {"truncated": False, "stop_reason": "eos"}} for name in ("orchard", "harbor")]
    result = e.validate(rows, suite)
    assert observed == [{"schema_version": 1, "return_retention": "digest"}] * 2
    assert result["tasks_failed"] == 0 and result["tasks_blocked"] == 2
    assert not result["completed"] and not result["engineering_complete"]
    assert result["pass_rate"] is None and not result["return_values_authoritative"]
    assert result["cases"][0]["oracle"]["return_trace"]["returns"][0]["value"] == 1


def fake_build_setup(tmp_path, monkeypatch, reject_recovery=False):
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    put(teacher / "config.json", {"model_type": "qwen2"})
    (teacher / "model.safetensors").write_bytes(b"0" * 100)
    recipe = recipe_at(tmp_path, recovery={"steps": 1})
    final = suite_at(tmp_path / "final-suite.json")
    suite_at(tmp_path / "validation-suite.json")
    preflight = {"hashes": {}, "counts": {"validation": 8}, "ids": {"final": ["orchard", "harbor"]},
        "final_suite_path": str(final), "final_suite_sha256": file_hash(final)["sha256"]}
    monkeypatch.setattr(w, "data_preflight", lambda *a: preflight)
    calls = []
    def child(argv, timeout):
        calls.append(argv[0])
        assert 0 < timeout <= 3600
        options = dict(zip(argv[1::2], argv[2::2]))
        if argv[0] in ("reconstruct", "recover"):
            assert options["--receipt-mode"] == "compact"
            assert Path(options["--report"]).name == argv[0] + ".receipt.json"
        completed = True
        if argv[0] == "evaluate":
            put(Path(options["--output"]), {"status": "completed", "completed": True,
                "engineering_complete": True, "tasks_total": 2, "tasks_passed": 1,
                "tasks_failed": 1, "tasks_graded": 2, "tasks_blocked": 0, "operational_failures": 0,
                "pass_rate": 0.5, "quality_pass": False})
        elif argv[0] == "recover" and reject_recovery:
            completed = False
        else:
            store = Path(options["--output-dir"])
            store.mkdir()
            put(store / "config.json", {"model_type": "qwen2"})
            (store / "model.safetensors").write_bytes(b"0" * 50)
            if argv[0] == "recover":
                put(store / "recovery_report.json", {"status": "completed", "artifact_admitted": True, "actual_steps": 1,
                    "standalone_native": True, "standalone_teacher_independent": True, "standalone_reload_probe": {"passed": True}})
        return {"returncode": 0 if completed else 1, "receipt": {"completed": completed,
                "status": ("RECONSTRUCTED_UNVALIDATED" if argv[0] == "reconstruct" else "completed") if completed else "rejected", "result": {"actual_steps": 0 if not completed else 1}}}
    monkeypatch.setattr(w, "run_child", child)
    return recipe, calls, final


def test_build_stage_order_frozen_smaller_and_no_quality_certificate(tmp_path, monkeypatch):
    recipe, calls, _ = fake_build_setup(tmp_path, monkeypatch)
    result = w.build(recipe, tmp_path / "study")
    assert calls == ["evaluate", "reconstruct", "evaluate", "recover", "evaluate"]
    assert result["status"] == "BUILT_UNCERTIFIED" and result["engineering_complete"]
    assert result["completed"] and not result["quality_pass"] and not result["certificate"]
    assert result["candidate_frozen"] and result["source_unchanged"]
    assert result["recovered_weight_bytes"] < result["source_weight_bytes"]
    assert result["comparison"]["noninferiority"] == "insufficient_evidence_no_power98_gate"
    with pytest.raises(ValueError):
        w.build(recipe, tmp_path / "study")


def test_rejected_recovery_preserved_no_completed_or_output(tmp_path, monkeypatch):
    recipe, calls, _ = fake_build_setup(tmp_path, monkeypatch, reject_recovery=True)
    root = tmp_path / "study"
    result = w.build(recipe, root)
    assert calls == ["evaluate", "reconstruct", "evaluate", "recover"]
    assert result["status"] == "REJECTED" and not result["completed"]
    assert not (root / "recovered").exists()
    failure = json.loads((root / "recover.result.json").read_text())
    assert failure["receipt"]["result"]["actual_steps"] == 0
    assert failure["state"] == "REJECTED"


def test_finalize_consumes_before_final_read_and_never_retries(tmp_path, monkeypatch):
    recipe, calls, final = fake_build_setup(tmp_path, monkeypatch)
    root = tmp_path / "study"
    assert w.build(recipe, root)["completed"]
    old = w.file_hash
    def guarded(path):
        if Path(path) == final:
            assert (root / "final-consumed.json").exists()
        return old(path)
    monkeypatch.setattr(w, "file_hash", guarded)
    result = w.finalize(root, final, tmp_path / "final-result.json")
    assert result["completed"] and not result["certificate"]
    assert calls[-3:] == ["evaluate"] * 3
    assert not result["training_on_final"]
    with pytest.raises(FileExistsError):
        w.finalize(root, final, tmp_path / "retry.json")


def test_finalize_changed_candidate_blocks_before_consumption(tmp_path, monkeypatch):
    recipe, _, final = fake_build_setup(tmp_path, monkeypatch)
    root = tmp_path / "study"
    w.build(recipe, root)
    (root / "recovered" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        w.finalize(root, final, tmp_path / "final-result.json")
    assert not (root / "final-consumed.json").exists()


def test_public_child_rejection_json_and_fixed_command(tmp_path):
    receipt = w.run_child(["recover", "--teacher-dir", str(tmp_path / "missing"), "--student-dir", str(tmp_path / "student"),
        "--output-dir", str(tmp_path / "output"), "--training-path", str(tmp_path / "train"), "--validation-path", str(tmp_path / "dev")], 20)
    assert receipt["returncode"] == 1 and not receipt["receipt"]["completed"]
    assert receipt["receipt"]["result"]["actual_steps"] == 0
    with pytest.raises(ValueError):
        w.run_child(["sh", "-c", "echo nope"], 1)


def test_actual_external_oracle_is_evidence_not_skipped(tmp_path):
    suite = suite_at(tmp_path / "suite.json")
    rows = [{"id": name, "status": "completed", "text": "def f(x): return x",
             "generation": {"truncated": False, "stop_reason": "eos"}} for name in ("orchard", "harbor")]
    report = e.validate(rows, suite)
    assert report["tasks_total"] == 2
    for row in report["cases"]:
        assert row["oracle"] is not None
        if row["oracle"]["supported"]:
            assert row["passed"] and row["oracle"]["status"] == "PASSED"
            assert row["oracle"]["return_trace"]["policy"]["return_retention"] == "digest"
        else:
            assert not row["passed"] and row["failure_stage"] == "function_oracle"
    assert not report["certificate"]


@pytest.mark.parametrize("script,limit,timeout,error", [
    ("import time; time.sleep(30)", 16000, 0.1, "deadline"),
    ("print('x'*20000)", 256, 3, "1 MiB")])
def test_child_timeout_and_output_bound_reap(monkeypatch, script, limit, timeout, error):
    original = w.subprocess.Popen
    def bounded_fixture(command, **kwargs):
        assert command[:3] == [sys.executable, "-I", "-S"]
        assert Path(command[3]) == Path(w.__file__).with_name("stage_worker.py")
        assert command[5:] == ["evaluate"]
        assert kwargs["shell"] is False and kwargs["start_new_session"] is True
        return original([sys.executable, "-c", script], **kwargs)
    monkeypatch.setattr(w.subprocess, "Popen", bounded_fixture)
    monkeypatch.setattr(w, "CAPTURE_LIMIT", limit)
    with pytest.raises(ValueError, match=error):
        w.run_child(["evaluate"], timeout)


@pytest.mark.parametrize("export_mode,steps", [("native_merged", 1), ("factor_preserving", 2)])
def test_tiny_random_native_reconstruct_recover_one_step_infer(tmp_path, capsys, monkeypatch, export_mode, steps):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from asea.specialist.recovery import _encode_details
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    try:
        source = tmp_path / "source"
        source.mkdir()
        vocab = {token: i for i, token in enumerate(["<pad>", "</s>", "<unk>", "<bos>", "<|user|>", "<|assistant|>",
            "orchard", "train", "apple", "pear", "harbor", "validation", "boat", "test"])}
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="</s>",
            unk_token="<unk>", bos_token="<bos>", additional_special_tokens=["<|user|>", "<|assistant|>"])
        tok.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + (eos_token if message['role'] == 'assistant' else '\\n') }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
        # Real newline template, not a handwritten Qwen production approximation.
        tok.chat_template = tok.chat_template.replace("\\n", "\n")
        tok.save_pretrained(source)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
            bos_token_id=3, eos_token_id=1, pad_token_id=0))
        model.save_pretrained(source, safe_serialization=True)
        del model
        before = model_inventory(source)
        train = put(tmp_path / "train.json", {"samples": [{"id": "orchard-train", "family": "orchard",
            "prompt": "orchard train", "response": "apple pear"}]})
        dev = put(tmp_path / "dev.json", {"samples": [{"id": "harbor-dev", "family": "harbor",
            "prompt": "harbor validation", "response": "boat"}]})
        small, recovered = tmp_path / "small", tmp_path / "recovered"
        reconstruct_transport = (["--receipt-mode", "compact", "--report", str(tmp_path / "reconstruct.receipt.json")]
                                 if export_mode == "factor_preserving" else [])
        assert main(["reconstruct", "--source-dir", str(source), "--output-dir", str(small),
            "--calibration-path", str(train), "--method", "uniform", "--retention", "0.5", "--dtype", "float32", "--max-length", "32"] + reconstruct_transport) == 0
        reconstructed = json.loads(capsys.readouterr().out)
        assert reconstructed["completed"]
        if reconstruct_transport:
            assert reconstructed["stdout_contract"] == "reconstruct_compact_v1"
            assert reconstructed["reconstruction_manifest_sha256"] == file_hash(small / "reconstruction_manifest.json")["sha256"]
        assert main(["recover", "--teacher-dir", str(source), "--student-dir", str(small), "--output-dir", str(recovered),
            "--training-path", str(train), "--validation-path", str(dev), "--steps", str(steps), "--rank", "2",
            "--learning-rate", "0.01", "--max-length", "32", "--dtype", "float32"] +
            (["--export-mode", export_mode, "--receipt-mode", "compact", "--report", str(tmp_path / "recover.receipt.json")]
             if export_mode == "factor_preserving" else [])) == 0
        trained = json.loads(capsys.readouterr().out)
        assert trained["result"]["actual_steps"] == steps and trained["result"]["adapter_delta_l2"] > 0
        if export_mode == "factor_preserving":
            assert trained["stdout_contract"] == "recover_compact_v1"
            assert len(w.compact_payload(trained).encode()) < w.COMPACT_LIMIT
            trained = json.loads(Path(trained["full_receipt_file"]).read_text())
        assert model_inventory(source) == before
        from asea.specialist import reconstruction as reconstruction_module
        monkeypatch.setattr(reconstruction_module, "_admit", lambda *a, **k:
            pytest.fail("NativeGenerator must not charge the reconstruction full-model formula"))
        output = e.infer(recovered, "orchard train", dtype="float32", max_new_tokens=8)
        assert output["output"] == {"type": "text", "text": output["text"]}
        assert output["generation"]["generate_config"]["do_sample"] is False
        expected_prefix = tok.apply_chat_template([{"role": "user", "content": "orchard train"}], tokenize=False, add_generation_prompt=True)
        expected_ids = tok.encode(expected_prefix, add_special_tokens=False)
        assert output["generation"]["input_token_ids"] == expected_ids
        assert output["encoding"]["format"] == "native_chat_response_only_v1"
        assert output["encoding"] == trained["result"]["encoding"]
        assert output["generation"]["generated_token_count"] <= 8
        assert "high_water" in output["resources"]["measurement"]
        assert output["resources"]["current_rss_bytes"] > 0
        assert output["resources"]["weights_loaded"]["current_rss_bytes"] > 0
        admission = output["resources"]["generation_admission"]
        assert admission["actual_prefill_tokens"] == len(expected_ids)
        assert admission["resident_weights_charged_bytes"] == 0
        assert output["resources"]["admission"]["phase"] == "loader_pre_weights"
        assert output["resources"]["phase"] == "generation_complete"
        generator = e.NativeGenerator(recovered, "float32", 256)
        try:
            with pytest.raises(ValueError, match="context limit"):
                generator.generate("orchard " * 300)
            from asea.specialist.reconstruction import ReconstructionBlocked
            token_calls = []
            original_call = type(generator.tokenizer).__call__
            def counted(self, *args, **kwargs):
                token_calls.append(kwargs)
                return original_call(self, *args, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(type(generator.tokenizer), "__call__", counted)
                patch.setattr(generator.model, "forward", lambda *a, **k: pytest.fail("over-budget forward"))
                generator.memory_budget_bytes = 1
                with pytest.raises(ReconstructionBlocked):
                    generator.generate("orchard train")
            assert len(token_calls) == 1 and token_calls[0]["truncation"] is False
            assert generator.last_trace["memory_admission"]["phase"] == "inference_incremental"
            assert not generator.last_trace["memory_admission"]["admitted"]
        finally:
            generator.close()
    finally:
        torch.set_num_threads(old_threads)


@pytest.fixture
def tiny_random_llama(tmp_path):
    """Random safetensors mechanics, not downloaded or trained competitor evidence."""
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path / "native-llama"
    root.mkdir()
    vocab = {token: i for i, token in enumerate(["<pad>", "</s>", "<unk>", "<bos>",
        "<|user|>", "<|assistant|>", "orchard", "harbor", "prompt"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="</s>",
        unk_token="<unk>", bos_token="<bos>", additional_special_tokens=["<|user|>", "<|assistant|>"])
    tok.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    tok.save_pretrained(root)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
            tie_word_embeddings=True, bos_token_id=3, eos_token_id=1, pad_token_id=0))
        model.save_pretrained(root, safe_serialization=True)
        del model
    try:
        yield root, tok
    finally:
        torch.set_num_threads(old_threads)


def test_tiny_random_llama_evaluation_only_native_flow(tiny_random_llama, tmp_path, monkeypatch, capsys):
    root, tok = tiny_random_llama
    from asea.specialist import reconstruction as r
    before = model_inventory(root)
    with pytest.raises(r.ReconstructionBlocked, match="unsupported native model_type"):
        r._artifact_preflight(root)  # Shared reconstruction/recovery gate stays closed.
    monkeypatch.setattr(r, "_native_meta", lambda *a: pytest.fail("Llama must use evaluation-only constructor"))
    monkeypatch.setattr(r, "_admit", lambda *a, **k: pytest.fail("no reconstruction admission or training"))
    assert main(["infer", "--model", str(root), "--prompt", "orchard prompt",
                 "--dtype", "float32", "--max-new-tokens", "2"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    output = receipt["result"]
    expected = tok.encode(tok.apply_chat_template([{"role": "user", "content": "orchard prompt"}],
        tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    trace = output["generation"]
    assert trace["input_token_ids"] == expected
    assert trace["model_kind"] == "llama" and trace["model_class"] == "LlamaForCausalLM"
    assert trace["native_sequence_token_ids"] == expected + trace["generated_token_ids"]
    assert trace["attention_implementation"] == "eager" and trace["device"] == "cpu"
    assert output["source_model_class"] == "LlamaForCausalLM" and not output["factor_preserving"]
    assert output["resources"]["generation_admission"]["dimensions"]["causal"]
    assert output["encoding"]["format"] == "native_chat_response_only_v1"
    # Use original suite input verbatim, same adapter as teacher/student evaluation.
    # Only oracle availability is stubbed; native tokenization/load/forward are real.
    monkeypatch.setattr(e, "oracle_preflight", lambda: {"available": True})
    suite = suite_at(tmp_path / "llama-suite.json")
    report = e.evaluate(root, suite, dtype="float32", max_new_tokens=2)
    assert report["inputs_unchanged"] and report["tasks_total"] == 2
    for case in report["cases"]:
        item = case["generation"]
        assert item["status"] == "completed"
        prefix = tok.apply_chat_template([{"role": "user", "content": case["id"] + " prompt"}],
            tokenize=False, add_generation_prompt=True)
        assert item["generation"]["input_token_ids"] == tok.encode(prefix, add_special_tokens=False)
    assert model_inventory(root) == before


def test_llama_untied_bfloat16_and_total_memory_guard(tiny_random_llama, monkeypatch):
    root, _ = tiny_random_llama
    from safetensors.torch import load_file, save_file
    from transformers import LlamaForCausalLM
    from asea.specialist.reconstruction import ReconstructionBlocked
    config = json.loads((root / "config.json").read_text())
    put(root / "config.json", dict(config, tie_word_embeddings=False))
    tensors = load_file(root / "model.safetensors")
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    output = e.infer(root, "orchard prompt", dtype="bfloat16", max_new_tokens=2)
    assert output["generation"]["dtype"] == "bfloat16"
    assert output["generation"]["model_kind"] == "llama"
    monkeypatch.setattr(LlamaForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("tiny cap must precede weights"))
    with pytest.raises(ReconstructionBlocked) as refused:
        e.NativeGenerator(root, "float32", 2, memory_budget_bytes=1)
    assert refused.value.memory_admission["phase"] == "loader_pre_import"
    assert not refused.value.memory_admission["admitted"]


@pytest.mark.parametrize("kind", ["qwen2", "t5", "switch_transformers", "unsupported"])
def test_evaluation_keeps_original_non_llama_preflight(tmp_path, monkeypatch, kind):
    from asea.specialist import reconstruction as r
    put(tmp_path / "config.json", {"model_type": kind})
    sentinel = object()
    seen = []
    def original(path):
        seen.append(path)
        return sentinel
    monkeypatch.setattr(r, "_artifact_preflight", original)
    monkeypatch.setattr(e, "_validate_llama_config", lambda *a: pytest.fail("non-Llama must retain shared gate"))
    assert e._evaluation_preflight(tmp_path) is sentinel
    assert seen == [tmp_path]


@pytest.mark.parametrize("mutation,match", [
    ({"architectures": ["Qwen2ForCausalLM"]}, "strict native"),
    ({"architectures": None}, "strict native"),
    ({"auto_map": {"AutoModel": "remote.Code"}}, "auto_map"),
    ({"quantization_config": {}}, "quantized"),
    ({"hidden_size": True}, "dimension"),
    ({"num_hidden_layers": 257}, "layer count"),
    ({"num_key_value_heads": 3}, "head geometry"),
    ({"pretraining_tp": 2}, "untensorized"),
    ({"tie_word_embeddings": "true"}, "boolean"),
])
def test_llama_rejects_unsafe_config_before_native_load(tiny_random_llama, monkeypatch, mutation, match):
    root, _ = tiny_random_llama
    from transformers import AutoTokenizer, LlamaForCausalLM
    config = json.loads((root / "config.json").read_text())
    put(root / "config.json", dict(config, **mutation))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("unsafe tokenizer load"))
    monkeypatch.setattr(LlamaForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("unsafe weight load"))
    with pytest.raises(ValueError, match=match):
        e.NativeGenerator(root, "float32", 2)


@pytest.mark.parametrize("mutation,match", [("missing", "missing source keys"),
    ("extra", "unmatched source keys"), ("shape", "shape mismatch"), ("alias", "redundant tied-alias")])
def test_llama_strict_safetensors_keys_and_tied_alias(tiny_random_llama, monkeypatch, mutation, match):
    root, _ = tiny_random_llama
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, LlamaForCausalLM
    tensors = load_file(root / "model.safetensors")
    key = "model.layers.0.self_attn.q_proj.weight"
    if mutation == "missing":
        tensors.pop(key)
    elif mutation == "extra":
        tensors["unmatched.weight"] = tensors[key].clone()
    elif mutation == "shape":
        tensors[key] = tensors[key][:1].clone()
    else:
        assert "lm_head.weight" not in tensors
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("header check must precede tokenizer"))
    monkeypatch.setattr(LlamaForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("header check must precede weights"))
    with pytest.raises(ValueError, match=match):
        e.NativeGenerator(root, "float32", 2)


@pytest.mark.parametrize("unsafe", ["python", "symlink", "tokenizer_auto_map", "broken_header"])
def test_llama_local_artifact_security(tiny_random_llama, monkeypatch, unsafe):
    root, _ = tiny_random_llama
    from transformers import AutoTokenizer
    if unsafe == "python":
        (root / "remote.py").write_text("raise RuntimeError('must not run')")
    elif unsafe == "symlink":
        (root / "link.json").symlink_to(root / "config.json")
    elif unsafe == "tokenizer_auto_map":
        config = json.loads((root / "tokenizer_config.json").read_text())
        put(root / "tokenizer_config.json", dict(config, auto_map={"AutoTokenizer": "remote.Code"}))
    else:
        (root / "model.safetensors").write_bytes(b"bad")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("unsafe tokenizer load"))
    with pytest.raises(ValueError):
        e.NativeGenerator(root, "float32", 2)


def native_rows(text="def f(x): return x"):
    return [{"id": name, "status": "completed", "text": text,
             "generation": {"truncated": False, "stop_reason": "eos"}}
            for name in ("orchard", "harbor")]


@pytest.mark.parametrize("text", ["def f(: broken", "", "```python\ndef f(x): return x", None])
def test_malformed_candidates_are_complete_failures_without_oracle(tmp_path, monkeypatch, text):
    import asea.certification.function_oracle as oracle
    monkeypatch.setattr(oracle, "evaluate_functions", lambda *a, **k: pytest.fail("malformed candidate must not execute"))
    report = e.validate(native_rows(text), suite_at(tmp_path / "suite.json"))
    assert report["completed"] and report["engineering_complete"]
    assert report["tasks_failed"] == report["tasks_graded"] == report["tasks_total"] == 2
    assert report["tasks_blocked"] == report["oracle_executed"] == 0
    assert not report["quality_pass"] and report["pass_rate"] == 0


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT", "RESOURCE_LIMIT"])
def test_supported_candidate_failures_and_timeouts_are_completed_tasks(tmp_path, monkeypatch, status):
    import asea.certification.function_oracle as oracle
    monkeypatch.setattr(oracle, "evaluate_functions", lambda *a, **k:
        {"status": status, "supported": True, "passed": False, "tests_total": 1, "tests_run": 0})
    report = e.validate(native_rows(), suite_at(tmp_path / "suite.json"))
    assert report["completed"] and report["tasks_failed"] == report["oracle_executed"] == 2
    assert report["tasks_blocked"] == 0


def test_oracle_exception_is_unavailable_not_bad_code(tmp_path, monkeypatch):
    import asea.certification.function_oracle as oracle
    def unavailable(*a, **k):
        raise RuntimeError("profile unavailable")
    monkeypatch.setattr(oracle, "evaluate_functions", unavailable)
    report = e.validate(native_rows(), suite_at(tmp_path / "suite.json"))
    assert report["status"] == "BLOCKED" and report["tasks_blocked"] == 2
    assert report["tasks_failed"] == report["tasks_graded"] == 0
    assert report["generation_completed"] == 2 and report["pass_rate"] is None


@pytest.mark.parametrize("phase", ["oracle_preflight", "model_load", "model_runtime"])
def test_live_infrastructure_failures_block_and_preserve_all_tasks(tmp_path, monkeypatch, phase):
    suite = suite_at(tmp_path / "suite.json")
    seen = []
    monkeypatch.setattr(e, "oracle_preflight", lambda: {"available": phase != "oracle_preflight"})
    monkeypatch.setattr(e, "model_inventory", lambda *a: {})
    class Generator:
        last_trace = {"stop_reason": "error"}
        def __init__(self, *a, **k):
            seen.append("load")
            if phase == "model_load":
                raise ImportError("missing runtime dependency")
        def generate(self, prompt):
            seen.append("generate")
            raise RuntimeError("model runtime OOM")
        def close(self):
            seen.append("close")
    monkeypatch.setattr(e, "NativeGenerator", Generator)
    report = e.evaluate(tmp_path / "model", suite)
    assert not report["completed"] and not report["engineering_complete"]
    assert report["tasks_total"] == report["tasks_blocked"] == 2
    assert report["tasks_failed"] == 0 and report["pass_rate"] is None
    assert all(r["failure_stage"] == phase for r in report["cases"])
    if phase == "oracle_preflight":
        assert seen == []


def test_stage_does_not_trust_zero_exit_over_blocked_output(tmp_path, monkeypatch):
    recipe, calls, _ = fake_build_setup(tmp_path, monkeypatch)
    child = w.run_child
    def blocked(argv, timeout):
        receipt = child(argv, timeout)
        if argv[0] == "evaluate":
            output = Path(argv[argv.index("--output") + 1])
            put(output, {"status": "BLOCKED", "completed": False, "engineering_complete": False,
                         "tasks_total": 2, "tasks_blocked": 2, "pass_rate": None})
        return receipt
    monkeypatch.setattr(w, "run_child", blocked)
    report = w.build(recipe, tmp_path / "study")
    assert report["status"] == "BLOCKED" and not report["engineering_complete"]
    assert calls == ["evaluate"] and report["attempted_stages"] == ["source-validation"]
    evidence = json.loads((tmp_path / "study/source-validation.result.json").read_text())
    assert evidence["state"] == "BLOCKED" and evidence["returncode"] == 0


def test_memory_cli_and_recipe_wire_exact_api_bytes(tmp_path, monkeypatch, capsys):
    from asea.specialist import recovery, reconstruction
    seen = {}
    def recover(**kwargs):
        seen.update(kwargs)
        return {"status": "completed", "artifact_admitted": True, "standalone_native": True,
                "standalone_teacher_independent": True, "standalone_reload_probe": {"passed": True}}
    monkeypatch.setattr(recovery, "recover", recover)
    assert main(["recover", "--teacher-dir", "t", "--student-dir", "s", "--output-dir", "o",
        "--training-path", "train", "--validation-path", "dev", "--memory-budget-mib", "8192.5",
        "--teacher-mode", "cache"]) == 0
    capsys.readouterr()
    assert seen["memory_budget_bytes"] == 8192 * 1024**2 + 512 * 1024
    assert seen["teacher_mode"] == "cached" and seen["max_length"] == 256
    config = w.recipe_config(recipe_at(tmp_path, memory_budget_bytes=4097, recovery={"teacher_mode": "resident"}))
    opts = w._options({"memory_budget_bytes": config["memory_budget_bytes"]})
    args = parser().parse_args(["infer", "--model", "m", "--prompt", "p"] + opts)
    assert args.memory_budget_bytes == 4097
    assert w.recipe_config(recipe_at(tmp_path))["memory_budget_bytes"] is None
    for invalid in (0, -1, True, "4096"):
        with pytest.raises(ValueError):
            w.recipe_config(recipe_at(tmp_path, memory_budget_bytes=invalid))


def test_build_propagates_memory_mode_and_continues_below_source_floor(tmp_path, monkeypatch):
    recipe, calls, _ = fake_build_setup(tmp_path, monkeypatch)
    values = json.loads(recipe.read_text())
    values.update(memory_budget_bytes=8192 * 1024**2, source_quality_floor=0.75,
                  recovery={"steps": 1, "teacher_mode": "resident"})
    put(recipe, values)
    child = w.run_child
    def inspect_child(argv, timeout):
        assert argv[argv.index("--memory-budget-mib") + 1] == "8192"
        if argv[0] == "recover":
            assert argv[argv.index("--teacher-mode") + 1] == "resident"
            assert argv[argv.index("--max-length") + 1] == "256"
        return child(argv, timeout)
    monkeypatch.setattr(w, "run_child", inspect_child)
    report = w.build(recipe, tmp_path / "study")
    assert report["completed"] and report["engineering_complete"]
    assert not report["qualified_source"] and report["study_role"] == "RESEARCH"
    assert not report["certificate"] and calls.count("recover") == 1
    assert report["outputs"]["candidate"]["role"] == "frozen_recovered_candidate_not_certified"
    assert "memory_budget_bytes" in report["implementation"]["api_signatures"]["recover"]


def test_child_timeout_preserves_actual_exit_logs_and_stage_resources(tmp_path, monkeypatch):
    original = w.subprocess.Popen
    monkeypatch.setattr(w.subprocess, "Popen", lambda command, **kwargs:
        original([sys.executable, "-u", "-c", "import time; print('partial receipt'); time.sleep(30)"], **kwargs))
    import time
    with pytest.raises(w.StageBlocked):
        w._stage(tmp_path, "source-validation", ["evaluate"], time.monotonic() + 0.2)
    evidence = json.loads((tmp_path / "source-validation.result.json").read_text())
    logs = json.loads(Path(evidence["logs"]).read_text())
    assert evidence["state"] == "BLOCKED" and evidence["returncode"] is not None
    assert evidence["returncode"] < 0 and evidence["resources"]["capture_limit_bytes"] == 1024**2
    assert "partial receipt" in logs["stdout"]


def test_recovery_history_has_separate_full_sidecar(tmp_path, monkeypatch, capsys):
    from asea.specialist import recovery
    history = [{"step": i + 1, "loss": 1.0} for i in range(64)]
    monkeypatch.setattr(recovery, "recover", lambda **kwargs:
        {"status": "rejected", "artifact_admitted": False, "actual_steps": 64, "training_history": history})
    report = tmp_path / "recovery-receipt.json"
    assert main(["recover", "--teacher-dir", "t", "--student-dir", "s", "--output-dir", "o",
        "--training-path", "train", "--validation-path", "dev", "--report", str(report)]) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["result"]["recovery_history_rows"] == 64
    assert "training_history" not in receipt["result"]
    assert json.loads(report.read_text())["result"]["training_history"] == history
    assert json.loads(Path(receipt["result"]["recovery_history_file"]).read_text())["history"] == history


def test_quarantine_prefix_rejected_before_any_data_read(tmp_path, monkeypatch):
    def never(*args):
        pytest.fail("quarantine must be rejected before reading files")
    monkeypatch.setattr(w, "_bounded_json", never)
    config = {key: str(tmp_path / "final-quarantined" / filename) for key, filename in
              (("training", "train.json"), ("calibration", "calibration.json"),
               ("validation_data", "validation.json"), ("validation_suite", "validation-suite.json"),
               ("data_manifest", "manifest.json"), ("selection_lock", "selection-lock.json"))}
    with pytest.raises(ValueError, match="quarantined"):
        w.data_preflight(config, {})


def test_actual_collaborator_api_signatures_and_shared_memory_guard(monkeypatch):
    import inspect
    from asea.specialist import reconstruction, recovery
    for function in (reconstruction.reconstruct, recovery.recover, e.infer, e.evaluate):
        assert inspect.signature(function).parameters["memory_budget_bytes"].default is None
    assert inspect.signature(recovery.recover).parameters["teacher_mode"].default == "cached"
    monkeypatch.setattr(reconstruction, "_available_ram", lambda: 16 * 1024**3)
    headers = {"weights": {"numel": 1200000000, "dtype": "BF16"}}
    admitted = reconstruction._admit(headers, {"vocab_size": 100}, "bfloat16", 256)
    assert admitted["estimated_peak_bytes"] > 4 * 1024**3
    assert admitted["requested_memory_budget_bytes"] is None
    monkeypatch.setattr(reconstruction, "_available_ram", lambda: 1024**3)
    with pytest.raises(reconstruction.ReconstructionBlocked, match="effective"):
        reconstruction._admit(headers, {"vocab_size": 100}, "bfloat16", 256,
                              memory_budget_bytes=128 * 1024**3)


def test_cli_completed_taskfail_vs_unavailable_oracle(tmp_path, monkeypatch, capsys):
    import asea.certification.function_oracle as oracle
    suite = suite_at(tmp_path / "suite.json")
    generations = put(tmp_path / "generations.json", native_rows("def f(: broken"))
    assert main(["validate", "--suite", str(suite), "--generations", str(generations),
                 "--output", str(tmp_path / "failed.json")]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["completed"] and receipt["result"]["tasks_failed"] == 2
    assert not receipt["result"]["quality_pass"] and not receipt["result"]["certificate"]
    put(generations, native_rows())
    monkeypatch.setattr(oracle, "evaluate_functions", lambda *a, **k:
        {"status": "BLOCKED", "supported": False, "reason": "profile unavailable"})
    assert main(["validate", "--suite", str(suite), "--generations", str(generations),
                 "--output", str(tmp_path / "blocked.json")]) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "BLOCKED" and not receipt["completed"]
    assert receipt["result"]["tasks_blocked"] == 2 and receipt["result"]["tasks_failed"] == 0


def test_cli_preserves_recovery_runtime_rejection_as_blocked_receipt(tmp_path, monkeypatch, capsys):
    from asea.specialist import recovery
    original = {"status": "rejected", "artifact_admitted": False, "actual_steps": 3,
                "error": {"type": "RuntimeError", "message": "runtime allocation failed"}}
    monkeypatch.setattr(recovery, "recover", lambda **kwargs: original)
    assert main(["recover", "--teacher-dir", "t", "--student-dir", "s", "--output-dir", "o",
                 "--training-path", "train", "--validation-path", "dev"]) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "BLOCKED" and not receipt["engineering_complete"]
    assert receipt["result"] == original


def transport_argv(tmp_path, command="reconstruct"):
    argv = [command, "--output-dir", str(tmp_path / "model")]
    if command == "reconstruct":
        argv += ["--source-dir", "source", "--calibration-path", "calibration"]
    else:
        argv += ["--teacher-dir", "teacher", "--student-dir", "student", "--training-path", "train",
                 "--validation-path", "dev"]
    return argv


def fake_transport_api(tmp_path, monkeypatch, command="reconstruct", large=False, **changes):
    from asea.specialist import reconstruction, recovery
    if command == "reconstruct":
        result = {"schema_version": 1, "status": "RECONSTRUCTED_UNVALIDATED",
            "selected_indices": {"layer" + str(i): list(range(20000)) for i in range(80 if large else 1)},
            "weights_provenance": {"tensor": {"selected_indices_ref": "layer0", "details": "x" * (2000000 if large else 1)}},
            "removed_source_tensors": [], "source": {"family": "qwen2", "files": {"a": "sha"}, "config": {"width": 20000}},
            "output": {"standalone": True, "teacher_required_at_serve": False, "files": {}, "config": {}},
            "counts": {"source": {"parameters": 8}, "output": {"parameters": 6}},
            "verification": {"capability_evaluated": False, "source_loading_strict": True,
                             "per_layer_vectors": [1, 2, 3]},
            "calibration": {"max_samples": 32, "encoding_audit": [1, 2, 3]}}
    else:
        result = {"schema_version": 1, "status": "completed", "artifact_admitted": True,
            "standalone_native": True, "standalone_teacher_independent": True,
            "standalone_reload_probe": {"passed": True, "pre_logits": [1, 2, 3], "generated_ids": [4, 5]},
            "requested_steps": 64, "actual_steps": 64, "training_history": [{"step": i} for i in range(64)],
            "teacher_bank": {"required_at_serve": False, "removed_before_export": True, "rows": [1, 2, 3]},
            "native_target_hashes_before": {"unused": "vector"}, "data": {"encoding_audit": [1, 2, 3]}}
    result.update(changes)
    seen = []
    def api(**kwargs):
        assert "report" not in kwargs and "receipt_mode" not in kwargs
        seen.append(kwargs)
        if result["status"] in ("completed", "RECONSTRUCTED_UNVALIDATED"):
            output = Path(kwargs["output_dir"])
            output.mkdir(exist_ok=True)
            w.write_json(output / ("reconstruction_manifest.json" if command == "reconstruct" else "recovery_report.json"), result)
        return result
    monkeypatch.setattr(reconstruction if command == "reconstruct" else recovery, command, api)
    return result, seen


def emitted_child(tmp_path, monkeypatch, payload, returncode=0):
    """Test-only subprocess: production command construction/capture still executes."""
    wire = tmp_path / "wire.txt"
    wire.write_text(payload)
    original = w.subprocess.Popen
    script = "import pathlib,sys; sys.stdout.write(pathlib.Path(sys.argv[1]).read_text()); sys.exit(int(sys.argv[2]))"
    monkeypatch.setattr(w.subprocess, "Popen", lambda command, **kwargs:
        original([sys.executable, "-c", script, str(wire), str(returncode)], **kwargs))


@pytest.mark.parametrize("command", ["reconstruct", "recover"])
def test_compact_requires_report_before_api_and_mode_is_strict(tmp_path, monkeypatch, capsys, command):
    _, seen = fake_transport_api(tmp_path, monkeypatch, command)
    argv = transport_argv(tmp_path, command)
    assert parser().parse_args(argv).receipt_mode == "full"
    assert main(argv + ["--receipt-mode", "compact"]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert not seen and not failure["completed"] and failure["transport_error"]
    for bad in ("unknown", "True", "false"):
        with pytest.raises(SystemExit):
            parser().parse_args(argv + ["--receipt-mode", bad])
    for bad in (True, False, None, 1, "unknown"):
        with pytest.raises(ValueError, match="receipt_mode"):
            w._options({"receipt_mode": bad})


@pytest.mark.parametrize("mode,report", [(None, False), ("full", False), (None, True), ("full", True)])
def test_legacy_reconstruction_stdout_keeps_full_manifest(tmp_path, monkeypatch, capsys, mode, report):
    original, seen = fake_transport_api(tmp_path, monkeypatch)
    argv = transport_argv(tmp_path)
    if mode:
        argv += ["--receipt-mode", mode]
    if report:
        argv += ["--report", str(tmp_path / "receipt.json")]
    assert main(argv) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert seen and receipt["result"] == original and "stdout_contract" not in receipt
    if report:
        assert json.loads((tmp_path / "receipt.json").read_text())["result"] == original


def test_ten_megabyte_manifest_compact_child_stage_and_legacy_cap(tmp_path, monkeypatch, capsys):
    import time
    original, seen = fake_transport_api(tmp_path, monkeypatch, large=True)
    report = tmp_path / "receipt.json"
    argv = transport_argv(tmp_path) + ["--report", str(report), "--receipt-mode", "compact"]
    assert main(argv) == 0
    wire = capsys.readouterr().out
    receipt = json.loads(wire)
    assert seen and len(wire.encode()) < 64 * 1024
    assert 10 * 1024**2 < report.stat().st_size < w.REPORT_LIMIT
    manifest = tmp_path / "model/reconstruction_manifest.json"
    assert json.loads(manifest.read_text()) == original
    assert json.loads(report.read_text())["result"] == original
    assert receipt["result"]["selection_summary"]["selected_indices_total"] == 1600000
    assert receipt["result"]["source"]["files_sha256"] == w.json_hash(original["source"]["files"])
    assert receipt["result"]["verification"] == {"capability_evaluated": False, "source_loading_strict": True}
    for prefix, path in (("full_receipt", report), ("reconstruction_manifest", manifest)):
        assert receipt[prefix + "_file"] == str(path)
        assert receipt[prefix + "_sha256"] == file_hash(path)["sha256"]
        assert receipt[prefix + "_bytes"] == path.stat().st_size
    assert "selected_indices" not in receipt["result"] and "weights_provenance" not in receipt["result"]
    with monkeypatch.context() as patch:
        emitted_child(tmp_path, patch, wire)
        value = w._stage(tmp_path, "reconstruct", argv, time.monotonic() + 10)
    assert value == receipt
    terminal = json.loads((tmp_path / "reconstruct.result.json").read_text())
    assert terminal["state"] == "COMPLETED" and (tmp_path / "reconstruct.result.json").stat().st_size < 64 * 1024
    # The old full CLI contract still correctly exceeds the UNCHANGED stream cap.
    with monkeypatch.context() as patch:
        emitted_child(tmp_path, patch, report.read_text())
        with pytest.raises(w.ChildFailure, match="1 MiB") as failed:
            w.run_child(transport_argv(tmp_path), 10)
    assert failed.value.evidence["stdout_bytes"] + failed.value.evidence["stderr_bytes"] == w.CAPTURE_LIMIT == 1024**2
    assert w.REPORT_LIMIT == 16 * 1024**2


@pytest.mark.parametrize("layers,width", [(28, 14208), (48, 10368), (64, 20736)])
def test_hypothetical_qwen_selection_projection_bounded(layers, width):
    full = {"selected_indices": {"layer" + str(i): list(range(width)) for i in range(layers)}}
    result = w.compact_result("reconstruct", full)
    assert result["selection_summary"]["selected_indices_total"] == layers * width
    assert len(w.compact_payload(result)) < 64 * 1024
    assert full["selected_indices"]["layer0"] == list(range(width))


@pytest.mark.parametrize("mutation", ["report_hash", "report_path", "report_size_bool", "manifest_hash",
    "manifest_path", "summary", "status", "symlink", "missing", "oversize", "contract"])
def test_compact_child_rejects_unverified_claims(tmp_path, monkeypatch, capsys, mutation):
    fake_transport_api(tmp_path, monkeypatch)
    report = tmp_path / "receipt.json"
    argv = transport_argv(tmp_path) + ["--report", str(report), "--receipt-mode", "compact"]
    assert main(argv) == 0
    receipt = json.loads(capsys.readouterr().out)
    if mutation == "report_hash":
        receipt["full_receipt_sha256"] = "0" * 64
    elif mutation == "report_path":
        receipt["full_receipt_file"] = str(tmp_path / "arbitrary.json")
    elif mutation == "report_size_bool":
        receipt["full_receipt_bytes"] = True
    elif mutation == "manifest_hash":
        receipt["reconstruction_manifest_sha256"] = "0" * 64
    elif mutation == "manifest_path":
        receipt["reconstruction_manifest_file"] = str(tmp_path / "arbitrary.json")
    elif mutation == "summary":
        receipt["result"]["verification"]["capability_evaluated"] = True
    elif mutation == "status":
        receipt["completed"] = False
    elif mutation == "contract":
        del receipt["stdout_contract"]
    elif mutation == "symlink":
        report.rename(tmp_path / "moved.json")
        report.symlink_to(tmp_path / "moved.json")
    elif mutation == "missing":
        report.unlink()
    else:
        report.write_bytes(b" " * (w.REPORT_LIMIT + 1))
        receipt["full_receipt_bytes"] = w.REPORT_LIMIT + 1
    emitted_child(tmp_path, monkeypatch, json.dumps(receipt))
    with pytest.raises(w.ChildFailure, match="compact") as failed:
        w.run_child(argv, 10)
    assert failed.value.evidence["receipt"]["transport_validation_failed"]
    assert not failed.value.evidence["receipt"]["completed"]


@pytest.mark.parametrize("command", ["reconstruct", "recover"])
def test_compact_preserves_api_failure_and_no_manifest_required(tmp_path, monkeypatch, capsys, command):
    import time
    original, _ = fake_transport_api(tmp_path, monkeypatch, command, status="rejected", actual_steps=7,
        artifact_admitted=False, engineering_complete=False, error={"type": "RuntimeError", "message": "OOM"})
    report = tmp_path / "receipt.json"
    argv = transport_argv(tmp_path, command) + ["--report", str(report), "--receipt-mode", "compact"]
    assert main(argv) == 1
    wire = capsys.readouterr().out
    receipt = json.loads(wire)
    assert receipt["status"] == "BLOCKED" and receipt["completed"] is False
    assert json.loads(report.read_text())["result"] == original
    emitted_child(tmp_path, monkeypatch, wire, returncode=1)
    with pytest.raises(w.StageBlocked):
        w._stage(tmp_path, command, argv, time.monotonic() + 10)
    terminal = json.loads((tmp_path / (command + ".result.json")).read_text())
    assert not terminal["completed"] and terminal["receipt"]["result"]["engineering_complete"] is False
    assert terminal["receipt"]["full_receipt_sha256"] == file_hash(report)["sha256"]


def test_compact_recovery_history_and_vectors_stay_on_disk(tmp_path, monkeypatch, capsys):
    original, seen = fake_transport_api(tmp_path, monkeypatch, "recover")
    report = tmp_path / "receipt.json"
    argv = transport_argv(tmp_path, "recover") + ["--report", str(report), "--receipt-mode", "compact"]
    assert main(argv) == 0
    wire = capsys.readouterr().out
    receipt = json.loads(wire)
    assert seen and receipt["completed"] and receipt["result"]["actual_steps"] == 64
    assert receipt["result"]["standalone_reload_probe"] == {"passed": True}
    assert len(wire.encode()) < 64 * 1024
    for field in ("pre_logits", "generated_ids", "training_history", "encoding_audit", "native_target_hashes_before"):
        assert field not in wire
    history = Path(receipt["result"]["recovery_history_file"])
    assert json.loads(history.read_text())["history"] == original["training_history"]
    assert receipt["result"]["recovery_history_rows"] == 64
    assert receipt["result"]["recovery_history_sha256"] == file_hash(history)["sha256"]
    assert json.loads(report.read_text())["result"] == original
    emitted_child(tmp_path, monkeypatch, wire)
    assert w.run_child(argv, 10)["receipt"] == receipt


@pytest.mark.parametrize("failure", ["nan", "summary_growth", "report_exists", "missing_manifest", "write_failure", "huge_error"])
def test_compact_serialization_failure_is_small_and_never_success(tmp_path, monkeypatch, capsys, failure):
    from asea.specialist import reconstruction
    changes = {"method": {"unexpected": "x" * (100000 if failure == "summary_growth" else 1)}}
    if failure == "nan":
        changes["method"] = {"unexpected": float("nan")}
    original, seen = fake_transport_api(tmp_path, monkeypatch, **changes)
    report = tmp_path / "receipt.json"
    argv = transport_argv(tmp_path) + ["--report", str(report), "--receipt-mode", "compact"]
    if failure == "report_exists":
        report.write_text("untouched")
    elif failure == "missing_manifest" or failure == "nan":
        monkeypatch.setattr(reconstruction, "reconstruct", lambda **kwargs: original)
    elif failure == "write_failure":
        monkeypatch.setattr(w, "write_json", lambda *a: (_ for _ in ()).throw(OSError("read only")))
    elif failure == "huge_error":
        monkeypatch.setattr(reconstruction, "reconstruct", lambda **kwargs:
            (_ for _ in ()).throw(RuntimeError("x" * 100000)))
    assert main(argv) == 1
    wire = capsys.readouterr().out
    receipt = json.loads(wire)
    assert len(wire.encode()) < 64 * 1024 and not receipt["completed"] and "result" not in receipt
    assert receipt["transport_error"]
    if failure == "summary_growth":
        assert json.loads(report.read_text())["result"] == original
        assert receipt["full_receipt_sha256"] == file_hash(report)["sha256"]
        assert w._resolve_compact_receipt(argv, receipt)["completed"] is False
    if failure == "report_exists":
        assert not seen and report.read_text() == "untouched"
    if failure == "huge_error":
        assert receipt["error_truncated"] and receipt["status"] == "BLOCKED"


def test_transport_source_lock_covers_all_changed_files():
    files = w.implementation_manifest()["source_files"]
    root = Path(__file__).resolve().parents[1]
    for relative in ("src/asea/specialist/__main__.py", "src/asea/specialist/workflow.py",
                     "tests/test_specialist_workflow.py", "docs/SPECIALIST_WORKFLOW.md"):
        assert files[str(root / relative)] == file_hash(root / relative)


# Fixed stand-in CLI bodies live only in tests. Production accepts no helper,
# source, module, executable, environment or import-path override.
def stage_standin(tmp_path, mode="sleep"):
    helper = tmp_path / "fixed_stage_fixture.py"
    worker = Path(w.__file__).with_name("stage_worker.py")
    helper.write_text(f'''
import runpy, sys
namespace = runpy.run_path({str(worker)!r}, run_name="stage_fixture")
def fixed_cli(argv):
    assert "site" not in sys.modules and "torch" not in sys.modules
    import ctypes, json, os, resource, signal, time
    actual = ctypes.c_int()
    assert ctypes.CDLL(None).prctl(2, ctypes.byref(actual), 0, 0, 0) == 0
    assert actual.value == signal.SIGKILL
    info = {{"worker": os.getpid(), "group": os.getpgrp(), "as_limit": resource.getrlimit(resource.RLIMIT_AS)[0]}}
    mode = {mode!r}
    if mode in ("descendant", "leader_exit", "closed_pipes"):
        child = os.fork()
        if child == 0:
            if mode == "closed_pipes":
                os.close(1); os.close(2)
            time.sleep(30)
            os._exit(0)
        info["descendant"] = child
    with open({str(tmp_path / 'ready.json')!r}, "w") as out:
        json.dump(info, out)
    os.write(1, b'{{"status":"completed","completed":true}}\\n')
    os.write(2, b'fixed fixture diagnostic\\n')
    if mode in ("leader_exit", "closed_pipes", "receipt"):
        return 0
    time.sleep(30)
    return 0
namespace["main"].__globals__["_public_cli"] = fixed_cli
raise SystemExit(namespace["main"]())
''')
    return helper


def inject_stage_standin(monkeypatch, helper):
    original = w.subprocess.Popen
    def launch(command, **kwargs):
        assert command[:3] == [sys.executable, "-I", "-S"]
        assert Path(command[3]) == Path(w.__file__).with_name("stage_worker.py")
        return original(command[:3] + [str(helper)] + command[4:], **kwargs)
    monkeypatch.setattr(w.subprocess, "Popen", launch)


def wait_fixture(path, process):
    import time
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass
        assert process.poll() is None, "supervisor exited before fixture readiness"
        time.sleep(0.01)
    pytest.fail("fixed stage fixture did not become ready")


def assert_process_dead(pid):
    import time
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = Path(f"/proc/{pid}/stat")
        try:
            if status.read_text().split(") ", 1)[1].split()[0] == "Z":
                return  # Reparented zombies await init, but execute no code.
        except (FileNotFoundError, ProcessLookupError):
            # /proc may disappear before open (ENOENT) or during read (ESRCH).
            # Both mean the process is gone; other read errors must still fail.
            return
        time.sleep(0.01)
    pytest.fail(f"process {pid} remains live")


@pytest.mark.parametrize("error", [FileNotFoundError(2, "gone before open"),
                                  ProcessLookupError(3, "gone during read")])
def test_process_dead_helper_accepts_proc_disappearance(monkeypatch, error):
    # Test-only /proc read double: no process is launched or signalled here.
    import sys
    class GoneStatus:
        def read_text(self):
            raise error
    monkeypatch.setattr(sys.modules[__name__], "Path", lambda path: GoneStatus())
    assert_process_dead(123456789)


@pytest.mark.parametrize("error", [PermissionError(13, "denied"), OSError(5, "I/O error")])
def test_process_dead_helper_preserves_unexpected_read_errors(monkeypatch, error):
    import sys
    class UnreadableStatus:
        def read_text(self):
            raise error
    monkeypatch.setattr(sys.modules[__name__], "Path", lambda path: UnreadableStatus())
    with pytest.raises(type(error)) as caught:
        assert_process_dead(123456789)
    assert caught.value is error


@pytest.mark.parametrize("mode", ["descendant", "leader_exit", "closed_pipes"])
def test_stage_known_group_cleanup_and_bounded_inherited_pipe_drain(tmp_path, monkeypatch, mode):
    import time
    helper = stage_standin(tmp_path, mode)
    inject_stage_standin(monkeypatch, helper)
    start = time.monotonic()
    if mode == "closed_pipes":
        result = w.run_child(["recover"], 1)
        assert result["returncode"] == 0 and result["receipt"]["completed"]
    else:
        with pytest.raises(w.ChildFailure, match="deadline") as error:
            w.run_child(["recover"], 1)
        assert error.value.evidence["returncode"] == (0 if mode == "leader_exit" else -9)
    assert time.monotonic() - start < 1.5
    info = json.loads((tmp_path / "ready.json").read_text())
    assert info["worker"] == info["group"]
    assert_process_dead(info["worker"])
    assert_process_dead(info["descendant"])


def test_stage_group_killed_before_leader_is_reaped(tmp_path, monkeypatch):
    import os
    helper = stage_standin(tmp_path, "receipt")
    inject_stage_standin(monkeypatch, helper)
    original = os.killpg
    observed = []
    def kill_before_reap(pid, sig):
        # Child still belongs to us, even if it has already exited: no PID reuse.
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        observed.append(pid)
        return original(pid, sig)
    monkeypatch.setattr(os, "killpg", kill_before_reap)
    result = w.run_child(["recover"], 3)
    assert observed == [result["worker_pid"]]
    assert result["returncode"] == 0
    assert result["stdout"] == '{"status":"completed","completed":true}\n'
    assert result["stderr"] == "fixed fixture diagnostic\n"
    assert result["stdout_bytes"] == len(result["stdout"].encode())


def test_stage_bootstrap_parent_race_fails_before_cli_or_site(tmp_path):
    import os, signal, subprocess
    worker = Path(w.__file__).with_name("stage_worker.py")
    result = subprocess.run([sys.executable, "-I", "-S", str(worker), str(os.getpid() + 1000000),
        "evaluate", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, timeout=3)
    assert result.returncode == -signal.SIGKILL
    assert result.stdout == result.stderr == b""


@pytest.mark.parametrize("sig_name", ["SIGKILL", "SIGTERM", "SIGINT"])
def test_build_parent_death_and_managed_interruption_no_promotion(tmp_path, sig_name):
    import os, signal, subprocess, time
    helper = stage_standin(tmp_path)
    supervisor = tmp_path / "fixed_supervisor_fixture.py"
    supervisor.write_text(f'''
import importlib.util, json, sys
from pathlib import Path
import pytest
from asea.specialist import workflow as w
spec = importlib.util.spec_from_file_location("workflow_tests", {__file__!r})
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
root = Path({str(tmp_path)!r})
real = w.run_child
patch = pytest.MonkeyPatch()
recipe, calls, _ = fixtures.fake_build_setup(root, patch)
fake = w.run_child
w.run_child = lambda argv, timeout: real(argv, timeout) if argv[0] == "recover" else fake(argv, timeout)
fixtures.inject_stage_standin(patch, Path({str(helper)!r}))
report = w.build(recipe, root / "study")
(root / "done.json").write_text(json.dumps(report))
raise SystemExit(0 if report["completed"] else 1)
''')
    env = dict(os.environ, PYTHONPATH=str(Path(w.__file__).resolve().parents[2]))
    process = subprocess.Popen([sys.executable, str(supervisor)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    info = None
    try:
        info = wait_fixture(tmp_path / "ready.json", process)
        start = time.monotonic()
        process.send_signal(getattr(signal, sig_name))
        stdout, stderr = process.communicate(timeout=3)
        assert_process_dead(info["worker"])
        assert time.monotonic() - start < 3
        study = tmp_path / "study"
        assert (study / "recover.started.json").exists()
        if sig_name == "SIGKILL":
            assert process.returncode == -signal.SIGKILL
            assert not (study / "manifest.json").exists()
        else:
            assert process.returncode == 1, (stdout, stderr)
            report = json.loads((study / "manifest.json").read_text())
            evidence = json.loads((study / "recover.result.json").read_text())
            assert report["status"] == evidence["state"] == "BLOCKED"
            assert report["interrupted"] and evidence["interrupted"]
            assert not report["completed"] and not report["engineering_complete"]
            assert not report.get("candidate_frozen") and not report["certificate"]
            assert report["attempted_stages"][-1] == "recover"
            assert not (study / "recovered-validation.started.json").exists()
            assert evidence["returncode"] == -signal.SIGKILL
            assert evidence["worker_pid"] == info["worker"]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=3)
        if info is not None:
            try:
                os.killpg(info["group"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_stage_baseexception_preserves_failed_report_and_restores_handlers(tmp_path, monkeypatch):
    import signal, time
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(*args):
        raise SystemExit("fixture interrupted before result")
    monkeypatch.setattr(w, "run_child", interrupted)
    with pytest.raises(w.StageBlocked):
        w._stage(tmp_path, "recover", ["recover"], time.monotonic() + 1)
    report = json.loads((tmp_path / "recover.result.json").read_text())
    assert not report["completed"] and report["interrupted"] and report["state"] == "BLOCKED"
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


def test_stage_launcher_in_implementation_lock():
    lock = w.implementation_manifest()
    worker = str(Path(w.__file__).with_name("stage_worker.py"))
    assert lock["source_files"][worker] == file_hash(worker)


def test_stage_real_child_baseexception_kills_before_preserving_evidence(tmp_path, monkeypatch):
    helper = stage_standin(tmp_path)
    inject_stage_standin(monkeypatch, helper)
    selector_type = w.selectors.DefaultSelector
    class InterruptedSelector(selector_type):
        def select(self, timeout=None):
            events = super().select(timeout)
            if (tmp_path / "ready.json").exists():
                raise SystemExit("fixed supervisor exception")
            return events
    monkeypatch.setattr(w.selectors, "DefaultSelector", InterruptedSelector)
    with pytest.raises(w.ChildFailure, match="fixed supervisor exception") as error:
        w.run_child(["recover"], 3)
    evidence = error.value.evidence
    assert evidence["interrupted"] and evidence["returncode"] == -9
    assert_process_dead(evidence["worker_pid"])


def test_stage_bootstrap_does_not_raise_inherited_kernel_memory_limit(tmp_path, monkeypatch):
    import resource
    helper = stage_standin(tmp_path, "receipt")
    inject_stage_standin(monkeypatch, helper)
    launch = w.subprocess.Popen
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = min([512 * 1024**2] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY])
    def limited(command, **kwargs):
        # Test-only stand-in injection; production exposes no preexec or profile.
        kwargs["preexec_fn"] = lambda: resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        return launch(command, **kwargs)
    monkeypatch.setattr(w.subprocess, "Popen", limited)
    result = w.run_child(["recover"], 3)
    assert result["returncode"] == 0
    assert json.loads((tmp_path / "ready.json").read_text())["as_limit"] == ceiling
    assert resource.getrlimit(resource.RLIMIT_AS) == (soft, hard)


def test_stage_isolated_bootstrap_ignores_caller_site_and_pythonpath(tmp_path, monkeypatch):
    marker = tmp_path / "poisoned"
    (tmp_path / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').write('bad')")
    (tmp_path / "asea.py").write_text("raise AssertionError('caller module loaded')")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = w.run_child(["recover", "--teacher-dir", str(tmp_path / "missing"),
        "--student-dir", str(tmp_path / "student"), "--output-dir", str(tmp_path / "out"),
        "--training-path", str(tmp_path / "train"), "--validation-path", str(tmp_path / "dev")], 3)
    assert result["returncode"] == 1 and not result["receipt"]["completed"]
    assert not marker.exists()
    assert not (tmp_path / "out").exists()


def inference_config(**updates):
    # Dimensions only: no downloaded model or allocations proportional to weights.
    config = dict(model_type="qwen2", vocab_size=151936, hidden_size=896,
        intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14,
        num_key_value_heads=2, max_position_embeddings=32768)
    return dict(config, **updates)


def test_loader_and_inference_do_not_double_charge_resident_weights(monkeypatch):
    from asea.specialist import reconstruction as r
    mib = 1024**2
    monkeypatch.setattr(e, "_current_rss", lambda: 300 * mib)
    monkeypatch.setattr(r, "_available_ram", lambda: 4 * 1024**3)
    headers = {"weight": {"numel": 494032768, "dtype": "BF16"}}
    loader = e._admit_loader(headers, inference_config(), "bfloat16")
    assert loader["estimated_additional_peak_bytes"] == 494032768 * 4 + 512 * mib
    assert loader["unknown_inference_workspace_charged_bytes"] == 0
    assert loader["phase"] == "loader_pre_import"
    # Remaining memory after weights is much smaller than a second model load,
    # yet enough for the actual short prompt's incremental eager workspace.
    monkeypatch.setattr(e, "_current_rss", lambda: 1700 * mib)
    monkeypatch.setattr(r, "_available_ram", lambda: 600 * mib)
    result = e._admit_inference(inference_config(), "bfloat16", 150, 256)
    assert result["admitted"] and result["resident_weights_charged_bytes"] == 0
    assert result["estimated_additional_peak_bytes"] < 344 * mib < loader["header_loaded_bytes"]
    assert result["actual_prefill_tokens"] == 150 and result["cache_tokens"] == 406
    assert result["kv_cache_bytes"] == 2 * 24 * 2 * 64 * 406 * 2
    assert result["estimated_total_process_peak_bytes"] == 1700 * mib + result["estimated_additional_peak_bytes"]


def test_loader_source_cast_and_native_f32_are_not_undercounted(monkeypatch):
    from asea.specialist import reconstruction as r
    monkeypatch.setattr(r, "_available_ram", lambda: 16 * 1024**3)
    monkeypatch.setattr(e, "_current_rss", lambda: 1024**2)
    headers = {"weight": {"numel": 1000, "dtype": "F32"},
        "ffn.wo.weight": {"numel": 200, "dtype": "BF16"},
        "router.classifier.weight": {"numel": 100, "dtype": "F32"}}
    report = e._admit_loader(headers, inference_config(), "bfloat16")
    assert report["header_loaded_bytes"] == 2000 + 800 + 400
    assert report["source_payload_bytes"] == 4000 + 400 + 400
    assert report["source_mmap_and_cast_transient_bytes"] == 4800
    assert report["estimated_additional_peak_bytes"] == 3200 + 4800 + 512 * 1024**2
    upcast = e._admit_loader({"w": {"numel": 1000, "dtype": "BF16"}}, inference_config(), "float32")
    assert upcast["source_mmap_and_cast_transient_bytes"] == 4000


def test_inference_actual_long_prefill_quadratic_attention_and_linear_kv(monkeypatch):
    from asea.specialist import reconstruction as r
    monkeypatch.setattr(r, "_available_ram", lambda: 16 * 1024**3)
    short = e._admit_inference(inference_config(), "bfloat16", 100, 256)
    long = e._admit_inference(inference_config(), "bfloat16", 1000, 256)
    assert long["native_prefill_logits_bytes"] == 10 * short["native_prefill_logits_bytes"]
    assert long["kv_cache_bytes"] - short["kv_cache_bytes"] == 2 * 24 * 2 * 64 * 900 * 2
    assert long["prefill_activation_bytes"] > 10 * short["prefill_activation_bytes"]
    assert long["estimated_additional_peak_bytes"] > short["estimated_additional_peak_bytes"]
    assert long["stage_peak_bytes"] == max(long["stage_additional_bytes"].values())
    assert long["stage_additional_bytes"]["prefill_decode_overlap"] >= long["native_prefill_logits_bytes"] + long["native_decode_logits_bytes"]
    assert long["retained_generation_score_bytes"] == 0


def test_inference_total_operator_cap_and_dynamic_host_ceiling(monkeypatch):
    from asea.specialist import reconstruction as r
    mib = 1024**2
    monkeypatch.setattr(e, "_current_rss", lambda: 1700 * mib)
    monkeypatch.setattr(r, "_available_ram", lambda: 4 * 1024**3)
    base = e._admit_inference(inference_config(), "bfloat16", 150, 256)
    required = base["estimated_additional_peak_bytes"]
    with pytest.raises(r.ReconstructionBlocked) as refused:
        e._admit_inference(inference_config(), "bfloat16", 150, 256, 1700 * mib + required - 1)
    report = refused.value.memory_admission
    assert report["operator_remaining_bytes"] == required - 1
    assert report["limit_bytes"] == required - 1 and not report["admitted"]
    admitted = e._admit_inference(inference_config(), "bfloat16", 150, 256, 1700 * mib + required)
    assert admitted["admitted"]
    monkeypatch.setattr(r, "_available_ram", lambda: 300 * mib)
    with pytest.raises(r.ReconstructionBlocked) as refused:
        e._admit_inference(inference_config(), "bfloat16", 150, 256, 64 * 1024**3)
    assert refused.value.memory_admission["limit_bytes"] == 44 * mib


@pytest.mark.parametrize("mutation", [{"model_type": "unknown"}, {"num_key_value_heads": None},
    {"num_attention_heads": 13}, {"hidden_size": None}, {"num_hidden_layers": None}])
def test_unknown_inference_geometry_refuses_not_zero_estimate(mutation):
    from asea.specialist.reconstruction import ReconstructionBlocked
    with pytest.raises(ReconstructionBlocked):
        e._admit_inference(inference_config(**mutation), "bfloat16", 150, 256)
    with pytest.raises(ReconstructionBlocked):
        e._admit_loader({"w": {"numel": 100, "dtype": "BF16"}}, inference_config(**mutation), "bfloat16")


def test_seq2seq_cache_contains_encoder_cross_attention(monkeypatch):
    from asea.specialist import reconstruction as r
    monkeypatch.setattr(r, "_available_ram", lambda: 16 * 1024**3)
    config = dict(model_type="t5", vocab_size=100, d_model=32, d_ff=64, d_kv=8,
        num_heads=4, num_layers=2, num_decoder_layers=3)
    report = e._admit_inference(config, "float32", 100, 20)
    assert report["kv_cache_bytes"] == 2 * 3 * 4 * 8 * (100 + 21) * 4
    assert report["retained_encoder_bytes"] == 2 * 100 * 32 * 4
    assert report["native_prefill_logits_bytes"] == 0


def test_checkpoint_unfaulted_mmaps_are_additional_not_resident(monkeypatch):
    from asea.specialist import reconstruction as r
    original = Path.read_text
    def maps(path, *args, **kwargs):
        if str(path) == "/proc/self/smaps":
            return "1000-5000 r--p 00000000 00:01 10 /model/model.safetensors\nSize: 16 kB\nRss: 4 kB\n5000-9000 rw-p 00000000 00:00 0\nSize: 16 kB\nRss: 16 kB\n"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", maps)
    missing = e._unresident_weight_mmaps(Path("/model"), {"w": {"file": "model.safetensors"}})
    assert missing == 12 * 1024
    monkeypatch.setattr(r, "_available_ram", lambda: 16 * 1024**3)
    base = e._admit_inference(inference_config(), "bfloat16", 150, 256)
    mapped = e._admit_inference(inference_config(), "bfloat16", 150, 256, unresident_weight_mmap_bytes=missing)
    assert mapped["estimated_additional_peak_bytes"] - base["estimated_additional_peak_bytes"] == missing


@pytest.mark.parametrize("smaps", ["", "1000-5000 r--p 00000000 00:01 10 /model/model.safetensors\nSize: 16 kB\n"])
def test_missing_mmap_residency_refuses(monkeypatch, smaps):
    from asea.specialist.reconstruction import ReconstructionBlocked
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k:
        smaps if str(path) == "/proc/self/smaps" else original(path, *a, **k))
    with pytest.raises(ReconstructionBlocked, match="mmap residency"):
        e._unresident_weight_mmaps(Path("/model"), {"w": {"file": "model.safetensors"}})


def test_tiny_budget_refuses_loader_before_optional_ml(tmp_path, monkeypatch):
    from asea.specialist import reconstruction as r
    monkeypatch.setattr(e, "_evaluation_preflight", lambda *a:
        ({}, inference_config(), {"w": {"numel": 10, "dtype": "BF16"}}))
    monkeypatch.setattr(r, "_native_meta", lambda *a: pytest.fail("tiny total cap reached native loader"))
    with pytest.raises(r.ReconstructionBlocked) as refused:
        e.NativeGenerator(tmp_path, memory_budget_bytes=1)
    assert refused.value.memory_admission["phase"] == "loader_pre_import"
    assert refused.value.memory_admission["operator_remaining_bytes"] == 0


@pytest.mark.parametrize("mode", ["native_merged", "factor_preserving"])
def test_recovery_completion_is_representation_specific(mode):
    report = {"status": "completed", "artifact_admitted": True, "export_mode": mode,
        "standalone_native": mode == "native_merged", "factor_preserving": mode == "factor_preserving",
        "standalone_teacher_independent": True, "standalone_reload_probe": {"passed": True}}
    assert w.recovery_complete(report, mode)
    assert not w.recovery_complete(dict(report, standalone_reload_probe={"passed": False}), mode)
    assert not w.recovery_complete(dict(report, standalone_teacher_independent=False), mode)
    if mode == "factor_preserving":
        assert not w.recovery_complete(dict(report, factor_preserving=False, standalone_native=True), mode)
    else:
        assert not w.recovery_complete(dict(report, standalone_native=False), mode)


def test_receipt_inside_output_refused_before_training(tmp_path, monkeypatch, capsys):
    from asea.specialist import recovery
    monkeypatch.setattr(recovery, "recover", lambda **k: pytest.fail("must not train"))
    assert main(["recover", "--teacher-dir", "t", "--student-dir", "s",
        "--output-dir", str(tmp_path / "bundle"), "--training-path", "train", "--validation-path", "dev",
        "--export-mode", "factor_preserving", "--report", str(tmp_path / "bundle/receipt.json")]) == 1
    assert "outside" in json.loads(capsys.readouterr().out)["error"]


def test_actual_two_step_factor_build_frozen_bundle_independent_infer_validate(tmp_path, monkeypatch, capsys):
    """Real training/loading; stubbed stage quality scores are NOT quality evidence.

    Only tiny random source and synthetic suites. No real final artifacts opened.
    """
    import contextlib
    import io
    import shutil
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from test_specialist_recovery import tiny_stores
    from asea.specialist import standalone
    from asea.specialist.reconstruction import ReconstructionBlocked
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        teacher, _, train, dev = tiny_stores(tmp_path, "causal")
        from transformers import Qwen2Config, Qwen2ForCausalLM
        config = Qwen2Config.from_pretrained(teacher, local_files_only=True)
        config.intermediate_size = 128  # reconstruction requires a positive multiple of 64
        larger = Qwen2ForCausalLM(config)
        larger.save_pretrained(teacher, safe_serialization=True)
        del larger
        suite = suite_at(tmp_path / "validation-suite.json")
        final = suite_at(tmp_path / "synthetic-final-suite.json")
        recipe = recipe_at(tmp_path, source_path=str(teacher), training=str(train), calibration=str(train),
            validation_data=str(dev), validation_suite=str(suite), dtype="float32", max_length=12,
            max_new_tokens=2, reconstruction={"method": "uniform", "retention": 0.5},
            recovery={"steps": 2, "rank": 1, "learning_rate": 0.01, "export_mode": "factor_preserving"})
        monkeypatch.setattr(w, "data_preflight", lambda *a: {"hashes": {}, "ids": {"final": ["orchard", "harbor"]},
            "final_suite_path": str(final), "final_suite_sha256": file_hash(final)["sha256"]})
        seen = []
        def child(argv, timeout):
            options = dict(zip(argv[1::2], argv[2::2]))
            if argv[0] == "evaluate":
                # Exercise actual source/pruned/bundle loading, not a fake generator.
                generated = e.infer(options["--model"], "hello dev", dtype="float32", max_new_tokens=2)
                seen.append(generated["representation"])
                put(Path(options["--output"]), {"status": "completed", "completed": True,
                    "engineering_complete": True, "tasks_total": 2, "tasks_passed": 0,
                    "tasks_failed": 2, "tasks_graded": 2, "tasks_blocked": 0, "operational_failures": 0,
                    "pass_rate": 0.0, "quality_pass": False})
                return {"returncode": 0, "receipt": {"status": "completed", "completed": True}}
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main(argv)
            return {"returncode": code, "receipt": json.loads(stream.getvalue())}
        monkeypatch.setattr(w, "run_child", child)
        root = tmp_path / "study"
        result = w.build(recipe, root)
        assert result["completed"], result
        assert result["factor_preserving"] and not result["standalone_native"]
        assert result["standalone_teacher_independent"] and result["standalone_reload_probe"]["passed"]
        assert not result["quality_pass"] and not result["certificate"]
        assert seen == ["native_merged", "native_merged", "factor_preserving"]
        bundle = root / "recovered"
        manifest = standalone.inspect_bundle(bundle)
        files = result["frozen_models"]["recovered"]["files"]
        assert set(files) == set(manifest["files"]) | {"specialist_bundle.json", "recovery_report.json"}
        assert result["recovered_weight_bytes"] == manifest["counts"]["total_safetensors_bytes"]
        assert result["recovered_total_bundle_bytes"] == sum(v["size"] for v in files.values())
        assert result["recovered_weight_bytes"] < result["source_weight_bytes"]
        receipt = json.loads((root / "recover.receipt.json").read_text())
        assert receipt["result"]["actual_steps"] == 2 and receipt["result"]["adapter_delta_l2"] > 0
        assert (root / "recover.receipt.json.history.json").exists()
        assert not (bundle / "recover.receipt.json.history.json").exists()
        assert any(Path(k).name == "standalone.py" for k in result["implementation"]["source_files"])
        # Even audit-only sidecars belong to the full immutable bundle freeze.
        audit = bundle / "recovery_report.json"
        audit_bytes = audit.read_bytes()
        audit.write_bytes(audit_bytes + b"\n")
        with pytest.raises(ValueError, match="changed"):
            w.finalize(root, final, tmp_path / "blocked-synthetic-final.json")
        assert not (root / "final-consumed.json").exists()
        audit.write_bytes(audit_bytes)
        # Three frozen arms, one marker, evaluation only on the synthetic suite.
        finished = w.finalize(root, final, tmp_path / "synthetic-final-result.json")
        assert finished["completed"] and not finished["training_on_final"]
        assert seen[-3:] == ["native_merged", "native_merged", "factor_preserving"]
        with pytest.raises(FileExistsError):
            w.finalize(root, final, tmp_path / "synthetic-retry.json")
        shutil.rmtree(teacher)
        shutil.rmtree(root / "reconstructed")
        output = e.infer(bundle, "hello dev", dtype="float32", max_new_tokens=2)
        assert output["representation"] == "factor_preserving" and not output["external_teacher_loaded"]
        assert output["source_model_class"] == "Qwen2ForCausalLM"
        assert output["generation"]["model_class"] == "PeftModelForCausalLM"
        assert output["model_files"] == files
        with monkeypatch.context() as patch:
            patch.setattr(e, "oracle_preflight", lambda: {"available": True, "fixture": True})
            evaluation = e.evaluate(bundle, suite, dtype="float32", max_new_tokens=2)
        assert evaluation["model_files"] == files and evaluation["inputs_unchanged"]
        assert evaluation["representation"] == "factor_preserving"
        assert evaluation["source_model_class"] == "Qwen2ForCausalLM"
        assert evaluation["tasks_total"] == 2
        assert output["encoding"] == receipt["result"]["encoding"]
        assert output["resources"]["generation_admission"]["resident_weights_charged_bytes"] == 0
        assert output["resources"]["generation_admission"]["factor_workspace_bytes"] > 0
        assert output["resources"]["admission"]["header_loaded_bytes"] >= manifest["counts"]["total_parameters"] * 4
        # Public validation of actual generated records; operational oracle failures
        # retain their existing status and are never counted as model-quality passes.
        rows = [dict(output, id=name) for name in ("orchard", "harbor")]
        generations = put(tmp_path / "generations.json", rows)
        code = main(["validate", "--generations", str(generations), "--suite", str(suite),
            "--output", str(tmp_path / "public-validation.json")])
        public = json.loads(capsys.readouterr().out)
        assert code == (0 if public["completed"] else 1)
        assert public["result"]["tasks_total"] == 2 and public["result"]["certificate"] is False
        with pytest.raises(ValueError, match="dtype must match"):
            e.NativeGenerator(bundle, "bfloat16", 2)
        with monkeypatch.context() as patch:
            patch.setattr(standalone, "load_standalone", lambda *a, **k: pytest.fail("no full weights before admission"))
            with pytest.raises(ReconstructionBlocked, match="memory admission"):
                e.NativeGenerator(bundle, "float32", 2, memory_budget_bytes=1)
        # Rehash tampering to prove adapter option validation, not just hash checks.
        adapter_path = bundle / "adapter/adapter_config.json"
        adapter = json.loads(adapter_path.read_text())
        adapter["inference_mode"] = False
        put(adapter_path, adapter)
        digest = file_hash(adapter_path)
        manifest["files"]["adapter/adapter_config.json"] = {"sha256": digest["sha256"], "bytes": digest["size"]}
        put(bundle / "specialist_bundle.json", manifest)
        with monkeypatch.context() as patch:
            patch.setattr(standalone, "load_standalone", lambda *a, **k: pytest.fail("no base-only fallback"))
            with pytest.raises(ValueError, match="adapter options"):
                e.NativeGenerator(bundle, "float32", 2)
        with pytest.raises(ValueError, match="adapter options"):
            e.representation_inventory(bundle)
        with pytest.raises(ValueError, match="incomplete bundle base"):
            e.NativeGenerator(bundle / "base", "float32", 2)
    finally:
        torch.set_num_threads(old_threads)
