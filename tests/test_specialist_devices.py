"""CPU mechanics, simulated admission boundaries, optional REAL NVIDIA execution.

Fake CUDA tests validate resolver/control flow ONLY, never backend execution.
No pretrained downloads; CUDA tests skip unless an actual NVIDIA runtime exists.
"""
from contextlib import contextmanager
import inspect
from types import SimpleNamespace

import pytest

from asea.specialist import devices as d

MiB = 1024**2


class FakeCuda:
    """Memory observations only: deliberately cannot allocate/execute tensors."""
    def __init__(self, free=(2048 * MiB,), reserved=0, major=8):
        self.free = free
        self.reserved = reserved
        self.major = major
        self.current = 0

    @contextmanager
    def device(self, index):
        previous, self.current = self.current, index
        try:
            yield
        finally:
            self.current = previous

    def is_available(self):
        return True

    def init(self):
        pass

    def device_count(self):
        return len(self.free)

    def get_device_properties(self, index):
        return SimpleNamespace(name="FAKE admission-only device", total_memory=4096 * MiB)

    def get_device_capability(self, index):
        return (self.major, 0)

    def is_bf16_supported(self):
        return True  # emulate torch's old-device emulation claim; major still checked

    def mem_get_info(self):
        return self.free[self.current], 4096 * MiB

    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return self.reserved


def fake_torch(**kwargs):
    return SimpleNamespace(version=SimpleNamespace(cuda="FAKE", hip=None), cuda=FakeCuda(**kwargs))


@pytest.fixture
def ample_host(monkeypatch):
    from asea.specialist import reconstruction
    monkeypatch.setattr(reconstruction, "_available_ram", lambda: 16 * 1024**3)


@pytest.mark.parametrize("device_request", ["mps", "rocm", "hip:0", "multi_gpu", "cuda:0,cuda:1", "cpu_offload"])
def test_explicit_unsupported_never_falls_back(device_request):
    with pytest.raises(d.DeviceRejected, match="explicitly unsupported"):
        d.validate_device_request(device_request)


@pytest.mark.parametrize("device_request", [None, 0, {}, "cuda:-1", "cuda:01", "CUDA:0", " cuda:0", "xpu"])
def test_invalid_device_request(device_request):
    with pytest.raises(d.DeviceRejected):
        d.validate_device_request(device_request)


def test_probe_rejects_hip_cpu_build_and_unavailable_index():
    torch = fake_torch()
    torch.version.hip = "6.0"
    with pytest.raises(d.DeviceRejected, match="not CPU or HIP"):
        d._cuda_probe(torch, "cuda:0", "float32", operators=False)
    torch.version.hip, torch.version.cuda = None, None
    with pytest.raises(d.DeviceRejected, match="NVIDIA CUDA torch build"):
        d._cuda_probe(torch, "cuda:0", "float32", operators=False)
    torch.version.cuda = "FAKE"
    with pytest.raises(d.DeviceRejected, match="index is unavailable"):
        d._cuda_probe(torch, "cuda:1", "float32", operators=False)


def test_native_bf16_requires_capability_not_emulation():
    torch = fake_torch(major=7)
    with pytest.raises(d.DeviceRejected, match="Native BF16 unsupported"):
        d._cuda_probe(torch, "cuda:0", "bfloat16", operators=False)
    result = d._cuda_probe(torch, "cuda:0", "float32", operators=False)
    assert result["operator_probe"] == "not_repeated"
    assert not result["native_bfloat16_supported"]


def test_per_device_free_vram_and_operator_cap_boundaries():
    torch = fake_torch(free=(1024 * MiB, 2048 * MiB), reserved=100 * MiB)
    torch.cuda.current = 1
    boundary = 1024 * MiB - 256 * MiB
    assert d.device_admission(torch, "cuda:0", boundary)["admitted"]
    assert torch.cuda.current == 1  # context always restored
    with pytest.raises(d.DeviceRejected, match="CUDA memory admission"):
        d.device_admission(torch, "cuda:0", boundary + 1)
    assert d.device_admission(torch, "cuda:1", boundary + 1)["admitted"]
    assert d.device_admission(torch, "cuda:0", 23, 100 * MiB + 23)["admitted"]
    with pytest.raises(d.DeviceRejected):
        d.device_admission(torch, "cuda:0", 24, 100 * MiB + 23)
    for invalid in (0, -1, True, 2.5):
        with pytest.raises(d.DeviceRejected):
            d.device_admission(torch, "cuda:0", 0, invalid)


def test_auto_selects_only_individually_fitting_device_and_host(ample_host, monkeypatch):
    torch = fake_torch(free=(300 * MiB, 2048 * MiB))
    probes = []
    def fake_probe(torch, device, dtype, *, operators=True):
        probes.append((device, operators))
        return {"operator_probe": "FAKE resolver test; NO backend execution"}
    monkeypatch.setattr(d, "_cuda_probe", fake_probe)
    result = d.resolve_device("auto", torch=torch, host_required_bytes=512 * MiB,
                              device_required_bytes=512 * MiB)
    assert result["device"] == "cuda:1"
    assert ("cuda:0", True) not in probes  # don't even probe operators before fit
    assert result["probe"]["operator_probe"].startswith("FAKE")
    assert result["cross_device_validation"] == "unknown"
    with pytest.raises(d.DeviceRejected, match="CUDA memory admission"):
        d.resolve_device("cuda:0", torch=torch, host_required_bytes=512 * MiB,
                          device_required_bytes=512 * MiB)
    with pytest.raises(d.DeviceRejected, match="Host memory"):
        d.resolve_device("auto", torch=torch, host_required_bytes=512 * MiB,
                          device_required_bytes=512 * MiB, memory_budget_bytes=1)
    with pytest.raises(d.DeviceRejected, match="header/config-derived"):
        d.resolve_device("auto", torch=torch)


def test_auto_cpu_fit_and_no_fabricated_cuda_on_hip(ample_host):
    torch = fake_torch()
    torch.version.hip = "6.0"
    result = d.resolve_device("auto", torch=torch, host_required_bytes=MiB, device_required_bytes=MiB)
    assert result["device"] == "cpu"
    assert "probe" not in result
    with pytest.raises(d.DeviceRejected):
        d.resolve_device("cuda:0", torch=torch, host_required_bytes=MiB, device_required_bytes=MiB)


def test_public_api_cpu_defaults():
    from asea.specialist.recovery import recover
    from asea.specialist.evaluation import NativeGenerator, infer, evaluate
    from asea.specialist.standalone import load_standalone
    for function in (recover, NativeGenerator, infer, evaluate, load_standalone):
        signature = inspect.signature(function)
        assert signature.parameters["execution_device"].default == "cpu"
        assert signature.parameters["device_memory_budget_bytes"].default is None


@pytest.fixture
def native_torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_actual_cpu_two_step_recovery_factor_parity_and_inference(tmp_path, native_torch, family, dtype):
    from test_specialist_recovery import tiny_stores
    from asea.specialist.recovery import recover
    from asea.specialist.standalone import load_standalone, inspect_bundle
    from asea.specialist.evaluation import infer
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    output = tmp_path / "factor"
    # Omit execution_device deliberately: old CPU callers stay CPU.
    report = recover(teacher, student, output, train, dev, steps=2, learning_rate=0.01,
                     rank=2, max_length=12, dtype=dtype, export_mode="factor_preserving")
    assert report["status"] == "completed", report.get("error")
    assert report["actual_steps"] == 2
    assert report["teacher_forward_calls"] == 2
    assert report["standalone_reload_probe"]["forward_bitwise_equal"]
    assert report["standalone_reload_probe"]["generation_bitwise_equal"]
    assert report["device"] == "cpu"
    assert report["execution_runtime"]["gpu_validation_status"] == "GPU_UNVERIFIED"
    assert report["cross_device_validation"] == "unknown"
    assert report["resources"]["release_checks"][-1]["all_collected"]
    assert "execution_device" not in inspect_bundle(output)
    loaded = load_standalone(output)
    assert {str(p.device) for p in loaded.model.parameters()} == {"cpu"}
    assert all(p.dtype == native_torch.float32 for name, p in loaded.model.named_parameters() if "lora_" in name)
    if family == "seq2seq":
        assert all(p.dtype == native_torch.float32 for name, p in loaded.model.named_parameters()
                   if ".wo.base_layer.weight" in name)
    result = infer(output, "hello world", dtype=dtype, max_new_tokens=2)
    assert result["status"] == "completed"
    assert result["resources"]["device"] == "cpu"
    assert result["execution_runtime"]["parameter_devices"] == ["cpu"]


def test_unavailable_cuda_recovery_rejected_before_weights(tmp_path, native_torch, monkeypatch):
    from test_specialist_recovery import tiny_stores
    from asea.specialist import recovery
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    monkeypatch.setattr(native_torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(recovery, "_load_recovery_model", lambda *a, **k: pytest.fail("weights read before device rejection"))
    output = tmp_path / "blocked"
    result = recovery.recover(teacher, student, output, train, dev, steps=1,
                              rank=2, dtype="float32", execution_device="cuda:0")
    assert result["status"] == "rejected", result
    assert result["error"]["type"] == "DeviceRejected"
    assert not output.exists()


def _actual_cuda():
    try:
        import torch
        return bool(torch.version.cuda and not torch.version.hip and torch.cuda.is_available())
    except ImportError:
        return False


@pytest.mark.skipif(not _actual_cuda(), reason="GPU_UNVERIFIED: actual NVIDIA CUDA runtime required")
@pytest.mark.parametrize("family", ["causal", "seq2seq"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_cuda_two_step_teacher_student_export_cpu_reload_same_cuda(tmp_path, native_torch, family, dtype):
    """No mocks: actual CUDA forward/backward/AdamW and strict same-device reload."""
    torch = native_torch
    if dtype == "bfloat16" and (torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported()):
        pytest.skip("Actual selected CUDA lacks native BF16")
    from test_specialist_recovery import tiny_stores
    from asea.specialist.recovery import recover
    from asea.specialist.standalone import load_standalone, inspect_bundle
    from asea.specialist.evaluation import infer
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    output = tmp_path / "gpu-factor"
    original_device = torch.cuda.current_device()
    report = recover(teacher, student, output, train, dev, steps=2, learning_rate=0.01,
                     rank=2, max_length=12, dtype=dtype, export_mode="factor_preserving", execution_device="cuda:0")
    assert report["status"] == "completed", report.get("error")
    assert report["actual_steps"] == 2 and report["teacher_forward_calls"] == 2
    assert report["teacher_bank"]["full_vocabulary"]
    assert report["teacher_bank"]["storage_dtype"] == "float32"
    assert report["teacher_unloaded_before_student"]
    assert report["adapter_hash_before"] != report["adapter_hash_after"]
    assert report["frozen_parameter_hash_before"] == report["frozen_parameter_hash_after"]
    assert report["factor_export_host_transfer"]["full_host_base_and_factors"]
    assert not report["factor_export_host_transfer"]["dtype_recast"]
    assert report["bounded_export"]["base_and_factor_live_hashes_verified_before_after"]
    probe = report["standalone_reload_probe"]
    assert probe["forward_bitwise_equal"] and probe["generation_bitwise_equal"]
    assert probe["before_execution_device"] == probe["after_execution_device"] == "cuda:0"
    assert probe["atol"] == probe["rtol"] == 0
    assert probe["cross_device_validation"] == "unknown"
    assert report["resources"]["release_checks"][-1]["all_collected"]
    runtime = report["execution_runtime"]
    assert runtime["parameter_devices"] == ["cuda:0"] and runtime["cuda_execution_verified"]
    assert runtime["gpu_peak_allocated_bytes"] > 0
    assert runtime["gpu_peak_reserved_bytes"] >= runtime["gpu_peak_allocated_bytes"]
    assert "tf32_matmul" in runtime and "deterministic_algorithms" in runtime
    assert "execution_device" not in inspect_bundle(output)
    cpu = load_standalone(output)  # artifact not hard-bound to GPU; NO cross-device numeric assertion
    assert {str(p.device) for p in cpu.model.parameters()} == {"cpu"}
    result = infer(output, "hello world", dtype=dtype, max_new_tokens=2, execution_device="cuda:0")
    assert result["generation"]["device"] == "cuda:0"
    assert torch.cuda.current_device() == original_device


@pytest.mark.parametrize("family", ["causal", "seq2seq"])
def test_default_cpu_and_explicit_cpu_identical_trained_tensors(tmp_path, native_torch, family):
    from test_specialist_recovery import tiny_stores
    from asea.specialist.recovery import recover
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    options = dict(steps=2, learning_rate=0.01, rank=2, max_length=12, dtype="float32", export_mode="factor_preserving")
    default = recover(teacher, student, tmp_path / "default", train, dev, **options)
    explicit = recover(teacher, student, tmp_path / "explicit", train, dev, execution_device="cpu", **options)
    assert default["status"] == explicit["status"] == "completed", (default.get("error"), explicit.get("error"))
    for key in ("adapter_hash_after", "frozen_parameter_hash_after", "training_history", "output_weight_sha256"):
        assert default[key] == explicit[key]


def test_unavailable_inference_and_standalone_cuda_fail_before_weights(tmp_path, native_torch, monkeypatch):
    from test_specialist_recovery import tiny_stores
    from asea.specialist import recovery
    from asea.specialist.standalone import load_standalone
    from asea.specialist.evaluation import NativeGenerator
    from transformers import Qwen2ForCausalLM
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    output = tmp_path / "factor"
    result = recovery.recover(teacher, student, output, train, dev, steps=1, rank=2, dtype="float32", export_mode="factor_preserving")
    assert result["status"] == "completed", result.get("error")
    monkeypatch.setattr(native_torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(recovery, "_load_recovery_model", lambda *a, **k: pytest.fail("loaded factor base before device rejection"))
    monkeypatch.setattr(Qwen2ForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("loaded native model before device rejection"))
    with pytest.raises(d.DeviceRejected):
        load_standalone(output, execution_device="cuda:0")
    with pytest.raises(d.DeviceRejected):
        NativeGenerator(student, dtype="float32", execution_device="cuda:0")


@pytest.mark.skipif(not _actual_cuda(), reason="GPU_UNVERIFIED: actual NVIDIA CUDA runtime required")
@pytest.mark.parametrize("family", ["causal", "seq2seq"])
def test_real_cuda_native_merged_keeps_original_tolerances(tmp_path, native_torch, family):
    from test_specialist_recovery import tiny_stores
    from asea.specialist.recovery import recover
    teacher, student, train, dev = tiny_stores(tmp_path, family)
    report = recover(teacher, student, tmp_path / "merged", train, dev, steps=2, learning_rate=0.01,
                     rank=2, max_length=12, dtype="float32", execution_device="cuda:0", export_mode="native_merged")
    assert report["status"] == "completed", report.get("error")
    assert report["merge_parity_probe"]["atol"] == report["merge_parity_probe"]["rtol"] == 2e-5
    assert report["standalone_reload_probe"]["passed"]
    assert report["standalone_reload_probe"]["execution_device"] == "cuda:0"
    assert report["resources"]["release_checks"][-1]["all_collected"]


def test_actual_cpu_evaluation_aggregate_device(tmp_path, native_torch, monkeypatch):
    """Real tiny CPU native generation; no coding-quality or GPU claim."""
    from test_specialist_recovery import tiny_stores
    from test_specialist_workflow import suite_at
    from asea.specialist import evaluation as e
    teacher, student, train, dev = tiny_stores(tmp_path, "causal")
    # Scope this regression to actual native generation/reporting, not containment.
    monkeypatch.setattr(e, "oracle_preflight", lambda: {"available": True, "evidence": "stubbed oracle preflight only"})
    report = e.evaluate(student, suite_at(tmp_path / "suite.json"), dtype="float32", max_new_tokens=2)
    assert report["generation_completed"] == 2, report
    assert report["resources"]["device"] == report["resources"]["execution_device_requested"] == "cpu"
    assert report["resources"]["device_selection"]["device"] == "cpu"
    assert report["resources"]["peak_rss_bytes"] > 0
    assert "gpu_measurements" not in report["resources"]
    for row in report["cases"]:
        generation = row["generation"]
        assert generation["resources"]["device"] == "cpu"
        assert generation["execution_runtime"]["parameter_devices"] == ["cpu"]
        assert generation["execution_runtime"]["gpu_validation_status"] == "GPU_UNVERIFIED"


@pytest.mark.parametrize("requested,effective", [("cuda:0", "cuda:0"), ("auto", "cuda:1"), ("auto", "cpu")])
def test_synthetic_evaluation_aggregate_reports_effective_device(tmp_path, monkeypatch, requested, effective):
    """SYNTHETIC report routing only; absolutely no CUDA model/operator execution."""
    from test_specialist_workflow import suite_at
    from asea.specialist import evaluation as e
    evidence = "SYNTHETIC aggregation only; GPU_UNVERIFIED"
    gpu_metadata = {"gpu_peak_allocated_bytes": 123, "gpu_validation_status": evidence}
    class StubGenerator:
        representation = "synthetic_not_a_model"
        manifest = None
        def __init__(self, *args, **kwargs):
            assert kwargs["execution_device"] == requested
            self.execution_device = effective
            self.device_selection = {"requested": requested, "device": effective, "evidence": evidence}
        def generate(self, prompt):
            return {"status": "completed", "text": "", "generation": {"truncated": True, "stop_reason": "max_new_tokens"},
                    "resources": {"device": effective, "gpu_measurements": gpu_metadata},
                    "execution_runtime": {"device": effective, "evidence": evidence}}
        def close(self):
            pass
    monkeypatch.setattr(e, "NativeGenerator", StubGenerator)
    monkeypatch.setattr(e, "oracle_preflight", lambda: {"available": True})
    monkeypatch.setattr(e, "representation_inventory", lambda path: {})
    monkeypatch.setattr(e, "profile_local_inputs", lambda *a: {"max_input_tokens": 2})
    report = e.evaluate(tmp_path, suite_at(tmp_path / "suite.json"), execution_device=requested)
    assert report["resources"]["device"] == effective
    assert report["resources"]["execution_device_requested"] == requested
    assert report["resources"]["device_selection"]["evidence"] == evidence
    # No synthetic GPU metrics copied into aggregate host RSS or relabeled real.
    assert "gpu_peak_allocated_bytes" not in report["resources"]
    assert all(row["generation"]["resources"]["gpu_measurements"] == gpu_metadata for row in report["cases"])
    assert all(row["generation"]["execution_runtime"]["evidence"] == evidence for row in report["cases"])


@pytest.mark.parametrize("stage", ["oracle_preflight", "model_load"])
@pytest.mark.parametrize("requested", ["cpu", "auto", "cuda:0"])
def test_blocked_evaluation_device_is_unresolved_not_cpu(tmp_path, monkeypatch, stage, requested):
    from test_specialist_workflow import suite_at
    from asea.specialist import evaluation as e
    def blocked(*args, **kwargs):
        raise d.DeviceRejected("synthetic unavailable device before any model loaded")
    monkeypatch.setattr(e, "NativeGenerator", blocked)
    monkeypatch.setattr(e, "oracle_preflight", lambda: {"available": stage != "oracle_preflight"})
    monkeypatch.setattr(e, "representation_inventory", lambda path: {})
    monkeypatch.setattr(e, "profile_local_inputs", lambda *a: {"max_input_tokens": 2})
    report = e.evaluate(tmp_path, suite_at(tmp_path / "suite.json"), execution_device=requested)
    assert report["status"] == "BLOCKED"
    assert report["resources"]["device"] is None
    assert report["resources"]["device_selection"] is None
    assert report["resources"]["execution_device_requested"] == requested
    assert all(row["failure_stage"] == stage for row in report["cases"])
