"""Device-dispatch mechanics with explicit fakes; not CUDA/quality evidence."""
import json
import sys
import types

import pytest

from asea.specialist import workflow as w
from asea.specialist.__main__ import parser
from test_specialist_workflow import fake_build_setup, recipe_at


@pytest.mark.parametrize("device", ["cpu", "auto", "cuda:0", "cuda:12"])
def test_recipe_and_cli_accept_known_device_format(tmp_path, device):
    assert w.recipe_config(recipe_at(tmp_path, execution_device=device))["execution_device"] == device
    args = parser().parse_args(["infer", "--model", "fixture", "--prompt", "fixture", "--device", device])
    assert args.execution_device == device


@pytest.mark.parametrize("device", ["cuda", "cuda:-1", "cuda:00", "cuda:1,2", "CUDA:0", "mps", "rocm", "", None, True])
def test_invalid_device_recipe_rejected(tmp_path, device):
    with pytest.raises(ValueError):
        w.recipe_config(recipe_at(tmp_path, execution_device=device))


def test_cpu_default_and_reconstruction_has_no_device(tmp_path):
    assert w.recipe_config(recipe_at(tmp_path))["execution_device"] == "cpu"
    assert parser().parse_args(["infer", "--model", "fixture", "--prompt", "fixture"]).execution_device == "cpu"
    with pytest.raises(SystemExit):
        parser().parse_args(["reconstruct", "--source-dir", "t", "--output-dir", "o", "--calibration-path", "c", "--device", "cuda:0"])
    assert w._options({"execution_device": "cuda:2"}) == ["--device", "cuda:2"]


def fake_planner(monkeypatch, device="cuda:0", status="READY"):
    calls = []
    def plan(recipe, hardware=None, workspace_parent=None, requested_device=None, inference_scope="dev"):
        assert inference_scope in ("dev", "final")
        calls.append((dict(recipe), requested_device))
        return {"schema_version": 1, "status": status, "execution_device": device, "reasons": ["FAKE_TEST_ONLY"]}
    monkeypatch.setitem(sys.modules, "asea.hardware", types.SimpleNamespace(plan_specialist=plan))
    return calls


@pytest.mark.parametrize("requested,chosen", [("cpu", "cpu"), ("auto", "cuda:0"), ("cuda:0", "cuda:0")])
def test_workflow_all_execution_stages_same_device_reconstruction_cpu(tmp_path, monkeypatch, requested, chosen):
    recipe, calls, final = fake_build_setup(tmp_path, monkeypatch)
    values = json.loads(recipe.read_text())
    values["execution_device"] = requested
    recipe.write_text(json.dumps(values))
    fake_planner(monkeypatch, device=chosen)
    child = w.run_child
    observed = []
    def capture(argv, timeout):
        observed.append(argv)
        return child(argv, timeout)
    monkeypatch.setattr(w, "run_child", capture)
    result = w.build(recipe, tmp_path / "study")
    assert result["status"] == "BUILT_UNCERTIFIED", result
    assert result["execution_device"] == result["config"]["execution_device"] == chosen
    assert result["requested_execution_device"] == requested
    assert result["quality_pass"] is False and result["certificate"] is False
    for argv in observed:
        if argv[0] == "reconstruct":
            assert "--device" not in argv
        else:
            assert argv[argv.index("--device") + 1] == chosen
    # Finalization preserves the frozen execution device; never reevaluates auto.
    finalized = w.finalize(tmp_path / "study", final, tmp_path / "result.json")
    assert finalized["completed"] and finalized["execution_device"] == chosen
    for argv in observed[-3:]:
        assert argv[argv.index("--device") + 1] == chosen


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_final_device_capability_loss_blocks_before_consumption(tmp_path, monkeypatch, device):
    recipe, _, final = fake_build_setup(tmp_path, monkeypatch)
    values = json.loads(recipe.read_text())
    values["execution_device"] = device
    recipe.write_text(json.dumps(values))
    fake_planner(monkeypatch, device=device)
    result = w.build(recipe, tmp_path / "study")
    assert result["completed"]
    def blocked_plan(*args, **kwargs):
        assert kwargs['inference_scope'] == 'final'
        assert not (tmp_path / 'study/final-consumed.json').exists()
        return {'schema_version':1, 'status':'BLOCKED', 'execution_device':None,
            'reasons':['synthetic conservative FINAL envelope does not fit']}
    monkeypatch.setitem(sys.modules, 'asea.hardware', types.SimpleNamespace(plan_specialist=blocked_plan))
    with pytest.raises(w.StageBlocked):
        w.finalize(tmp_path / "study", final, tmp_path / "result")
    assert not (tmp_path / "study/final-consumed.json").exists()


def test_final_rejects_auto_or_tampered_frozen_config(tmp_path, monkeypatch):
    recipe, _, final = fake_build_setup(tmp_path, monkeypatch)
    result = w.build(recipe, tmp_path / "study")
    assert result["completed"]
    result["config"]["execution_device"] = "auto"
    (tmp_path / "study/manifest.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="frozen recipe"):
        w.finalize(tmp_path / "study", final, tmp_path / "out")
    assert not (tmp_path / "study/final-consumed.json").exists()


def test_implementation_freezes_shared_backend_files():
    files = w.implementation_manifest()["source_files"]
    assert any(path.endswith("/specialist/devices.py") for path in files)
    assert any("/hardware/" in path and path.endswith(".py") for path in files)


@pytest.mark.parametrize("budget", [None, 1, 6 * 1024**3, 128 * 1024**3, 2**63 - 1])
def test_device_budget_recipe_cli_roundtrip(tmp_path, budget):
    config = w.recipe_config(recipe_at(tmp_path, device_memory_budget_bytes=budget))
    assert config["device_memory_budget_bytes"] == budget
    argv = w._options({"device_memory_budget_bytes": budget})
    args = parser().parse_args(["infer", "--model", "fixture", "--prompt", "fixture"] + argv)
    assert args.device_memory_budget_bytes == budget
    assert args.memory_budget_bytes is None


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "123", 2**63, float("nan"), float("inf")])
def test_device_budget_invalid(tmp_path, budget):
    with pytest.raises(ValueError):
        w.recipe_config(recipe_at(tmp_path, device_memory_budget_bytes=budget))


@pytest.mark.parametrize("budget", [4 * 1024**3, 6 * 1024**3, 128 * 1024**3, 2**63 - 1])
def test_output_budget_representational_not_machine_ceiling(tmp_path, budget):
    assert w.recipe_config(recipe_at(tmp_path, output_budget_bytes=budget))["output_budget_bytes"] == budget


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "123", 2**63, 10**1000, float("nan"), float("inf")])
def test_output_budget_invalid(tmp_path, budget):
    with pytest.raises(ValueError):
        w.recipe_config(recipe_at(tmp_path, output_budget_bytes=budget))


def test_budget_default_and_safety_caps_unchanged(tmp_path):
    assert w.recipe_config(recipe_at(tmp_path))["output_budget_bytes"] == 4 * 1024**3
    for values in ({"timeout_seconds": 3601}, {"recovery": {"steps": 4097}}):
        with pytest.raises(ValueError):
            w.recipe_config(recipe_at(tmp_path, **values))


def test_device_budget_passes_recovery_and_evaluation_not_reconstruction(tmp_path, monkeypatch):
    recipe, _, final = fake_build_setup(tmp_path, monkeypatch)
    config = json.loads(recipe.read_text())
    config.update(memory_budget_bytes=1024**3, device_memory_budget_bytes=2 * 1024**3)
    recipe.write_text(json.dumps(config))
    child = w.run_child
    observed = []
    def capture(argv, timeout):
        observed.append(argv)
        return child(argv, timeout)
    monkeypatch.setattr(w, "run_child", capture)
    assert w.build(recipe, tmp_path / "study")["completed"]
    assert w.finalize(tmp_path / "study", final, tmp_path / "out")["completed"]
    for argv in observed:
        assert argv[argv.index("--memory-budget-mib") + 1] == "1024"
        if argv[0] == "reconstruct":
            assert "--device-memory-budget-mib" not in argv
        else:
            assert argv[argv.index("--device-memory-budget-mib") + 1] == "2048"


def test_actual_cli_forwards_separate_device_budget(tmp_path, monkeypatch, capsys):
    from asea.specialist.__main__ import main
    from test_specialist_workflow import fake_transport_api, transport_argv
    _, calls = fake_transport_api(tmp_path, monkeypatch, "recover")
    assert main(transport_argv(tmp_path, "recover") + ["--memory-budget-mib", "512", "--device-memory-budget-mib", "1024"]) == 0
    capsys.readouterr()
    assert calls[0]["memory_budget_bytes"] == 512 * 1024**2
    assert calls[0]["device_memory_budget_bytes"] == 1024**3
