"""Controller review regressions. Synthetic only; BUG proofs inverted to safe invariants."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from scripts import run_specialist_quality_experiment as r
from asea.specialist import workflow as w
from asea.specialist.__main__ import main as cli_main
from test_specialist_workflow import fake_build_setup, fake_transport_api, transport_argv
from test_specialist_quality_experiment_runner import bound_plan, built_manifest

EVIDENCE = Path(__file__).parent
REPO = Path(r.__file__).parents[1]


def evidence(name, data):
    assert data  # No rewrite of independent historical evidence.


def put(path, value):
    path.write_text(json.dumps(value))
    return path


def receipt(command='build', quality=True):
    v = {'schema_version': 1, 'command': command, 'status': 'BUILT_UNCERTIFIED', 'completed': True,
         'workspace': '/synthetic', 'manifest': '/synthetic/manifest.json',
         'result': {'engineering_complete': True, 'quality_pass': quality, 'qualified_source': True}}
    if command == 'finalize':
        v['result'].update(completed=True, status='BUILT_UNCERTIFIED', final_consumed=True, training_on_final=False)
    return {'returncode': 0, 'stdout': json.dumps(v), 'stderr': ''}


def quiet(monkeypatch):
    monkeypatch.setattr(r, 'environment_report', lambda *a: {'synthetic_environment': True})
    monkeypatch.setattr(r, 'specialist_help', lambda *a: {})


def final_fixture(tmp_path, monkeypatch):
    quiet(monkeypatch)
    lock = put(tmp_path / 'selection-lock.json', {'selection': [{'id': 'fresh', 'family': 'fresh-family', 'split': 'final'}]})
    recipe = put(tmp_path / 'recipe.json', {'source_path': str(tmp_path / 'teacher'), 'selection_lock': str(lock),
        'source_metadata': {'canonical_id': 'SYNTHETIC/model'}, 'execution_device': 'auto', 'recovery': {'steps': 2},
        **{key: str(tmp_path / name) for key, name in [('training','train.json'),('calibration','calibration.json'),('validation_data','validation.json'),('validation_suite','validation-suite.json')]}})
    ledger = put(tmp_path / 'ledger.json', {'schema_version': 1, 'authoritative': True,
        'consumed': [{'id': 'old', 'family': 'old-family'}]})
    plan = bound_plan(recipe)
    plan['plan_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    approved = put(tmp_path / 'approval.json', {'schema_version': 1, 'status': 'APPROVED', 'policy': 'engineering_all_pass_v1',
        'plan_sha256': plan['plan_sha256'], 'recipe_sha256': r.sha256_file(recipe)['sha256'],
        'exclude_ledger_sha256': r.sha256_file(ledger)['sha256'], 'compact_baseline': 'not_run_engineering_only'})
    deployment = put(tmp_path / 'deployment.json', {'schema_version': 1, 'status': 'REVIEWED', 'teacher_unavailable': True,
        'passed': True, 'tasks_exercised': 1, 'frozen_models_sha256': 'synthetic-frozen', 'execution_device': 'cpu'})
    report = tmp_path / 'report.json'
    argv = ['--recipe', str(recipe), '--workspace', str(tmp_path / 'study'), '--report', str(report), '--run-final',
        '--final-suite', str(tmp_path / 'DO-NOT-READ-final-suite.json'), '--final-output', str(tmp_path / 'final-output.json'),
        '--reviewed-plan-sha256', plan['plan_sha256'], '--acceptance-record', str(approved), '--exclude-ledger', str(ledger),
        '--deployment-receipt', str(deployment)]
    monkeypatch.setattr(r, 'hardware_plan', lambda *a: (plan, {'SYNTHETIC': True}))
    return argv, report, plan


def fake_wrapper_child(tmp_path, plan, calls, *, quality=True, edit_recipe=False):
    def fake(argv, *a, **kw):
        stage = argv[3]
        calls.append(stage)
        if stage == 'build':
            effective = Path(argv[argv.index('--recipe') + 1])
            values = json.loads(effective.read_text())
            if edit_recipe:
                values['recovery']['steps'] = 987
                put(effective, values)
            root = tmp_path / 'study'
            root.mkdir()
            put(root / 'manifest.json', built_manifest(argv, plan))
            return receipt()
        assert stage == 'finalize'
        return receipt('finalize', quality=quality)
    return fake


def test_outer_controller_death_kills_direct_child(tmp_path):
    # Tiny Python stand-in only. Its survival implies a real build supervisor and
    # therefore its own PDEATH-protected worker can also remain alive.
    child = tmp_path / 'standin.py'
    child.write_text("import os,time,json,pathlib,sys\np=pathlib.Path(sys.argv[1]);p.write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgrp()}))\ntime.sleep(30)\n")
    marker = tmp_path / 'ready.json'
    outer = tmp_path / 'outer.py'
    outer.write_text('from pathlib import Path\nfrom scripts.run_specialist_quality_experiment import run\nimport sys\nrun([sys.executable,' + repr(str(child)) + ',' + repr(str(marker)) + '],Path(' + repr(str(tmp_path)) + '),timeout=25)\n')
    proc = subprocess.Popen([sys.executable, str(outer)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    info = None
    try:
        end = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < end:
            time.sleep(.02)
        info = json.loads(marker.read_text())
        proc.kill()
        proc.wait(timeout=3)
        time.sleep(.2)
        proc_stat = Path('/proc/' + str(info['pid']) + '/stat')
        state = proc_stat.read_text().split(') ')[1].split()[0] if proc_stat.exists() else 'GONE'
        assert state in ('Z', 'GONE')
        evidence('BUG-outer-orphan', dict(outer_returncode=proc.returncode, child=info, state_after_outer_death=state,
            result='direct child is alive after controller SIGKILL; explicitly cleaned up by test'))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if info:
            try:
                os.killpg(info['pgid'], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_wrapper_kills_group_before_reaping(tmp_path, monkeypatch):
    original = os.killpg
    observations = []
    def killpg(pid, sig):
        try:
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            observations.append('still-owned')
        except ChildProcessError:
            observations.append('already-reaped')
        return original(pid, sig)
    monkeypatch.setattr(os, 'killpg', killpg)
    result = r.run([sys.executable, '-c', 'print("synthetic")'], tmp_path)
    assert observations == ['still-owned']
    evidence('BUG-wrapper-reap-order', dict(returncode=result['returncode'], killpg_observations=observations,
        consequence='group leader PID no longer pinned during killpg; unlike workflow.run_child'))


def test_effective_recipe_mutation_invalidates_review(tmp_path, monkeypatch):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(r, 'run', fake_wrapper_child(tmp_path, plan, calls, edit_recipe=True))
    assert r.main(argv) != 0
    summary = json.loads(report.read_text())
    actual = r.sha256_file(Path(summary['effective_recipe']['path']))
    assert actual['sha256'] != summary['effective_recipe']['sha256']
    assert summary['status'] == 'BLOCKED' and calls == ['build']
    assert json.loads((tmp_path / 'study/manifest.json').read_text())['config']['recovery']['steps'] == 987
    evidence('BUG-effective-recipe-review-bypass', dict(calls=calls, final_status=summary['status'],
        reviewed_steps=2, executed_mock_manifest_steps=987, declared_effective_recipe=summary['effective_recipe'],
        actual_effective_recipe=actual, certificate=summary['certificate']))


def test_effective_recipe_tamper_before_actual_workflow_has_no_effects(tmp_path, monkeypatch):
    # Real wrapper -> real workflow build; only data/model stages mocked by the
    # existing mechanics fixture. No actual model or final evaluator is called.
    recipe, calls, _ = fake_build_setup(tmp_path, monkeypatch)
    quiet(monkeypatch)
    plan = bound_plan(recipe)
    plan['model']['files'] = w.representation_inventory(tmp_path / 'teacher')
    monkeypatch.setattr(r, 'hardware_plan', lambda *a: (plan, {}))
    report = tmp_path / 'wrapper.json'
    def public_build(argv, *a, **kw):
        assert argv[3] == 'build'
        path = Path(argv[argv.index('--recipe') + 1])
        cfg = json.loads(path.read_text())
        cfg['source_quality_floor'] = .01
        put(path, cfg)
        result = w.build(path, tmp_path / 'study', expected_recipe_sha256=argv[argv.index('--expected-recipe-sha256') + 1])
        v = {'schema_version': 1, 'command': 'build', 'status': result['status'], 'completed': result['completed'],
             'workspace': str(tmp_path / 'study'), 'manifest': str(tmp_path / 'study/manifest.json'), 'result': result}
        return {'returncode': 0, 'stdout': json.dumps(v), 'stderr': ''}
    monkeypatch.setattr(r, 'run', public_build)
    rc = r.main(['--recipe', str(recipe), '--workspace', str(tmp_path / 'study'), '--report', str(report)])
    result = json.loads(report.read_text())
    assert rc != 0 and result['status'] == 'BLOCKED'
    assert not (tmp_path / 'study').exists() and calls == []



@pytest.mark.parametrize('collision', ['history', 'missing-parent'])
def test_cli_recovery_report_targets_checked_before_effects(tmp_path, monkeypatch, capsys, collision):
    _, calls = fake_transport_api(tmp_path, monkeypatch, 'recover')
    path = tmp_path / 'report.json'
    if collision == 'history':
        Path(str(path) + '.history.json').write_text('USER-OWNED')
    else:
        path = tmp_path / 'absent' / 'report.json'
    code = cli_main(transport_argv(tmp_path, 'recover') + ['--receipt-mode', 'compact', '--report', str(path)])
    emitted = json.loads(capsys.readouterr().out)
    assert code == 1 and not calls and not (tmp_path / 'model').exists()
    assert emitted['completed'] is False
    full_completed = json.loads(path.read_text())['completed'] if path.exists() else None
    if collision == 'history':
        assert Path(str(path) + '.history.json').read_text() == 'USER-OWNED'
        assert full_completed is None
    evidence('BUG-cli-report-' + collision, dict(exit=code, mock_api_calls=len(calls), model_output_created=True,
        stdout=emitted, durable_full_receipt_completed=full_completed))


def test_final_quality_false_never_promoted(tmp_path, monkeypatch):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(r, 'run', fake_wrapper_child(tmp_path, plan, calls, quality=False))
    assert r.main(argv) == 1
    value = json.loads(report.read_text())
    assert value['status'] == 'REJECTED' and value['reason_code'] == 'FINAL_ALL_PASS_GATE'
    evidence('PASS-final-quality-false', dict(status=value['status'], reason=value['reason_code'], calls=calls))


@pytest.mark.parametrize('which', ['terminal.json', 'execution-recipe.json'])
def test_wrapper_sidecars_preexist_fail_before_effects(tmp_path, monkeypatch, which):
    quiet(monkeypatch)
    monkeypatch.setattr(r, 'environment_report', lambda *a: pytest.fail('effects before path checks'))
    report = tmp_path / 'report.json'
    sidecar = Path(str(report) + '.' + which)
    sidecar.write_text('USER-OWNED')
    recipe = put(tmp_path / 'recipe.json', {'source_path': 'synthetic'})
    assert r.main(['--report', str(report), '--recipe', str(recipe), '--workspace', str(tmp_path / 'study')]) == 2
    assert not report.exists() and not (tmp_path / 'study').exists() and sidecar.read_text() == 'USER-OWNED'


def test_terminal_collision_during_effects_not_overwritten(tmp_path, monkeypatch):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    base = fake_wrapper_child(tmp_path, plan, calls)
    def fake(argv, *a, **kw):
        value = base(argv, *a, **kw)
        if argv[3] == 'finalize':
            Path(str(report) + '.terminal.json').write_text('OTHER-OWNER')
        return value
    monkeypatch.setattr(r, 'run', fake)
    with pytest.raises(FileExistsError):
        r.main(argv)
    assert Path(str(report) + '.terminal.json').read_text() == 'OTHER-OWNER'
    value = json.loads(report.read_text())
    assert value['status'] == 'FINAL_STARTING'
    evidence('PASS-terminal-collision-failclosed', dict(exception='FileExistsError', summary_status=value['status'],
        terminal_preserved='OTHER-OWNER', note='completion evidence lost, but no silent successful exit or overwrite'))


def test_actual_timeout_invalid_bytes_and_bounded_streams(tmp_path):
    result = r.run([sys.executable, '-c', "import sys,time;sys.stdout.buffer.write(b'\\xff'*1100000);sys.stdout.flush();time.sleep(10)"], tmp_path, timeout=.2)
    assert result['timeout'] and result['stdout_capture']['invalid_utf8']
    assert result['stdout_capture']['captured_bytes'] == r.CAPTURE_LIMIT
    assert result['stdout_capture']['bytes'] == 1100000
    assert r.classify_cli(result)[0] == 'BLOCKED'
    json.dumps(result)
    evidence('PASS-timeout-invalid-bounded', dict(metadata={k: v for k, v in result['stdout_capture'].items() if k != 'raw_sample_base64'}, timeout=result['timeout'],
        elapsed_seconds=result['elapsed_seconds'], classification=r.classify_cli(result)))


def test_environment_values_not_dumped(tmp_path):
    env = dict(os.environ, SYNTHETIC_SENTINEL='DO_NOT_LEAK_SYNTHETIC_SENTINEL')
    result = r.run([sys.executable, '-c', 'print("SYNTHETIC")'], tmp_path, env=env)
    assert env['SYNTHETIC_SENTINEL'] not in json.dumps(result)


def test_recover_gpu_keyword_actual_cli_to_backend(tmp_path, monkeypatch, capsys):
    _, calls = fake_transport_api(tmp_path, monkeypatch, 'recover')
    assert cli_main(transport_argv(tmp_path, 'recover') + ['--device', 'cuda:3']) == 0
    capsys.readouterr()
    assert calls[0]['execution_device'] == 'cuda:3' and 'device' not in calls[0]


@pytest.mark.parametrize('field', ['plan', 'acceptance', 'ledger', 'deployment'])
def test_required_final_evidence_blocks(tmp_path, monkeypatch, field):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    option = {'plan': '--reviewed-plan-sha256', 'acceptance': '--acceptance-record',
              'ledger': '--exclude-ledger', 'deployment': '--deployment-receipt'}[field]
    index = argv.index(option)
    del argv[index:index+2]
    calls = []
    monkeypatch.setattr(r, 'run', fake_wrapper_child(tmp_path, plan, calls))
    assert r.main(argv) != 0
    summary = json.loads(report.read_text())
    assert 'finalize' not in calls
    if field != 'deployment':
        assert calls == []
    else:
        assert calls == ['build'] and summary['reason_code'] == 'AWAITING_DEPLOYMENT_RECEIPT'


@pytest.mark.parametrize('mutation', ['plan-digest', 'recipe-sha', 'ledger-sha', 'ledger-overlap', 'frozen-model', 'device'])
def test_tampered_final_evidence_blocks(tmp_path, monkeypatch, mutation):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    if mutation == 'plan-digest':
        argv[argv.index('--reviewed-plan-sha256') + 1] = '0' * 64
    elif mutation in ('recipe-sha', 'ledger-sha'):
        path = tmp_path / 'approval.json'
        v = json.loads(path.read_text())
        v['recipe_sha256' if mutation == 'recipe-sha' else 'exclude_ledger_sha256'] = '0' * 64
        put(path, v)
    elif mutation == 'ledger-overlap':
        put(tmp_path / 'selection-lock.json', {'selection': [{'id': 'old', 'family': 'fresh-family'}]})
    else:
        path = tmp_path / 'deployment.json'
        v = json.loads(path.read_text())
        v['frozen_models_sha256' if mutation == 'frozen-model' else 'execution_device'] = 'wrong'
        put(path, v)
    monkeypatch.setattr(r, 'run', fake_wrapper_child(tmp_path, plan, calls))
    assert r.main(argv) != 0
    assert 'finalize' not in calls


def test_terminal_fsync_error_is_not_acknowledged(tmp_path, monkeypatch):
    import stat
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    base = fake_wrapper_child(tmp_path, plan, calls)
    original_fsync = os.fsync
    def fsync(fd):
        if (stat.S_ISDIR(os.fstat(fd).st_mode)
                and Path(str(report) + '.terminal.json').exists()):
            raise OSError('SYNTHETIC directory fsync failure')
        return original_fsync(fd)
    monkeypatch.setattr(r, 'run', base)
    monkeypatch.setattr(os, 'fsync', fsync)
    with pytest.raises(OSError, match='SYNTHETIC'):
        r.main(argv)
    terminal = json.loads(Path(str(report) + '.terminal.json').read_text())
    summary = json.loads(report.read_text())
    assert terminal['status'] == 'COMPLETED' and summary['status'] == 'FINAL_STARTING'
    assert terminal['final_acknowledgement']['state'] == 'REQUIRES_SEPARATE_ACK'
    assert not Path(terminal['final_acknowledgement']['path']).exists()
    evidence('CAVEAT-terminal-fsync-failure', dict(terminal_status=terminal['status'], summary_status=summary['status'],
        raises='OSError (not a silent successful exit)', note='authoritative terminal is COMPLETE even though its durability operation failed'))


def test_cuda_capability_loss_cli_classified_blocked(tmp_path, monkeypatch, capsys):
    import asea.hardware as hardware
    recipe, calls, final = fake_build_setup(tmp_path, monkeypatch)
    cfg = json.loads(recipe.read_text())
    cfg['execution_device'] = 'auto'
    put(recipe, cfg)
    monkeypatch.setattr(hardware, 'plan_specialist', lambda *a, **k:
        {'schema_version': 1, 'status': 'READY', 'execution_device': 'cuda:0'})
    built = w.build(recipe, tmp_path / 'study')
    assert built['completed'] and built['config']['execution_device'] == 'cuda:0'
    before = list(calls)
    monkeypatch.setattr(hardware, 'plan_specialist', lambda *a, **k:
        {'schema_version': 1, 'status': 'BLOCKED', 'execution_device': None, 'reasons': ['SYNTHETIC GPU disappeared']})
    code = cli_main(['finalize', '--study', str(tmp_path / 'study'), '--suite', str(final), '--output', str(tmp_path / 'no-output')])
    emitted = json.loads(capsys.readouterr().out)
    assert code == 1 and emitted['error_type'] == 'StageBlocked'
    assert emitted['status'] == 'BLOCKED' and emitted['operational_failure'] is True
    assert calls == before and not (tmp_path / 'study/final-consumed.json').exists()
    evidence('BUG-cuda-loss-classification', dict(cli_exit=code, cli_receipt=emitted,
        wrapper_classification=r.classify_cli({'returncode': code, 'stdout': json.dumps(emitted)}),
        final_consumed=False, final_model_stages=0, build_all_model_stages='mocked'))


@pytest.mark.parametrize("field,value", [("steps", 987), ("learning_rate", .02), ("source_quality_floor", .01)])
def test_complete_built_config_must_equal_approval(tmp_path, monkeypatch, field, value):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    base = fake_wrapper_child(tmp_path, plan, calls)
    def fake(command, *args, **kwargs):
        result = base(command, *args, **kwargs)
        if command[3] == "build":
            path = tmp_path / "study/manifest.json"
            manifest = json.loads(path.read_text())
            if field == "source_quality_floor":
                manifest["config"][field] = value
            else:
                manifest["config"]["recovery"][field] = value
            manifest["config_sha256"] = w.json_hash(manifest["config"])
            put(path, manifest)
        return result
    monkeypatch.setattr(r, "run", fake)
    assert r.main(argv) != 0
    assert calls == ["build"] and json.loads(report.read_text())["status"] == "BLOCKED"


def test_sidecar_tamper_during_publication_before_cli(tmp_path, monkeypatch):
    argv, report, _ = final_fixture(tmp_path, monkeypatch)
    original = r.write_report
    calls = []
    def tamper(path, value):
        original(path, value)
        changed = json.loads(path.read_text())
        changed["source_quality_floor"] = .01
        put(path, changed)
    monkeypatch.setattr(r, "write_report", tamper)
    monkeypatch.setattr(r, "run", lambda *a, **k: calls.append(a))
    assert r.main(argv) != 0 and calls == []
    assert not (tmp_path / "study").exists()


def test_sidecar_change_at_final_boundary_is_blocked(tmp_path, monkeypatch):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(r, "run", fake_wrapper_child(tmp_path, plan, calls))
    original = r.ReservedReport.write
    def mutate(self, value):
        original(self, value)
        if value.get("status") == "FINAL_STARTING":
            path = Path(value["effective_recipe"]["path"])
            changed = json.loads(path.read_text())
            changed["recovery"]["steps"] = 987
            put(path, changed)
    monkeypatch.setattr(r.ReservedReport, "write", mutate)
    assert r.main(argv) != 0 and calls == ["build"]


def test_core_hash_mismatch_precedes_model_and_workspace_effects(tmp_path, monkeypatch):
    recipe, calls, _ = fake_build_setup(tmp_path, monkeypatch)
    expected = r.sha256_file(recipe)["sha256"]
    changed = json.loads(recipe.read_text())
    changed["source_quality_floor"] = .01
    put(recipe, changed)
    monkeypatch.setattr(w, "_execution_plan", lambda *a: pytest.fail("planning before recipe-byte approval"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        w.build(recipe, tmp_path / "study", expected_recipe_sha256=expected)
    assert not (tmp_path / "study").exists() and calls == []


def test_core_parses_the_same_verified_bytes(tmp_path, monkeypatch):
    recipe, _, _ = fake_build_setup(tmp_path, monkeypatch)
    expected = r.sha256_file(recipe)["sha256"]
    original_loads = json.loads
    def swap_after_read(raw, *args, **kwargs):
        value = original_loads(raw, *args, **kwargs)
        changed = dict(value, source_quality_floor=.01)
        put(recipe, changed)
        return value
    monkeypatch.setattr(json, "loads", swap_after_read)
    config = w.recipe_config(recipe, expected_recipe_sha256=expected)
    assert config["source_quality_floor"] == 1.0
    assert r.sha256_file(recipe)["sha256"] != expected


def test_core_final_guard_before_consumed_even_if_manifest_rehashed(tmp_path, monkeypatch):
    recipe, calls, final = fake_build_setup(tmp_path, monkeypatch)
    expected = r.sha256_file(recipe)["sha256"]
    built = w.build(recipe, tmp_path / "study", expected_recipe_sha256=expected)
    assert built["completed"]
    before = list(calls)
    built["config"]["recovery"]["steps"] = 999
    built["config_sha256"] = w.json_hash(built["config"])
    put(tmp_path / "study/manifest.json", built)
    with pytest.raises(ValueError, match="effective config"):
        w.finalize(tmp_path / "study", final, tmp_path / "result", expected_recipe_sha256=expected)
    assert calls == before and not (tmp_path / "study/final-consumed.json").exists()


@pytest.mark.parametrize("collision", ["report", "history", "parent-symlink", "output", "output-parent", "history-output-alias", "input-alias"])
def test_all_recovery_targets_preadmitted_without_effects(tmp_path, monkeypatch, capsys, collision):
    _, calls = fake_transport_api(tmp_path, monkeypatch, "recover")
    report = tmp_path / "receipt.json"
    argv = transport_argv(tmp_path, "recover")
    if collision in ("report", "history"):
        Path(str(report) + (".history.json" if collision == "history" else "")).write_text("KEEP")
    elif collision == "parent-symlink":
        (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
        report = tmp_path / "link/receipt.json"
    elif collision == "output":
        (tmp_path / "model").mkdir()
    elif collision == "output-parent":
        argv[argv.index("--output-dir") + 1] = str(tmp_path / "absent/model")
    elif collision == "history-output-alias":
        argv[argv.index("--output-dir") + 1] = str(report) + ".history.json"
    else:
        argv[argv.index("--training-path") + 1] = str(report)
    assert cli_main(argv + ["--report", str(report), "--receipt-mode", "compact"]) == 1
    assert not calls and not json.loads(capsys.readouterr().out)["completed"]


def test_late_history_failure_leaves_primary_incomplete(tmp_path, monkeypatch, capsys):
    _, calls = fake_transport_api(tmp_path, monkeypatch, "recover")
    report = tmp_path / "receipt.json"
    original = w.ReceiptReservation.write
    def fail(self, value):
        if self.path.name.endswith(".history.json") and "history" in value:
            raise OSError("synthetic history write failed")
        original(self, value)
    monkeypatch.setattr(w.ReceiptReservation, "write", fail)
    assert cli_main(transport_argv(tmp_path, "recover") + ["--report", str(report), "--receipt-mode", "compact"]) == 1
    emitted = json.loads(capsys.readouterr().out)
    assert calls and emitted["status"] == "BLOCKED" and not emitted["completed"]
    assert json.loads(report.read_text())["completed"] is False


def test_receipt_owned_fd_never_overwrites_racing_replacement(tmp_path):
    report = tmp_path / "receipt"
    owner = w.ReceiptReservation(report)
    try:
        report.rename(tmp_path / "original")
        report.write_text("OTHER_OWNER")
        with pytest.raises(FileExistsError):
            owner.write({"completed": True})
    finally:
        owner.close()
    assert report.read_text() == "OTHER_OWNER"


def test_wrapper_inherited_pipe_deadline_and_unreaped_leader(tmp_path, monkeypatch):
    observed = []
    kill = os.killpg
    def killpg(pid, sig):
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        observed.append(pid)
        return kill(pid, sig)
    monkeypatch.setattr(os, "killpg", killpg)
    # Exactly one fork, tiny sleeping stand-in; no adversarial resource exhaustion.
    code = "import os,time; p=os.fork(); time.sleep(10) if p==0 else None"
    result = r.run([sys.executable, "-c", code], tmp_path, timeout=.4)
    assert result["timeout"] and result["elapsed_seconds"] < 1.5 and len(observed) == 1


def test_controller_bootstrap_parent_race_precedes_target(tmp_path):
    worker = REPO / "src/asea/specialist/controller_worker.py"
    marker = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, "-I", "-S", str(worker), "2147483647", sys.executable,
        "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"],
        start_new_session=True, timeout=3, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode != 0 and not marker.exists()


def test_wrapper_nonlinux_explicitly_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unsupported kernel execution"))
    result = r.run([sys.executable, "--version"], tmp_path)
    assert result["containment"] == "UNSUPPORTED" and r.classify_cli(result)[0] == "BLOCKED"


def test_controller_worker_in_implementation_lock():
    assert any(path.endswith("/specialist/controller_worker.py") for path in w.implementation_manifest()["source_files"])


def test_acknowledgement_normal_success(tmp_path, monkeypatch):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(r, "run", fake_wrapper_child(tmp_path, plan, calls))
    assert r.main(argv) == 0
    terminal = json.loads(Path(str(report) + ".terminal.json").read_text())
    acknowledgement = json.loads(Path(terminal["final_acknowledgement"]["path"]).read_text())
    assert acknowledgement["state"] == "TERMINAL_AND_SUMMARY_FSYNC_RETURNED"
    assert acknowledgement["terminal_sha256"] == r.sha256_file(Path(str(report) + ".terminal.json"))["sha256"]
    assert acknowledgement["power_failure_guarantee"] is False


@pytest.mark.parametrize("late", [False, True])
def test_acknowledgement_collision_never_overwrites_or_succeeds(tmp_path, monkeypatch, late):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    ack = Path(str(report) + ".ack.json")
    calls = []
    base = fake_wrapper_child(tmp_path, plan, calls)
    def child(command, *args, **kwargs):
        result = base(command, *args, **kwargs)
        if command[3] == "finalize":
            ack.write_text("OTHER_OWNER")
        return result
    monkeypatch.setattr(r, "run", child)
    if late:
        with pytest.raises(FileExistsError):
            r.main(argv)
        assert calls == ["build", "finalize"]
    else:
        ack.write_text("OTHER_OWNER")
        monkeypatch.setattr(r, "environment_report", lambda *a: pytest.fail("inventory before admission"))
        assert r.main(argv) == 2 and calls == []
    assert ack.read_text() == "OTHER_OWNER"


@pytest.mark.parametrize("binding", ["recipe_sha256", "inventory_sha256", "workspace_parent", "requested_device"])
def test_plan_source_and_config_binding_checked_before_build(tmp_path, monkeypatch, binding):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    # Exercise the build-only guard, independent of automatic-final acceptance.
    argv = argv[:argv.index("--run-final")]
    plan["bindings"][binding] = "mismatch"
    calls = []
    monkeypatch.setattr(r, "run", lambda *a, **k: calls.append(a))
    assert r.main(argv) != 0 and calls == []
    assert not (tmp_path / "study").exists()
