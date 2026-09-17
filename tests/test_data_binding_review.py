"""Static-final findings reproduced with synthetic bytes and forbidden-read sentinels.

No model construction/training. Tiny tokenizer/profile fixtures are not quality evidence.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from asea.hardware import data, plan_specialist
from asea.specialist import workflow as w
from scripts import run_specialist_quality_experiment as runner
from test_hardware_planning import recipe, profile, write_permitted_data
from test_specialist_workflow import fake_build_setup
from test_controller_integrity_review import final_fixture, fake_wrapper_child


def put(path, value):
    path.write_text(json.dumps(value))
    return path


def watch_inode(monkeypatch, path):
    """Track actual read calls, not eventual rejection or just filenames."""
    st = path.stat()
    forbidden = (st.st_dev, st.st_ino)
    reads = []
    original = io.open
    class Watched:
        def __init__(self, handle):
            self.handle = handle
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def read(self, *args):
            st = os.fstat(self.handle.fileno())
            if (st.st_dev, st.st_ino) == forbidden:
                reads.append(st.st_ino)
                raise AssertionError('FORBIDDEN FINAL INODE READ')
            return self.handle.read(*args)
        def __iter__(self):
            return self
        def __next__(self):
            st = os.fstat(self.handle.fileno())
            if (st.st_dev, st.st_ino) == forbidden:
                reads.append(st.st_ino)
                raise AssertionError('FORBIDDEN FINAL INODE READ')
            return next(self.handle)
    def opened(*args, **kwargs):
        return Watched(original(*args, **kwargs))
    monkeypatch.setattr(io, 'open', opened)  # covers Path.open AND os.fdopen
    return reads


@pytest.mark.parametrize('name', ['manifest.json', 'selection-lock.json'])
@pytest.mark.parametrize('core', [False, True])
def test_metadata_alias_before_final_declarations_never_read(tmp_path, monkeypatch, name, core):
    cfg = write_permitted_data(tmp_path)
    final = tmp_path / 'unlabelled-heldout.json'
    final.write_text('FORBIDDEN ANSWERS')
    target = tmp_path / name
    target.unlink()
    os.link(final, target)
    reads = watch_inode(monkeypatch, final)
    with pytest.raises(ValueError, match='hardlink'):
        if core:
            w.data_preflight(cfg, {})
        else:
            data.read_permitted(cfg, {})
    assert reads == []


@pytest.mark.parametrize('name', ['train.json', 'validation.json', 'validation-suite.json', 'calibration.json'])
@pytest.mark.parametrize('core', [False, True])
def test_declared_final_data_alias_is_rejected_by_stat_before_read(tmp_path, monkeypatch, name, core):
    cfg = write_permitted_data(tmp_path)
    final = tmp_path / 'sealed.json'
    final.write_text('FORBIDDEN ANSWERS')
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['final_suite_path'] = str(final)
    put(tmp_path / 'manifest.json', manifest)
    (tmp_path / name).unlink()
    os.link(final, tmp_path / name)
    reads = watch_inode(monkeypatch, final)
    with pytest.raises(ValueError, match='aliases a final'):
        (w.data_preflight if core else data.read_permitted)(cfg, {})
    assert reads == []


@pytest.mark.parametrize('which', ['train.json', 'manifest.json', 'selection-lock.json', 'validation-suite.json'])
def test_late_pin_swap_to_final_never_hashes_forbidden_inode(tmp_path, monkeypatch, which):
    cfg = write_permitted_data(tmp_path)
    _, pins, _ = data.data_snapshots(cfg)
    final, target = tmp_path / 'final-suite.json', tmp_path / which
    target.unlink()
    os.link(final, target)
    reads = watch_inode(monkeypatch, final)
    with pytest.raises(ValueError, match='changed after reading'):
        data.check_pins(pins)
    assert reads == []


@pytest.mark.parametrize('hardlink', [False, True])
def test_open_race_checks_descriptor_identity_before_first_read(tmp_path, monkeypatch, hardlink):
    target = tmp_path / 'train.json'
    target.write_text('permitted snapshot')
    _, pin = data.snapshot(target)
    final = tmp_path / 'final-suite.json'
    final.write_text('FORBIDDEN ANSWERS')
    replacement = final
    if not hardlink:
        replacement = tmp_path / 'new-single-link.json'
        replacement.write_text('FORBIDDEN ANSWERS')
    reads = watch_inode(monkeypatch, replacement)
    original = os.open
    def race(path, flags, *args, **kwargs):
        if str(path) == str(target):
            target.unlink()
            if hardlink:
                os.link(final, target)
            else:
                replacement.rename(target)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', race)
    with pytest.raises(ValueError, match='descriptor identity'):
        data.snapshot(target, expected=pin)
    assert reads == []


def test_planner_probe_intervening_swap_never_reads_final(tmp_path, monkeypatch):
    from asea.hardware import planning
    cfg = recipe(tmp_path)
    final = tmp_path / 'final-suite.json'
    reads = watch_inode(monkeypatch, final)
    def probe(*a, **kw):
        target = tmp_path / 'train.json'
        target.unlink()
        os.link(final, target)
        return profile(tmp_path)
    monkeypatch.setattr(planning, 'probe_hardware', probe)
    result = plan_specialist(cfg, workspace_parent=tmp_path)
    assert result['status'] == 'BLOCKED'
    assert any('changed after reading' in r for r in result['reasons'])
    assert reads == []


@pytest.mark.parametrize('asset', ['config.json', 'tokenizer.json'])
def test_source_asset_final_hardlink_rejected_before_inventory(tmp_path, monkeypatch, asset):
    cfg = recipe(tmp_path)
    final = tmp_path / 'final-suite.json'
    path = Path(cfg['source_path']) / asset
    path.unlink()
    os.link(final, path)
    reads = watch_inode(monkeypatch, final)
    result = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert result['status'] == 'BLOCKED'
    assert reads == []


def valid_core_fixture(tmp_path):
    cfg = recipe(tmp_path)
    train = json.loads((tmp_path / 'train.json').read_text())
    put(tmp_path / 'calibration.json', {'samples': train['samples'][:1]})
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['artifact_sha256']['calibration.json'] = data.fingerprint(tmp_path / 'calibration.json')['sha256']
    manifest['artifact_sha256']['final-suite.json'] = '0' * 64  # deliberately never hash sentinel
    put(tmp_path / 'manifest.json', manifest)
    return cfg


def test_consistent_dataset_and_manifest_substitution_blocks_public_build_before_model_stage(tmp_path, monkeypatch):
    cfg = valid_core_fixture(tmp_path)
    plan = plan_specialist(cfg, profile(tmp_path), workspace_parent=tmp_path)
    assert plan['status'] == 'READY', plan['reasons']
    expected = data.binding_from_plan(plan)
    binding = put(tmp_path / 'approved-data.json', expected)
    recipe_file = put(tmp_path / 'recipe.json', cfg)
    train = json.loads((tmp_path / 'train.json').read_text())
    train['samples'][0]['response'] = 'internally consistent but unapproved response'
    put(tmp_path / 'train.json', train)
    put(tmp_path / 'calibration.json', {'samples': train['samples'][:1]})
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    for name in ('train.json', 'calibration.json'):
        manifest['artifact_sha256'][name] = data.fingerprint(tmp_path / name)['sha256']
    put(tmp_path / 'manifest.json', manifest)
    calls = []
    monkeypatch.setattr(w, '_stage', lambda *a, **kw: calls.append(a))
    from asea.specialist.__main__ import main
    rc = main(['build', '--recipe', str(recipe_file), '--workspace', str(tmp_path / 'study'),
               '--expected-recipe-sha256', data.fingerprint(recipe_file)['sha256'],
               '--expected-data-binding-file', str(binding),
               '--expected-data-binding-sha256', data.fingerprint(binding)['sha256']])
    assert rc != 0 and calls == []
    result = json.loads((tmp_path / 'study/manifest.json').read_text())
    assert result['completed'] is False and 'changed' in result['error']
    assert result['attempted_stages'] == []


def test_core_validates_same_snapshots_and_accepts_distinct_exact_train_subset(tmp_path, monkeypatch):
    cfg = valid_core_fixture(tmp_path)
    original = w._read_samples
    seen = []
    def parsed(raw):
        assert isinstance(raw, bytes)
        seen.append(hashlib.sha256(raw).hexdigest())
        return original(raw)
    monkeypatch.setattr(w, '_read_samples', parsed)
    result = w.data_preflight(cfg, {})
    assert len(seen) == 3
    assert not os.path.samefile(cfg['training'], cfg['calibration'])
    assert set(seen) == {result['hashes'][cfg[k]]['sha256'] for k in ('training', 'validation_data', 'calibration')}


def test_original_data_change_after_stage_is_not_suppressed(tmp_path, monkeypatch):
    recipe_file, calls, final = fake_build_setup(tmp_path, monkeypatch)
    path = tmp_path / 'immutable-permitted.json'
    path.write_text('approved')
    _, pin = data.snapshot(path)
    old = w.data_preflight
    def preflight(*a):
        value = old(*a)
        value['hashes'] = {str(path): pin}
        return value
    monkeypatch.setattr(w, 'data_preflight', preflight)
    child = w.run_child
    def changing(*a, **kw):
        value = child(*a, **kw)
        path.write_text('edited after model stage')
        return value
    monkeypatch.setattr(w, 'run_child', changing)
    result = w.build(recipe_file, tmp_path / 'study')
    assert calls == ['evaluate'] and not result['completed']
    assert 'changed after reading' in result['error']


@pytest.mark.parametrize('argument', ['recipe', 'acceptance_record', 'exclude_ledger', 'deployment_receipt', 'reviewed_plan'])
def test_wrapper_prefinal_artifact_aliases_never_read(tmp_path, monkeypatch, argument):
    final = tmp_path / 'fresh-final-suite.json'
    final.write_text('FORBIDDEN ANSWERS')
    alias = tmp_path / 'neutral.json'
    os.link(final, alias)
    args = ['--report', str(tmp_path / 'report.json'), '--final-suite', str(final),
            '--' + argument.replace('_', '-'), str(alias), '--preflight-only']
    reads = watch_inode(monkeypatch, final)
    monkeypatch.setattr(runner, 'environment_report', lambda *a: pytest.fail('no effects before input admission'))
    assert runner.main(args) != 0
    assert reads == []


def test_wrapper_rejects_built_data_receipt_drift_before_final_eligibility(tmp_path, monkeypatch):
    argv, report, plan = final_fixture(tmp_path, monkeypatch)
    calls = []
    child = fake_wrapper_child(tmp_path, plan, calls)
    def drift(*a, **kw):
        result = child(*a, **kw)
        path = tmp_path / 'study/manifest.json'
        manifest = json.loads(path.read_text())
        next(iter(manifest['data']['hashes'].values()))['sha256'] = 'f' * 64
        put(path, manifest)
        return result
    monkeypatch.setattr(runner, 'run', drift)
    assert runner.main(argv) != 0
    assert calls == ['build']
    assert 'data differs from approved' in json.loads(report.read_text())['error']


def test_final_boundary_rejects_changed_build_data_before_consumed_marker(tmp_path, monkeypatch):
    recipe_file, calls, final = fake_build_setup(tmp_path, monkeypatch)
    path = tmp_path / 'immutable-permitted.json'
    path.write_text('approved')
    _, pin = data.snapshot(path)
    original = w.data_preflight
    def preflight(*a):
        value = original(*a)
        value['hashes'] = {str(path): pin}
        return value
    monkeypatch.setattr(w, 'data_preflight', preflight)
    study = tmp_path / 'study'
    result = w.build(recipe_file, study)
    assert result['completed'] is True
    calls.clear()
    path.write_text('edited after build')
    reads = watch_inode(monkeypatch, final)
    with pytest.raises(ValueError, match='changed after reading'):
        w.finalize(study, final, tmp_path / 'output.json')
    assert not (study / 'final-consumed.json').exists()
    assert reads == [] and calls == []


def test_expected_binding_digest_refuses_before_workspace_or_model_effects(tmp_path, monkeypatch):
    cfg = valid_core_fixture(tmp_path)
    recipe_file = put(tmp_path / 'recipe.json', cfg)
    binding = put(tmp_path / 'binding.json', {})
    monkeypatch.setattr(w, '_stage', lambda *a, **kw: pytest.fail('no model stage'))
    with pytest.raises(ValueError, match='binding SHA256 mismatch'):
        w.build(recipe_file, tmp_path / 'study', expected_data_binding_file=binding,
                expected_data_binding_sha256='f' * 64)
    assert not (tmp_path / 'study').exists()
