"""Synthetic-only observability regressions; no models or consumed references."""
import copy
import json
import signal
import subprocess
import time

import pytest

from asea.certification import _function_passed
from asea.certification import function_oracle as oracle
from asea.certification import sandbox


def case(expected=5, **changes):
    return {'id': 'synthetic-one', 'function': 'add', 'args': [2, 3],
            'kwargs': {}, 'expected': expected, **changes}


def policy(mode):
    return {'schema_version': 1, 'return_retention': mode}


def error_frame(**changes):
    data = {'schema_version': 1, 'phase': 'call', 'type_code': 'ValueError', **changes}
    return {'jsonrpc': '2.0', 'id': 'now', 'error': {
        'code': -32000, 'message': 'candidate error', 'data': data}}


@pytest.fixture(scope='module')
def enabled():
    probe = sandbox.probe_code_sandbox()
    if not probe.supported:
        pytest.skip('actual Linux containment unavailable: ' + probe.reason + probe.stderr)
    assert probe.passed


@pytest.mark.parametrize('bad', ['value', {}, {'schema_version': True, 'return_retention': 'value'},
    policy('all'), {**policy('value'), 'expected': 1}])
def test_bad_policy_before_preflight(monkeypatch, bad):
    monkeypatch.setattr(sandbox, 'probe_code_sandbox', lambda _: pytest.fail('must not launch'))
    with pytest.raises(ValueError):
        oracle.evaluate_functions('def add(a,b): return a+b', [case()], trace_policy=bad)


def test_setup_failure_and_preflight_are_distinct(monkeypatch):
    monkeypatch.setattr(sandbox, 'probe_code_sandbox', lambda _: sandbox.SandboxResult('BLOCKED', False))
    result = oracle.evaluate_functions('def add(a,b): return 5', [case()], trace_policy=policy('value'))
    assert result['diagnostic']['host_phase'] == 'preflight'
    assert result['evidence_flags'] == ['sensitive_actual_value_trace']
    monkeypatch.setattr(sandbox, 'probe_code_sandbox', lambda _: sandbox.SandboxResult('PASSED', True, True))
    def fail_runtime(root):
        raise OSError('synthetic setup failure')
    monkeypatch.setattr(sandbox, '_runtime', fail_runtime)
    result = oracle.evaluate_functions('def add(a,b): return 5', [case()])
    assert result['status'] == 'BLOCKED'
    assert result['diagnostic']['host_phase'] == 'setup'
    assert result['diagnostic']['event'] == 'transport_error'
    assert result['termination']['observed_exit_state'] == 'unknown'
    assert result['termination']['cleanup_reason'] == 'setup_failed_before_execution'


def test_error_metadata_is_not_a_result():
    with pytest.raises(oracle.CandidateError) as exc:
        oracle._response(oracle._canonical(error_frame()), 'now')
    assert exc.value.metadata == {'trust': 'untrusted_candidate', 'schema_version': 1,
                                 'phase': 'call', 'type_code': 'ValueError'}
    assert 'ValueError' not in str(exc.value)


@pytest.mark.parametrize('obj', [error_frame(passed=True), error_frame(tests_run=999),
    error_frame(phase='host'), error_frame(type_code='MyPIIException'),
    error_frame(message='private'), error_frame(schema_version=True),
    {**error_frame(), 'result': 5}, {**error_frame(), 'id': 'unissued'}])
def test_forged_error_fields_rejected(obj):
    with pytest.raises(oracle.ProtocolError) as exc:
        oracle._response(oracle._canonical(obj), 'now')
    assert not isinstance(exc.value, oracle.CandidateError)


@pytest.mark.parametrize('mode', ['none', 'digest', 'value'])
def test_policies_do_not_change_transport_limits(mode):
    value = 'é' * (oracle.STRING_BYTES // 2 + 1)
    with pytest.raises(oracle.ProtocolError):
        oracle._response(json.dumps({'jsonrpc': '2.0', 'id': 'now', 'result': value},
                                     ensure_ascii=False).encode(), 'now')
    with pytest.raises(oracle.ProtocolError):
        oracle._response(json.dumps(error_frame(type_code=value), ensure_ascii=False).encode(), 'now')
    assert (oracle.FRAME_BYTES, oracle.TOTAL_BYTES, oracle.MAX_DEPTH,
            oracle.MAX_NODES, oracle.STRING_BYTES, oracle.MAX_CASES) == (65536, 1048576, 16, 4096, 16384, 128)


def test_replay_independent_of_counters_and_bool_int():
    entry = oracle._return_trace('synthetic-one', {'x': [True]}, policy('value'))
    trace = {'schema_version': 1, 'policy': policy('value'), 'returns': [entry]}
    assert oracle.replay_returns(trace, [case({'x': [True]})])['cases'][0]['matched']
    assert not oracle.replay_returns(trace, [case({'x': [1]})])['cases'][0]['matched']
    changed = copy.deepcopy(trace)
    changed['returns'][0]['value'] = 5
    with pytest.raises(ValueError, match='mismatch'):
        oracle.replay_returns(changed, [case()])
    changed = copy.deepcopy(trace)
    changed['returns'][0]['id'] = 'other'
    with pytest.raises(ValueError):
        oracle.replay_returns(changed, [case()])
    for mode in ('none', 'digest'):
        trace = {'schema_version': 1, 'policy': policy(mode), 'returns': [
            oracle._return_trace('synthetic-one', 5, policy(mode))]}
        assert oracle.replay_returns(trace, [case()]) == {
            'schema_version': 1, 'available': False, 'reason': 'values_not_retained', 'cases': []}


def test_replay_rejects_oversized_value_not_truncation():
    entry = oracle._return_trace('synthetic-one', 5, policy('value'))
    entry['value'] = ['x' * 16000] * 5
    with pytest.raises(ValueError):
        oracle.replay_returns({'schema_version': 1, 'policy': policy('value'),
                               'returns': [entry]}, [case()])


class FakeProcess:
    pid = 54321
    def __init__(self, code=None, grace_code=None):
        self.returncode = code
        self.grace_code = grace_code
        self.waits = []
    def poll(self):
        return self.returncode
    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            if self.grace_code is not None:
                self.returncode = self.grace_code
            else:
                raise subprocess.TimeoutExpired('synthetic', timeout)
        return self.returncode


def test_observed_exit_1_never_relabelled_cleanup_kill(monkeypatch):
    monkeypatch.setattr(oracle.os, 'killpg', lambda *a: pytest.fail('must not kill exited process'))
    observed = oracle._termination(FakeProcess(1), time.monotonic() + 1, 'response_eof')
    assert observed['observed_exit'] == observed['final_exit'] == 1
    assert observed['cleanup_action'] == 'none'
    race = oracle._termination(FakeProcess(grace_code=1), time.monotonic() + 1, 'response_eof')
    assert race['observed_exit'] is None and race['observed_exit_state'] == 'unknown'
    assert race['exit_before_signal'] == race['final_exit'] == 1
    assert race['cleanup_action'] == 'none'


def test_cleanup_kill_is_separate_and_deadline_bounded(monkeypatch):
    proc = FakeProcess()
    def kill(pid, sig):
        assert pid == proc.pid and sig == signal.SIGKILL
        proc.returncode = -signal.SIGKILL
    monkeypatch.setattr(oracle.os, 'killpg', kill)
    result = oracle._termination(proc, time.monotonic() - 1, 'wall deadline exceeded')
    assert result['observed_exit'] is None and result['exit_before_signal'] is None
    assert result['final_exit'] == -9 and result['cleanup_signal'] == 'SIGKILL'
    assert result['grace_seconds'] == 0 and proc.waits == [0]
    assert result['oom_proven'] is False


@pytest.mark.parametrize('mode', ['none', 'digest', 'value'])
def test_real_wrong_value_trace_and_gate(enabled, mode):
    result = oracle.evaluate_functions('def add(a,b): return 9', [case()], trace_policy=policy(mode))
    assert result['supported'] and not result['passed'] and not _function_passed(result, [case()])
    assert result['cases'] == [{'id': 'synthetic-one', 'matched': False}]
    entry = result['return_trace']['returns'][0]
    assert entry['phase'] == 'response' and entry['retention'] == mode
    if mode == 'value':
        assert entry['value'] == 9
        assert result['evidence_flags'] == ['sensitive_actual_value_trace']
        assert not oracle.replay_returns(result['return_trace'], [case()])['cases'][0]['matched']
        # Expectations supplied afresh; never recovered from candidate verdict fields.
        assert oracle.replay_returns(result['return_trace'], [case(9)])['cases'][0]['matched']
    elif mode == 'digest':
        assert 'value' not in entry and entry['sha256'] == oracle._digest(b'9')
    else:
        assert set(entry) == {'id', 'phase', 'retention', 'reason'}
        assert entry['reason'] == 'policy_none'
    assert 'expected' not in json.dumps(result['return_trace'])


@pytest.mark.parametrize('source,phase,type_code', [
    ('raise ValueError("SECRET PII")', 'import', 'ValueError'),
    ('def add(a,b): raise ZeroDivisionError("SECRET PII")', 'call', 'ZeroDivisionError'),
    ('def add(a,b): return object()', 'serialize', 'opaque'),
    ('class PrivateError(Exception):\n def __str__(self): raise AssertionError("STR CALLED")\n'
     ' def __repr__(self): raise AssertionError("REPR CALLED")\n'
     'def add(a,b): raise PrivateError("SECRET PII")', 'call', 'opaque'),
])
def test_real_exception_envelopes_without_exception_text(enabled, source, phase, type_code):
    result = oracle.evaluate_functions(source, [case()], trace_policy=policy('value'))
    assert not result['passed'] and result['tests_run'] == 0, result
    assert result['diagnostic']['candidate_error'] == {'schema_version': 1,
        'trust': 'untrusted_candidate', 'phase': phase, 'type_code': type_code}
    assert result['diagnostic']['host_phase'] == 'response'
    assert result['diagnostic']['requested_case_id'] == 'synthetic-one'
    assert result['return_trace']['returns'] == []
    assert 'SECRET PII' not in json.dumps(result) and 'STR CALLED' not in json.dumps(result)
    assert result['termination']['final_exit'] == 1
    assert result['termination']['cleanup_action'] == 'none'


def forge(expression, suffix='os._exit(0)'):
    return ('import __main__ as m, os, json, struct\n'
            'payload = ' + expression + '\n'
            'raw=json.dumps(payload,ensure_ascii=False).encode()\n'
            "os.write(m.config['response_fd'],struct.pack('!I',len(raw))+raw)\n" + suffix)


def test_real_forged_error_is_failure_not_counter_or_resource_authority(enabled):
    obj = error_frame(type_code='MemoryError', phase='import')
    expression = repr(obj).replace("'now'", "m.request['id']")
    result = oracle.evaluate_functions(forge(expression), [case()], trace_policy=policy('value'))
    assert not result['passed'] and result['tests_run'] == result['tests_passed'] == 0
    assert result['diagnostic']['candidate_error']['type_code'] == 'MemoryError'
    assert result['resource_flags'] == [] and not result['termination']['oom_proven']
    assert result['return_trace']['returns'] == []


def test_structured_result_not_passed_fields_authoritative(enabled):
    value = {'passed': True, 'tests_run': 999}
    result = oracle.evaluate_functions('def add(a,b): return ' + repr(value), [case()],
                                       trace_policy=policy('value'))
    assert not result['passed'] and result['tests_passed'] == 0
    assert result['return_trace']['returns'][0]['value'] == value
    assert not oracle.replay_returns(result['return_trace'], [case()])['cases'][0]['matched']


def test_crash_before_diagnostic_is_not_claimed_import_failure(enabled):
    result = oracle.evaluate_functions('import os; os._exit(1)', [case()])
    assert not result['passed'] and result['tests_run'] == 0
    assert result['diagnostic']['event'] == 'response_eof'
    assert result['diagnostic']['candidate_error'] is None
    assert result['termination']['final_exit'] == 1
    assert result['termination']['cleanup_action'] == 'none'


@pytest.mark.parametrize('source', [
    'def add(a,b): return "é" * 8193',
    'def add(a,b): raise ValueError("é" * 100000)',
])
def test_real_unicode_oversize_no_truncation_or_exception_message(enabled, source):
    result = oracle.evaluate_functions(source, [case()], trace_policy=policy('value'))
    assert not result['passed'] and result['return_trace']['returns'] == []
    assert len(json.dumps(result).encode()) < 10000


def test_synthetic_seed_shaped_correct_and_false_positive_regression(enabled):
    cases = [case(), case(0, id='synthetic-two', args=[-3, 3])]
    result = oracle.evaluate_functions('def add(a,b): return a+b', cases, trace_policy=policy('value'))
    assert result['passed'] and _function_passed(result, cases), result
    assert result['cases'] == [{'id': c['id'], 'matched': True} for c in cases]
    assert oracle.replay_returns(result['return_trace'], cases)['cases'] == result['cases']
    result['passed'] = True
    result['tests_passed'] = 999
    assert not _function_passed(result, cases)


def test_replay_total_bound_and_exact_frame_survives_metadata():
    value = ['é' * 8192] * 3 + ['']
    used = len(oracle._canonical({'jsonrpc': '2.0', 'id': '0' * 32, 'result': value}))
    value[-1] = 'a' * (oracle.FRAME_BYTES - used)
    entry = oracle._return_trace('synthetic-one', value, policy('value'))
    trace = {'schema_version': 1, 'policy': policy('value'), 'returns': [entry]}
    # Metadata does not shrink the unchanged response frame budget.
    replay = oracle.replay_returns(trace, [case()])
    assert replay['available'] and not replay['cases'][0]['matched']
    trace['returns'] = [oracle._return_trace(i, value, policy('value')) for i in range(16)]
    with pytest.raises(ValueError, match='total retained response'):
        oracle.replay_returns(trace, [case(id=i) for i in range(16)])


def test_real_response_eof_then_hang_records_cleanup_not_original_sigkill(enabled):
    source = ('import __main__ as m, os, time\n'
              "os.close(m.config['response_fd'])\ntime.sleep(5)")
    result = oracle.evaluate_functions(source, [case()], sandbox.SandboxLimits(wall_seconds=1))
    assert not result['passed'] and result['diagnostic']['event'] == 'response_eof'
    termination = result['termination']
    assert termination['observed_exit'] is None and termination['exit_before_signal'] is None
    assert termination['cleanup_action'] == 'kill_process_group'
    assert termination['cleanup_signal'] == 'SIGKILL' and termination['final_exit'] == -9
    assert not termination['oom_proven'] and result['resource_flags'] == []


def test_real_value_policy_never_staged_with_expectations(enabled):
    source = ('import __main__ as m, json, os\n'
              "print(json.dumps({'files':{p:open('/work/'+p).read() for p in os.listdir('/work')},"
              "'config':m.config,'request':m.request}))\ndef add(a,b): return 9")
    expected = 'synthetic-private-host-expectation-not-staged'
    result = oracle.evaluate_functions(source, [case(expected)], trace_policy=policy('value'))
    assert not result['passed'] and result['return_trace']['returns'][0]['value'] == 9
    assert expected not in result['stdout']
    exposed = json.loads(result['stdout'])
    assert set(exposed['config']) == {'limits', 'namespaces', 'python', 'request_fd', 'response_fd', 'setup_fd'}
    assert set(exposed['request']['params']) == {'function', 'args', 'kwargs'}
    assert 'return_retention' not in exposed['files']['runner.py']


def test_cli_help_and_value_opt_in(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        oracle.main(['--help'])
    assert exc.value.code == 0
    assert '--trace-policy' in capsys.readouterr().out
    source = tmp_path / 'source.py'
    cases = tmp_path / 'cases.json'
    source.write_text('def add(a,b): return a+b')
    cases.write_text(json.dumps([case()]))
    seen = []
    def fake(source, cases, limits=None, trace_policy=None):
        seen.append(trace_policy)
        return {'passed': True, 'evidence_flags': ['sensitive_actual_value_trace']}
    monkeypatch.setattr(oracle, 'evaluate_functions', fake)
    assert oracle.main(['--source', str(source), '--cases', str(cases)]) == 0
    capsys.readouterr()
    output = tmp_path / 'private-trace.json'
    assert oracle.main(['--source', str(source), '--cases', str(cases), '--trace-policy', 'value',
                        '--output', str(output)]) == 0
    assert seen == [policy('none'), policy('value')]
    assert output.stat().st_mode & 0o077 == 0
    assert json.loads(capsys.readouterr().out)['evidence_flags'] == ['sensitive_actual_value_trace']
