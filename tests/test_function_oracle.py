"""Finite host-parser, protocol tamper and actual Linux containment tests."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from asea.certification import function_oracle as oracle
from asea.certification import sandbox


def case(expected=5, **changes):
    value = {'id': 'one', 'function': 'add', 'args': [2, 3], 'kwargs': {}, 'expected': expected}
    value.update(changes)
    return value


@pytest.fixture(scope='module')
def enabled():
    probe = sandbox.probe_code_sandbox()
    if not probe.supported:
        pytest.skip('actual Linux containment unavailable: ' + probe.reason + probe.stderr)
    assert probe.passed


@pytest.mark.parametrize('raw', [
    b'{"x":1,"x":2}', b'NaN', b'Infinity', b'1.0', b'1e400',
    b'9223372036854775808', b'1' * 1000, b'"\\ud800"', b'"\xff"',
    b'[' * 30 + b'0' + b']' * 30,
    b'[' + b'0,' * 5000 + b'0]', b'"' + b'a' * 17000 + b'"',
    b'x' * (oracle.FRAME_BYTES + 1),
])
def test_reject_malformed_bounded_data(raw):
    with pytest.raises(oracle.ProtocolError):
        oracle._decode(raw)


@pytest.mark.parametrize('value', [1.5, float('nan'), (1,), {1: 2}, b'x', object(), 2**64, '\udfff'])
def test_only_exact_bounded_builtins(value):
    with pytest.raises(ValueError):
        oracle._validate_value(value)


def test_no_host_methods_and_bool_int_trap():
    class Evil(int):
        def __eq__(self, _):
            raise AssertionError('must never invoke')
    with pytest.raises(ValueError):
        oracle._validate_value(Evil(1))
    assert not oracle._equal(True, 1)
    assert not oracle._equal({'x': [True]}, {'x': [1]})
    assert oracle._equal({'x': [None, False, -4, 'é']}, {'x': [None, False, -4, 'é']})


@pytest.mark.parametrize('cases', [[], [case(), case()], [case(function='x.y')],
                                  [case(id=True)], [case(args=())], [case(extra=1)],
                                  [case(expected=True)] * 129])
def test_invalid_suite_never_preflights(monkeypatch, cases):
    monkeypatch.setattr(sandbox, 'probe_code_sandbox', lambda _: pytest.fail('invalid data must not launch'))
    with pytest.raises(ValueError):
        oracle.evaluate_functions('def add(a,b): return a+b', cases)


def test_failed_fresh_preflight_blocks_every_time(monkeypatch):
    calls = []
    def fail(limits):
        calls.append(limits)
        return sandbox.SandboxResult('BLOCKED', False, reason='denied')
    monkeypatch.setattr(sandbox, 'probe_code_sandbox', fail)
    monkeypatch.setattr(oracle, '_exchange', lambda *a: pytest.fail('must not execute'))
    for _ in range(2):
        result = oracle.evaluate_functions('raise AssertionError("never")', [case()])
        assert result['status'] == 'BLOCKED' and result['tests_run'] == 0
    assert len(calls) == 2


@pytest.mark.parametrize('obj', [
    {'jsonrpc': '2.0', 'id': 'future', 'result': 5},
    {'jsonrpc': '2.0', 'id': 'now', 'result': 5, 'passed': True},
    {'jsonrpc': '2.0', 'id': 'now', 'result': 5, 'error': {}},
    {'passed': True, 'tests_run': 999}, [],
])
def test_no_candidate_verdict_or_unissued_id(obj):
    with pytest.raises(ValueError):
        oracle._response(json.dumps(obj).encode(), 'now')


def test_real_positive_and_exact_case_count(enabled):
    result = oracle.evaluate_functions('def add(a,b): return a+b',
                                       [case(), case(0, id=2, args=[-3, 3])])
    assert result['status'] == 'PASSED', result
    assert result['tests_total'] == result['tests_run'] == result['tests_passed'] == 2
    assert all(len(result[k]) == 64 for k in ('source_sha256', 'inputs_sha256', 'data_sha256', 'oracle_sha256', 'sandbox_sha256'))
    assert result['resources']['memory_enforcement'] == 'rlimit_as_single_process'
    assert result['resources']['aggregate_memory_enforced'] is False


@pytest.mark.parametrize('source,expected', [
    ('def add(a,b): return True', 1),
    ('import unittest\nunittest.TestResult.wasSuccessful=lambda s: True\ndef add(a,b): return 9', 5),
    ('import os; os._exit(0)', 5),
    ('print("{\\"passed\\": true, \\"tests_run\\": 999}")\ndef add(a,b): return 9', 5),
    ('def add(a,b): return {"passed":True,"tests_run":999}', 5),
])
def test_candidate_tampering_cannot_change_host_verdict(enabled, source, expected):
    result = oracle.evaluate_functions(source, [case(expected)])
    assert result['supported'] and not result['passed'], result
    assert result['tests_run'] <= 1 and result['tests_passed'] == 0


def forge(payload_expression, after='os._exit(0)'):
    return '''import __main__ as m, os, json, struct
payload = ''' + payload_expression + '''
raw = json.dumps(payload).encode()
os.write(m.config['response_fd'], struct.pack('!I', len(raw)) + raw)
''' + after + '\n'


@pytest.mark.parametrize('expression', [
    '{"passed": True, "tests_run": 999}',
    '{"jsonrpc":"2.0", "id":"future-case", "result":5}',
    '{"jsonrpc":"2.0", "id":m.request["id"], "result":5,"passed":True}',
    '{"jsonrpc":"2.0", "id":m.request["id"], "result":99}',
])
def test_real_forged_frames_do_not_become_verdicts(enabled, expression):
    result = oracle.evaluate_functions(forge(expression), [case(), case(id='future-case')])
    assert not result['passed'] and result['tests_run'] <= 1, result


def test_exact_forged_answer_is_only_observable_output(enabled):
    source = forge('{"jsonrpc":"2.0", "id":m.request["id"], "result":5}')
    result = oracle.evaluate_functions(source, [case()])
    assert result['passed'] and result['tests_run'] == 1, result
    # The same observable answer cannot certify a different host expectation.
    changed = oracle.evaluate_functions(source, [case(6)])
    assert not changed['passed'] and changed['tests_run'] == 1, changed
    assert changed['source_sha256'] == result['source_sha256']
    assert changed['inputs_sha256'] == result['inputs_sha256']
    assert changed['data_sha256'] != result['data_sha256']


@pytest.mark.parametrize('suffix', [
    "os.write(m.config['response_fd'], struct.pack('!I',len(raw))+raw); os._exit(0)",
    "os.write(m.config['response_fd'], b'x'); os._exit(0)",
    "os.write(1, b'x'*70000); os._exit(0)",
    'os._exit(2)',
])
def test_valid_answer_then_trailing_data_flood_or_crash_never_pass(enabled, suffix):
    result = oracle.evaluate_functions(forge('{"jsonrpc":"2.0", "id":m.request["id"], "result":5}', suffix), [case()])
    assert not result['passed'], result
    assert len(result['stdout'].encode()) <= sandbox.SandboxLimits().output_bytes


def test_setup_closed_no_suite_config_host_fds_or_network(enabled, tmp_path, monkeypatch):
    import fcntl
    marker = tmp_path / 'host-marker'
    marker.write_text('scratch canary only')
    monkeypatch.setenv('ORACLE_HOST_CANARY', 'not-in-candidate')
    with marker.open('rb') as handle:
        extra = fcntl.fcntl(handle.fileno(), fcntl.F_DUPFD, 80)
        os.set_inheritable(extra, True)
        try:
            source = '''import os, socket, json, __main__ as m

def check(path, extra):
    c = json.load(open('/work/config.json'))
    out = {'config_keys': sorted(c), 'tests': os.path.exists('/work/tests.py'),
           'host': os.path.exists(path), 'proc': os.path.exists('/proc'),
           'dev': os.path.exists('/dev'), 'env': os.environ.get('ORACLE_HOST_CANARY')}
    for name, fn in [('setup',lambda: os.write(c['setup_fd'],b'forged')),
                     ('extra',lambda: os.read(extra,64)),
                     ('socket',socket.socket), ('fork',os.fork),
                     ('root_write',lambda: open('/work/forbidden','w'))]:
        try: fn(); out[name] = 'unexpected success'
        except OSError as e: out[name] = e.errno
    return out
'''
            expected = {'config_keys': ['limits', 'namespaces', 'python', 'request_fd', 'response_fd', 'setup_fd'],
                        'tests': False, 'host': False, 'proc': False, 'dev': False, 'env': None,
                        'setup': errno.EBADF, 'extra': errno.EBADF, 'socket': errno.EPERM,
                        'fork': errno.EPERM, 'root_write': errno.EROFS}
            result = oracle.evaluate_functions(source, [case(expected, function='check', args=[str(marker), extra])])
            assert result['passed'], result
        finally:
            os.close(extra)
    assert marker.read_text() == 'scratch canary only'


def test_large_prefix_is_rejected_without_body(enabled):
    source = "import __main__ as m,os,struct\nos.write(m.config['response_fd'],struct.pack('!I',2**32-1))\ndef add(a,b): return 5"
    result = oracle.evaluate_functions(source, [case()])
    assert not result['passed'] and 'protocol_limit' in result['resource_flags'], result


def test_no_reader_is_deadline_bounded(enabled):
    # First answer is forged, then candidate does not consume the next request.
    source = forge('{"jsonrpc":"2.0", "id":m.request["id"], "result":5}', 'import time; time.sleep(3)')
    cases = [case(), case(5, id='large', args=['x'*15000]*3)]
    result = oracle.evaluate_functions(source, cases, sandbox.SandboxLimits(wall_seconds=1))
    assert result['status'] == 'TIMEOUT' and result['tests_run'] == 1, result


def test_cli_json_authority_and_exclusive_output(enabled, tmp_path):
    source = tmp_path / 'source.py'
    cases = tmp_path / 'cases.json'
    output = tmp_path / 'result.json'
    source.write_text('print("forged pass")\ndef add(a,b): return a+b')
    cases.write_text(json.dumps([case()]))
    command = [sys.executable, '-m', 'asea.certification.function_oracle',
               '--source', str(source), '--cases', str(cases), '--output', str(output)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    value = json.loads(run.stdout)
    assert run.returncode == 0 and value['passed'], (run.stdout, run.stderr)
    assert json.loads(output.read_text()) == value
    assert value['stdout'] == 'forged pass\n'
    before = output.read_bytes()
    rerun = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert rerun.returncode == 1 and 'output_error' in json.loads(rerun.stdout)
    assert output.read_bytes() == before


def test_cli_invalid_json_is_structured(tmp_path):
    source = tmp_path / 'source.py'
    cases = tmp_path / 'cases.json'
    source.write_text('raise AssertionError("never execute")')
    cases.write_text('[{"id":"a","id":"b"}]')
    run = subprocess.run([sys.executable, '-m', 'asea.certification.function_oracle',
                          '--source', str(source), '--cases', str(cases)], capture_output=True, text=True)
    assert run.returncode == 1 and json.loads(run.stdout)['status'] == 'INVALID_INPUT'


def near_cap_result(char, extra=0):
    """Four individually bounded strings; response body exactly FRAME_BYTES."""
    width = len(char.encode('utf-8'))
    value = [char * (oracle.STRING_BYTES // width)] * 3 + ['']
    envelope = {'jsonrpc': '2.0', 'id': '0' * 32, 'result': value}
    used = len(json.dumps(envelope, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    remaining = oracle.FRAME_BYTES - used + extra
    value[-1] = char * (remaining // width) + 'a' * (remaining % width)
    return value


@pytest.mark.parametrize('char', ['a', 'é', '漢', '😀'])
def test_exact_utf8_frame_bytes_and_envelope_prevalidation(monkeypatch, char):
    import struct
    value = near_cap_result(char)
    envelope = {'jsonrpc': '2.0', 'id': '0' * 32, 'result': value}
    framed = oracle._frame(envelope)
    assert struct.unpack('!I', framed[:4])[0] == oracle.FRAME_BYTES
    assert len(framed) == oracle.FRAME_BYTES + 4
    assert oracle._response(framed[4:], '0' * 32) == value
    oracle._suite('def f(): pass', [case(value, id=0, function='f', args=[])], sandbox.SandboxLimits())
    # A legal case can have an impossible response because response ID/envelope
    # bytes are larger. Reject that suite before an OS probe or serialization.
    too_large = case(near_cap_result(char, 1), id=0, function='f', args=[])
    assert len(oracle._canonical(too_large)) <= oracle.FRAME_BYTES
    monkeypatch.setattr(sandbox, 'probe_code_sandbox', lambda _: pytest.fail('invalid response must not launch'))
    with pytest.raises(oracle.ProtocolError, match='frame budget'):
        oracle.evaluate_functions('def f(): pass', [too_large])


@pytest.mark.parametrize('char', ['a', 'é', '漢', '😀'])
def test_real_near_cap_unicode_roundtrip(enabled, char):
    value = near_cap_result(char)
    source = 'def f(): return ' + repr(value)
    result = oracle.evaluate_functions(source, [case(value, id=0, function='f', args=[])])
    assert result['passed'] and result['tests_run'] == 1, result


def test_real_original_unicode_counterexample_and_unicode_request(enabled):
    value = ['é' * 8192] * 2
    result = oracle.evaluate_functions('def f(): return [chr(233)*8192]*2',
                                       [case(value, function='f', args=[])])
    assert result['passed'], result
    value = {'😀': ['é' * 3000, '漢' * 2000, ''.join(map(chr, (0, 10, 9, 34, 92)))]}
    result = oracle.evaluate_functions('def f(x): return x',
                                       [case(value, function='f', args=[value])])
    assert result['passed'], result


@pytest.mark.parametrize('value', [
    None, True, False, -(2**63), 2**63-1, [], {},
    {'é😀': [''.join(chr(n) for n in range(128)), '漢', chr(0x2028) + chr(0x2029), True, None]},
])
def test_preallocation_byte_counter_matches_json(value):
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')
    assert oracle._validate_value(value) == len(raw)
    assert oracle._canonical(value) == raw


@pytest.mark.parametrize('value', [
    ['é' * 8192] * 4, ['😀' * 4096] * 4, chr(0) * 16384,
    'é' * 8193, '😀' * 4097, chr(0xd800), {chr(0xdfff): 0},
])
def test_reject_before_aggregate_json_allocation(monkeypatch, value):
    monkeypatch.setattr(json, 'dumps', lambda *a, **kw: pytest.fail('oversized/invalid JSON must not be allocated'))
    with pytest.raises(oracle.ProtocolError):
        oracle._canonical(value)


def test_total_response_bytes_include_each_header_before_preflight(monkeypatch):
    value = near_cap_result('é')
    cases = [case(value, id=i, function='f', args=[]) for i in range(16)]
    assert len(json.dumps(cases, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) < oracle.TOTAL_BYTES
    monkeypatch.setattr(sandbox, 'probe_code_sandbox', lambda _: pytest.fail('invalid response total must not launch'))
    with pytest.raises(ValueError, match='total response byte budget'):
        oracle.evaluate_functions('def f(): pass', cases)


@pytest.mark.parametrize('expression', ['chr(0xd800)', '{chr(0xdfff): 0}',
                                        'chr(233)*8193', 'chr(128512)*4097', 'chr(0)*16384'])
def test_real_invalid_or_oversized_candidate_output_still_fails(enabled, expression):
    result = oracle.evaluate_functions('def f(): return ' + expression,
                                       [case('', function='f', args=[])])
    assert result['supported'] and not result['passed'] and result['tests_passed'] == 0, result


def test_real_public_codec_tampering_does_not_control_host(enabled):
    source = '''import __main__ as m, json
m._validate_value = lambda *a, **kw: 0
m._canonical = lambda value: json.dumps({'jsonrpc':'2.0','id':value['id'],'result':True}).encode()
def add(a,b): return 5
'''
    result = oracle.evaluate_functions(source, [case(1)])
    assert result['supported'] and not result['passed'] and result['tests_run'] == 1, result


def test_real_visible_source_config_and_request_do_not_leak_expected(enabled):
    import secrets
    hidden = 'host-only-' + secrets.token_hex(24)
    source = '''import __main__ as m, json, os
print(json.dumps({'files': {p: open('/work/'+p).read() for p in os.listdir('/work')},
                  'request': m.request, 'config': m.config}))
def add(a,b): return 'observable-wrong-answer'
'''
    result = oracle.evaluate_functions(source, [case(hidden)])
    assert result['supported'] and not result['passed'] and result['tests_run'] == 1, result
    exposed = json.loads(result['stdout'])
    assert exposed['files']['candidate.py'] == source
    assert set(exposed['files']) == {'candidate.py', 'runner.py', 'config.json'}
    assert sorted(exposed['config']) == ['limits', 'namespaces', 'python', 'request_fd', 'response_fd', 'setup_fd']
    assert set(exposed['request']['params']) == {'function', 'args', 'kwargs'}
    assert hidden not in result['stdout']
    assert 'def _equal(' not in exposed['files']['runner.py']
    assert 'def _suite(' not in exposed['files']['runner.py']


def test_unicode_scalars_not_surrogates():
    assert oracle._decode(b'"\\ud83d\\ude00"') == '😀'
    for value in (chr(0xd800), chr(0xdfff), chr(0xd83d) + chr(0xde00)):
        with pytest.raises(oracle.ProtocolError, match='Unicode scalars'):
            oracle._canonical(value)


def test_real_split_header_and_body(enabled):
    source = '''import __main__ as m, os, json, struct, time
raw = json.dumps({'jsonrpc':'2.0','id':m.request['id'],'result':5}).encode()
header = struct.pack('!I', len(raw))
for part in [header[:1], header[1:3], header[3:], raw[:3], raw[3:17], raw[17:]]:
    os.write(m.config['response_fd'], part)
    time.sleep(0.01)
os._exit(0)
'''
    result = oracle.evaluate_functions(source, [case()])
    assert result['passed'] and result['tests_run'] == 1, result


@pytest.mark.parametrize('statement', [
    "os.write(fd, b'xx')",
    "os.write(fd, struct.pack('!I', 30) + b'{}')",
    "raw=b'{bad-json}'; os.write(fd, struct.pack('!I',len(raw))+raw)",
    "raw=b'{\"x\":1,\"x\":2}'; os.write(fd, struct.pack('!I',len(raw))+raw)",
    "raw=b'NaN'; os.write(fd, struct.pack('!I',len(raw))+raw)",
    "raw=bytes([34,255,34]); os.write(fd, struct.pack('!I',len(raw))+raw)",
])
def test_real_truncated_and_malformed_frames_fail_closed(enabled, statement):
    source = ('import __main__ as m, os, struct\nfd=m.config[\"response_fd\"]\n'
              + statement + '\nos._exit(0)\n')
    result = oracle.evaluate_functions(source, [case()])
    assert result['supported'] and not result['passed'] and result['tests_run'] == 0, result

