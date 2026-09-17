"""Fixed selected-Python tokenizer worker. No model objects or weights loaded."""
import json
from pathlib import Path
import sys

# -I ignores caller PYTHONPATH; import only this checkout's implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from asea.hardware.data import native_profile, TokenizerUnavailable
from asea.hardware.probe import _linux_resources


def main():
    try:
        raw = sys.stdin.buffer.read(64 * 1024**2 + 1)
        if len(raw) > 64 * 1024**2:
            raise ValueError('tokenizer preflight input exceeds bounded transport')
        request = json.loads(raw)
        value = native_profile(request['recipe'], request['config'], request['splits'])
        # Sample after native library imports and complete encoding, not a stale
        # pre-import estimate. Never credit the worker's later teardown as free.
        value['observed_memory_after_processing'] = _linux_resources()['memory']
    except TokenizerUnavailable as exc:
        value = {'status': 'unavailable', 'reason': str(exc)}
    except Exception as exc:
        value = {'status': 'rejected', 'reason': type(exc).__name__ + ': ' + str(exc)[:2048]}
    print('SILT_TOKENIZER_JSON=' + json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
