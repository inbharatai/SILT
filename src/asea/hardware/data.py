"""Bounded TRAIN/DEV-only preflight. No model construction or FINAL suite reads.

Use recovery's complete-row validator and encoder, not an approximate tokenizer.
Final paths are metadata only: never open/hash them, including for identity checks.
"""
from __future__ import annotations

import hashlib
import stat
import os
from pathlib import Path
import random
import re
import sys

from asea.artifacts import safe_file, safe_path
from asea.specialist.recovery import (
    MAX_DATA_BYTES, _bounded_json, _read_samples, _sample_families,
    _encode_details, _encoding_contract, _json_hash,
)


class TokenizerUnavailable(ValueError):
    pass


def profile_native_selected(cfg, config, splits, python_executable=None):
    """Profile in the selected environment, with bounded isolated transport."""
    executable = os.fspath(python_executable) if python_executable is not None else sys.executable
    if Path(executable).absolute() == Path(sys.executable).absolute():
        from .probe import _linux_resources
        try:
            value = native_profile(cfg, config, splits)
        except TokenizerUnavailable as exc:
            return {'status': 'unavailable', 'reason': str(exc)}
        value['observed_memory_after_processing'] = _linux_resources()['memory']
        return value
    import json
    import subprocess
    import threading
    payload = json.dumps({'recipe': cfg, 'config': config, 'splits': splits}, ensure_ascii=False).encode()
    if len(payload) > 64 * 1024**2:
        raise ValueError('tokenizer preflight input exceeds bounded transport')
    try:
        process = subprocess.Popen([executable, '-I', str(Path(__file__).with_name('tokenizer_worker.py'))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false'))
    except OSError:
        return {'status': 'unavailable', 'reason': 'selected Python unavailable for native tokenizer preflight'}
    captured = []
    def drain():
        captured.append(process.stdout.read(4 * 1024**2 + 1))
        if len(captured[0]) > 4 * 1024**2:
            process.kill()
    def send():
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            process.stdin.close()
    reader, writer = threading.Thread(target=drain, daemon=True), threading.Thread(target=send, daemon=True)
    reader.start()
    writer.start()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        return {'status': 'unavailable', 'reason': 'bounded native tokenizer preflight timeout'}
    finally:
        writer.join(timeout=5)
        reader.join(timeout=5)
        process.stdout.close()
    if process.returncode or not captured or len(captured[0]) > 4 * 1024**2:
        raise ValueError('invalid or oversized native tokenizer profile output')
    lines = [line[len('SILT_TOKENIZER_JSON='):] for line in captured[0].decode().splitlines()
             if line.startswith('SILT_TOKENIZER_JSON=')]
    if len(lines) != 1:
        raise ValueError('invalid native tokenizer profile receipt')
    value = json.loads(lines[0])
    if value.get('status') == 'rejected':
        raise ValueError(value.get('reason', 'native encoding rejected'))
    return value


def admit_file(path, *, expected=None, forbidden=(), single_link=False):
    """Stat-only admission; no forbidden file is opened, even to establish identity."""
    path = safe_file(path)
    current = path.stat()
    identity = {'device': current.st_dev, 'inode': current.st_ino}
    if not stat.S_ISREG(current.st_mode):
        raise ValueError('bounded regular file required')
    if single_link and current.st_nlink != 1:
        raise ValueError('operator metadata must have exactly one hardlink: ' + str(path))
    if expected is not None and expected.get('identity') != identity:
        raise ValueError('preflight input identity changed before read: ' + str(path))
    for value in forbidden:
        final = Path(value)
        if path.resolve() == final.resolve():
            raise ValueError('permitted input aliases a final artifact')
        try:
            other = final.stat()
        except FileNotFoundError:
            continue
        if (current.st_dev, current.st_ino) == (other.st_dev, other.st_ino):
            raise ValueError('permitted input aliases a final artifact')
    return path, identity


def snapshot(path, limit=MAX_DATA_BYTES, *, expected=None, forbidden=(), single_link=False):
    """Open once, admit descriptor BEFORE reading, hash and parse the same bytes.

    This detects pathname substitution, not malicious same-owner in-place writes.
    A concurrent in-place edit can only be detected after reading. Not a sandbox.
    """
    if expected is not None:
        forbidden = expected.get('forbidden', forbidden)
        single_link = expected.get('single_link', single_link)
    path, identity = admit_file(path, expected=expected, forbidden=forbidden, single_link=single_link)
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_size > limit
                or identity != {'device': before.st_dev, 'inode': before.st_ino}
                or (single_link and before.st_nlink != 1)):
            raise ValueError('preflight descriptor identity/size changed before read')
        # Repeat stat-only quarantine admission after open, before the first read.
        admit_file(path, expected={'identity': identity}, forbidden=forbidden, single_link=single_link)
        raw = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
        if (len(raw) > limit or len(raw) != after.st_size
                or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                   (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError('preflight file changed during bounded reading')
    entry = {'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw), 'identity': identity,
             'forbidden': sorted(map(str, forbidden)), 'single_link': single_link}
    if expected is not None and any(entry[k] != expected[k] for k in ('sha256', 'size') if k in expected):
        raise ValueError('preflight input changed: ' + str(path))
    return raw, entry


def fingerprint(path, limit=MAX_DATA_BYTES):
    raw, entry = snapshot(path, limit)
    return {k: entry[k] for k in ('sha256', 'size')}


def _stable_read(path, reader, pins, *, forbidden=(), single_link=False):
    raw, entry = snapshot(path, expected=pins.get(str(path)), forbidden=forbidden, single_link=single_link)
    pins[str(path)] = entry
    return reader(raw)


def check_pins(pins):
    for path, entry in pins.items():
        if 'identity' in entry:
            try:
                snapshot(path, max(MAX_DATA_BYTES, entry['size']), expected=entry)
            except ValueError as exc:
                raise ValueError('preflight input changed after reading: ' + path + ': ' + str(exc)) from exc
        elif fingerprint(path, max(MAX_DATA_BYTES, entry['size'])) != entry:
            raise ValueError('preflight input changed after reading: ' + path)


def binding_from_plan(plan):
    data = plan.get('bindings', {}).get('data_metadata', {})
    names = ('training', 'validation_data', 'data_manifest', 'selection_lock', 'validation_suite')
    result = {}
    for name in names:
        value = data.get(name, {})
        if (not isinstance(value.get('path'), str) or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', ''))
                or type(value.get('size')) is not int or not isinstance(value.get('identity'), dict)):
            raise ValueError('approved plan lacks immutable permitted-data binding: ' + name)
        result[value['path']] = {k: value[k] for k in ('sha256', 'size', 'identity')}
    return result


def compare_binding(expected, actual):
    if not expected:
        raise ValueError('empty approved data binding')
    for path, entry in expected.items():
        if path not in actual or any(actual[path].get(k) != v for k, v in entry.items()):
            raise ValueError('data differs from approved plan: ' + path)


def data_snapshots(cfg, expected=None, *, calibration=False):
    """Shared planner/legacy CPU admission. Metadata has a single-link constraint.

    Calibration is runtime-only and remains an exact TRAIN subset, not a samefile
    requirement. The approved manifest binds its hash even when not profiled.
    """
    paths = permitted_paths(cfg)
    manifest_path, lock_path = paths['data_manifest'], paths['selection_lock']
    if manifest_path.name != 'manifest.json' or lock_path != manifest_path.parent / 'selection-lock.json':
        raise ValueError('only designated manifest.json and selection-lock.json metadata may be read')
    if expected is not None:
        required = {str(paths[k]) for k in ('training', 'validation_data', 'data_manifest', 'selection_lock', 'validation_suite')}
        if not isinstance(expected, dict) or set(expected) != required:
            raise ValueError('approved data binding path set mismatch')
        for entry in expected.values():
            if (not isinstance(entry, dict) or set(entry) != {'sha256', 'size', 'identity'}
                    or not isinstance(entry['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])
                    or type(entry['size']) is not int or not 0 <= entry['size'] <= MAX_DATA_BYTES
                    or not isinstance(entry['identity'], dict) or set(entry['identity']) != {'device', 'inode'}
                    or any(type(v) is not int or v < 0 for v in entry['identity'].values())):
                raise ValueError('malformed approved data SHA/size/identity binding')
    final = {manifest_path.parent / 'final-suite.json', manifest_path.parent / 'final.json'}
    raw, pins, admitted = {}, {}, {}
    # Save pre-read identities for ALL paths; metadata final declarations are unknown.
    for path in paths.values():
        _, identity = admit_file(path, forbidden=final, single_link=path in (manifest_path, lock_path),
                   expected=None if expected is None else expected.get(str(path)))
        admitted[str(path)] = dict((expected or {}).get(str(path), {}), identity=identity)
    def read(name, single=False):
        path = paths[name]
        raw[name], pins[str(path)] = snapshot(path, forbidden=final, single_link=single,
            expected=admitted[str(path)])
    read('data_manifest', True)
    manifest = _bounded_json(raw['data_manifest'])
    if not isinstance(manifest, dict) or manifest.get('schema') != 'silt.specialist.manifest.v1':
        raise ValueError('versioned specialist manifest required')
    if manifest.get('source_calibration_only') is True or manifest.get('usage') == 'source_calibration_only':
        raise ValueError('source-calibration-only data is not eligible for training')
    final = _final_paths(manifest, manifest_path.parent)
    for path in paths.values():
        admit_file(path, forbidden=final, single_link=path in (manifest_path, lock_path),
                   expected=admitted[str(path)])
    pins[str(manifest_path)]['forbidden'] = sorted(map(str, final))
    read('selection_lock', True)
    lock = _bounded_json(raw['selection_lock'])
    if not isinstance(lock, dict) or lock.get('schema') != 'silt.specialist.selection-lock.v1':
        raise ValueError('versioned specialist selection lock required')
    if lock.get('frozen_before_model_generation') is not True or lock.get('model_outputs_consulted') is not False:
        raise ValueError('selection must be frozen without model outputs')
    if pins[str(lock_path)]['sha256'] != manifest.get('selection_lock_sha256'):
        raise ValueError('selection lock hash mismatch')
    for name in ('training', 'validation_data', 'validation_suite') + (('calibration',) if calibration else ()):
        read(name)
    if expected is not None:
        compare_binding(expected, pins)
    return raw, pins, final


def admit_source_assets(root, forbidden=()):
    """Conservative single-link rule for non-weight native metadata/tokenizer assets.

    Weight hardlinks (including local cache layouts) are not globally forbidden.
    """
    for path in Path(root).rglob('*'):
        if path.is_file():
            admit_file(path, forbidden=forbidden, single_link=not path.name.endswith('.safetensors'))


def check_asset_pins(root, pins):
    names = {str(safe_path(p)) for p in Path(root).rglob('*') if p.is_file() and not p.name.endswith('.safetensors')}
    if names != set(pins):
        raise ValueError('native tokenizer/source asset set changed after inventory')
    check_pins(pins)


def _quarantined(path):
    return any(part.lower().replace('-', '_').startswith(('final', 'source_calibration_only'))
               for part in path.parts)


def permitted_paths(cfg):
    paths = {k: safe_path(cfg[k]) for k in ('training', 'calibration', 'validation_data',
                                          'validation_suite', 'data_manifest', 'selection_lock')}
    if any(_quarantined(p) for p in paths.values()):
        raise ValueError('quarantined final/source-calibration-only path is not a permitted preflight input')
    return paths


def _final_paths(manifest, root):
    """Parse path-valued metadata only, never interpret counts/SHA as paths.

    The v1 canonical final-suite.json is always quarantined. Optional explicit
    final path declarations are supported; malformed declarations fail closed.
    Unknown final path containers are rejected rather than guessed.
    """
    paths = {root / 'final-suite.json', root / 'final.json'}

    def add(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 4096 or '\x00' in value:
            raise ValueError('malformed final path metadata')
        p = Path(value)
        paths.add((p if p.is_absolute() else root / p).resolve())

    def declaration(value):
        if isinstance(value, str):
            add(value)
        elif isinstance(value, dict) and value:
            found = False
            for key, item in value.items():
                key = key.lower().replace('-', '_')
                if key in ('path', 'file', 'suite', 'data', 'suite_path', 'data_path', 'file_path'):
                    add(item)
                    found = True
                elif key not in ('sha256', 'count', 'ids', 'families'):
                    raise ValueError('unrecognized final path metadata')
            if not found:
                raise ValueError('missing path in final path metadata')
        else:
            raise ValueError('malformed final path metadata')

    for key, value in manifest.items():
        normalized = key.lower().replace('-', '_')
        if normalized in ('final', 'final_suite', 'final_data', 'final_test') or (normalized.startswith('final') and normalized.endswith(('_path', '_file'))):
            declaration(value)
    for container in ('paths', 'splits', 'artifacts'):
        if container in manifest:
            values = manifest[container]
            if not isinstance(values, dict):
                raise ValueError('malformed manifest path container')
            for key, value in values.items():
                if key.lower().replace('-', '_').startswith('final'):
                    declaration(value)
    artifacts = manifest.get('artifact_sha256')
    if not isinstance(artifacts, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               or not re.fullmatch('[0-9a-f]{64}', v) for k, v in artifacts.items()):
        raise ValueError('malformed artifact_sha256 metadata')
    for name in artifacts:
        if _quarantined(Path(name)):
            add(name)
    return {p.resolve() for p in paths}


def read_permitted(cfg, source_files, read_audit=None):
    """Validate governance first; read TRAIN/DEV and reviewed DEV-suite inputs.

    Calibration remains path/stat metadata only. The DEV suite is fully hash-bound
    but only its input strings enter tokenization. FINAL is never opened/hashed.
    """
    paths = permitted_paths(cfg)
    raw, pins, final = data_snapshots(cfg)
    if read_audit is not None:
        read_audit.extend(pins)
    def read(path, reader):
        name = next(k for k, v in paths.items() if v == path)
        return reader(raw[name])
    manifest_path, lock_path = paths['data_manifest'], paths['selection_lock']
    manifest = read(manifest_path, _bounded_json)
    if not isinstance(manifest, dict) or manifest.get('schema') != 'silt.specialist.manifest.v1':
        raise ValueError('versioned specialist manifest required')
    if manifest.get('source_calibration_only') is True or manifest.get('usage') == 'source_calibration_only':
        raise ValueError('source-calibration-only data is not eligible for training')
    final = _final_paths(manifest, manifest_path.parent)
    # resolve() handles path aliases; samefile catches hardlinks using stat only.
    for path in paths.values():
        for forbidden in final:
            if path.resolve() == forbidden or (path.exists() and forbidden.exists() and os.path.samefile(path, forbidden)):
                raise ValueError('permitted input aliases a final artifact; no data may be read')
    lock = read(safe_file(lock_path), _bounded_json)
    if not isinstance(lock, dict) or lock.get('schema') != 'silt.specialist.selection-lock.v1':
        raise ValueError('versioned specialist selection lock required')
    if lock.get('frozen_before_model_generation') is not True or lock.get('model_outputs_consulted') is not False:
        raise ValueError('selection must be frozen without model outputs')
    if pins[str(lock_path)]['sha256'] != manifest.get('selection_lock_sha256'):
        raise ValueError('selection lock hash mismatch')
    selected = lock.get('selection')
    consumed = lock.get('prior_consumed_selection', [])
    if not isinstance(selected, list) or not selected or not isinstance(consumed, list):
        raise ValueError('malformed selection metadata')
    for row in selected + consumed:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('id', 'family')):
            raise ValueError('malformed selected ID/family metadata')
    if len({r['id'] for r in selected}) != len(selected):
        raise ValueError('duplicate selected IDs')
    if any(r.get('split') not in ('train', 'validation', 'final') for r in selected):
        raise ValueError('unknown selected split')
    ids = {s: {r['id'] for r in selected if r['split'] == s} for s in ('train', 'validation', 'final')}
    families = {s: {r['family'] for r in selected if r['split'] == s} for s in ids}
    for split in ids:
        if not ids[split] or ids[split] & {r['id'] for r in consumed} or families[split] & {r['family'] for r in consumed}:
            raise ValueError('missing split or previously consumed task/family selected')
    for a, b in (('train', 'validation'), ('train', 'final'), ('validation', 'final')):
        if ids[a] & ids[b] or families[a] & families[b]:
            raise ValueError('cross-split ID/family overlap')
    # Never use metadata-declared row counts as a substitute for complete validation.
    splits, pairs = [], []
    index = {r['id']: r for r in selected}
    for name, split, basename in (('training', 'train', 'train.json'), ('validation_data', 'validation', 'validation.json')):
        path = paths[name]
        if path != manifest_path.parent / basename:
            raise ValueError('permitted data path mismatch: ' + name)
        rows, row_ids, hashes = read(safe_file(path), _read_samples)
        if pins[str(path)]['sha256'] != manifest['artifact_sha256'].get(basename):
            raise ValueError('permitted data hash mismatch: ' + name)
        if row_ids != ids[split]:
            raise ValueError('sample IDs differ from frozen selection')
        for row in rows:
            if row.get('family') != index[row['id']]['family']:
                raise ValueError('sample family differs from selection lock')
            for key in ('response_sha256', 'canonical_prompt_sha256', 'canonical_source_sha256', 'source_record_sha256', 'alpha_source_sha256'):
                if key in index[row['id']] and row.get(key) != index[row['id']][key]:
                    raise ValueError('sample provenance differs from selection lock')
        splits.append(rows)
        pairs.append(hashes)
    if pairs[0] & pairs[1] or _sample_families(splits[0]) & _sample_families(splits[1]):
        raise ValueError('training/validation content or family leakage')
    for section in ('tokenizer', 'tokenizer_check', 'environment'):
        value = manifest.get(section, {})
        if not isinstance(value, dict):
            raise ValueError('malformed tokenizer metadata')
        token_pins = value.get('tokenizer_files_sha256', {})
        if not isinstance(token_pins, dict) or any(source_files.get(k, {}).get('sha256') != v for k, v in token_pins.items()):
            raise ValueError('source tokenizer differs from locked data encoding')
    # Only after split/final-alias, lock, provenance and tokenizer binding checks
    # may the reviewed DEV suite be read. No final artifact is opened or hashed.
    suite_path = paths['validation_suite']
    if suite_path != manifest_path.parent / 'validation-suite.json':
        raise ValueError('permitted validation suite path mismatch')
    expected_suite_hash = manifest['artifact_sha256'].get('validation-suite.json')
    if not isinstance(expected_suite_hash, str):
        raise ValueError('validation suite lacks governed artifact hash; cannot profile inputs')
    from asea.specialist.evaluation import load_suite
    suite = read(safe_file(suite_path), load_suite)
    if pins[str(suite_path)]['sha256'] != expected_suite_hash:
        raise ValueError('validation suite hash mismatch')
    if len(suite.cases) != len(ids['validation']) or {c.id for c in suite.cases} != ids['validation']:
        raise ValueError('validation suite IDs differ from frozen DEV selection')
    # Transport only input strings, not tests/reference answers/oracle metadata.
    splits.append([case.input for case in suite.cases])
    check_pins(pins)
    return splits, {'hashes': pins, 'ids': {k: sorted(v) for k, v in ids.items()},
        'families': {k: sorted(v) for k, v in families.items()},
        'counts': {k: len(v) for k, v in ids.items()}, 'selection_lock_sha256': pins[str(lock_path)]['sha256'],
        'final_paths_metadata_only': sorted(map(str, final)), 'final_opened': False,
        'calibration_contents_read': False, 'validation_suite_contents_read': True,
        'validation_suite_profiled_fields': ['input'], 'prior_consumed_checked': True,
        'row_validator': 'asea.specialist.recovery._read_samples/_sample_families'}


def native_profile(cfg, config, splits):
    """Only native local tokenizer; all rows encoded before seeded subsampling."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg['source_path'], local_files_only=True, trust_remote_code=False)
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        raise TokenizerUnavailable('compatible local native tokenizer unavailable: ' + type(exc).__name__) from exc
    vocabulary = tokenizer.get_vocab()
    if not vocabulary or max(vocabulary.values()) >= config['vocab_size']:
        raise ValueError('native tokenizer IDs exceed model vocabulary')
    family = 'causal' if config['model_type'] == 'qwen2' else 'seq2seq'
    encoded, token_hashes = [], []
    for rows in splits[:2]:
        items, hashes = [], set()
        for row in rows:
            example, audit = _encode_details(row, tokenizer, family, cfg['max_length'])
            labels = example['labels'][1:] if family == 'causal' else example['labels']
            response = sum(v != -100 for v in labels)
            items.append({'id': row['id'], 'input_tokens': len(example['input_ids']),
                          'label_tokens': len(example['labels']), 'response_tokens': response,
                          'terminal_eos_positions': audit['terminal_eos_positions']})
            hashes.add(_json_hash(example))
        encoded.append(items)
        token_hashes.append(hashes)
    if token_hashes[0] & token_hashes[1]:
        raise ValueError('training/validation complete tokenized content leakage')
    all_rows = [item for split in encoded for item in split]
    rng = random.Random(cfg['seed'])
    for split in encoded:
        rng.shuffle(split)
    chosen = [split[:cfg['recovery']['max_samples']] for split in encoded]
    rows = [item for split in chosen for item in split]
    # Drop template text; hash/format and tokenizer assets pin the exact contract.
    contract = _encoding_contract(tokenizer, family)
    from asea.specialist.evaluation import profile_prompts, _current_rss
    if len(splits) != 3:
        raise ValueError('governed complete DEV suite input profile required')
    input_profile = profile_prompts(tokenizer, family, config, splits[2], cfg['max_new_tokens'])
    return {'status': 'verified', 'evaluation_input_profile': input_profile,
        'profile_process_rss_bytes': _current_rss(), 'kind': 'native_tokenizer_train_dev_preflight_not_header_only',
        'model_weights_materialized': False, 'model_execution': False,
        'python_executable': str(Path(sys.executable).resolve()),
        'encoding_contract': {k: v for k, v in contract.items() if k != 'template'},
        'all_rows_validated': True, 'tokenized_split_disjoint': True,
        'max_complete_input_tokens': max(r['input_tokens'] for r in all_rows),
        'max_complete_label_tokens': max(r['label_tokens'] for r in all_rows),
        'selected_max_complete_length': max(max(r['input_tokens'], r['label_tokens']) for r in rows),
        'max_response_tokens': max(r['response_tokens'] for r in rows),
        'total_response_tokens': sum(r['response_tokens'] for r in rows),
        'selected_rows': len(rows), 'selected': dict(zip(('training', 'validation'), chosen)),
        'seed': cfg['seed'], 'max_samples_per_split': cfg['recovery']['max_samples'],
        'selection_algorithm': 'random.Random(seed); shuffle(train); shuffle(dev); each[:max_samples]',
        'encoder': 'asea.specialist.recovery._encode_details',
        'response_positions': 'causal labels[1:] != -100; seq2seq labels != -100; includes terminal EOS'}
