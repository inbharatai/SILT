"""Explicit, pinned public-model acquisition. Never called by inference.
Downloads only inert native-model assets from Hugging Face, never Python or pickle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request

from . import Blocked, safe_path

ASSETS = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
          'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'vocab.txt', 'merges.txt',
          'spiece.model', 'tokenizer.model', 'preprocessor_config.json', 'processor_config.json',
          'normalizer.json', 'chat_template.json', 'chat_template.jinja', 'README.md', 'LICENSE',
          'LICENSE.txt', 'model.safetensors.index.json'}

def _get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise Blocked('model metadata too large')
    return json.loads(raw)

def download(repo, revision, output, max_bytes):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo) or '..' in repo:
        raise Blocked('expected public namespace/model identifier')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise Blocked('an exact 40-character model commit is required, not a mutable branch')
    destination = safe_path(output)
    if destination.exists() or not destination.parent.is_dir():
        raise Blocked('output must not exist and its parent must exist')
    if max_bytes <= 0 or max_bytes > 12 * 1024**3:
        raise Blocked('download byte ceiling must be positive and at most 12 GiB')
    metadata = _get_json('https://huggingface.co/api/models/' + repo + '/revision/' + revision + '?blobs=true')
    if metadata.get('sha') != revision:
        raise Blocked('resolved model revision mismatch')
    selected = []
    for entry in metadata.get('siblings', []):
        name = entry['rfilename']
        if name in ASSETS or re.fullmatch(r'model(?:-\d{5}-of-\d{5})?\.safetensors', name):
            if type(entry.get('size')) is not int or entry['size'] < 0:
                raise Blocked('missing declared size for ' + name)
            selected.append(entry)
    if not any(e['rfilename'].endswith('.safetensors') for e in selected):
        raise Blocked('revision has no permitted safetensors weights; pickle fallback is forbidden')
    total = sum(e['size'] for e in selected)
    if total > max_bytes or total + 1024**3 > shutil.disk_usage(destination.parent).free:
        raise Blocked('model exceeds configured byte/disk budget')
    stage = Path(tempfile.mkdtemp(prefix='.silt-model-', dir=str(destination.parent)))
    files = {}
    try:
        for entry in selected:
            name = entry['rfilename']
            url = 'https://huggingface.co/' + repo + '/resolve/' + revision + '/' + urllib.parse.quote(name)
            sha = hashlib.sha256()
            git_blob = hashlib.sha1(('blob ' + str(entry['size']) + '\x00').encode())
            count = 0
            request = urllib.request.Request(url, headers={'User-Agent': 'SILT-explicit-model-acquisition/1'})
            with urllib.request.urlopen(request, timeout=90) as response, (stage/name).open('xb') as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > entry['size']:
                        raise Blocked('download exceeds declared size: ' + name)
                    target.write(chunk)
                    sha.update(chunk)
                    git_blob.update(chunk)
            if count != entry['size']:
                raise Blocked('truncated asset: ' + name)
            expected = (entry.get('lfs') or {}).get('sha256')
            blob_id = entry.get('blobId')
            if expected:
                if sha.hexdigest() != expected:
                    raise Blocked('remote LFS digest mismatch: ' + name)
            elif not isinstance(blob_id, str) or not re.fullmatch(r'[0-9a-f]{40}', blob_id) or git_blob.hexdigest() != blob_id:
                raise Blocked('missing or mismatched remote Git-blob digest: ' + name)
            files[name] = {'bytes': count, 'sha256': sha.hexdigest(), 'remote_lfs_verified': bool(expected), 'remote_git_blob_verified': not bool(expected)}
            print('downloaded ' + name + ': ' + str(count) + ' bytes', file=sys.stderr, flush=True)
        receipt = {'schema_version': 1, 'repo': repo, 'revision': revision, 'files': files,
                   'license': (metadata.get('cardData') or {}).get('license'),
                   'scope': 'downloaded source identity/integrity only; no inference or capability certification'}
        (stage/'acquisition.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        if destination.exists():
            raise Blocked('output appeared during download')
        # The exists() check alone races with another creator; never replace even
        # an empty user directory that appears immediately before publication.
        from .bundle import _publish_directory
        _publish_directory(stage, destination)
        return {'output': str(destination), 'bytes': total, **receipt}
    finally:
        if stage.exists():
            shutil.rmtree(stage)

def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m asea.artifacts')
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('download', help='Explicit pinned safe model acquisition; never implicit inference fallback')
    fetch.add_argument('--repo', required=True)
    fetch.add_argument('--revision', required=True)
    fetch.add_argument('--output', required=True)
    fetch.add_argument('--max-bytes', type=int, default=4 * 1024**3)
    ingest = sub.add_parser('import-bundle', help='Import dependencies and rebind a spec; never imports admission authority')
    ingest.add_argument('--archive', required=True)
    ingest.add_argument('--output', required=True)
    ingest.add_argument('--spec-output', required=True)
    ingest.add_argument('--max-bytes', type=int, default=4 * 1024**3)
    args = parser.parse_args(argv)
    try:
        if args.command == 'import-bundle':
            from .bundle import import_bundle
            result = import_bundle(args.archive, args.output, args.spec_output, args.max_bytes)
        else:
            result = download(args.repo, args.revision, args.output, args.max_bytes)
        print(json.dumps({'ok': True, 'result': result}, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': {'type': type(exc).__name__, 'message': str(exc)}}))
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
