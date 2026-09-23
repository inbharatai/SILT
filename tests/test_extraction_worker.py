"""Extraction-program worker contracts (brief sections C3/C4/C5).

These tests run the REAL worker script (``workers/glm53/worker.py``) as a
subprocess and exercise the paths that do NOT need the Transformers 5.x
runtime: the protocol loop, exit semantics, and the EXACT checkpoint
manifest op. They are MECHANISM tests on the real code path, not
FIXTURE_VERIFIED GLM capability evidence and never real-model evidence.

C3: the ``manifest`` op hashes EVERY file of the checkpoint tree
    (per-file sha256 + byte size), disk-only, no model load, no preflight.
C4: ``_chat_encode`` uses the checkpoint's own chat template
    (apply_chat_template + add_generation_prompt=True); no raw-string
    encoding path exists.
C5 (worker side): no GLM_CHECKPOINT -> one honest blocked frame, exit 2;
    unknown op -> typed invalid_op frame.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from asea.capability_build import worker_protocol as proto
from asea.capability_build.errors import InterventionInvalid

_REPO = Path(__file__).resolve().parents[1]
_WORKER = _REPO / "workers" / "glm53" / "worker.py"


def _load_worker_module():
    """Import workers/glm53/worker.py directly from its file path (it is
    not a package member; the Docker image copies it standalone)."""
    spec = importlib.util.spec_from_file_location("extraction_worker", _WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_worker(checkpoint: str, frames, timeout=60):
    """Run the real worker script as a subprocess; feed JSONL request
    frames, return (exit_code, [response frames])."""
    process = subprocess.run(
        [sys.executable, str(_WORKER)],
        input="".join(json.dumps(frame) + "\n" for frame in frames),
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "GLM_CHECKPOINT": checkpoint},
    )
    responses = [json.loads(line) for line in process.stdout.splitlines()
                 if line.strip()]
    return process, responses


def _fixture_checkpoint(tmp_path: Path) -> Path:
    """A tiny fake checkpoint tree: config, index, one 'shard', tokenizer
    assets -- enough to prove the manifest walks EVERY file."""
    root = tmp_path / "GLM-5.3-Flash-BF16"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "glm5_next"}',
                                      encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        '{"weight_map": {}}', encoding="utf-8")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# C3 -- EXACT SourceCheckpointManifest (per-file sha256)
# ---------------------------------------------------------------------------

def test_manifest_op_hashes_every_checkpoint_file(tmp_path):
    """:op:`manifest` returns per-file sha256 + size for EVERY file in the
    tree -- weight shards included, not just the structural files the
    identity pin covers. The digests are recomputed independently in this
    test and must match byte for byte."""
    checkpoint = _fixture_checkpoint(tmp_path)
    request = proto.make_request("manifest", {})
    process, responses = _run_worker(str(checkpoint), [request])
    assert process.returncode == 0
    assert len(responses) == 1
    response = responses[0]
    assert response["ok"] is True
    result = response["result"]
    assert result["file_count"] == 5
    expected_total = 0
    for name in ("config.json", "model.safetensors.index.json",
                 "model-00001-of-00002.safetensors",
                 "model-00002-of-00002.safetensors",
                 "tokenizer_config.json"):
        raw = (checkpoint / name).read_bytes()
        expected_total += len(raw)
        entry = result["files"][name]
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["bytes"] == len(raw)
    assert result["total_bytes"] == expected_total


def test_manifest_op_requires_a_directory(tmp_path):
    """A checkpoint path that is a FILE (not a tree) is a typed refusal,
    never a crash."""
    bogus = tmp_path / "checkpoint.bin"
    bogus.write_bytes(b"not a directory")
    process, responses = _run_worker(
        str(bogus), [proto.make_request("manifest", {})])
    assert process.returncode == 0  # the loop refuses, it does not crash
    assert responses[0]["ok"] is False
    assert responses[0]["error"]["kind"] == "arch_mismatch"
    assert "not a directory" in responses[0]["error"]["message"]


def test_protocol_registers_the_manifest_op():
    assert "manifest" in proto.OPS
    request = proto.make_request("manifest", {})
    assert request["op"] == "manifest"
    with pytest.raises(ValueError):
        proto.make_request("definitely-not-an-op", {})


# ---------------------------------------------------------------------------
# C4 -- official GLM inference contract (chat template, never raw encoding)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Records the apply_chat_template invocation and returns a dict the
    generate/routing handlers can consume."""

    def __init__(self, return_dict=True, result=None):
        self.calls = []
        self.return_dict = return_dict
        self.result = result if result is not None else {"input_ids": [1, 2, 3]}

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self.return_dict and "return_dict" in kwargs:
            # mimics a stack whose signature lacks the return_dict kwarg
            raise TypeError("unexpected keyword argument 'return_dict'")
        return self.result


def test_chat_encode_uses_the_official_chat_template():
    worker = _load_worker_module()
    tokenizer = _FakeTokenizer()
    encoded = worker._chat_encode(tokenizer, "write a python function")
    assert encoded == {"input_ids": [1, 2, 3]}
    assert len(tokenizer.calls) == 1
    call = tokenizer.calls[0]
    assert call["kwargs"]["add_generation_prompt"] is True
    assert call["kwargs"]["tokenize"] is True
    # the prompt rides as a USER message, never a raw string encoding
    assert call["messages"] == [{
        "role": "user",
        "content": [{"type": "text", "text": "write a python function"}],
    }]
    # C4: BOTH thinking-related template arguments are PINNED (verified
    # against the real GLM-5.3-Flash chat template 2026-09-23: it
    # declares reasoning_effort, default 'max', and clear_thinking,
    # default false; enable_thinking does NOT exist in it) so the clean
    # and intervention arms provably share one generation policy.
    assert call["kwargs"]["clear_thinking"] is True
    assert call["kwargs"]["reasoning_effort"] == "low"
    # the pinned policy is exported for the callers' policy hash
    assert worker.PINNED_CHAT_TEMPLATE_KWARGS == {
        "clear_thinking": True, "reasoning_effort": "low"}
    assert worker.PINNED_DECODING["do_sample"] is False


def test_chat_encode_falls_back_without_raw_encoding():
    """A stack whose apply_chat_template lacks ``return_dict`` retries the
    documented form -- it never falls back to encoding the raw prompt
    string (that path produced off-distribution measurements)."""
    worker = _load_worker_module()
    tokenizer = _FakeTokenizer(return_dict=False)
    encoded = worker._chat_encode(tokenizer, "hello")
    assert encoded == {"input_ids": [1, 2, 3]}
    assert len(tokenizer.calls) == 2
    assert "return_dict" not in tokenizer.calls[1]["kwargs"]
    # BOTH invocations are still chat-template invocations
    assert all("add_generation_prompt" in c["kwargs"] for c in tokenizer.calls)


def test_chat_encode_refuses_template_output_without_input_ids():
    worker = _load_worker_module()
    tokenizer = _FakeTokenizer(result={"attention_mask": [1]})
    with pytest.raises(InterventionInvalid) as excinfo:
        worker._chat_encode(tokenizer, "hello")
    assert "input_ids" in str(excinfo.value)


class _FakeTensor(list):
    """A list with the one tensor attribute the budget check reads."""

    @property
    def shape(self):
        return (1, len(self))


def test_chat_encode_refuses_over_budget_rather_than_truncating():
    """max_length is a BUDGET, not a truncation: slicing a chat-templated
    encoding would break the template structure the inference contract
    depends on, so an over-budget prompt is a typed refusal."""
    worker = _load_worker_module()
    tokenizer = _FakeTokenizer(result={"input_ids": _FakeTensor([1, 2, 3])})
    with pytest.raises(InterventionInvalid) as excinfo:
        worker._chat_encode(tokenizer, "hello", 2)
    assert "never silently truncated" in str(excinfo.value)
    # within budget: returned unchanged
    ok = worker._chat_encode(tokenizer, "hello", 3)
    assert list(ok["input_ids"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# C5 (worker side) -- exit semantics at the subprocess level
# ---------------------------------------------------------------------------

def test_worker_exit_2_without_checkpoint(tmp_path):
    """No GLM_CHECKPOINT: exactly one honest blocked frame on stdout and
    exit code 2 -- the BLOCKED_RESOURCE semantics, before any runtime
    import happens."""
    process, responses = _run_worker("", [proto.make_request("hello", {})])
    assert process.returncode == 2
    assert len(responses) == 1
    response = responses[0]
    assert response["ok"] is False
    assert response["error"]["kind"] == "blocked"
    assert "GLM_CHECKPOINT" in response["error"]["requirement"]
    assert "remedy" in response["error"]


def test_worker_unknown_op_is_a_typed_refusal(tmp_path):
    checkpoint = _fixture_checkpoint(tmp_path)
    frame = {"protocol": proto.PROTOCOL,
             "protocol_version": proto.PROTOCOL_VERSION,
             "op": "make-coffee", "id": "x1", "payload": {}}
    process, responses = _run_worker(str(checkpoint), [frame])
    assert process.returncode == 0  # the loop survives one bad request
    assert responses[0]["ok"] is False
    assert responses[0]["error"]["kind"] == "invalid_op"
    assert "make-coffee" in responses[0]["error"]["message"]