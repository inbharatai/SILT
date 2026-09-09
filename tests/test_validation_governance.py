"""Unit fixtures only: no real quality evaluation, ratings, models or approvals."""
import copy
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import wave

import pytest
from pydantic import ValidationError

from asea.artifacts import Blocked, canonical, digest
import asea.validation as validation
from asea.validation import (ValidationRegistry, exclusive_json, license_status, load_json,
                             prepare_listening_batch, prepare_voice_evidence,
                             proportion_summary, validate_listening_review)
from asea.validation.schema import Suite


def card(spdx="CC-BY-NC-4.0"):
    return {"spdx": spdx, "source": "https://example.invalid/unit-fixture", "revision": "fixture-v1",
            "license_text_sha256": "a" * 64, "attribution": "Unit fixture; not real data",
            "usage_restrictions": [], "consent_privacy": "not_applicable", "redistribution": "prohibited"}


def suite(name="fixture"):
    members = []
    for i, (split, role) in enumerate((("development", "target"), ("final", "target"), ("final", "control"))):
        content = {"input": "fixture input %d" % i, "expected": "fixture expected %d" % i}
        members.append({"member_id": "case-%d" % i, "dataset_id": "fixture", "family_id": "family-%d" % i,
                        "split": split, "role": role, "language": "en", **content, "content_sha256": digest(content)})
    return {"suite_id": name, "version": "1.0.0", "owner": "fixture-not-custodian",
            "format": "data_only_text_v1", "exposure": "local_training_overlap_unknown",
            "selection_method": "all unit fixtures", "selection_seed": 0, "sampling_frame": "unit fixtures only",
            "datasets": {"fixture": card()}, "members": members}


def policy():
    return {"purpose": "noncommercial_experimental", "evaluator_revision": "unit-oracle-v1",
            "metric": "exact_text", "threshold": 1.0, "max_cases": 3, "max_seconds": 5,
            "stopping_rule": "one_attempt_all_cases", "missing_output": "failure"}


def setup_registry(tmp_path):
    registry = ValidationRegistry(tmp_path / "candidate")
    registry.register(suite())
    reservation = registry.freeze("fixture", "b" * 64, policy())
    return registry, reservation


def test_lifecycle_exact_binding_consumed_even_failure(tmp_path):
    registry, reservation = setup_registry(tmp_path)
    assert registry.root.parent == tmp_path and registry.root != registry.workspace
    assert reservation["state"] == "frozen_final"
    assert "not independent custodian" in reservation["boundary"]
    with pytest.raises(Blocked, match="binding"):
        registry.begin(reservation["reservation_id"], "c" * 64, policy())
    changed = {**policy(), "threshold": 0.5}
    with pytest.raises(Blocked, match="binding"):
        registry.begin(reservation["reservation_id"], "b" * 64, changed)
    run = registry.begin(reservation["reservation_id"], "b" * 64, policy())
    assert run["state"] == "consumed"
    assert len(registry.final_inputs(run["run_id"])) == 2
    result = registry.finish(run["run_id"], {"status": "infrastructure_failure", "reason": "fixture failed",
                                            "evaluator_revision": "unit-oracle-v1"})
    assert result["quality_admission"] is False
    with pytest.raises(Blocked, match="consumed"):
        registry.begin(reservation["reservation_id"], "b" * 64, policy())
    with pytest.raises(Blocked, match="already finished"):
        registry.finish(run["run_id"], result["result_input"])
    with pytest.raises(Blocked):
        registry.final_inputs(run["run_id"])
    with pytest.raises(Blocked, match="reuse"):
        registry.freeze("fixture", "b" * 64, policy())
    assert any(e["kind"] == "access" for e in registry.history()["events"])


def test_never_accept_pass_or_python_and_no_rename_reuse(tmp_path):
    registry, reservation = setup_registry(tmp_path)
    run = registry.begin(reservation["reservation_id"], "b" * 64, policy())
    for result in ({"passed": True}, {"status": "passed"}, {"status": "completed", "passed": True,
                   "reason": "fixture", "evaluator_revision": "unit-oracle-v1"}):
        with pytest.raises(ValidationError):
            registry.finish(run["run_id"], result)
    other = suite("renamed")
    registry.register(other)
    with pytest.raises(Blocked, match="already reserved"):
        registry.freeze("renamed", "d" * 64, policy())
    malicious = suite("code")
    malicious["trusted_tests"] = "__import__('os').system('false')"
    with pytest.raises(ValidationError):
        registry.register(malicious)
    malicious = suite("code")
    malicious["members"][0]["python_test"] = "assert True"
    with pytest.raises(ValidationError):
        registry.register(malicious)


def test_suite_hash_immutability_and_group_split(tmp_path):
    registry = ValidationRegistry(tmp_path / "candidate")
    registry.register(suite())
    with pytest.raises(Blocked, match="immutable"):
        registry.register(suite())
    changed = suite("other")
    changed["members"][0]["split"] = "final"
    with pytest.raises(Blocked, match="crosses"):
        registry.register(changed)
    changed = suite("bad")
    changed["members"][0]["expected"] = "tampered"
    with pytest.raises(ValidationError, match="hash"):
        Suite.model_validate(changed)
    changed = suite("family")
    changed["members"][0]["family_id"] = changed["members"][1]["family_id"]
    with pytest.raises(ValidationError, match="family"):
        Suite.model_validate(changed)


def test_license_cards_block_unknown_and_nc_commercial(tmp_path):
    assert license_status([card()])["commercial_product_eligible"] is False
    assert license_status([card()])["licenses"][0]["spdx"] == "CC-BY-NC-4.0"
    assert license_status([card("unknown")])["commercial_terms_status"] == "blocked"
    registry = ValidationRegistry(tmp_path / "candidate")
    registry.register(suite())
    with pytest.raises(Blocked, match="commercial"):
        registry.freeze("fixture", "b" * 64, {**policy(), "purpose": "commercial"})
    unknown = suite("unknown")
    unknown["datasets"]["fixture"] = card("LicenseRef-Unknown")
    registry.register(unknown)
    with pytest.raises(Blocked, match="unknown license"):
        registry.freeze("unknown", "b" * 64, policy())


def test_recovery_does_not_reset_consumed_or_read_keys(tmp_path):
    registry, reservation = setup_registry(tmp_path)
    registry.begin(reservation["reservation_id"], "b" * 64, policy())
    pending = registry.root / "events" / ".pending-fixture"
    pending.write_text("partial bytes")
    result = ValidationRegistry(tmp_path / "candidate").recover()
    assert result["removed_uncommitted_files"] == [".pending-fixture"]
    assert not pending.exists()
    with pytest.raises(Blocked, match="consumed"):
        registry.begin(reservation["reservation_id"], "b" * 64, policy())
    assert not (tmp_path / "candidate").exists()  # no Workspace(), keys or runtime setup
    assert not list(registry.root.rglob("*key*"))


def test_paths_exclusive_corruption_and_writer_lock(tmp_path):
    workspace = tmp_path / "candidate"
    with pytest.raises(Blocked, match="disjoint"):
        ValidationRegistry(workspace, workspace / "inside")
    with pytest.raises(Blocked, match="traversal"):
        ValidationRegistry(tmp_path / ".." / "unsafe")
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(Blocked, match="symlink"):
        ValidationRegistry(link / "candidate")
    output = tmp_path / "output.json"
    exclusive_json(output, {"fixture": 1})
    with pytest.raises(FileExistsError):
        exclusive_json(output, {"fixture": 2})
    assert load_json(output) == {"fixture": 1}
    registry, _ = setup_registry(tmp_path)
    import fcntl
    with (registry.root / ".writer.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(Blocked, match="busy"):
            registry.history()
    first = registry.root / "events" / "00000001.json"
    changed = load_json(first)
    changed["data"]["suite"]["owner"] = "tampered"
    first.write_text(json.dumps(changed))
    with pytest.raises(Blocked, match="corrupted"):
        registry.recover()


def registry_bytes(registry):
    """Observe without history(), which intentionally appends an access event."""
    return {str(p.relative_to(registry.root)): p.read_bytes()
            for p in registry.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("payload", ["members", "escaped_exclusions"])
def test_aggregate_serialized_suite_budget_rejects_without_mutation(tmp_path, payload):
    registry = ValidationRegistry(tmp_path / "candidate")
    data = suite()
    if payload == "members":
        template = data["members"][0]
        data["members"] = []
        for i in range(70):
            content = {"input": str(i) + "x" * 65530, "expected": "y" * 65536}
            data["members"].append({**template, "member_id": "large-%d" % i,
                "family_id": "large-%d" % i, **content, "content_sha256": digest(content)})
    else:
        # JSON escaping, not character count or UTF-8 input length, sets the budget.
        data["exclusions"] = ["\u0001" * (validation.MAX_JSON_BYTES // 6)]
    normalized = Suite.model_validate(data).model_dump()  # individually valid fields
    assert len(canonical(normalized)) > validation.MAX_JSON_BYTES
    before = registry_bytes(registry)
    with pytest.raises(Blocked, match="reader byte budget"):
        registry.register(data)
    assert registry_bytes(registry) == before
    reopened = ValidationRegistry(registry.workspace)
    assert registry_bytes(reopened) == before
    reopened.register(suite())  # rejection did not reserve the ID or poison replay
    assert len(reopened._events()) == 1


@pytest.mark.parametrize("headroom", [-1, 0])
def test_event_envelope_included_in_exact_byte_boundary(tmp_path, monkeypatch, headroom):
    monkeypatch.setattr(validation.time, "time_ns", lambda: 123456789)
    probe = ValidationRegistry(tmp_path / "probe")
    data = probe.register(suite())
    event_size = len((probe.root / "events" / "00000001.json").read_bytes())
    assert len(canonical(data)) < event_size - 1
    registry = ValidationRegistry(tmp_path / "candidate")
    before = registry_bytes(registry)
    monkeypatch.setattr(validation, "MAX_JSON_BYTES", event_size + headroom)
    if headroom < 0:
        with pytest.raises(Blocked, match="reader byte budget"):
            registry.register(suite())
        assert registry_bytes(registry) == before
    else:
        registry.register(suite())
        assert (registry.root / "events" / "00000001.json").stat().st_size == event_size
    reopened = ValidationRegistry(registry.workspace)
    assert len(reopened._events()) == (1 if headroom == 0 else 0)


@pytest.mark.parametrize("invalid_policy", [{"max_cases": 1}, {"max_seconds": 0}])
def test_reservation_budget_rejected_before_freeze(tmp_path, invalid_policy):
    registry = ValidationRegistry(tmp_path / "candidate")
    registry.register(suite())
    before = registry_bytes(registry)
    with pytest.raises((Blocked, ValidationError)):
        registry.freeze("fixture", "b" * 64, {**policy(), **invalid_policy})
    assert registry_bytes(registry) == before
    reopened = ValidationRegistry(registry.workspace)
    reservation = reopened.freeze("fixture", "b" * 64, policy())
    assert reservation["state"] == "frozen_final"
    assert [e["kind"] for e in reopened._events()] == ["register", "freeze"]


def test_duplicate_policy_and_failed_finish_event_timing(tmp_path):
    registry, reservation = setup_registry(tmp_path)
    before = registry_bytes(registry)
    for operation, message in [
        (lambda: registry.register(suite()), "immutable"),
        (lambda: registry.freeze("fixture", "c" * 64, policy()), "reuse"),
        (lambda: registry.begin(reservation["reservation_id"], "b" * 64,
                                {**policy(), "threshold": 0.5}), "binding"),
        (lambda: registry.final_inputs("not-begun"), "absent"),
    ]:
        with pytest.raises(Blocked, match=message):
            operation()
        assert registry_bytes(registry) == before
    run = registry.begin(reservation["reservation_id"], "b" * 64, policy())
    assert [e["kind"] for e in registry._events()] == ["register", "freeze", "begin"]
    assert registry._events()[-1]["data"]["state"] == "consumed"
    consumed = registry_bytes(registry)
    result = {"status": "completed", "reason": "unit fixture only", "evaluator_revision": "unit-oracle-v1"}
    with pytest.raises(ValidationError):
        registry.finish(run["run_id"], {"passed": True})
    with pytest.raises(Blocked, match="revision"):
        registry.finish(run["run_id"], {**result, "evaluator_revision": "wrong"})
    assert registry_bytes(registry) == consumed
    reopened = ValidationRegistry(registry.workspace)
    with pytest.raises(Blocked, match="consumed"):
        reopened.begin(reservation["reservation_id"], "b" * 64, policy())
    assert registry_bytes(reopened) == consumed
    reopened.finish(run["run_id"], result)
    finished = registry_bytes(reopened)
    with pytest.raises(Blocked, match="already finished"):
        reopened.finish(run["run_id"], result)
    assert registry_bytes(reopened) == finished
    assert [e["kind"] for e in reopened._events()] == ["register", "freeze", "begin", "finish"]


@pytest.mark.parametrize("failure", ["file_fsync", "link", "directory_fsync"])
@pytest.mark.parametrize("operation", ["register", "freeze", "begin", "finish"])
def test_durable_publication_failure_rolls_back_and_reopens(tmp_path, monkeypatch, failure, operation):
    if operation == "freeze":
        registry = ValidationRegistry(tmp_path / "candidate")
        registry.register(suite())
    else:
        registry, reservation = setup_registry(tmp_path)
    result = {"status": "candidate_failure", "reason": "unit fixture only", "evaluator_revision": "unit-oracle-v1"}
    if operation == "register":
        action = lambda r: r.register(suite("another"))
    elif operation == "freeze":
        action = lambda r: r.freeze("fixture", "b" * 64, policy())
    elif operation == "begin":
        action = lambda r: r.begin(reservation["reservation_id"], "b" * 64, policy())
    else:
        run = registry.begin(reservation["reservation_id"], "b" * 64, policy())
        action = lambda r: r.finish(run["run_id"], result)
    before = registry_bytes(registry)
    committed_count = len(registry._events())
    real_fsync, real_link = validation.os.fsync, validation.os.link
    injected = []

    def faulty_fsync(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if not injected and ((failure == "directory_fsync" and directory)
                             or (failure == "file_fsync" and not directory)):
            injected.append(failure)
            raise OSError("injected " + failure)
        return real_fsync(fd)

    def faulty_link(*args, **kwargs):
        if failure == "link":
            injected.append(failure)
            raise OSError("injected link")
        return real_link(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(validation.os, "fsync", faulty_fsync)
        patch.setattr(validation.os, "link", faulty_link)
        with pytest.raises(OSError, match="injected " + failure):
            action(registry)
    assert injected == [failure]
    assert registry_bytes(registry) == before  # no committed link or pending debris
    reopened = ValidationRegistry(registry.workspace)
    assert registry_bytes(reopened) == before
    action(reopened)
    assert len(reopened._events()) == committed_count + 1
    assert reopened._events()[-1]["kind"] == operation
    if operation in {"begin", "finish"}:
        with pytest.raises(Blocked, match="consumed"):
            reopened.begin(reservation["reservation_id"], "b" * 64, policy())


@pytest.mark.parametrize("corruption", ["truncated", "non_object", "gap", "duplicate_key", "oversized"])
def test_corrupt_log_fails_closed_without_repair_or_reset(tmp_path, corruption):
    registry, reservation = setup_registry(tmp_path)
    registry.begin(reservation["reservation_id"], "b" * 64, policy())
    first = registry.root / "events" / "00000001.json"
    if corruption == "truncated":
        first.write_bytes(b'{"sequence":')
    elif corruption == "non_object":
        first.write_text("null")
    elif corruption == "gap":
        first.unlink()
    elif corruption == "duplicate_key":
        first.write_text('{"sequence":1,"sequence":2}')
    else:
        first.write_bytes(b" " * (validation.MAX_JSON_BYTES + 1))
    (registry.root / "events" / ".pending-keep-on-corruption").write_bytes(b"partial")
    before = registry_bytes(registry)
    for action in (lambda: ValidationRegistry(registry.workspace), registry.recover, registry.history,
                   lambda: registry.begin(reservation["reservation_id"], "b" * 64, policy())):
        with pytest.raises(Blocked):
            action()
        assert registry_bytes(registry) == before


def wave_fixture(tmp_path, width=2):
    path = tmp_path / ("unit-%d.wav" % width)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(width)
        output.setframerate(8000)
        scale = 2 ** (8 * width - 1)
        values = [0, -scale, scale - 1, 0] * 200
        raw = bytes(v + 128 for v in values) if width == 1 else b"".join(v.to_bytes(width, "little", signed=True) for v in values)
        output.writeframes(raw)
    manifest = {"model_id": "unit-fixture-no-model", "model_sha256": "c" * 64, "licenses": [card()]}
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(manifest))
    return path, model_path


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_voice_waveform_is_not_perceptual_quality(tmp_path, width):
    path, manifest = wave_fixture(tmp_path, width)
    original = path.read_bytes()
    evidence = prepare_voice_evidence(path, "unit fixture", "en", manifest)
    assert path.read_bytes() == original
    assert evidence["waveform_facts"]["duration_seconds"] == 0.1
    assert evidence["waveform_facts"]["clipping_fraction"] == 0.5
    assert evidence["waveform_facts"]["near_silence_fraction"] == 0.5
    assert evidence["status"] == "pending_human"
    assert evidence["quality_admission"] is False
    assert "mos" not in evidence
    assert evidence["model_rights"]["commercial_product_eligible"] is False
    assert evidence["model_rights"]["use_scope"] == "noncommercial_experimental"
    assert set(evidence["dimensions"].values()) == {"pending_human"}


def test_voice_invalid_and_existing_manifest_without_model_access(tmp_path):
    path, _ = wave_fixture(tmp_path)
    native = {"kind": "hf_tts", "task": "vits", "model_path": "/not/read/secret/key", "risk": "high",
              "provenance": [{"source": "fixture", "license": "CC-BY-NC-4.0", "risk": "high"}]}
    evidence = prepare_voice_evidence(path, "fixture", "en", native)
    assert evidence["model_rights"]["commercial_product_eligible"] is False
    path.write_bytes(b"not wav")
    with pytest.raises(Blocked, match="PCM WAV"):
        prepare_voice_evidence(path, "fixture", "en", native)


def test_listening_packet_blank_and_even_dummy_complete_never_approves(tmp_path):
    path, model = wave_fixture(tmp_path)
    manifest = {"protocol_id": "fixture-only", "clips": [{"clip_id": "unit", "audio": path.name,
                "text": "fixture", "language": "en", "model_manifest": model.name}]}
    packet = prepare_listening_batch(manifest, tmp_path)
    review = copy.deepcopy(packet["review_template"])
    checked = validate_listening_review(review, packet)
    assert checked["schema_complete"] is False
    assert checked["status"] == "pending_human"
    # Deliberately dummy test values exercise the validator, NOT real human evidence.
    review["trials"][0].update(listener_id="dummy-unit-fixture", language_proficiency="dummy",
                              playback_setup="dummy", blind_transcript="", script_hidden_during_transcription=True,
                              naturalness_rating=3, pronunciation_notes="dummy", listened_at="dummy")
    checked = validate_listening_review(review, packet)
    assert checked["schema_complete"] is True
    assert checked["authenticated_human_evidence"] is False
    assert checked["quality_admission"] is False
    assert checked["status"] == "pending_human"
    review["approved_by_human"] = True
    with pytest.raises(ValidationError):
        validate_listening_review(review, packet)


def test_statistics_are_conditional_and_never_change_threshold():
    summary = proportion_summary(48, 64)
    assert summary["proportion"] == 0.75
    assert summary["wilson_95"] == pytest.approx([0.6318, 0.8399], abs=0.001)
    assert not summary["threshold_changed"] and not summary["quality_admission"]
    assert proportion_summary(6, 6)["wilson_95"][0] == pytest.approx(0.6097, abs=0.001)
    for args in [(True, 2), (2, 1), (0, 0)]:
        with pytest.raises(ValueError):
            proportion_summary(*args)


def cli(tmp_path, *args):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    return subprocess.run([sys.executable, "-m", "asea.validation", *map(str, args)],
                          cwd=str(tmp_path), env=env, capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["suite", "policy"])
def test_cli_duplicate_suite_or_policy_keys_rejected_before_event(tmp_path, kind):
    registry = ValidationRegistry(tmp_path / "candidate")
    prefix = ["--workspace", registry.workspace]
    source = tmp_path / (kind + ".json")
    if kind == "suite":
        source.write_text(json.dumps(suite())[:-1] + ',"suite_id":"fixture"}')
        command = ["register", "--suite", source]
    else:
        registry.register(suite())
        source.write_text(json.dumps(policy())[:-1] + ',"threshold":0.5}')
        command = ["freeze", "--suite", "fixture", "--candidate-hash", "b" * 64, "--policy", source]
    before = registry_bytes(registry)
    result = cli(tmp_path, *prefix, *command)
    assert result.returncode == 2
    assert "duplicate JSON key" in result.stderr
    assert registry_bytes(registry) == before
    assert registry_bytes(ValidationRegistry(registry.workspace)) == before


def test_public_cli_end_to_end(tmp_path):
    s, p = tmp_path / "suite.json", tmp_path / "policy.json"
    s.write_text(json.dumps(suite()))
    p.write_text(json.dumps(policy()))
    prefix = ["--workspace", tmp_path / "candidate"]
    result = cli(tmp_path, *prefix, "register", "--suite", s)
    assert result.returncode == 0, result.stderr
    result = cli(tmp_path, *prefix, "freeze", "--suite", "fixture", "--candidate-hash", "b" * 64, "--policy", p)
    assert result.returncode == 0, result.stderr
    reservation = json.loads(result.stdout)["reservation_id"]
    result = cli(tmp_path, *prefix, "begin", "--reservation", reservation, "--candidate-hash", "b" * 64, "--policy", p)
    assert result.returncode == 0, result.stderr
    run = json.loads(result.stdout)["run_id"]
    r = tmp_path / "result.json"
    r.write_text(json.dumps({"status": "completed", "reason": "unit fixture only", "evaluator_revision": "unit-oracle-v1"}))
    result = cli(tmp_path, *prefix, "finish", "--run", run, "--result", r)
    assert result.returncode == 0 and json.loads(result.stdout)["quality_admission"] is False
    audio, model = wave_fixture(tmp_path)
    output = tmp_path / "voice.json"
    result = cli(tmp_path, "voice-check", "--audio", audio, "--text", "fixture", "--language", "en", "--model-manifest", model, "--output", output)
    assert result.returncode == 0, result.stderr
    assert load_json(output)["status"] == "pending_human"
    result = cli(tmp_path, "voice-check", "--audio", audio, "--text", "fixture", "--language", "en", "--model-manifest", model, "--output", output)
    assert result.returncode == 2
    result = cli(tmp_path, "schema", "Suite")
    assert result.returncode == 0 and json.loads(result.stdout)["additionalProperties"] is False
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({"protocol_id": "unit-only", "clips": [{"clip_id": "unit",
                     "audio": audio.name, "text": "fixture", "language": "en", "model_manifest": model.name}]}))
    packet_path = tmp_path / "packet.json"
    result = cli(tmp_path, "export-listen-batch", "--manifest", batch, "--output", packet_path)
    assert result.returncode == 0, result.stderr
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(load_json(packet_path)["review_template"]))
    result = cli(tmp_path, "check-review", "--review", review_path, "--packet", packet_path)
    assert result.returncode == 0, result.stderr
    assert not json.loads(result.stdout)["schema_complete"]
    assert json.loads(result.stdout)["status"] == "pending_human"
    result = cli(tmp_path, "approve-human", "--review", review_path)
    assert result.returncode == 2  # no approval-creation command exists


def test_interrupted_first_marker_recovery_and_json_duplicates(tmp_path):
    root = tmp_path / "candidate.validation-registry"
    root.mkdir()
    (root / ".pending-marker").write_text("interrupted marker")
    registry = ValidationRegistry(tmp_path / "candidate")
    assert not (root / ".pending-marker").exists()
    assert registry.history()["events"] == []
    source = tmp_path / "duplicate.json"
    source.write_text('{"passed":false,"passed":true}')
    with pytest.raises(Blocked, match="duplicate"):
        load_json(source)


def test_same_input_different_reference_cannot_cross_splits(tmp_path):
    data = suite()
    data["members"][0]["input"] = data["members"][1]["input"]
    m = data["members"][0]
    m["content_sha256"] = digest({"input": m["input"], "expected": m["expected"]})
    with pytest.raises(ValidationError, match="input content crosses"):
        Suite.model_validate(data)
