"""Issue 3 guard test — every MiniCPM-o skill path hits the BLOCKED guard.

The receiver declares 9 Assamese capabilities but loads no weights on this CPU
machine. This test proves that NOT ONE of those 9 paths can return a fabricated
answer: every `infer` path AND the skill-conditioned `infer_with_skills` path
raise the BLOCKED RuntimeError before any output is produced. Parametrized over
all 9 capabilities so a future capability addition cannot silently bypass the
guard.

Deliberately lives under experiments/ (NOT tests/), so it is NOT collected by
the core `pytest tests/` run (which stays green: all pass, 4 skipped). Run it on
its own:

    PYTHONPATH=src python -m pytest experiments/minicpmo45_assamese/test_receiver_guard.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for minicpmo_receiver

from minicpmo_receiver import MiniCPMOReceiver, assamese_capabilities  # noqa: E402

BLOCKED_MARKER = "BLOCKED"


@pytest.fixture(scope="module")
def receiver():
    # Force the checkpoint-absent branch regardless of any HF cache state, so the
    # test is deterministic on any machine (it asserts the guard fires when the
    # checkpoint is not loadable, which is the GPU-box precondition too).
    r = MiniCPMOReceiver()
    r._model = None
    if r.checkpoint_available():  # pragma: no cover - only on a GPU box
        pytest.skip("checkpoint present on this machine; guard test assumes absent")
    return r


@pytest.mark.parametrize("cap", assamese_capabilities(),
                         ids=[c.as_str() for c in assamese_capabilities()])
def test_every_infer_path_is_blocked(receiver, cap):
    """Each of the 9 capability paths must raise BLOCKED on `infer`."""
    with pytest.raises(RuntimeError) as exc_info:
        receiver.infer(cap, prompt="probe")
    assert BLOCKED_MARKER in str(exc_info.value), (
        "infer path for {} did not hit the BLOCKED guard: {!r}".format(
            cap.as_str(), exc_info.value))


@pytest.mark.parametrize("cap", assamese_capabilities(),
                         ids=[c.as_str() for c in assamese_capabilities()])
def test_every_infer_with_skills_path_is_blocked(receiver, cap):
    """Each of the 9 capability paths must raise BLOCKED on `infer_with_skills`
    too — skill-conditioned inference cannot slip past the guard."""
    with pytest.raises(RuntimeError) as exc_info:
        receiver.infer_with_skills(cap, prompt="probe", skills=[{"k": "v"}])
    assert BLOCKED_MARKER in str(exc_info.value), (
        "infer_with_skills path for {} did not hit the BLOCKED guard: {!r}".format(
            cap.as_str(), exc_info.value))


def test_guard_is_not_only_the_main_answer_path():
    """The non-translate capabilities (reason, stt, g2p, tts, sts, vision, tool,
    agent) are the 8 'other' paths the user flagged. Sample each and prove the
    guard fires — they are not implicitly covered by the translate/answer path.
    """
    caps = {c.as_str(): c for c in assamese_capabilities()}
    other_paths = ["reason/text/general/as", "transcribe/audio_asr/general/as",
                  "grapheme_to_phoneme/speech_tts/pronunciation/as-ipa",
                  "synthesize/speech_tts/general/as", "sts/speech_tts/general/as",
                  "describe/ocr/general/as", "tool_call/text/software/as",
                  "agent_reason/text/general/as"]
    assert set(other_paths).issubset(caps.keys()), (
        "expected 8 non-answer paths; got {}".format(sorted(caps.keys())))
    r = MiniCPMOReceiver()
    r._model = None
    if r.checkpoint_available():  # pragma: no cover
        pytest.skip("checkpoint present; guard test assumes absent")
    for key in other_paths:
        with pytest.raises(RuntimeError) as exc_info:
            r.infer(caps[key], prompt="probe")
        assert BLOCKED_MARKER in str(exc_info.value), (
            "{} slipped past the guard: {!r}".format(key, exc_info.value))


def test_infer_with_skills_does_not_silently_succeed():
    """A skill-conditioned call must not return *anything* (the inherited default
    used to delegate to infer; the explicit override now guards directly)."""
    r = MiniCPMOReceiver()
    r._model = None
    if r.checkpoint_available():  # pragma: no cover
        pytest.skip("checkpoint present; guard test assumes absent")
    cap = assamese_capabilities()[0]
    with pytest.raises(RuntimeError) as exc_info:
        r.infer_with_skills(cap, prompt="probe", skills=[])
    assert BLOCKED_MARKER in str(exc_info.value)


if __name__ == "__main__":
    # Allow running without pytest for a quick guard smoke check.
    r = MiniCPMOReceiver()
    r._model = None
    if r.checkpoint_available():
        print("SKIP: checkpoint present on this machine")
        os._exit(0)
    ok = True
    for cap in assamese_capabilities():
        for method_name, call in (("infer", lambda c: r.infer(c, "probe")),
                                  ("infer_with_skills",
                                   lambda c: r.infer_with_skills(c, "probe", []))):
            try:
                call(cap)
                print("FAIL  {}  {} -> returned without BLOCKED".format(
                    method_name, cap.as_str()))
                ok = False
            except RuntimeError as exc:
                if BLOCKED_MARKER not in str(exc):
                    print("FAIL  {}  {} -> wrong error: {!r}".format(
                        method_name, cap.as_str(), exc))
                    ok = False
    print("guard ok" if ok else "guard FAILED")
    os._exit(0 if ok else 1)