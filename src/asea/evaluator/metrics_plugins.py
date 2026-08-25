"""Modality-specific task-success metrics.

These exist because "did it succeed?" is not one question. A translation is
partially right; a phoneme string is right or wrong; a code fix is judged by
whether the test passes, not by how it looks.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..core.interfaces import MetricPlugin
from ..core.protocol import Modality, SkillPacket
from .similarity import LexicalSimilarity, normalize


class TextMetric(MetricPlugin):
    """Graded credit via lexical similarity. A proxy -- see similarity.py."""

    modality = Modality.TEXT

    def __init__(self) -> None:
        self._sim = LexicalSimilarity()

    def score(self, expected: Any, actual: Any, packet: Optional[SkillPacket] = None) -> float:
        return self._sim.similarity(str(expected), str(actual))


class TTSMetric(MetricPlugin):
    """Phoneme strings are symbolic: near-enough is wrong.

    Compares after stripping IPA delimiters and whitespace so /kɔ.ma/ and
    [kɔ.ma] are treated as the same transcription.
    """

    modality = Modality.SPEECH_TTS

    _DELIM = re.compile(r"[\[\]/\s]")

    def _canon(self, value: Any) -> str:
        return self._DELIM.sub("", normalize(value))

    def score(self, expected: Any, actual: Any, packet: Optional[SkillPacket] = None) -> float:
        return 1.0 if self._canon(expected) == self._canon(actual) else 0.0


class CodeMetric(MetricPlugin):
    """Whitespace-insensitive structural comparison.

    HONEST LIMITATION: this is a text comparison, not test execution. A real
    deployment must run the test command in a sandbox and score on exit status.
    The interface is here; the runner is deliberately not, because executing
    model-authored code inside this process would be unsafe.
    """

    modality = Modality.CODE

    _WS = re.compile(r"\s+")

    def _canon(self, value: Any) -> str:
        return self._WS.sub(" ", str(value or "")).strip()

    def score(self, expected: Any, actual: Any, packet: Optional[SkillPacket] = None) -> float:
        if self._canon(expected) == self._canon(actual):
            return 1.0
        exp_tokens = set(self._canon(expected).split())
        act_tokens = set(self._canon(actual).split())
        if not exp_tokens:
            return 0.0
        return len(exp_tokens & act_tokens) / len(exp_tokens | act_tokens)
