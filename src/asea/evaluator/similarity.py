"""Similarity backends.

HONEST LIMITATION: the bundled backend is *lexical*, not semantic. It combines
normalised Levenshtein distance with token-level F1. It cannot tell you that
"the patient is febrile" and "the patient has a fever" mean the same thing, and
it will happily score a fluent-but-wrong translation highly if it shares
surface tokens with the reference.

Everywhere a score from this backend is reported, ``is_semantic`` is surfaced
alongside it so downstream readers know the number is a proxy. Swap in an
embedding model (LaBSE / IndicSBERT / any sentence-transformer) by implementing
SimilarityBackend and passing it to the Evaluator.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from ..core.interfaces import SimilarityBackend

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Unicode-aware word tokenisation.

    ``\\w`` with re.UNICODE covers Bengali-Assamese, Devanagari and Meitei Mayek
    code points, so this works for the Indic targets without a language pack.
    """
    return _TOKEN_RE.findall(normalize(text))


def normalize(text: str) -> str:
    """NFC-normalise, casefold and collapse whitespace.

    NFC matters for Indic scripts: the same visual grapheme can be encoded with
    different code point sequences, and comparing unnormalised strings produces
    false mismatches.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.casefold().split())


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance, O(len(a) * len(b)) time, O(len(b)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def char_ratio(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - (levenshtein(a, b) / longest)


def token_f1(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    # Multiset overlap so repeated tokens are not double-counted.
    remaining = list(tb)
    overlap = 0
    for tok in ta:
        if tok in remaining:
            remaining.remove(tok)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


class LexicalSimilarity(SimilarityBackend):
    """Default backend. Deterministic, offline, and only a proxy for meaning."""

    def __init__(self, char_weight: float = 0.4, token_weight: float = 0.6) -> None:
        total = char_weight + token_weight
        self.char_weight = char_weight / total
        self.token_weight = token_weight / total

    def similarity(self, a: str, b: str) -> float:
        score = self.char_weight * char_ratio(a, b) + self.token_weight * token_f1(a, b)
        return max(0.0, min(1.0, score))

    @property
    def is_semantic(self) -> bool:
        return False

    def fingerprint(self) -> str:
        # Instance config matters: char_weight/token_weight change every score,
        # so two differently-weighted LexicalSimilarity instances MUST key
        # differently or a score cached under one weighting is served under the
        # other (audit 2026-08-17, F1). Use the NORMALISED weights, since those
        # are what actually drive similarity(); (0.4,0.6) and (2,3) normalise to
        # the same pair and must collide (they produce identical scores). Keep
        # the class identity (via super) so this never collides with another
        # backend that happens to stringify similarly.
        return "{}|cw={:.10f}|tw={:.10f}".format(
            super().fingerprint(), self.char_weight, self.token_weight
        )


class ExactMatch(SimilarityBackend):
    """Strict backend for tasks where partial credit is meaningless."""

    def similarity(self, a: str, b: str) -> float:
        return 1.0 if normalize(a) == normalize(b) else 0.0

    @property
    def is_semantic(self) -> bool:
        return False
