"""Teacher-score cache (A1, audit 2026-08-17).

Gap negotiation scores BOTH modules on every suite's extraction split. The
receiver is the thing being improved -- always measured fresh. The sender is
the TEACHER: for a fixed model + suite its extraction output is a pure
function of (teacher, suite, harness) under the SILT deterministic-decoding
discipline (``do_sample=False``), so re-measuring it on a re-run bills a
second full sender evaluation for an identical result. On a real GPU that is
wasted VRAM-time; on a 4-bit 7B it is a second 5+ GB load. This module keys
that result and serves it read-through.

It is OFF by default: ``GapEngine`` takes an optional ``teacher_cache`` and,
absent one, behaves byte-identically to before (every sender run is fresh).
The opt-in is explicit so the existing suite -- which asserts nothing about
sender measurement counts -- is unaffected, and so an operator cannot
accidentally cache a teacher they did not mean to.

HONESTY CONTRACT (binding -- these are why the key looks heavier than
"hash the module id"):

  * Only the SENDER role is ever cached. The receiver is always measured
    fresh; caching a receiver would freeze the very thing the transfer is
    supposed to change. The :class:`GapEngine` enforces this by only
    consulting the cache on the sender path.
  * The cache key is (teacher fingerprint, suite fingerprint, harness
    fingerprint). The harness fingerprint covers the similarity backend
    (including INSTANCE config -- e.g. :class:`LexicalSimilarity` weights --
    via :meth:`SimilarityBackend.fingerprint`) AND the per-modality metric
    plugin (via :meth:`MetricPlugin.fingerprint`), so swapping the metric
    (lexical -> embedding), RECONFIGURING a backend's weights, or swapping a
    metric plugin invalidates the cache -- a score computed under one harness
    is never silently served under another (audit 2026-08-17, F1/F4).
  * A clean MISS (key absent on disk) is silent: the cache returns None and the
    caller re-runs the harness. A CORRUPT entry (file present but unreadable) is
    NOT silent: it raises :class:`asea.core.errors.CacheCorruptionError`. The
    cache does not self-heal by discarding a corrupt entry and recomputing --
    that would hide a disk/serialisation problem. ``put`` writes atomically
    (temp file + ``os.replace``) so a crash mid-write cannot produce the
    half-written file that would trigger the error (audit 2026-08-17, F2).
  * The suite fingerprint covers the extraction cases (case_id, prompt,
    expected) and the capability, so a suite whose prompts OR reference
    answers change is a miss -- a stale score is never served.
  * The teacher fingerprint is :meth:`ModuleAdapter.fingerprint`; the
    default is manifest identity and real connectors override it to include
    weights/revision/quantization (see that method's docstring).
  * Nothing here weakens a gate. The gate reads scores produced by the
    evaluator, not by this cache; the cache only avoids recomputing the
    teacher's *extraction* score during gap negotiation.

DISK backing is opt-in (``backing_dir``). In-memory only (the default for
tests) never persists; on-disk persists across process restarts, which is
where the harness-fingerprint invalidation above earns its keep.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ..core.errors import CacheCorruptionError
from .harness import BenchmarkSuite, SuiteResult


def suite_fingerprint(suite: BenchmarkSuite) -> str:
    """A stable hash of everything about a suite that determines a sender's
    extraction score on it.

    Covers the suite_id, the capability (which infer path the module takes),
    and the extraction cases (case_id, prompt, expected). ``prompt`` drives
    the module's output; ``expected`` drives the similarity/task score; both
    must change the key or a stale score is served. The split itself is
    fixed ("extraction") by construction -- this function is only called on
    the extraction path -- so it is not hashed.
    """
    cases = suite.split("extraction")
    payload = "\x1f".join(
        "{}\x1e{}\x1e{}".format(c.case_id, c.prompt, c.expected) for c in cases
    )
    raw = "\x1f".join([suite.suite_id, suite.capability().as_str(), payload])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class TeacherScoreCache:
    """Keyed store of teacher (sender) extraction ``SuiteResult``s.

    The key is opaque and built by the caller (:class:`GapEngine`), which
    composes it from the teacher fingerprint, the suite fingerprint and the
    harness fingerprint. This class does not know what a teacher or a harness
    is -- it is a pure key->SuiteResult store with optional on-disk backing,
    so the identity policy lives in exactly one place (the engine) and the
    storage policy lives in exactly one place (here).
    """

    def __init__(self, backing_dir: Optional[Path] = None) -> None:
        self._mem: Dict[str, SuiteResult] = {}
        self._backing: Optional[Path] = (
            Path(backing_dir) if backing_dir is not None else None
        )
        if self._backing is not None:
            self._backing.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Hash the key for the filename so any key (it contains '|') is a safe,
        # fixed-length, collision-resistant file name.
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self._backing / "{}.json".format(digest)  # type: ignore[union-attr]

    def get(self, key: str) -> Optional[SuiteResult]:
        """Return the cached result for ``key``, or None on a clean miss.

        A clean miss (no file on disk, or no backing) is silent: the caller
        re-runs the harness. A CORRUPT entry (file present but unreadable) is
        NOT silent -- it raises :class:`CacheCorruptionError`. The cache does
        not self-heal by discarding a corrupt entry, because that would hide a
        disk/serialisation problem behind a silent recompute (audit 2026-08-17,
        F2). In-memory hits are served without touching disk, so corruption
        only matters across process restarts, and ``put`` writes atomically so
        normal operation cannot produce a corrupt file in the first place.
        """
        if key in self._mem:
            return self._mem[key]
        if self._backing is not None:
            path = self._path(key)
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    result = SuiteResult.model_validate_json(text)
                except Exception as exc:  # noqa: BLE001 -- any read/parse failure
                    # is a corrupt entry, surfaced as a typed error (not a
                    # silent miss). pydantic.ValidationError, UnicodeDecodeError,
                    # json.JSONDecodeError, OSError all land here.
                    raise CacheCorruptionError(str(path), exc) from exc
                self._mem[key] = result
                return result
        return None

    def put(self, key: str, result: SuiteResult) -> None:
        """Store ``result`` under ``key``.

        Writes the in-memory entry first (so an in-process re-read always hits
        even if the disk write were to fail), then the disk entry ATOMICALLY:
        write to a temp file in the same directory, then ``os.replace``. A crash
        between the temp write and the rename leaves the temp file orphaned and
        the real file untouched -- never a half-written entry that ``get``
        could not read (audit 2026-08-17, F2).
        """
        self._mem[key] = result
        if self._backing is not None:
            final = self._path(key)
            payload = result.model_dump_json(indent=2)
            # Named temp file in the SAME directory so os.replace is atomic on
            # the same filesystem (rename is atomic; cross-fs rename is not).
            fd, tmp_name = tempfile.mkstemp(
                prefix=".tmp-", suffix=".json", dir=str(self._backing)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                os.replace(tmp_name, final)
            except BaseException:
                # Clean up the orphaned temp on ANY failure (including crash mid-
                # write) so a later run does not trip over it.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    def __len__(self) -> int:
        return len(self._mem)

    def __contains__(self, key: str) -> bool:
        return key in self._mem