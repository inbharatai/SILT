"""The sealed final split (extraction-brief C6).

The final split lives in a PHYSICALLY SEPARATE artifact from the
development data. The rules, enforced here:

* development commands (:class:`SealedSplit.access` with
  ``authority="development"``) can NEVER open it -- every attempt is
  recorded and answered with a typed refusal;
* the final evaluation is the only command that can open it, and only
  ONCE: the consumption record is written BEFORE the cases are
  returned, so a crash mid-evaluation cannot grant a second attempt;
* EVERY access attempt -- refused or permitted -- is appended to an
  access log beside the artifact, recording the command, the
  authority, the timestamp and the verdict.

This is a physical separation plus a one-way latch, not encryption:
the artifact is separate, the refusal is typed, and the latch makes
"evaluate until it passes" impossible on this machine.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from asea.artifacts import digest, safe_path
from asea.extraction.errors import Refused

SEAL_SCHEMA = "silt.extraction.sealed_split.v1"

SEAL_FILENAME = "final-split.sealed.json"
ACCESS_LOG_FILENAME = "final-split.access.jsonl"
CONSUMPTION_FILENAME = "final-split.consumed.json"

DEVELOPMENT = "development"
FINAL_EVALUATION = "final_evaluation"


class SealedSplit:
    """The sealed final-split artifact. Construct with the directory
    that holds (or will hold) the seal."""

    def __init__(self, directory):
        self.directory = safe_path(directory)
        self.seal_path = self.directory / SEAL_FILENAME
        self.access_log_path = self.directory / ACCESS_LOG_FILENAME
        self.consumption_path = self.directory / CONSUMPTION_FILENAME

    # -- sealing ---------------------------------------------------------

    def seal(self, *, capability_id: str, spec_sha256: str,
             cases: Dict[str, Any], note: str = "") -> Dict[str, Any]:
        """Create the physically separate sealed artifact. The cases
        (already validated, near-duplicate free, never retuned) are
        written ONCE with the spec fingerprint they are pinned to."""
        if self.seal_path.exists():
            raise Refused(
                "a sealed final split already exists at %s; resealing "
                "would let the final split be replaced after the fact"
                % self.seal_path,
                "delete the sealed artifact deliberately and re-run the "
                "dataset builder; this is recorded as an operator action, "
                "never a routine one")
        self.directory.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema": SEAL_SCHEMA,
            "capability_id": capability_id,
            "spec_sha256": spec_sha256,
            "sealed_unix": time.time(),
            "case_count": len(cases.get("cases", [])),
            "cases_sha256": digest(cases),
            "cases": cases,
            "note": note,
        }
        artifact["seal_sha256"] = digest(
            {key: value for key, value in artifact.items()
             if key != "seal_sha256"})
        with open(str(self.seal_path), "w", encoding="utf-8") as stream:
            stream.write(json.dumps(artifact, sort_keys=True))
        return artifact

    # -- access ----------------------------------------------------------

    def _record_access(self, *, command: str, authority: str,
                       verdict: str, detail: str = "") -> None:
        entry = {
            "schema": "silt.extraction.seal_access.v1",
            "command": command,
            "authority": authority,
            "verdict": verdict,
            "detail": detail,
            "recorded_unix": time.time(),
        }
        entry["entry_sha256"] = digest(entry)
        with open(str(self.access_log_path), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")

    def access_log(self):
        if not self.access_log_path.exists():
            return []
        result = []
        with open(str(self.access_log_path), "r",
                  encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    result.append(json.loads(line))
        return result

    def is_consumed(self) -> bool:
        return self.consumption_path.exists()

    def assert_intact(self) -> None:
        """The sealed artifact must still hash to its own seal; a
        tampered seal is a refusal, never a warning."""
        if not self.seal_path.exists():
            raise Refused(
                "no sealed final split at %s" % self.seal_path,
                "run the dataset builder to produce one before "
                "attempting the final evaluation")
        with open(str(self.seal_path), "r", encoding="utf-8") as stream:
            artifact = json.load(stream)
        expected = artifact.get("seal_sha256")
        body = {key: value for key, value in artifact.items()
                if key != "seal_sha256"}
        if digest(body) != expected:
            raise Refused(
                "the sealed final split at %s does not match its seal "
                "hash -- the artifact was modified after sealing"
                % self.seal_path,
                "investigate who edited the sealed artifact; a modified "
                "final split can never be evaluated as 'sealed'")

    def open_for_final_evaluation(self, *, command: str,
                                  spec_sha256: str) -> Dict[str, Any]:
        """The ONLY permitted opener: the final-evaluation command. The
        consumption record is written BEFORE the cases are returned and
        a second open is refused -- the split is evaluated exactly
        once."""
        self.assert_intact()
        with open(str(self.seal_path), "r", encoding="utf-8") as stream:
            artifact = json.load(stream)
        if self.is_consumed():
            self._record_access(command=command,
                                authority=FINAL_EVALUATION,
                                verdict="REFUSED_ALREADY_CONSUMED")
            raise Refused(
                "the sealed final split has already been opened for "
                "evaluation; evaluating it twice would turn the final "
                "measurement into a tunable one",
                "a failed final evaluation is a recorded FAILED_HYPOTHESIS "
                "-- rerun the whole program with a new capability id, do "
                "not rerun the final evaluation")
        if artifact.get("spec_sha256") != spec_sha256:
            self._record_access(command=command,
                                authority=FINAL_EVALUATION,
                                verdict="REFUSED_SPEC_MISMATCH")
            raise Refused(
                "the sealed final split is pinned to spec %s, not %s"
                % (artifact.get("spec_sha256"), spec_sha256),
                "reseal from the dataset builder under the current spec; "
                "a spec change invalidates the sealed evaluation")
        consumption = {
            "schema": "silt.extraction.seal_consumption.v1",
            "command": command,
            "spec_sha256": spec_sha256,
            "consumed_unix": time.time(),
            "cases_sha256": artifact["cases_sha256"],
            "recorded_before_cases_returned": True,
        }
        consumption["consumption_sha256"] = digest(
            {key: value for key, value in consumption.items()
             if key != "consumption_sha256"})
        with open(str(self.consumption_path), "w",
                  encoding="utf-8") as stream:
            stream.write(json.dumps(consumption, sort_keys=True))
        self._record_access(command=command, authority=FINAL_EVALUATION,
                            verdict="PERMITTED", detail="consumed exactly once")
        return artifact["cases"]

    def access(self, *, command: str,
               authority: str = DEVELOPMENT) -> Dict[str, Any]:
        """Development-facing access: ALWAYS refused, ALWAYS recorded.
        Development commands never receive the cases under any flag."""
        self._record_access(command=command, authority=authority,
                            verdict="REFUSED_DEVELOPMENT_ACCESS")
        raise Refused(
            "development commands cannot open the sealed final split "
            "(%s)" % self.seal_path,
            "the final split is opened only by the final evaluation, "
            "exactly once; development uses the training, development, "
            "heldout and controls splits")