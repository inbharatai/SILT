"""Real corpus-backed sender.

REAL, not a mock -- and the distinction matters. A sender does not have to be a
neural network: the feasibility review has said from day one that an expert
source can be "a curated corpus, an API, or a human panel". This module wraps a
reviewed reference corpus (a JSON file of prompt->answer records) and serves it
verbatim. There is no fake inference and nothing simulated about it -- retrieval
from curated data is the *safest* kind of teacher, which is exactly why it is
the recommended sender for high-risk domains:

  * ``is_mock = False``      -- it is a real knowledge source
  * ``synthetic_depth = 0``  -- content is human-authored, zero model generations
  * confidence comes from the corpus record itself, not from a model's opinion

For medical/legal/finance this is the sender you should be using anyway. A
"high-medical-skill LLM" as a teacher launders model output into a learner;
a reviewed corpus keeps the provenance chain anchored to humans, and the
promotion gate's human-approval requirement still applies on top.

Corpus file format::

    {
      "corpus_id": "who-triage-v1",
      "reviewed_by": "..." ,
      "records": [
        {"prompt": "chest pain radiating to the left arm",
         "answer": "Red flag. ...",
         "confidence": 0.95}
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.interfaces import ModuleAdapter
from ...core.protocol import CapabilityKey, CapabilityManifest, LearningLevel

UNKNOWN = "<not-in-corpus>"


class CorpusSender(ModuleAdapter):
    is_mock = False

    def __init__(
        self,
        corpus_path: Path,
        capabilities: List[CapabilityKey],
        module_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> None:
        path = Path(corpus_path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        corpus_id = data.get("corpus_id", path.stem)
        super().__init__(
            module_id or "corpus-{}".format(corpus_id),
            display_name or "Curated corpus {}".format(corpus_id),
        )
        self.corpus_path = path
        self.corpus_id = corpus_id
        self.reviewed_by = data.get("reviewed_by")
        self._capabilities = list(capabilities)
        self._records: Dict[str, Dict[str, Any]] = {
            str(r["prompt"]).strip(): r for r in data.get("records", [])
        }

    # -- identity ---------------------------------------------------------

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=["sender"],  # a corpus cannot learn; sender only
            capabilities=self._capabilities,
            # A corpus can anchor any learning level up to dataset export.
            max_learning_level=LearningLevel.L5_DISTILL_DATASET,
            is_mock=False,
            version="corpus:{}".format(self.corpus_id),
        )

    # -- behaviour ----------------------------------------------------------

    def infer(self, capability: CapabilityKey, prompt: Any) -> Any:
        record = self._records.get(str(prompt).strip())
        if record is None:
            # Fail visibly. Substituting a nearest-match answer from a medical
            # corpus would be exactly the silent-wrong-answer failure this
            # system exists to prevent.
            return UNKNOWN
        return record["answer"]

    def confidence(self, capability: CapabilityKey, prompt: Any, output: Any) -> float:
        record = self._records.get(str(prompt).strip())
        if record is None:
            return 0.0
        return float(record.get("confidence", 0.9))

    def __len__(self) -> int:
        return len(self._records)
