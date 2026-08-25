"""Base class for MOCK modules.

=========================  READ THIS  =========================
Nothing in this package performs model inference. There is no torch, no
transformers, no weights and no API credentials in the execution environment
this was built in, so every "model" here is a lookup table with a documented
fallback behaviour.

What that means for the numbers this system produces:
  * A measured "improvement" from a mock receiver proves the PLUMBING works --
    that a distilled packet reached the receiver, was consumed, and changed the
    held-out output. It proves NOTHING about whether a real Qwen or Gemma would
    improve.
  * Every mock sets ``is_mock = True``. That flag propagates into packet
    provenance, and the promotion gate refuses mock-derived packets under the
    default strict policy. A mock cannot launder itself into approved data.

Replace these with real connectors (docs/connector_authoring.md) before drawing
any conclusion about model capability.
===============================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core.interfaces import ModuleAdapter
from ...core.protocol import CapabilityKey, CapabilityManifest, LearningLevel

#: Sentinel returned when a table-backed mock has no entry.
UNKNOWN = "<unknown>"


class MockModule(ModuleAdapter):
    """A deterministic lookup-table stand-in for a model.

    ``knowledge`` maps a capability string to {prompt: answer}. Unknown prompts
    fall back according to ``fallback``:

      * ``"echo"``    -- return the prompt unchanged (models often do this on
                         low-resource input, which is exactly the failure the
                         Assamese flow is designed to catch)
      * ``"unknown"`` -- return the UNKNOWN sentinel
      * ``"english"`` -- return a fixed English apology, simulating a model
                         silently answering in the wrong language
    """

    is_mock = True

    def __init__(
        self,
        module_id: str,
        display_name: str,
        capabilities: List[CapabilityKey],
        roles: List[str],
        knowledge: Optional[Dict[str, Dict[str, Any]]] = None,
        fallback: str = "echo",
        max_learning_level: LearningLevel = LearningLevel.L3_SKILL_PACKET,
        base_confidence: float = 0.5,
        consumes_skills: bool = True,
    ) -> None:
        super().__init__(module_id, display_name)
        self._capabilities = list(capabilities)
        self._roles = list(roles)
        self.knowledge = knowledge or {}
        self.fallback = fallback
        self.max_learning_level = max_learning_level
        self.base_confidence = base_confidence
        #: Some modules genuinely cannot consume injected skills. Modelling that
        #: honestly matters: such a receiver will never show improvement and so
        #: will never pass the gate.
        self.consumes_skills = consumes_skills

    # -- identity ---------------------------------------------------------

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            module_id=self.module_id,
            display_name=self.display_name,
            roles=self._roles,
            capabilities=self._capabilities,
            max_learning_level=self.max_learning_level,
            is_mock=self.is_mock,
            version="0.1.0-mock",
        )

    # -- behaviour --------------------------------------------------------

    def _table(self, capability: CapabilityKey) -> Dict[str, Any]:
        return self.knowledge.get(capability.as_str(), {})

    def _fallback(self, prompt: Any) -> Any:
        if self.fallback == "echo":
            return prompt
        if self.fallback == "english":
            return "I do not know this phrase."
        return UNKNOWN

    def infer(self, capability: CapabilityKey, prompt: Any) -> Any:
        table = self._table(capability)
        key = str(prompt).strip()
        if key in table:
            return table[key]
        return self._fallback(prompt)

    def infer_with_skills(
        self, capability: CapabilityKey, prompt: Any, skills: List[Dict[str, Any]]
    ) -> Any:
        if not self.consumes_skills:
            return self.infer(capability, prompt)
        hit = lookup_in_skills(skills, prompt)
        if hit is not None:
            return hit
        return self.infer(capability, prompt)

    def confidence(self, capability: CapabilityKey, prompt: Any, output: Any) -> float:
        table = self._table(capability)
        return 0.95 if str(prompt).strip() in table else self.base_confidence


#: Fraction of prompt tokens that must be covered by a glossary before the
#: compositional path is allowed to answer. Below this, guessing does more harm
#: than good and we prefer to fail visibly.
COMPOSITION_COVERAGE = 0.6


def lookup_in_skills(skills: List[Dict[str, Any]], prompt: Any) -> Optional[Any]:
    """Apply redacted skill payloads to a prompt.

    Four application modes, tried in order of decreasing confidence:

      1. **exact lookup**   -- the prompt is a known glossary source or grapheme.
      2. **rule trigger**   -- a rule's condition appears in the prompt.
      3. **fragment fix**   -- an exemplar's buggy fragment appears in the
         prompt; splice in the fixed fragment.
      4. **composition**    -- enough of the prompt's tokens are in the glossary
         to assemble an answer piecewise.

    Modes 2-4 generalise to inputs never seen during extraction, which is what
    makes the held-out evaluation meaningful rather than a memorisation check.

    Mode 4 is honest about its own crudeness: token-wise substitution produces
    the right *words* in the *source* language's order. For Assamese (SOV) into
    English (SVO) the word order will be wrong, and the bundled lexical metric
    under-penalises that because token-F1 is order-insensitive. This is a real
    limitation of the proxy metric, documented here and in the final report
    rather than hidden behind a flattering number.
    """
    text = str(prompt)
    key = text.strip()
    payloads = [s.get("distilled_skill") or {} for s in (skills or [])]

    # 1. exact
    for payload in payloads:
        for entry in payload.get("entries", []) or []:
            if str(entry.get("source", "")).strip() == key:
                return entry.get("target")
            if str(entry.get("grapheme", "")).strip() == key:
                return entry.get("phoneme")

    # 2. rule trigger (longest condition first, so specific beats generic)
    rules = [r for p in payloads for r in (p.get("rules") or [])]
    for rule in sorted(rules, key=lambda r: -len(str(r.get("condition", "")))):
        condition = str(rule.get("condition", "")).strip()
        if condition and condition.casefold() in text.casefold():
            return rule.get("action")

    # 3. fragment fix
    examples = [e for p in payloads for e in (p.get("examples") or [])]
    for example in sorted(examples, key=lambda e: -len(str(e.get("buggy", "")))):
        buggy = str(example.get("buggy", ""))
        fixed = example.get("fixed")
        if buggy and fixed is not None and buggy in text:
            if buggy.strip() == key:
                return fixed
            return text.replace(buggy, str(fixed))

    # 4. composition
    return _compose(payloads, text)


def _compose(payloads: List[Dict[str, Any]], text: str) -> Optional[Any]:
    glossary: Dict[str, Any] = {}
    grapheme_mode = False
    for payload in payloads:
        for entry in payload.get("entries", []) or []:
            if "source" in entry:
                glossary[str(entry["source"]).strip()] = entry.get("target")
            elif "grapheme" in entry:
                glossary[str(entry["grapheme"]).strip()] = entry.get("phoneme")
                grapheme_mode = True
    if not glossary:
        return None

    if grapheme_mode:
        return _compose_graphemes(glossary, text)

    tokens = text.split()
    if not tokens:
        return None

    pieces, covered = [], 0
    for token in tokens:
        stripped = token.strip(".,!?;:।")  # includes Devanagari danda
        if stripped in glossary:
            pieces.append(str(glossary[stripped]))
            covered += 1
        else:
            pieces.append(token)
    if covered / len(tokens) < COMPOSITION_COVERAGE:
        return None
    return " ".join(pieces)


def _compose_graphemes(lexicon: Dict[str, Any], text: str) -> Optional[Any]:
    """Greedy longest-match segmentation for grapheme-to-phoneme composition.

    A pronunciation lexicon keyed on aksharas can pronounce a word it has never
    seen, provided every akshara in it is known. That is genuine (if shallow)
    generalisation, and it is the honest limit of what transfers between TTS
    systems: symbols, not voices.
    """
    source = text.strip()
    if not source:
        return None
    keys = sorted(lexicon, key=len, reverse=True)
    out, i, matched = [], 0, 0
    while i < len(source):
        for key in keys:
            if key and source.startswith(key, i):
                out.append(str(lexicon[key]))
                i += len(key)
                matched += len(key)
                break
        else:
            if source[i].isspace():
                out.append(" ")
            i += 1
    if not source or matched / len(source) < COMPOSITION_COVERAGE:
        return None
    return "".join(out)
