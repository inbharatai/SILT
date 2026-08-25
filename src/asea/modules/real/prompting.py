"""Prompt construction for real connectors.

Two jobs, both shared across every real backend so that a change here changes
every connector consistently:

  1. turn a CapabilityKey into a task instruction,
  2. render approved skill payloads into text a model can condition on.

The second is the security-sensitive one. It accepts ONLY the output of
``SkillPacket.redacted_for_receiver()``, which structurally cannot contain
``sender_output``. If you extend this to read any other field, you have broken
the central invariant of the system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core.protocol import CapabilityKey, Domain, Modality

#: Human-readable language names for prompt text. Extend as needed.
LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "mni": "Manipuri (Meitei)",
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
}


def language_name(tag: Optional[str]) -> str:
    if not tag:
        return "the target language"
    return LANGUAGE_NAMES.get(tag, tag)


def split_pair(language: Optional[str]):
    """'as->en' -> ('as', 'en'); 'as' -> (None, 'as')."""
    if not language:
        return None, None
    if "->" in language:
        src, tgt = language.split("->", 1)
        return src.strip(), tgt.strip()
    return None, language.strip()


def system_for_capability(capability: CapabilityKey) -> str:
    """Task instruction. Terse on purpose: long preambles cost tokens and,
    on small instruct models, actively degrade instruction following."""
    src, tgt = split_pair(capability.language)

    if capability.task_type == "translate" and src and tgt:
        return (
            "Translate the user's text from {} to {}. "
            "Reply with the translation only, no explanation, no quotes."
        ).format(language_name(src), language_name(tgt))

    if capability.task_type == "generate" and tgt:
        return (
            "Reply in {} only. Output the {} text and nothing else."
        ).format(language_name(tgt), language_name(tgt))

    if capability.modality == Modality.CODE:
        return (
            "Fix the bug in the user's code. Reply with the corrected code only, "
            "no explanation, no code fences."
        )

    if capability.modality == Modality.SPEECH_TTS:
        return (
            "Give the IPA phonemic transcription of the user's text. "
            "Reply with the transcription only, no slashes or brackets."
        )

    if capability.domain in (Domain.MEDICAL, Domain.LEGAL, Domain.FINANCE):
        # A real deployment should route high-risk domains through a reviewed
        # template rather than a free-form instruction. Kept blunt and cautious.
        return (
            "You are a triage assistant. State the escalation guidance only. "
            "Never diagnose, never give dosages, and always advise seeking "
            "professional care where appropriate."
        )

    return "Answer the user concisely. Output the answer only."


def render_skills(skills: List[Dict[str, Any]], max_entries: int = 60) -> str:
    """Render redacted skill payloads as reference text for the prompt.

    Deliberately compact and tabular. Prose framing invites the model to
    paraphrase the reference instead of using it.
    """
    lines: List[str] = []
    for skill in skills or []:
        payload = skill.get("distilled_skill") or {}

        entries = payload.get("entries") or []
        for entry in entries[:max_entries]:
            if "source" in entry:
                lines.append("{} = {}".format(entry["source"], entry["target"]))
            elif "grapheme" in entry:
                lines.append("{} -> {}".format(entry["grapheme"], entry["phoneme"]))

        for example in (payload.get("examples") or [])[:max_entries]:
            lines.append(
                "wrong: {}\nright: {}".format(example.get("buggy"), example.get("fixed"))
            )

        for rule in (payload.get("rules") or [])[:max_entries]:
            lines.append(
                "if [{}] then: {}".format(rule.get("condition"), rule.get("action"))
            )

        for pair in (payload.get("pairs") or [])[:max_entries]:
            lines.append(
                "{} => {}".format(pair.get("observed"), pair.get("corrected"))
            )

    return "\n".join(lines)


def build_messages(
    capability: CapabilityKey,
    prompt: Any,
    skills: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Chat messages for an instruct model.

    Skills are folded into the SINGLE system message (not a second ``system``
    turn): several HF chat templates (e.g. Gemma family) honour only the first
    system message, so a second one carrying the verified reference entries
    would be silently dropped and ``infer_with_skills`` would run with skills
    un-injected -- a silent degradation. One system message is portable.    """
    system_content = system_for_capability(capability)
    if skills:
        reference = render_skills(skills)
        if reference:
            system_content = (
                system_content
                + "\n\nUse these verified reference entries. When an entry "
                "applies, follow it exactly.\n" + reference
            )
    messages = [{"role": "system", "content": system_content}]
    messages.append({"role": "user", "content": str(prompt)})
    return messages
