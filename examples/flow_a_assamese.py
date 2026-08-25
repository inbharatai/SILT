"""Flow A -- Assamese language transfer.

Sender:   MOCK curated Assamese source (stands in for IndicTrans2 / Samanantar)
Receiver: MOCK Qwen, weak on Assamese
Goal:     lift Assamese->English and English->Assamese on a HELD-OUT split by
          transferring a glossary, while not damaging Hindi->English.

Watch for three honest behaviours in the output:
  * one signal dropped because the receiver already knew it,
  * one dropped because the SENDER was wrong (the mock is seeded with a bad
    answer for চাহ/tea on purpose),
  * a Hindi regression check that must stay flat.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _common import demo_pipeline, knowledge_from, show, suite

from asea.core.protocol import Domain
from asea.modules.mock.zoo import make_generic_sender, make_qwen, text_cap


def main(workspace: Path) -> None:
    as_en = suite("assamese_english")
    en_as = suite("assamese_phrases")
    hi_en = suite("hindi_english")

    # Sender knows the reference answers -- except one, seeded wrong.
    sender_knowledge = knowledge_from(
        [as_en, en_as], splits=("extraction",), overrides={"চাহ": "coffee"}
    )
    sender = make_generic_sender(
        module_id="assamese-corpus-mock",
        display_name="Curated Assamese source (MOCK)",
        capabilities=[
            text_cap("translate", "as->en", Domain.TRANSLATION),
            text_cap("generate", "en->as", Domain.LANGUAGE),
        ],
        knowledge=sender_knowledge,
    )

    # Receiver knows Hindi well, and exactly one Assamese word.
    receiver_knowledge = knowledge_from([hi_en], splits=("regression",))
    receiver_knowledge.update(
        knowledge_from([as_en], splits=("extraction",), only_cases=["as_w11"])
    )
    receiver = make_qwen(knowledge=receiver_knowledge, fallback="echo")

    pipeline = demo_pipeline(workspace)
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter(
        "assamese-to-qwen", sender.module_id, receiver.module_id,
        description="Assamese glossary transfer into a general instruct model",
    )

    report = pipeline.run("assamese-to-qwen", suites=[as_en, en_as, hi_en])
    show(report, "FLOW A -- Assamese language transfer (all modules MOCK)")
    print("\nmemory store: {}".format(pipeline.store.stats()))
    print("audit chain : {}".format(pipeline.audit.verify()))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(Path(tmp))
