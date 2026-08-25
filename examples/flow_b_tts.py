"""Flow B -- TTS pronunciation transfer.

Sender:   MOCK AI4Bharat G2P source
Receiver: MOCK generic TTS front-end with no Assamese lexicon
Goal:     transfer a symbolic pronunciation lexicon so the receiver can
          pronounce words it has never seen.

SCOPE, stated plainly: what moves here is the SYMBOLIC layer -- grapheme to
phoneme mappings. Voice timbre and learned prosody live in the acoustic model
and vocoder and do not transfer through this adapter. The distilled packet says
so in its own payload (``not_transferable``).

This flow also demonstrates the handshake guard: an ASR module is registered and
an ASR->TTS binding is attempted, which is refused because the two share no
modality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _common import demo_pipeline, knowledge_from, show, suite

from asea.core.errors import HandshakeError, RegistrationError
from asea.modules.mock.zoo import make_ai4bharat_asr, make_ai4bharat_tts, make_generic_receiver, tts_cap


def main(workspace: Path) -> None:
    g2p = suite("tts_pronunciation_as")

    sender = make_ai4bharat_tts(
        knowledge=knowledge_from([g2p], splits=("extraction", "heldout"))
    )
    receiver = make_generic_receiver(
        module_id="generic-tts-mock",
        display_name="Generic TTS front-end (MOCK)",
        capabilities=[tts_cap("as-ipa")],
        knowledge=None,
        fallback="unknown",
    )
    asr = make_ai4bharat_asr()

    pipeline = demo_pipeline(workspace)
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.register_module(asr)

    # The guard: ASR and TTS share no modality, so no session can be opened.
    try:
        pipeline.bind_adapter("asr-to-tts", asr.module_id, receiver.module_id)
        pipeline.handshake.open("asr-to-tts", asr, receiver)
        print("UNEXPECTED: incompatible pair was accepted")
    except (HandshakeError, RegistrationError) as exc:
        print("handshake guard working as intended:\n   {}".format(exc))

    pipeline.bind_adapter(
        "g2p-to-tts", sender.module_id, receiver.module_id,
        description="Assamese pronunciation lexicon transfer (symbolic only)",
    )
    report = pipeline.run("g2p-to-tts", suites=[g2p])
    show(report, "FLOW B -- TTS pronunciation transfer (all modules MOCK)")

    for packet in report.distilled:
        payload = packet.distilled_skill or {}
        print("\nlexicon scope declared in payload: {}".format(payload.get("scope")))
        print("explicitly not transferable      : {}".format(payload.get("not_transferable")))
        print("entries                          : {}".format(len(payload.get("entries", []))))

    print("\nmemory store: {}".format(pipeline.store.stats()))
    print("audit chain : {}".format(pipeline.audit.verify()))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(Path(tmp))
