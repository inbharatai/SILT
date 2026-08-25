"""Registration, binding and handshake negotiation."""

from __future__ import annotations

import pytest

from asea.core.errors import HandshakeError, RegistrationError
from asea.core.handshake import Handshake
from asea.core.protocol import CapabilityManifest, LearningLevel, Modality
from asea.modules.mock.base import MockModule
from asea.modules.mock.zoo import (
    code_cap,
    make_ai4bharat_asr,
    make_gemma,
    make_generic_receiver,
    make_qwen,
    text_cap,
    tts_cap,
)
from asea.registry.registries import (
    AdapterRegistry,
    ModuleRegistry,
    ReceiverRegistry,
    SenderRegistry,
)


@pytest.fixture
def modules():
    reg = ModuleRegistry()
    reg.register_module(make_qwen())
    reg.register_module(make_gemma())
    reg.register_module(make_ai4bharat_asr())
    return reg


def test_registration_and_role_views(modules):
    assert set(modules.ids()) == {"qwen-mock", "gemma-mock", "ai4bharat-asr-mock"}
    senders = SenderRegistry(modules)
    receivers = ReceiverRegistry(modules)
    assert "ai4bharat-asr-mock" in senders.ids()
    assert "ai4bharat-asr-mock" not in receivers.ids()
    with pytest.raises(RegistrationError):
        receivers.get("ai4bharat-asr-mock")


def test_duplicate_registration_refused(modules):
    with pytest.raises(RegistrationError):
        modules.register_module(make_qwen())
    modules.register_module(make_qwen(), replace=True)  # explicit override is fine


def test_unknown_module_lookup(modules):
    with pytest.raises(RegistrationError):
        modules.get("does-not-exist")


def test_module_misreporting_mock_status_is_refused():
    class Liar(MockModule):
        is_mock = False  # adapter claims real...

        def manifest(self) -> CapabilityManifest:
            m = super().manifest()
            return CapabilityManifest(**{**m.model_dump(), "is_mock": True})  # ...manifest says mock

    liar = Liar("liar", "Liar", [text_cap("translate", "as->en")], ["sender"])
    with pytest.raises(RegistrationError):
        ModuleRegistry().register_module(liar)


def test_module_with_no_roles_refused():
    m = MockModule("roleless", "Roleless", [text_cap("translate", "as->en")], [])
    with pytest.raises(RegistrationError):
        ModuleRegistry().register_module(m)


def test_self_transfer_binding_refused(modules):
    adapters = AdapterRegistry(modules)
    with pytest.raises(RegistrationError, match="self-transfer"):
        adapters.bind("loop", "qwen-mock", "qwen-mock")


def test_binding_requires_correct_roles(modules):
    adapters = AdapterRegistry(modules)
    with pytest.raises(RegistrationError, match="cannot act as receiver"):
        adapters.bind("bad", "qwen-mock", "ai4bharat-asr-mock")


def test_handshake_refuses_self_transfer():
    qwen = make_qwen()
    with pytest.raises(HandshakeError, match="self-transfer"):
        Handshake().open("a", qwen, qwen)


def test_handshake_refuses_incompatible_modalities():
    asr = make_ai4bharat_asr()
    tts_receiver = make_generic_receiver(
        module_id="tts-recv", capabilities=[tts_cap("as-ipa")]
    )
    with pytest.raises(HandshakeError, match="no shared modality"):
        Handshake().open("a", asr, tts_receiver)


def test_handshake_negotiates_down_to_lowest_ceiling():
    """Qwen mock allows L4; Gemma mock caps at L3. The session must take L3."""
    sender = make_qwen()
    receiver = make_gemma()
    session = Handshake().open(
        "a", sender, receiver, requested_level=LearningLevel.L5_DISTILL_DATASET
    )
    assert session.negotiated_level == LearningLevel.L3_SKILL_PACKET
    assert session.involves_mock is True
    assert Modality.TEXT in session.shared_modalities


def test_session_dict_is_auditable():
    session = Handshake().open("a", make_qwen(), make_gemma())
    payload = session.to_dict()
    for key in ("session_id", "sender", "receiver", "involves_mock", "opened_at"):
        assert key in payload
