"""Closed, inert validation records; none is an admission certificate."""
from typing import Dict, List, Literal, Optional
from pydantic import Field, StrictStr, model_validator
from asea.compose.schema import StrictModel

Hash = StrictStr


class LicenseCard(StrictModel):
    spdx: StrictStr = Field(min_length=1, max_length=200)
    source: StrictStr = Field(min_length=1, max_length=2048)
    revision: StrictStr = Field(min_length=1, max_length=200)
    license_text_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    attribution: StrictStr = Field(min_length=1, max_length=4096)
    usage_restrictions: List[StrictStr] = Field(default_factory=list, max_length=32)
    consent_privacy: Literal["reviewed", "not_applicable", "unknown"]
    redistribution: Literal["allowed", "prohibited", "unknown"]


class Member(StrictModel):
    member_id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    dataset_id: StrictStr = Field(min_length=1, max_length=128)
    family_id: StrictStr = Field(min_length=1, max_length=128)
    split: Literal["development", "final"]
    role: Literal["target", "control"]
    language: StrictStr = Field(min_length=1, max_length=64)
    # Text is inert, including candidate code. No exec, imports, or test callbacks.
    input: StrictStr = Field(max_length=65536)
    expected: StrictStr = Field(max_length=65536)
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class Suite(StrictModel):
    schema_version: Literal[1] = 1
    suite_id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    version: StrictStr = Field(min_length=1, max_length=64)
    parent_suite_id: Optional[StrictStr] = None
    owner: StrictStr = Field(min_length=1, max_length=200)
    format: Literal["data_only_text_v1"]
    exposure: Literal["public_training_overlap_unknown", "local_training_overlap_unknown"]
    selection_method: StrictStr = Field(min_length=1, max_length=4096)
    selection_seed: int
    sampling_frame: StrictStr = Field(min_length=1, max_length=4096)
    exclusions: List[StrictStr] = Field(default_factory=list, max_length=128)
    datasets: Dict[StrictStr, LicenseCard] = Field(min_length=1, max_length=64)
    members: List[Member] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def membership(self):
        from asea.artifacts import digest
        seen, families, inputs = set(), {}, {}
        for member in self.members:
            if member.member_id in seen or member.dataset_id not in self.datasets:
                raise ValueError("duplicate member ID or missing dataset provenance")
            seen.add(member.member_id)
            family = member.family_id
            if family in families and families[family] != member.split:
                raise ValueError("family crosses development/final split")
            families[family] = member.split
            if member.input in inputs and inputs[member.input] != member.split:
                raise ValueError("input content crosses development/final split")
            inputs[member.input] = member.split
            if digest({"input": member.input, "expected": member.expected}) != member.content_sha256:
                raise ValueError("member content hash mismatch")
        return self


class Policy(StrictModel):
    schema_version: Literal[1] = 1
    purpose: Literal["noncommercial_experimental", "commercial"]
    evaluator_revision: StrictStr = Field(min_length=1, max_length=200)
    metric: Literal["exact_text"]
    threshold: float = Field(ge=0, le=1)
    max_cases: int = Field(ge=1, le=256)
    max_seconds: int = Field(ge=1, le=3600)
    stopping_rule: Literal["one_attempt_all_cases"]
    missing_output: Literal["failure"]


class ResultInput(StrictModel):
    """Untrusted evaluator output reference, never a user asserted pass."""
    schema_version: Literal[1] = 1
    status: Literal["completed", "candidate_failure", "infrastructure_failure"]
    evaluator_revision: StrictStr = Field(min_length=1, max_length=200)
    output_sha256: Optional[StrictStr] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reason: StrictStr = Field(min_length=1, max_length=4096)


class VoiceModelManifest(StrictModel):
    schema_version: Literal[1] = 1
    model_id: StrictStr = Field(min_length=1, max_length=200)
    model_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    licenses: List[LicenseCard] = Field(min_length=1, max_length=32)


class ListeningTrial(StrictModel):
    clip_id: StrictStr = Field(min_length=1, max_length=128)
    audio_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    listener_id: Optional[StrictStr] = None
    language_proficiency: Optional[StrictStr] = None
    playback_setup: Optional[StrictStr] = None
    blind_transcript: Optional[StrictStr] = None
    script_hidden_during_transcription: Optional[bool] = None
    naturalness_rating: Optional[int] = Field(default=None, ge=1, le=5)
    pronunciation_notes: Optional[StrictStr] = None
    listened_at: Optional[StrictStr] = None


class ListeningReview(StrictModel):
    schema_version: Literal[1] = 1
    packet_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: StrictStr = Field(min_length=1, max_length=200)
    trials: List[ListeningTrial] = Field(min_length=1, max_length=4096)


class BatchClip(StrictModel):
    clip_id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    audio: StrictStr
    text: StrictStr = Field(min_length=1, max_length=65536)
    language: StrictStr = Field(min_length=1, max_length=64)
    model_manifest: StrictStr


class ListeningBatch(StrictModel):
    schema_version: Literal[1] = 1
    protocol_id: StrictStr = Field(min_length=1, max_length=200)
    clips: List[BatchClip] = Field(min_length=1, max_length=64)
