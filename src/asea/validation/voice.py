"""Read-only PCM signal facts and proposed local listening packets, not quality scores."""
import hashlib
import io
import math
from pathlib import Path
import wave

from asea.artifacts import Blocked, digest, safe_path
from .schema import ListeningBatch, ListeningReview, VoiceModelManifest


def _model_rights(manifest):
    from . import license_status
    # Also accept the existing closed composition model manifest, without touching
    # its model_path, weights, signing keys, or any referenced credentials.
    if "kind" in manifest:
        from asea.compose.schema import ComponentManifest
        parsed = ComponentManifest.model_validate(manifest)
        licenses = [p.license for p in parsed.provenance]
        nc = any("NC" in value.upper() for value in licenses)
        return {"licenses": [p.model_dump() for p in parsed.provenance],
                "commercial_product_eligible": False,
                "commercial_terms_status": "blocked_pending_complete_license_card",
                "use_scope": "noncommercial_experimental" if nc else "subject_to_recorded_terms",
                "legal_authorization_granted": False,
                "missing": ["pinned license text hash", "usage/consent review"]}
    parsed = VoiceModelManifest.model_validate(manifest)
    return license_status(parsed.licenses)


def prepare_voice_evidence(audio, text, language, model_manifest):
    """Measure existing WAV bytes only. Noise can be structurally valid.

    1/2/3/4-byte integer PCM, <=120 seconds, <=32 MiB, <=8 channels.
    Silence = fraction of scalar samples with abs(normalized sample) <= 0.001;
    clipping = scalar samples exactly at either integer representable endpoint.
    Neither diagnostic is a perceptual pass/fail rule. No ASR or MOS is run.
    """
    from . import read_bytes, load_json
    if not isinstance(text, str) or not text.strip() or len(text) > 65536:
        raise Blocked("nonempty reference text required (at most 65536 characters)")
    if not isinstance(language, str) or not language.strip() or len(language) > 64:
        raise Blocked("explicit language required")
    if not isinstance(model_manifest, dict):
        model_manifest = load_json(model_manifest)
    rights = _model_rights(model_manifest)
    raw = read_bytes(audio)
    try:
        with wave.open(io.BytesIO(raw), "rb") as reader:
            channels, width, rate, frames = reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes()
            if (reader.getcomptype() != "NONE" or width not in (1, 2, 3, 4) or not 1 <= channels <= 8
                    or not 1 <= rate <= 384000 or frames <= 0 or frames / rate > 120):
                raise Blocked("unsupported/empty/oversized PCM WAV")
            pcm = reader.readframes(frames + 1)
            if len(pcm) != frames * channels * width:
                raise Blocked("truncated or inconsistent WAV frames")
    except (wave.Error, EOFError) as exc:
        raise Blocked("not a supported decodable PCM WAV") from exc
    scale = 2 ** (width * 8 - 1)
    count, clipped, silent, total, squared, peak = 0, 0, 0, 0.0, 0.0, 0.0
    for offset in range(0, len(pcm), width):
        value = pcm[offset] - 128 if width == 1 else int.from_bytes(pcm[offset:offset + width], "little", signed=True)
        sample = value / scale
        count += 1
        clipped += int(value in (-scale, scale - 1))
        silent += int(abs(sample) <= 0.001)
        total += sample
        squared += sample * sample
        peak = max(peak, abs(sample))
    facts = {"decoder": "stdlib.wave integer PCM", "sample_rate_hz": rate, "channels": channels,
             "sample_width_bytes": width, "frames": frames, "duration_seconds": frames / rate,
             "scalar_samples": count, "finite_samples": True, "peak": peak,
             "rms": math.sqrt(squared / count), "dc_offset": total / count,
             "clipping_count": clipped, "clipping_fraction": clipped / count,
             "clipping_definition": "integer representable endpoints, each scalar channel sample",
             "near_silence_fraction": silent / count, "near_silence_threshold_absolute": 0.001,
             "silence_definition": "sample fraction; not window/VAD/speech intelligibility",
             "signal_status": "structurally_valid_pcm_not_quality"}
    return {"schema_version": 1, "kind": "voice_evidence_pending",
            "audio": {"local_path": str(safe_path(audio)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
            "reference_text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "language": language,
            "model_manifest_sha256": digest(model_manifest), "model_rights": rights,
            "measurement_code_sha256": hashlib.sha256(read_bytes(Path(__file__))).hexdigest(),
            "waveform_facts": facts, "status": "pending_human",
            "dimensions": {"intelligibility": "pending_human", "naturalness": "pending_human", "pronunciation": "pending_human"},
            "automatic_text_recovery": "not_run; ASR roundtrip is not voice quality",
            "listening_approval": "unavailable_no_authenticated_human_collection",
            "quality_admission": False, "audio_output_admission": False,
            "allowed_claim": "Measured waveform facts for these exact bytes only",
            "blocked_claims": ["human intelligibility", "naturalness", "pronunciation", "population quality", "MOS", "commercial authorization"]}


def prepare_listening_batch(manifest, base_directory=None):
    """Local file references only; no uploads, ratings or approvals fabricated."""
    from . import load_json
    if not isinstance(manifest, dict):
        path = safe_path(manifest)
        base_directory = path.parent
        manifest = load_json(path)
    batch = ListeningBatch.model_validate(manifest)
    base = safe_path(base_directory or Path.cwd())
    if len({clip.clip_id for clip in batch.clips}) != len(batch.clips):
        raise Blocked("duplicate listening clip ID")
    clips, trials = [], []
    for clip in batch.clips:
        audio, model = safe_path(base / clip.audio), safe_path(base / clip.model_manifest)
        evidence = prepare_voice_evidence(audio, clip.text, clip.language, model)
        clips.append({"clip_id": clip.clip_id, "evidence": evidence})
        trials.append({"clip_id": clip.clip_id, "audio_sha256": evidence["audio"]["sha256"],
                       "listener_id": None, "language_proficiency": None, "playback_setup": None,
                       "blind_transcript": None, "script_hidden_during_transcription": None,
                       "naturalness_rating": None, "pronunciation_notes": None, "listened_at": None})
    packet = {"schema_version": 1, "kind": "proposed_listening_packet", "status": "pending_human",
              "protocol_id": batch.protocol_id, "clips": clips,
              "distribution": "local coordinator packet only; never upload automatically",
              "instructions": ["Coordinator packet contains scripts: do not expose to listeners before blind transcription.",
                  "Recruit qualified real humans and preregister randomized/blinded assignments, anchors and exclusions.",
                  "Collect blind transcription before revealing script; record actual playback setup and timestamps.",
                  "Naturalness question: How natural does this utterance sound? 1 very unnatural to 5 very natural.",
                  "Have qualified reviewers document pronunciation against predeclared accepted variants.",
                  "Real authenticated human collection and approval are unavailable in this CLI. Do not fill with agent or dummy ratings."],
              "human_collection_available": False, "quality_admission": False}
    packet["packet_sha256"] = digest(packet)
    packet["review_template"] = {"schema_version": 1, "packet_sha256": packet["packet_sha256"],
                                  "protocol_id": batch.protocol_id, "trials": trials}
    return packet


def validate_listening_review(review, packet):
    """Pure completeness checker. Even fully populated JSON is not human authentication.

    No storing, signing, accepting or fabricating approvals, and no MOS calculation.
    Caller supplied names/ratings remain unverified data, including fixture values.
    """
    parsed = ListeningReview.model_validate(review)
    problems = []
    body = {k: v for k, v in packet.items() if k not in {"packet_sha256", "review_template"}}
    if digest(body) != packet.get("packet_sha256") or parsed.packet_sha256 != packet.get("packet_sha256"):
        problems.append("packet digest mismatch")
    if parsed.protocol_id != packet.get("protocol_id"):
        problems.append("protocol mismatch")
    expected = {c["clip_id"]: c["evidence"]["audio"]["sha256"] for c in packet["clips"]}
    seen = set()
    for index, trial in enumerate(parsed.trials):
        data = trial.model_dump()
        if expected.get(trial.clip_id) != trial.audio_sha256:
            problems.append("trial %d: unknown clip or audio hash mismatch" % index)
        pair = (trial.clip_id, trial.listener_id)
        if pair in seen:
            problems.append("duplicate clip/listener trial")
        seen.add(pair)
        for field, value in data.items():
            if value is None or (isinstance(value, str) and not value.strip() and field != "blind_transcript"):
                problems.append("trial %d: missing %s" % (index, field))
        if trial.script_hidden_during_transcription is not True:
            problems.append("trial %d: no blind-transcription attestation" % index)
    if set(expected) - {trial.clip_id for trial in parsed.trials}:
        problems.append("missing clip coverage")
    return {"schema_complete": not problems, "problems": problems, "status": "pending_human",
            "authenticated_human_evidence": False, "quality_admission": False,
            "reason": "data-only completeness check cannot authenticate real listening, competence or approval"}
