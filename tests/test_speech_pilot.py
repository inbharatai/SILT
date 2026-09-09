"""Tiny SYNTHETIC audio fixtures test code only; none are FLEURS/dataset rows."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import tarfile
import wave

import pytest

SPEC = importlib.util.spec_from_file_location("speech_pilot", Path(__file__).resolve().parents[1] / "scripts/prepare_speech_pilot.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def synthetic_float(values=(0.0, 0.5, -0.5, 1.0, -1.0)):
    payload = b"".join(struct.pack("<f", x) for x in values)
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def synthetic_metadata():
    return "\n".join(f"{i}\t{i*2+j}.wav\tRaw original {i}.\toriginal {i}\tignored\t8000\tIGNORED" for i in range(90) for j in range(2)).encode()


def synthetic_tar(path, entries):
    with tarfile.open(path, "w:gz") as t:
        for name, kind, data in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = "/outside"
            t.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    return path


def selected():
    return [dict(sentence_id="1", source_filename="123.wav", case_id="fleurs_en_1", num_samples=5,
                 reference_text="synthetic fixture only", raw_transcription="Synthetic fixture only.", group="target")]


def test_selection_deterministic_unique_original_references():
    metadata = synthetic_metadata()
    rows, stats = p.select(metadata)
    shuffled = b"\n".join(reversed(metadata.splitlines()))
    assert rows == p.select(shuffled)[0]
    assert len(rows) == len({r["sentence_id"] for r in rows}) == 60
    assert sum(r["group"] == "target" for r in rows) == 48
    assert sum(r["group"] == "control" for r in rows) == 12
    assert all(r["reference_text"] == "original " + r["sentence_id"] for r in rows)
    assert stats == dict(total_eligible_utterances=180, eligible_distinct_sentences=90)
    assert all("gender" not in r for r in rows)


def test_selection_filters_duration_and_fails_short_population():
    metadata = synthetic_metadata().replace(b"\t8000\t", b"\t320001\t")
    with pytest.raises(ValueError, match="fewer than 60"):
        p.select(metadata)
    with pytest.raises(ValueError, match="unsafe"):
        p.select(synthetic_metadata().replace(b"0.wav", b"../0.wav"))


def test_float_conversion_metadata_and_digest():
    source = synthetic_float()
    output, info = p.parse_wav(source)
    assert info["source_encoding"] == "IEEE_FLOAT_32"
    assert info["source_sha256"] == p.sha(source)
    assert info["output_sha256"] == p.sha(output)
    assert info["num_samples"] == 5
    assert info["sample_rate"] == 16000
    with wave.open(io.BytesIO(output)) as wav:
        assert wav.getsampwidth() == 2 and wav.getnchannels() == 1
        assert struct.unpack("<5h", wav.readframes(5)) == (0, 16384, -16384, 32767, -32768)
    assert p.parse_wav(output)[0] == output


@pytest.mark.parametrize("values", [(0.0, 0.0), (float("nan"),), (float("inf"),)])
def test_reject_silence_nonfinite(values):
    with pytest.raises(ValueError):
        p.parse_wav(synthetic_float(values))


def test_reject_malformed_riff():
    with pytest.raises(ValueError, match="size mismatch"):
        p.parse_wav(synthetic_float()[:-1])


def test_safe_whitelist_exclusive_and_original_preserved(tmp_path):
    data = synthetic_float()
    archive = synthetic_tar(tmp_path / "test.tar.gz", [("test/999.wav", tarfile.REGTYPE, b"unselected"),
                                                       ("test/123.wav", tarfile.REGTYPE, data)])
    destination = tmp_path / "audio"
    rows, _ = p.extract_selected(archive, selected(), destination)
    assert len(rows) == 1
    assert (destination / "originals/123.wav").read_bytes() == data
    assert not (destination / "999.wav").exists()
    with pytest.raises(FileExistsError):
        p.extract_selected(archive, selected(), destination)


@pytest.mark.parametrize("name,kind", [("../escape.wav", tarfile.REGTYPE), ("/escape.wav", tarfile.REGTYPE),
                                      ("test/123.wav", tarfile.SYMTYPE), ("test/123.wav", tarfile.LNKTYPE)])
def test_reject_unsafe_archive(tmp_path, name, kind):
    archive = synthetic_tar(tmp_path / "bad.tar.gz", [(name, kind, synthetic_float())])
    with pytest.raises(ValueError):
        p.extract_selected(archive, selected(), tmp_path / "audio")
    assert not (tmp_path / "escape.wav").exists()


def test_duplicate_missing_oversize_members(tmp_path):
    data = synthetic_float()
    for label, entries in [("duplicate", [("test/123.wav", tarfile.REGTYPE, data)] * 2),
                           ("missing", [("test/456.wav", tarfile.REGTYPE, data)]),
                           ("oversize", [("test/123.wav", tarfile.REGTYPE, b"x" * (p.MAX_WAV+1))])]:
        archive = synthetic_tar(tmp_path / (label+".tar.gz"), entries)
        with pytest.raises(ValueError):
            p.extract_selected(archive, selected(), tmp_path / label)


def test_sample_metadata_mismatch(tmp_path):
    archive = synthetic_tar(tmp_path / "bad.tar.gz", [("test/123.wav", tarfile.REGTYPE, synthetic_float())])
    rows = selected()
    rows[0]["num_samples"] = 6
    with pytest.raises(ValueError, match="sample-count"):
        p.extract_selected(archive, rows, tmp_path / "audio")


def test_suite_schema_and_frozen_real_metadata():
    from asea.compose.schema import EvaluationSuite
    # Metadata-only test; actual recordings need not be shipped with code/tests.
    if not (p.OUT / "evaluation-suite.json").exists():
        pytest.skip("pilot metadata not acquired")
    value = json.loads((p.OUT / "evaluation-suite.json").read_text())
    suite = EvaluationSuite.model_validate(value)
    assert suite.claims == ["reference_text"]
    assert len(suite.cases) == 60
    assert all(c.metric == "word_error_rate" and c.threshold == 0.5 for c in suite.cases)
    assert len({c.reference for c in suite.cases}) == 60
    assert all(p.ROOT not in Path(c.input_file).parents for c in suite.cases)
