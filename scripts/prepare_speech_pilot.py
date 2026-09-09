"""Fetch/check a frozen, real-recording FLEURS English pilot. No model inference.

Only official google/fleurs pinned bytes; archive never extracted wholesale.
Synthetic waveforms occur only in unit tests, never in the dataset.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tarfile
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/speech-pilot-v2"
AUDIO = Path("/agent/workspace/silt-pass2-data/fleurs-en")
PIN = "4683b04af03d2d9549064c7d72060a9a94bb6046"
API = "https://huggingface.co/api/datasets/google/fleurs"
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/" + PIN + "/"
SEED = "silt-speech-pilot-v2-fleurs-en-test-before-inference-1"
MAX_ARCHIVE = 1_000_000_000
MAX_EXPANDED = 1_000_000_000
MAX_WAV = 2_000_000
COUNT = 60
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def save_json(path, obj):
    path.write_bytes(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")


def small_get(url, limit=5_000_000):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("metadata fetch exceeds bound: " + url)
    return data


def select(metadata):
    """No generated labels or outputs: TSV columns 0,1,2,3,5 only; ignore gender."""
    eligible = []
    for cols in csv.reader(io.StringIO(metadata.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(cols) != 7:
            raise ValueError("unexpected upstream TSV schema")
        sid, name, raw, transcription, _, samples, _ = cols
        if not re.fullmatch(r"[0-9]+", sid) or not re.fullmatch(r"[0-9]+\.wav", name):
            raise ValueError("unsafe upstream identifier")
        samples = int(samples)
        if 0 < samples <= 320000 and transcription.strip() and len(transcription) <= 65536:
            eligible.append(dict(sentence_id=sid, source_filename=name, reference_text=transcription,
                                 raw_transcription=raw, num_samples=samples))
    # Canonical cryptographic order, not interpreter RNG or source-row ordering.
    eligible.sort(key=lambda r: (sha((SEED + "|utterance|" + r["sentence_id"] + "|" + r["source_filename"]).encode()), r["source_filename"]))
    unique = {}
    for row in eligible:
        unique.setdefault(row["sentence_id"], row)
    population = sorted(unique.values(), key=lambda r: (sha((SEED + "|sentence|" + r["sentence_id"]).encode()), r["sentence_id"]))
    if len(population) < COUNT:
        raise ValueError("fewer than 60 eligible distinct sentence IDs")
    rows = []
    for i, row in enumerate(population[:COUNT]):
        rows.append(dict(row, case_id="fleurs_en_" + row["sentence_id"], group="target" if i < 48 else "control"))
    return rows, dict(total_eligible_utterances=len(eligible), eligible_distinct_sentences=len(unique))


def parse_wav(data):
    """Bounded RIFF parser for source IEEE float32 and PCM16, mono 16kHz."""
    if len(data) > MAX_WAV or len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("invalid/big WAV")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("RIFF size mismatch")
    chunks, pos = {}, 12
    while pos + 8 <= len(data):
        key = data[pos:pos+4]
        size = struct.unpack_from("<I", data, pos+4)[0]
        pos += 8
        if pos + size > len(data):
            raise ValueError("truncated WAV chunk")
        if key in (b"fmt ", b"data"):
            if key in chunks:
                raise ValueError("duplicate WAV chunk")
            chunks[key] = data[pos:pos+size]
        pos += size + (size % 2)
    fmt, payload = chunks.get(b"fmt ", b""), chunks.get(b"data", b"")
    if len(fmt) < 16 or not payload:
        raise ValueError("missing WAV fmt/data")
    code, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", fmt)
    if code == 65534:
        if len(fmt) < 40 or fmt[26:40] != bytes.fromhex("000000001000800000aa00389b71"):
            raise ValueError("unsupported extensible format")
        code = struct.unpack_from("<H", fmt, 24)[0]
    if channels != 1 or rate != 16000 or (code, bits) not in ((3, 32), (1, 16)):
        raise ValueError("requires mono 16kHz IEEE float32 or PCM16")
    if align != bits // 8 or byte_rate != rate * align or len(payload) % align:
        raise ValueError("inconsistent WAV layout")
    frames = len(payload) // align
    if not 0 < frames <= 320000:
        raise ValueError("WAV duration outside (0,20] seconds")
    if code == 3:
        values = [v[0] for v in struct.iter_unpack("<f", payload)]
        if any(not math.isfinite(v) for v in values):
            raise ValueError("nonfinite waveform")
        clipped = sum(v < -1 or v > 1 for v in values)
        pcm = b"".join(struct.pack("<h", max(-32768, min(32767, round(v * 32768)))) for v in values)
        transformation = "IEEE float32 to PCM16: round ties-to-even(x*32768), saturate [-32768,32767]; no resample/trim/normalization"
    else:
        pcm, clipped = payload, 0
        transformation = "PCM16 samples unchanged; canonical RIFF PCM16 container rewritten"
    if not any(v[0] for v in struct.iter_unpack("<h", pcm)):
        raise ValueError("silent waveform")
    result = io.BytesIO()
    with wave.open(result, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    info = dict(source_encoding="IEEE_FLOAT_32" if code == 3 else "PCM_16", sample_rate=rate,
                channels=channels, num_samples=frames, duration_seconds=frames/rate,
                source_bytes=len(data), source_sha256=sha(data), output_bytes=len(result.getvalue()),
                output_sha256=sha(result.getvalue()), output_encoding="PCM_16", transformation=transformation,
                source_out_of_range_samples=clipped)
    return result.getvalue(), info


def extract_selected(archive, selected, destination):
    """Only exact test/<numeric>.wav whitelist, exclusive files in exclusive dir."""
    destination.mkdir(parents=False, exist_ok=False)
    originals = destination / "originals"
    originals.mkdir()
    wanted = {"test/" + r["source_filename"]: r for r in selected}
    if len(wanted) != len(selected) or len(selected) > COUNT:
        raise ValueError("duplicate/too many selections")
    found, expanded, members = {}, 0, 0
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            members += 1
            expanded += member.size
            if members > 10000 or expanded > MAX_EXPANDED or member.size < 0:
                raise ValueError("archive expanded size/count limit")
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("unsafe archive path")
            if not (member.isfile() or member.isdir()):
                raise ValueError("archive contains nonregular entry")
            if member.name not in wanted:
                continue
            if not member.isfile() or member.name in found or not 0 < member.size <= MAX_WAV:
                raise ValueError("invalid/duplicate selected WAV member")
            row = wanted[member.name]
            source = tar.extractfile(member)
            data = source.read(MAX_WAV + 1)
            if len(data) != member.size:
                raise ValueError("selected member size mismatch")
            pcm, info = parse_wav(data)
            if info["num_samples"] != row["num_samples"]:
                raise ValueError("metadata/audio sample-count mismatch")
            with (originals / row["source_filename"]).open("xb") as f:
                f.write(data)
            output = destination / (row["case_id"] + ".wav")
            with output.open("xb") as f:
                f.write(pcm)
            found[member.name] = dict(row, audio=info, input_file=str(output))
    if set(found) != set(wanted):
        raise ValueError("selected archive members missing: " + str(sorted(set(wanted)-set(found))))
    return [found["test/" + row["source_filename"]] for row in selected], dict(archive_members=members, archive_expanded_member_bytes=expanded)


def suite(rows):
    return dict(schema_version=1, name="FLEURS en_us test real-recording 60-sentence pilot v2",
                reference_source=BASE + "data/en_us/test.tsv", policy_version="composition-admission-v2",
                claims=["reference_text"], cases=[dict(id=r["case_id"], group=r["group"], input_file=r["input_file"],
                reference=r["reference_text"], metric="word_error_rate", threshold=0.5) for r in rows])


def fetch(out, audio):
    if audio.exists() or out.exists():
        raise ValueError("fetch refuses existing output/audio directories; use --check")
    if ROOT == audio.resolve() or ROOT in audio.resolve().parents:
        raise ValueError("audio must be outside source tree")
    # Resolve exact revision through public HF API, before data downloads.
    api_bytes = small_get(API + "/revision/" + PIN)
    api = json.loads(api_bytes)
    if api["id"] != "google/fleurs" or api["sha"] != PIN:
        raise ValueError("repository/revision mismatch")
    card = small_get(BASE + "README.md")
    if b"license:\n- cc-by-4.0" not in card or "cc-by-4.0" not in api["cardData"]["license"]:
        raise ValueError("CC BY 4.0 not confirmed by both pinned API and card")
    script = small_get(BASE + "fleurs.py")  # evidence only, NEVER import/execute upstream
    if b'"audio/{split}.tar.gz"' not in script or b'"data/{langs}/"' not in script:
        raise ValueError("upstream archive path no longer matches inspected source")
    tree = json.loads(small_get(API + "/tree/" + PIN + "/data/en_us?expand=true"))
    audio_tree = json.loads(small_get(API + "/tree/" + PIN + "/data/en_us/audio?expand=true"))
    meta_info = next(x for x in tree if x["path"] == "data/en_us/test.tsv")
    archive_info = next(x for x in audio_tree if x["path"] == "data/en_us/audio/test.tar.gz")
    declared = archive_info["size"]
    if not 0 < declared <= MAX_ARCHIVE or archive_info.get("lfs", {}).get("size") != declared:
        raise ValueError("archive declared size invalid or over 1GB: " + str(declared))
    free = shutil.disk_usage(audio.parent if audio.parent.exists() else audio.parent.parent).free
    if free < 2_000_000_000:
        raise ValueError("less than 2GB disk free: " + str(free))
    metadata = small_get(BASE + "data/en_us/test.tsv")
    git_blob = hashlib.sha1(b"blob " + str(len(metadata)).encode() + b"\0" + metadata).hexdigest()
    if len(metadata) != meta_info["size"] or git_blob != meta_info["oid"]:
        raise ValueError("metadata size/git blob mismatch")
    if "lfs" in meta_info and sha(metadata) != meta_info["lfs"]["oid"]:
        raise ValueError("metadata LFS digest mismatch")
    rows, eligibility = select(metadata)
    license_method = "official legalcode URL"
    try:
        license_text = small_get(LICENSE_URL)
    except Exception as exc:
        # Legal text only, never alternate dataset/audio. A previously preserved
        # byte-identical CC legalcode copy is bundled in this source distribution.
        candidates = [OUT / "CC-BY-4.0.txt", ROOT / "data/pilot-v2/sources/CC-BY-4.0.txt"]
        cached = next((p for p in candidates if p.is_file()), None)
        if cached is None:
            raise
        license_text = cached.read_bytes()
        license_method = "verified cached legalcode after " + type(exc).__name__ + ": " + str(exc)
    if sha(license_text) != "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411":
        raise ValueError("CC legalcode digest mismatch")
    out.mkdir(parents=True, exist_ok=False)
    files = {"upstream-README.md": card, "upstream-fleurs.py.txt": script,
             "upstream-test.tsv": metadata, "CC-BY-4.0.txt": license_text, "hf-revision-api.json": api_bytes}
    for name, data in files.items():
        (out / name).write_bytes(data)
    selection = dict(seed=SEED, order="SHA256(seed|utterance|sentence_id|filename) chooses one; SHA256(seed|sentence|sentence_id) orders sentence population; first 60; first 48 target, last 12 control",
                     eligibility_filters=["official en_us test only", "0 < metadata num_samples <= 320000 (20 seconds at 16kHz)", "nonblank dataset transcription <=65536 chars", "numeric sentence ID and numeric WAV basename", "one utterance per sentence ID; no speaker independence claim"],
                     rows=rows, selection_sha256=sha(canonical(rows)), **eligibility)
    save_json(out / "selection.json", selection)  # frozen BEFORE archive or inference
    source = dict(repository="google/fleurs", revision=PIN, revision_api_url=API + "/revision/" + PIN,
                  card_url=BASE+"README.md", script_url=BASE+"fleurs.py", metadata_url=BASE+"data/en_us/test.tsv",
                  archive_url=BASE+"data/en_us/audio/test.tar.gz", license="CC-BY-4.0", license_url=LICENSE_URL, license_text_acquisition=license_method,
                  metadata_sha256=sha(metadata), metadata_git_blob_oid=git_blob,
                  metadata_lfs_sha256=meta_info.get("lfs", {}).get("oid"),
                  metadata_lfs_note="TSV is ordinary Git blob at this revision, not LFS; archive has LFS SHA256",
                  metadata_api_entry=meta_info, archive_api_entry=archive_info, archive_declared_bytes=declared,
                  free_disk_before_download=free, archived_evidence_sha256={name: sha(data) for name, data in files.items()})
    save_json(out / "provenance.json", source)
    print(json.dumps(dict(stage="preflight_passed_selection_frozen", archive_bytes=declared, disk_free=free, selection_sha256=selection["selection_sha256"])), flush=True)
    audio.parent.mkdir(parents=True, exist_ok=True)
    temporary = audio.parent / (audio.name + ".test.tar.gz.partial")
    transferred, hasher = 0, hashlib.sha256()
    try:
        with temporary.open("xb") as f, urllib.request.urlopen(source["archive_url"], timeout=120) as response:
            header = response.headers.get("Content-Length")
            if header and int(header) != declared:
                raise ValueError("HTTP declared archive size mismatch")
            while True:
                chunk = response.read(min(1024*1024, declared-transferred+1))
                if not chunk:
                    break
                transferred += len(chunk)
                if transferred > declared or transferred > MAX_ARCHIVE:
                    raise ValueError("archive download byte limit exceeded")
                f.write(chunk)
                hasher.update(chunk)
        if transferred != declared or hasher.hexdigest() != archive_info["lfs"]["oid"]:
            raise ValueError("archive length/LFS SHA256 mismatch")
        acquired, extraction = extract_selected(temporary, rows, audio)
        save_json(out / "manifest.json", dict(rows=acquired, selection_sha256=selection["selection_sha256"]))
        save_json(out / "evaluation-suite.json", suite(acquired))
        status = dict(status="acquired_no_inference", actual_count=len(acquired), target_count=48, control_count=12,
                      downloaded_archive_bytes=transferred, archive_sha256=hasher.hexdigest(),
                      source_audio_bytes=sum(r["audio"]["source_bytes"] for r in acquired),
                      pcm_audio_bytes=sum(r["audio"]["output_bytes"] for r in acquired), **extraction)
        save_json(out / "acquisition.json", status)
        print(json.dumps(status), flush=True)
    except Exception as exc:
        save_json(out / "failure.json", dict(error_type=type(exc).__name__, error=str(exc), downloaded_bytes=transferred,
                  actual_pcm_files=len(list(audio.glob("*.wav"))) if audio.exists() else 0,
                  disposition="STOP; no substitution, no inference; partial paths intentionally retained for inspection"))
        raise
    finally:
        if temporary.exists():
            temporary.unlink()  # no unnecessary complete corpus archive retained
    check(out)


def check(out):
    """Offline integrity check; NEVER imports a model/runtime or performs inference."""
    provenance = json.loads((out / "provenance.json").read_text())
    for name, expected in provenance["archived_evidence_sha256"].items():
        if sha((out / name).read_bytes()) != expected:
            raise ValueError("evidence digest mismatch: " + name)
    metadata = (out / "upstream-test.tsv").read_bytes()
    rows, _ = select(metadata)
    selection = json.loads((out / "selection.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    if rows != selection["rows"] or sha(canonical(rows)) != selection["selection_sha256"] or manifest["selection_sha256"] != selection["selection_sha256"]:
        raise ValueError("selection mismatch")
    acquired = manifest["rows"]
    if len(acquired) != 60 or len({r["sentence_id"] for r in acquired}) != 60:
        raise ValueError("count/independence mismatch")
    for expected, actual in zip(rows, acquired):
        if any(actual[k] != v for k, v in expected.items()):
            raise ValueError("manifest metadata mismatch")
        path = Path(actual["input_file"])
        pcm, info = parse_wav((path.parent / "originals" / actual["source_filename"]).read_bytes())
        if info != actual["audio"] or path.read_bytes() != pcm:
            raise ValueError("audio digest/transformation mismatch")
    parent = Path(acquired[0]["input_file"]).parent
    if {p.name for p in parent.glob("*.wav")} != {r["case_id"]+".wav" for r in acquired} or {p.name for p in (parent/"originals").iterdir()} != {r["source_filename"] for r in acquired}:
        raise ValueError("unexpected audio files")
    suite_data = json.loads((out / "evaluation-suite.json").read_text())
    if suite_data != suite(acquired):
        raise ValueError("suite mismatch")
    from asea.compose.schema import EvaluationSuite
    EvaluationSuite.model_validate(suite_data)
    result = dict(status="check_passed_no_inference", actual_count=60, target_count=48, control_count=12,
                  selection_sha256=selection["selection_sha256"], pcm_audio_sha256=sha(canonical([r["audio"]["output_sha256"] for r in acquired])))
    save_json(out / "validation.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--fetch", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO)
    args = parser.parse_args()
    if args.fetch:
        fetch(args.output, args.audio_dir)
    else:
        check(args.output)


if __name__ == "__main__":
    main()
