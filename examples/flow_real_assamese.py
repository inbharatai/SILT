"""REAL MODEL FLOW -- Assamese transfer with actual weights. No mocks.

    Sender   : NLLB-200-distilled-600M      (genuinely covers Assamese)
    Receiver : Qwen2.5 instruct             (genuinely weak at Assamese)
    Policy   : DEFAULT strict policy -- no mock bypass, because nothing is mocked

Backends, chosen with environment variables::

    # in-process HuggingFace weights (default)
    ASEA_RECEIVER=hf ASEA_RECEIVER_MODEL=Qwen/Qwen2.5-0.5B-Instruct python3 flow_real_assamese.py

    # local Ollama server -- the right choice on a laptop
    ASEA_RECEIVER=ollama ASEA_RECEIVER_MODEL=qwen2.5:7b-instruct python3 flow_real_assamese.py

    # real semantic similarity instead of the lexical proxy (downloads a model)
    ASEA_SIMILARITY=embedding python3 flow_real_assamese.py

A 0.5B receiver is a plumbing test. Use 7B+ via Ollama before drawing any
conclusion about whether this approach helps a model you would actually deploy.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from _common import show, suite

from asea.benchmarks.harness import BenchmarkHarness
from asea.core.pipeline import Pipeline
from asea.core.plugins import default_registry
from asea.evaluator.evaluator import Evaluator
from asea.core.gap import GapEngine
from asea.modules.real import (
    make_nllb_translator,
    make_qwen_hf,
    make_qwen_ollama,
    translation_capability,
)


def build_receiver():
    backend = os.environ.get("ASEA_RECEIVER", "hf").lower()
    caps = [translation_capability("as", "en"), translation_capability("hi", "en")]

    if backend == "ollama":
        model = os.environ.get("ASEA_RECEIVER_MODEL", "qwen2.5:7b-instruct")
        receiver = make_qwen_ollama(model=model, capabilities=caps)
        health = receiver.health()
        print("Ollama health: {}".format(health))
        if not health["model_present"]:
            print("\nFATAL: {}".format(health["hint"]))
            sys.exit(1)
        return receiver

    model = os.environ.get("ASEA_RECEIVER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    return make_qwen_hf(model, capabilities=caps, max_new_tokens=32)


def build_similarity():
    if os.environ.get("ASEA_SIMILARITY", "lexical").lower() == "embedding":
        from asea.modules.real import best_available_similarity

        backend = best_available_similarity()
        print("similarity backend: {} (semantic={})".format(
            type(backend).__name__, backend.is_semantic))
        return backend
    print("similarity backend: LexicalSimilarity (semantic=False) -- proxy only")
    return None


def main(workspace: Path) -> None:
    as_en = suite("assamese_english")
    hi_en = suite("hindi_english")

    sender = make_nllb_translator(
        pairs=["as->en", "hi->en"],
        dtype=os.environ.get("ASEA_SENDER_DTYPE", "bfloat16"),
    )
    receiver = build_receiver()

    plugins = default_registry()
    similarity = build_similarity()
    harness = BenchmarkHarness(plugins=plugins, similarity=similarity)

    # DEFAULT policy: strict_no_mock stays True. Both modules are real, so it
    # does not need to be relaxed -- which is the whole point of this script.
    pipeline = Pipeline(
        workspace=workspace,
        plugins=plugins,
        harness=harness,
        gap_engine=GapEngine(harness=harness),
        evaluator=Evaluator(harness=harness),
    )
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter(
        "nllb-to-qwen", sender.module_id, receiver.module_id,
        description="REAL Assamese glossary transfer, NLLB teacher -> Qwen student",
    )

    print("\nsender  : {} (mock={})".format(sender.display_name, sender.is_mock))
    print("receiver: {} (mock={})".format(receiver.display_name, receiver.is_mock))
    print("\nrunning -- real inference, this takes minutes on CPU ...\n")

    started = time.time()
    report = pipeline.run("nllb-to-qwen", suites=[as_en, hi_en])
    elapsed = time.time() - started

    show(report, "REAL MODEL FLOW -- {} -> {}, Assamese (NO MOCKS)".format(
        sender.module_id, receiver.module_id))
    print("\nwall clock  : {:.1f}s".format(elapsed))
    print("memory store: {}".format(pipeline.store.stats()))
    print("audit chain : {}".format(pipeline.audit.verify()))

    for packet in report.distilled:
        entries = (packet.distilled_skill or {}).get("entries", [])
        print("\nglossary learned from the real teacher ({} entries):".format(len(entries)))
        for entry in entries[:12]:
            print("   {!r} = {!r}  (conf {:.2f})".format(
                entry["source"], entry["target"], entry.get("confidence", 0.0)))

    for ev in report.evaluations:
        print("\nper-case held-out diff -- {}".format(ev["capability"]))
        for row in ev["case_diff"]:
            flag = "WORSE" if row["regressed"] else ("better" if row["delta"] > 0 else "same")
            print("   expected : {}".format(row["expected"]))
            print("   baseline : {}".format(str(row["baseline_output"])[:70]))
            print("   +packet  : {}".format(str(row["candidate_output"])[:70]))
            print("   {:+.4f} {}".format(row["delta"], flag))


if __name__ == "__main__":
    main(Path(os.environ.get("ASEA_WORKSPACE", ".work-real")))
