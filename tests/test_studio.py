"""SILT Studio: API structure, guards and catalog integrity.

Fast tests use FastAPI's TestClient and never load model weights. The full
end-to-end (real NLLB -> real Qwen through the HTTP API) was executed against
a live server and is recorded in docs/real_run_findings.md; re-run it locally
with ASEA_RUN_REAL=1 via test_studio_real_transfer.
"""

from __future__ import annotations

import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from asea.studio import catalog  # noqa: E402
from asea.studio.server import app, manager  # noqa: E402

client = TestClient(app)

REAL = pytest.mark.skipif(
    os.environ.get("ASEA_RUN_REAL") != "1",
    reason="loads real weights; set ASEA_RUN_REAL=1",
)


# -- health & static ----------------------------------------------------------


def test_health_declares_mock_free():
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert payload["mock_free"] is True


def test_ui_and_logo_are_served():
    assert client.get("/").status_code == 200
    logo = client.get("/logo.svg")
    assert logo.status_code == 200
    assert b"<svg" in logo.content


def test_readme_endpoint_serves_live_readme():
    """The landing README view is a single source of truth: /api/readme serves
    the actual README.md from disk, verbatim, so the on-page README can never
    drift from or contradict the repo file."""
    r = client.get("/api/readme")
    assert r.status_code == 200
    assert "text/markdown" in r.headers.get("content-type", "")
    body = r.text
    assert "SILT — Skill Interchange Layer with Trust-gating" in body  # real title
    assert "Architecture at a glance" in body                          # a section
    assert "```" in body                                               # a code fence
    assert "| Surface |" in body                                       # a table row


# -- catalog integrity ---------------------------------------------------------


def test_catalog_lists_no_mock_presets():
    """The Studio catalog must not contain any *_mock preset, by name or fact."""
    listing = client.get("/api/catalog").json()["modules"]
    assert listing, "catalog must not be empty"
    for module in listing:
        assert "mock" not in module["id"].lower()


def test_catalog_build_rejects_mocks_structurally():
    """Even if a mock were registered, build() must refuse to return it."""
    from asea.modules.mock.zoo import make_qwen

    catalog.CATALOG["smuggled"] = {
        "factory": make_qwen, "roles": ["receiver"],
        "description": "should never load", "requires": "",
    }
    try:
        with pytest.raises(RuntimeError, match="catalog integrity"):
            catalog.build("smuggled")
    finally:
        del catalog.CATALOG["smuggled"]


def test_catalog_unknown_module_raises():
    with pytest.raises(KeyError):
        catalog.build("does-not-exist")


def test_corpus_sender_is_real_and_cheap_to_build():
    """The corpus sender needs no ML deps -- verify it builds and is not a mock."""
    module = catalog.build("triage-corpus")
    assert module.is_mock is False
    assert module.manifest().roles == ["sender"]
    assert len(module) >= 4


# -- suites ---------------------------------------------------------------------


def test_suites_expose_splits_and_risk():
    suites = client.get("/api/suites").json()
    assert "medical_triage" in suites
    assert suites["medical_triage"]["high_risk"] is True
    assert suites["assamese_english"]["high_risk"] is False
    assert suites["assamese_english"]["splits"]["extraction"] > 0


# -- transfer validation guards ---------------------------------------------------


def test_transfer_rejects_self_transfer():
    response = client.post("/api/transfers", json={
        "sender": "qwen2.5-0.5b", "receiver": "qwen2.5-0.5b",
        "suites": ["assamese_english"],
    })
    assert response.status_code == 400
    assert "self-transfer" in response.json()["detail"]


def test_transfer_rejects_unknown_module_and_suite():
    assert client.post("/api/transfers", json={
        "sender": "nope", "receiver": "qwen2.5-0.5b",
        "suites": ["assamese_english"],
    }).status_code == 400
    assert client.post("/api/transfers", json={
        "sender": "nllb-teacher", "receiver": "qwen2.5-0.5b",
        "suites": ["nope"],
    }).status_code == 400


def test_transfer_rejects_empty_suites_and_bad_metric():
    assert client.post("/api/transfers", json={
        "sender": "nllb-teacher", "receiver": "qwen2.5-0.5b", "suites": [],
    }).status_code == 422
    assert client.post("/api/transfers", json={
        "sender": "nllb-teacher", "receiver": "qwen2.5-0.5b",
        "suites": ["assamese_english"], "similarity": "vibes",
    }).status_code == 422


def test_unknown_job_endpoints_404():
    assert client.get("/api/transfers/nope").status_code == 404
    assert client.get("/api/transfers/nope/packets").status_code == 404
    assert client.post("/api/transfers/nope/approve",
                       json={"packet_id": "x", "approver": "someone"}).status_code == 404
    assert client.get("/api/transfers/nope/export").status_code == 404
    # /api/skills/test resolves the job first (404 for an unknown job_id).
    assert client.post("/api/skills/test", json={
        "job_id": "nope", "module": "triage-corpus", "suite_id": "assamese_english",
    }).status_code == 404


# -- export endpoint (download the trained model) --------------------------------


def test_export_endpoint_returns_zip_of_approved_packet(
    tmp_path, capability, clean_provenance, packet_factory
):
    """The /export endpoint emits only already-approved packets as a zip.

    No real model weights are loaded: a job is injected into the manager with a
    pipeline whose store already holds one approved, non-mock skill packet.
    """
    import io
    import zipfile

    from asea.core.protocol import PacketType, PromotionStatus
    from asea.core.pipeline import Pipeline
    from asea.distill.export import export_artifact_bundle  # noqa: F401  (proves import path)
    from asea.studio.jobs import TransferJob
    from asea.studio.server import manager

    packet = packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "ভাত", "target": "rice"}]},
        rollback_token="snap",
        promotion_status=PromotionStatus.PROMOTED,
    )
    pipeline = Pipeline(workspace=tmp_path)
    pipeline.store.approve(packet)

    job = TransferJob({"sender": "x", "receiver": "y", "suites": ["assamese_english"]},
                      manager.workspace_root)
    job.pipeline = pipeline
    manager.jobs[job.job_id] = job
    try:
        response = client.get("/api/transfers/{}/export".format(job.job_id))
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        assert response.headers["content-disposition"].endswith(
            '{}_skill_bundle.zip"'.format(job.job_id)
        )

        zf = zipfile.ZipFile(io.BytesIO(response.content))
        names = zf.namelist()
        assert "manifest.json" in names
        assert "README.txt" in names
        assert "approved/{}.json".format(packet.packet_id) in names
        # No base_model query param => no job spec (L0-L3 honest default).
        assert not any(n.endswith(".job.json") for n in names)
        manifest = __import__("json").loads(zf.read("manifest.json"))
        assert manifest["packets"][0]["packet_id"] == packet.packet_id
        assert "SILT trains no weights" in manifest["note"]
    finally:
        del manager.jobs[job.job_id]


# -- skills library, runtime catalog, test-before-download ---------------------


def test_skills_endpoint_lists_approved_packets_across_jobs(
    tmp_path, monkeypatch, capability, clean_provenance, packet_factory
):
    """GET /api/skills is a cross-job library: it globs .studio/*/memory/approved
    and lists every approved packet, not just one job's. No job needs to be live."""
    from asea.memory.store import MemoryStore
    from asea.studio import server
    from asea.core.protocol import PacketType, PromotionStatus

    monkeypatch.setattr(server, "WORKSPACES", tmp_path)

    seen = []
    for job_id, entries in (
        ("jobaaaaaaaa", [{"source": "ভাত", "target": "rice"}]),
        ("jobbbbbbbbb", [{"source": "পানী", "target": "water"}]),  # distinct content
    ):
        store = MemoryStore(tmp_path / job_id / "memory")
        pkt = packet_factory(
            capability, clean_provenance,
            packet_type=PacketType.GLOSSARY,
            distilled_skill={"entries": entries},
            target_module="learner",
            rollback_token="snap",
            promotion_status=PromotionStatus.PROMOTED,
        )
        store.approve(pkt)
        seen.append((job_id, pkt.packet_id))

    payload = client.get("/api/skills").json()
    assert payload["count"] == 2
    job_ids = {s["job_id"] for s in payload["skills"]}
    assert job_ids == {"jobaaaaaaaa", "jobbbbbbbbb"}
    row = next(s for s in payload["skills"] if s["job_id"] == "jobaaaaaaaa")
    assert row["packet_id"] == seen[0][1]
    assert row["target_module"] == "learner"
    assert row["promotion_status"] == "promoted"
    assert row["is_mock"] is False


def test_skills_endpoint_skips_unreadable_approved_files(tmp_path, monkeypatch):
    """A corrupt approved/*.json must not crash the library endpoint."""
    from asea.studio import server

    monkeypatch.setattr(server, "WORKSPACES", tmp_path)
    approved = tmp_path / "jobcccccccc" / "memory" / "approved"
    approved.mkdir(parents=True)
    (approved / "broken.json").write_text("{not valid json", encoding="utf-8")

    response = client.get("/api/skills")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 0
    assert payload["skills"] == []


def test_catalog_post_adds_entry_and_preflight_refuses_missing_model():
    """POST /api/catalog registers a runtime Ollama entry but the preflight must
    refuse a model that is not reachable/pulled (400, never a silent 200)."""
    from asea.studio import catalog

    new_id = "test-runtime-model"
    assert new_id not in catalog.CATALOG
    try:
        response = client.post("/api/catalog", json={
            "module_id": new_id,
            "ollama_tag": "definitely-not-pulled:latest",
            "role": "receiver",
            "suite_id": "assamese_english",
        })
        # Honest guard: a model that is absent (not pulled, or ollama not
        # running) is refused with 400 — never accepted as a usable module.
        assert response.status_code == 400
        # The entry was registered (preflight runs after registration so the
        # user can retry after `ollama pull` without re-POSTing).
        assert new_id in catalog.CATALOG
        assert catalog.CATALOG[new_id]["roles"] == ["receiver"]
    finally:
        catalog.CATALOG.pop(new_id, None)
        catalog._cache.pop(new_id, None)


def test_catalog_post_rejects_unknown_suite_and_overwrite_of_builtin():
    # unknown suite -> 400 before any entry is created
    r = client.post("/api/catalog", json={
        "module_id": "whatever-x", "ollama_tag": "x:latest",
        "role": "receiver", "suite_id": "does-not-exist",
    })
    assert r.status_code == 400
    assert "unknown suite" in r.json()["detail"]
    # overwriting a built-in module id -> 400 (protected)
    r2 = client.post("/api/catalog", json={
        "module_id": "smollm2-360m", "ollama_tag": "x:latest",
        "role": "receiver", "suite_id": "assamese_english",
    })
    assert r2.status_code == 400
    assert "built-in" in r2.json()["detail"]


def test_catalog_post_rejects_html_in_description():
    """The catalog ``description`` is reflected back through ``/api/catalog``
    and interpolated into ``<option>`` text at several Studio sinks, so a
    crafted description with the HTML breakout chars ``<`` / ``>`` is rejected
    at the source with 422 -- the same bug class as the compress-table
    ``state`` XSS, one surface over (adversarial review 2026-08-18). Defense in
    depth: the UI also escapes the description at the sink; this guards a
    future sink that forgets to. Legitimate descriptions keep ``&`` / quotes
    (harmless in element text once escaped) -- only the breakout chars are
    refused, so a clean description reaches preflight and 400s on the unpulled
    tag, NOT a 422."""
    from asea.studio import catalog

    # HTML breakout -> 422 (pydantic validation runs before the handler, so the
    # entry is never registered).
    r = client.post("/api/catalog", json={
        "module_id": "xss-desc-model",
        "ollama_tag": "x:latest",
        "role": "receiver",
        "suite_id": "assamese_english",
        "description": "</option></select><img src=x onerror=alert(1)>",
    })
    assert r.status_code == 422
    assert "description" in r.text
    assert "xss-desc-model" not in catalog.CATALOG

    # A clean description with '&' and quotes (no breakout chars) passes
    # validation -> reaches preflight -> 400 on the unpulled tag, NOT 422.
    r2 = client.post("/api/catalog", json={
        "module_id": "clean-desc-model",
        "ollama_tag": "definitely-not-pulled:latest",
        "role": "receiver",
        "suite_id": "assamese_english",
        "description": 'Qwen & "GPT" style model',
    })
    assert r2.status_code == 400
    catalog.CATALOG.pop("clean-desc-model", None)
    catalog._cache.pop("clean-desc-model", None)


def test_skill_test_endpoint_runs_ab_and_is_read_only(
    tmp_path, capability, clean_provenance, packet_factory
):
    """POST /api/skills/test runs a read-only baseline-vs-candidate A/B and must
    NOT mutate the approved packet (it calls the harness directly, never
    Evaluator.evaluate and never the gate). Uses the cheap file-backed
    triage-corpus module so no model weights load."""
    from asea.core.pipeline import Pipeline
    from asea.core.protocol import PacketType, PromotionStatus
    from asea.studio.jobs import TransferJob
    from asea.studio.server import manager

    pkt = packet_factory(
        capability, clean_provenance,
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "chest pain", "target": "red flag"}]},
        target_module="triage-corpus",
        rollback_token="snap",
        promotion_status=PromotionStatus.PROMOTED,
    )
    pipeline = Pipeline(workspace=tmp_path)
    pipeline.store.approve(pkt)
    job = TransferJob(
        {"sender": "x", "receiver": "y", "suites": ["medical_triage"]},
        manager.workspace_root,
    )
    job.pipeline = pipeline
    manager.jobs[job.job_id] = job
    try:
        response = client.post("/api/skills/test", json={
            "job_id": job.job_id,
            "module": "triage-corpus",
            "suite_id": "medical_triage",
            "similarity": "lexical",  # no embedding model loaded -> stays fast
        })
        assert response.status_code == 200, response.text
        d = response.json()
        assert d["module"] == "triage-corpus"
        assert d["is_mock"] is False
        assert d["skills_active"] >= 1
        assert "score" in d["baseline"] and "score" in d["candidate"]
        assert isinstance(d["improvement"], (int, float))
        assert isinstance(d["cases"], list)

        # READ-ONLY guarantee: the approved packet is still "promoted",
        # not mutated to "evaluated" by Evaluator.evaluate.
        pkts = client.get(
            "/api/transfers/{}/packets".format(job.job_id)
        ).json()
        approved = [p for p in pkts["approved"] if p["packet_id"] == pkt.packet_id]
        assert approved, "approved packet must still be present"
        assert approved[0]["promotion_status"] == "promoted"
    finally:
        del manager.jobs[job.job_id]


def test_approval_requires_a_name():
    response = client.post("/api/transfers/nope/approve",
                           json={"packet_id": "x", "approver": ""})
    assert response.status_code == 422  # min_length guard before the 404


# -- real end-to-end (opt-in) -----------------------------------------------------


@REAL
def test_studio_real_transfer_corpus_to_smollm():
    """Full HTTP lifecycle with a real corpus sender and a real HF receiver.

    Medical domain: the packet must park at PENDING_HUMAN, then promote only
    after named approval through the API.
    """
    import time

    response = client.post("/api/transfers", json={
        "sender": "triage-corpus", "receiver": "smollm2-360m",
        "suites": ["medical_triage"], "similarity": "embedding",
    })
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    for _ in range(600):
        state = client.get("/api/transfers/{}".format(job_id)).json()
        if state["status"] in ("done", "failed"):
            break
        time.sleep(2)
    assert state["status"] == "done", state.get("error")

    report = state["report"]
    assert report["promoted"] == [], "medical must never auto-promote"

    if report["pending_human"]:
        packet_id = report["pending_human"][0]
        decision = client.post(
            "/api/transfers/{}/approve".format(job_id),
            json={"packet_id": packet_id, "approver": "reviewer@example.org"},
        ).json()
        assert decision["status"] in ("promoted", "rejected")

    audit = client.get("/api/transfers/{}/audit".format(job_id)).json()
    assert audit["integrity"]["ok"] is True


def test_favicon_is_served():
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert b"<svg" in response.content


# -- adversarial audit 2026-08-13: missing endpoint coverage + hardening pins --


def _knowledge(suites, splits=("extraction",)):
    """Local copy of the helper in test_end_to_end (kept here so this file stays
    self-contained for the offline medical-approve HTTP test below)."""
    table = {}
    for s in suites:
        bucket = table.setdefault(s.capability().as_str(), {})
        for case in s.cases:
            if case.split in splits:
                bucket[str(case.prompt).strip()] = case.expected
    return table


def _inject_job(pipeline, suites=("medical_triage",)):
    """Register a synthetic TransferJob wrapping an already-built pipeline."""
    from asea.studio.jobs import TransferJob
    from asea.studio.server import manager

    job = TransferJob({"sender": "x", "receiver": "y", "suites": list(suites)},
                      manager.workspace_root)
    job.pipeline = pipeline
    manager.jobs[job.job_id] = job
    return job


def test_http_approve_promotes_a_pending_human_packet(tmp_path, medical_suite):
    """POST /api/transfers/{id}/approve re-runs the full gate over HTTP and
    promotes a PENDING_HUMAN medical packet (audit #23: the real gate-rerun was
    only exercised by the ASEA_RUN_REAL-skipped test; the default-run tests
    covered only the 422 min_length and 404 unknown-job paths)."""
    from asea.core.pipeline import Pipeline
    from asea.core.protocol import Domain
    from asea.modules.mock.zoo import make_generic_receiver, make_generic_sender, rule_cap
    from asea.promotion.gate import PromotionGate, PromotionPolicy

    cap = rule_cap(Domain.MEDICAL, "triage")
    sender = make_generic_sender(module_id="triage-src", capabilities=[cap],
                                 knowledge=_knowledge([medical_suite]))
    receiver = make_generic_receiver(module_id="med-rcv", capabilities=[cap],
                                     fallback="english")
    pipeline = Pipeline(workspace=tmp_path / "med",
                        gate=PromotionGate(PromotionPolicy(strict_no_mock=False)))
    pipeline.register_module(sender)
    pipeline.register_module(receiver)
    pipeline.bind_adapter("med", sender.module_id, receiver.module_id)
    report = pipeline.run("med", suites=[medical_suite])
    assert report.pending_human, "medical packet must park for human approval"
    packet_id = report.pending_human[0]

    job = _inject_job(pipeline)
    try:
        r = client.post("/api/transfers/{}/approve".format(job.job_id),
                        json={"packet_id": packet_id, "approver": "dr.r@example.org"})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "promoted"
        pkts = client.get("/api/transfers/{}/packets".format(job.job_id)).json()
        assert any(p["packet_id"] == packet_id for p in pkts["approved"])

        # An unknown packet_id surfaces as 400 (the handler's except->400 path),
        # not a 500 and not a silent 200.
        r2 = client.post("/api/transfers/{}/approve".format(job.job_id),
                         json={"packet_id": "no-such-packet",
                               "approver": "dr.r@example.org"})
        assert r2.status_code == 400
        # Info-leakage pin (#20): the 400 body must NOT echo the raw exception
        # (which would carry store paths / packet ids); it carries only a ref.
        assert "ref" in r2.json()["detail"]
    finally:
        del manager.jobs[job.job_id]


def test_http_rollback_endpoint_round_trip(tmp_path, capability, clean_provenance,
                                           packet_factory):
    """POST /api/transfers/{id}/rollback over HTTP (audit #22: the endpoint had
    zero test coverage). A valid pre-approval snapshot token reverts the
    approved set; an unknown token returns 400."""
    from asea.core.pipeline import Pipeline
    from asea.core.protocol import PacketType, PromotionStatus

    pkt = packet_factory(capability, clean_provenance,
                         packet_type=PacketType.GLOSSARY,
                         distilled_skill={"entries": [{"source": "ভাত", "target": "rice"}]},
                         rollback_token="snap",
                         promotion_status=PromotionStatus.PROMOTED)
    pipeline = Pipeline(workspace=tmp_path)
    token_pre = pipeline.rollback.snapshot(label="before-approve")  # approved/ empty
    pipeline.store.approve(pkt)
    assert pipeline.store.stats()["approved"] == 1

    job = _inject_job(pipeline, suites=("assamese_english",))
    try:
        r = client.post("/api/transfers/{}/rollback".format(job.job_id),
                        json={"token": token_pre})
        assert r.status_code == 200, r.text
        assert r.json()["removed"] >= 1
        assert (client.get("/api/transfers/{}/packets".format(job.job_id))
                .json()["approved"] == [])

        r2 = client.post("/api/transfers/{}/rollback".format(job.job_id),
                         json={"token": "no-such-snapshot"})
        assert r2.status_code == 400
        assert "ref" in r2.json()["detail"]  # info-leakage pin (#20)
    finally:
        del manager.jobs[job.job_id]


def test_export_query_params_base_model_and_include_mock(
    tmp_path, monkeypatch, capability, clean_provenance, packet_factory
):
    """?base_model adds an L4/L5 NOT_EXECUTED job spec to the zip; ?include_mock
    lets a mock-derived approved packet through (audit #41: these query params
    were never exercised over HTTP, only at the unit level)."""
    import io
    import json
    import zipfile

    from asea.core.protocol import OriginKind, PacketType, PromotionStatus, Provenance
    from asea.core.pipeline import Pipeline

    pkt = packet_factory(capability, clean_provenance,
                         packet_type=PacketType.GLOSSARY,
                         distilled_skill={"entries": [{"source": "ভাত", "target": "rice"}]},
                         rollback_token="snap",
                         promotion_status=PromotionStatus.PROMOTED)
    mock_pkt = packet_factory(
        capability,
        Provenance(origin_kind=OriginKind.MODEL_GENERATED, chain=["qwen-mock"],
                   is_mock=True, synthetic_depth=0, source_reference="unit-test"),
        packet_type=PacketType.GLOSSARY,
        distilled_skill={"entries": [{"source": "a", "target": "b"}]},
        rollback_token="snap2",
        promotion_status=PromotionStatus.PROMOTED,
    )
    pipeline = Pipeline(workspace=tmp_path)
    pipeline.store.approve(pkt)
    pipeline.store.approve(mock_pkt)

    job = _inject_job(pipeline, suites=("assamese_english",))
    try:
        # base_model -> a .job.json appears in the zip.
        r = client.get("/api/transfers/{}/export".format(job.job_id),
                       params={"base_model": "Qwen/Qwen2.5-7B-Instruct"})
        assert r.status_code == 200, r.text
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        job_jsons = [n for n in names if n.endswith(".job.json")]
        assert job_jsons, names
        spec = json.loads(zf.read(job_jsons[0]))
        assert spec["status"] == "NOT_EXECUTED"
        assert spec["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert spec["eval_gate"]["source"] == "derived_from_gate_policy"

        # include_mock=true -> the mock-derived packet is bundled.
        r2 = client.get("/api/transfers/{}/export".format(job.job_id),
                        params={"include_mock": "true"})
        assert r2.status_code == 200
        zf2 = zipfile.ZipFile(io.BytesIO(r2.content))
        assert "approved/{}.json".format(mock_pkt.packet_id) in zf2.namelist()
        manifest = json.loads(zf2.read("manifest.json"))
        assert manifest["contains_mock_data"] is True

        # Default (no include_mock) -> the mock packet is skipped, not bundled.
        r3 = client.get("/api/transfers/{}/export".format(job.job_id))
        zf3 = zipfile.ZipFile(io.BytesIO(r3.content))
        assert "approved/{}.json".format(mock_pkt.packet_id) not in zf3.namelist()
    finally:
        del manager.jobs[job.job_id]


def test_audit_events_and_playground_endpoints_respond(
    tmp_path, capability, clean_provenance, packet_factory
):
    """GET /audit, GET /events, and POST /playground have non-skipped coverage
    (audit #41: they appeared only in the REAL-skipped test or not at all)."""
    from asea.core.pipeline import Pipeline
    from asea.core.protocol import PacketType, PromotionStatus

    pkt = packet_factory(capability, clean_provenance,
                         packet_type=PacketType.GLOSSARY,
                         distilled_skill={"entries": [{"source": "ভাত", "target": "rice"}]},
                         rollback_token="snap",
                         promotion_status=PromotionStatus.PROMOTED,
                         target_module="triage-corpus")
    pipeline = Pipeline(workspace=tmp_path)
    pipeline.store.approve(pkt)

    job = _inject_job(pipeline, suites=("medical_triage",))
    # The injected job never ran start(), so mark it done so the SSE stream()
    # generator terminates instead of polling forever on status "queued".
    job.status = "done"
    try:
        audit = client.get("/api/transfers/{}/audit".format(job.job_id)).json()
        assert audit["integrity"]["ok"] is True

        events = client.get("/api/transfers/{}/events".format(job.job_id))
        assert events.status_code == 200
        assert "text/event-stream" in events.headers.get("content-type", "")

        # Playground is read-only w.r.t. learning; use the cheap file-backed
        # triage-corpus sender with skills off so no second module is built.
        pg = client.post("/api/playground", json={
            "job_id": job.job_id, "module": "triage-corpus",
            "prompt": "anything",
            "capability": {"task_type": "triage", "modality": "structured",
                           "domain": "medical", "language": "en"},
            "use_skills": False,
        })
        assert pg.status_code == 200, pg.text
        assert "output" in pg.json()
        assert pg.json()["is_mock"] is False
    finally:
        del manager.jobs[job.job_id]


def test_request_payload_bounds_reject_oversized_and_duplicate_suites():
    """DoS / cost-blowup bounds (audit #19): an oversized suites list, a
    duplicate-stem list, and an oversized prompt are all rejected before any
    real model is built."""
    # 9 suites exceeds max_length=8 -> 422 at the schema layer (no handler run).
    r = client.post("/api/transfers", json={
        "sender": "nllb-teacher", "receiver": "qwen2.5-0.5b",
        "suites": ["assamese_english"] * 9,
    })
    assert r.status_code == 422

    # Duplicate stems within the bound -> 400 from the handler's dedup guard
    # (fires before the preflight builds any module).
    r2 = client.post("/api/transfers", json={
        "sender": "nllb-teacher", "receiver": "qwen2.5-0.5b",
        "suites": ["assamese_english", "assamese_english"],
    })
    assert r2.status_code == 400
    assert "duplicate" in r2.json()["detail"].lower()

    # Oversized prompt -> 422 at the schema layer.
    r3 = client.post("/api/playground", json={
        "job_id": "nope", "module": "triage-corpus",
        "prompt": "x" * 8001,
        "capability": {"task_type": "triage", "modality": "structured"},
        "use_skills": False,
    })
    assert r3.status_code == 422


def test_catalog_post_rejects_unsafe_ollama_tag():
    """The ollama_tag is reflected into the catalog and the Ollama model=
    argument, so a tag with whitespace / shell-meta / path chars is refused at
    the schema layer (audit #40)."""
    from asea.studio import catalog

    r = client.post("/api/catalog", json={
        "module_id": "test-bad-tag", "ollama_tag": "evil tag; rm -rf /",
        "role": "receiver", "suite_id": "assamese_english",
    })
    assert r.status_code == 422
    assert "test-bad-tag" not in catalog.CATALOG


def test_write_endpoints_are_currently_open_visible_decision():
    """PINNING (audit #21): the Studio's mutating endpoints have no auth today.
    This is an ACCEPTED RISK for a localhost dev tool, held for explicit
    sign-off on an auth scheme (see the GATE/CORE findings) -- not an oversight.
    The test pins the current open state: requests reach the schema/handler
    layers (422/404), they are NOT intercepted by a 401/403 auth gate. Adding
    auth later must update this test, making the change visible."""
    # /api/catalog reaches the schema validator (422), not an auth gate.
    assert client.post("/api/catalog", json={
        "module_id": "x", "ollama_tag": "bad tag!", "role": "receiver",
        "suite_id": "assamese_english",
    }).status_code == 422
    # /approve on an unknown job reaches the 404 path, not 401/403.
    assert client.post("/api/transfers/nope/approve",
                       json={"packet_id": "x", "approver": "someone"}).status_code == 404


# -- deep-apply (weights-mode training) endpoints ------------------------------


def test_deep_apply_unknown_source_job_404():
    """POST /api/deepapply on an unknown source job is 404 (the source pipeline
    must exist -- deep-apply trains from THAT job's approved packets)."""
    r = client.post("/api/deepapply", json={
        "job_id": "nope", "suite_id": "assamese_english", "backend": "standard",
    })
    assert r.status_code == 404


def test_deep_apply_unknown_id_404():
    assert client.get("/api/deepapply/nope").status_code == 404


def test_deep_apply_refuses_ollama_receiver_and_drains_telemetry(tmp_path):
    """Deep-apply needs an in-process HF receiver; an Ollama (external-weights)
    receiver is refused with a typed DeepApplyBlocked BEFORE any training or
    model load -- so the run is hermetic and deterministic on both no-torch and
    torch envs. The telemetry SSE stream drains the REAL phase events the
    runner/endpoint emit (job_started, job_failed) and closes with a status
    frame. No fabricated numbers; the failure is the system working (refusing a
    receiver it cannot LoRA-train)."""
    from asea.core.pipeline import Pipeline
    from asea.studio.jobs import TransferJob
    from asea.studio.server import manager, deepapply_manager
    from asea.studio.deepapply_jobs import DeepApplyJob

    # Source transfer job with an empty store (preflight fails before any
    # packet check, so empty is fine -- the receiver rejection comes first).
    pipeline = Pipeline(workspace=tmp_path)
    job = TransferJob({"sender": "x", "receiver": "y", "suites": ["assamese_english"]},
                      manager.workspace_root)
    job.pipeline = pipeline
    manager.jobs[job.job_id] = job
    try:
        r = client.post("/api/deepapply", json={
            "job_id": job.job_id,
            "receiver_id": "qwen2.5-7b-ollama",   # external-weights -> refused
            "suite_id": "assamese_english",
            "backend": "standard",
        })
        assert r.status_code == 200
        da_id = r.json()["deepapply_id"]

        # Poll to completion (preflight rejection is fast; no model loads). The
        # worker thread imports torch on first call (~seconds), so poll with a
        # sleep rather than a tight busy loop.
        import time as _time
        got = None
        for _ in range(200):
            got = client.get("/api/deepapply/{}".format(da_id)).json()
            if got["status"] in ("done", "failed"):
                break
            _time.sleep(0.1)
        assert got["status"] == "failed"
        assert got["error"] == "DeepApplyBlocked"

        # The listing includes the failed run.
        listing = client.get("/api/deepapply").json()["jobs"]
        assert any(j["job_id"] == da_id for j in listing)
    finally:
        manager.jobs.pop(job.job_id, None)
        deepapply_manager._jobs.pop(da_id, None)

    # Telemetry drain: construct a job directly and run synchronously so the
    # SSE generator can be consumed without a streaming TestClient.
    src = Pipeline(workspace=tmp_path / "src")
    da = DeepApplyJob(
        {"receiver_id": "qwen2.5-7b-ollama", "suite_id": "assamese_english",
         "backend": "standard"},
        tmp_path / "da", src,
    )
    da._run()  # synchronous; preflight rejects before any training
    assert da.status == "failed"
    assert da.error_type == "DeepApplyBlocked"

    frames = list(da.telemetry())
    # Telemetry frames are "event: telemetry\ndata: {...}\n\n"; last is status.
    telem = [f for f in frames if f.startswith("event: telemetry")]
    assert any('"phase": "job_started"' in f for f in telem)
    assert any('"phase": "job_failed"' in f for f in telem)
    assert frames[-1].startswith("event: status")
    assert '"status": "failed"' in frames[-1]


# -- SiltSpring compression certification endpoints ---------------------------


def test_spring_unknown_id_404():
    assert client.get("/api/spring/nope").status_code == 404


def test_spring_refuses_ollama_and_drains_telemetry(tmp_path):
    """SiltSpring certifies a model whose weights the process loads, so an
    Ollama (external-weights) module is refused with a typed DeepApplyBlocked
    BEFORE any model load -- hermetic and deterministic on both no-torch and
    torch envs. The telemetry SSE stream drains the REAL phase events
    (job_started, job_failed) and closes with a status frame."""
    from asea.studio.server import spring_manager
    from asea.studio.spring_jobs import SpringJob

    r = client.post("/api/spring", json={
        "module_id": "qwen2.5-7b-ollama",   # external-weights -> refused
        "suite_ids": ["assamese_english"],
    })
    assert r.status_code == 200
    sp_id = r.json()["spring_id"]

    import time as _time
    got = None
    for _ in range(200):
        got = client.get("/api/spring/{}".format(sp_id)).json()
        if got["status"] in ("done", "failed"):
            break
        _time.sleep(0.05)
    assert got["status"] == "failed"
    assert got["error"] == "DeepApplyBlocked"
    spring_manager._jobs.pop(sp_id, None)

    # Telemetry drain (synchronous, no streaming client).
    sp = SpringJob({"module_id": "qwen2.5-7b-ollama",
                    "suite_ids": ["assamese_english"]}, tmp_path / "sp")
    sp._run()
    assert sp.status == "failed"
    assert sp.error_type == "DeepApplyBlocked"
    frames = list(sp.telemetry())
    telem = [f for f in frames if f.startswith("event: telemetry")]
    assert any('"phase": "job_started"' in f for f in telem)
    assert any('"phase": "job_failed"' in f for f in telem)
    assert frames[-1].startswith("event: status")
    assert '"status": "failed"' in frames[-1]


def test_deep_apply_telemetry_unknown_id_404():
    """The SSE telemetry sub-path 404s on an unknown id (contract pinned; the
    non-telemetry GET was already pinned)."""
    assert client.get("/api/deepapply/nope/telemetry").status_code == 404


def test_spring_telemetry_unknown_id_404():
    assert client.get("/api/spring/nope/telemetry").status_code == 404


def test_jsonsafe_sanitizes_nonfinite_floats():
    """A non-finite float (nan/inf) in a served artifact would crash JSON
    serialization (ValueError: Out of range float values are not JSON compliant)
    which 500s the GET endpoint AND the listing. json_safe is the single
    chokepoint that turns nan/inf -> None (UI renders "—", honest) and rounds
    finite floats to 6 dp. (adversarial review 2026-08-19: an int2 state of a
    tiny model on an out-of-distribution suite collapses to NaN loss.)"""
    import math
    from asea.studio._jsonsafe import json_safe

    nan = float("nan")
    inf = float("inf")
    out = json_safe({"loss": {"as->en": nan, "en": 1.2345678},
                     "vram": inf, "n": 5, "ok": True, "name": "x", "lst": [nan, 2.0]})
    # non-finite -> None; finite -> rounded 6 dp; ints/bools/strs untouched
    assert out["loss"]["as->en"] is None
    assert out["loss"]["en"] == 1.234568
    assert out["vram"] is None
    assert out["n"] == 5 and out["ok"] is True and out["name"] == "x"
    assert out["lst"] == [None, 2.0]
    # the whole thing must round-trip through the stdlib JSON encoder FastAPI uses
    import json
    json.dumps(out)


def test_spring_report_with_nan_loss_serializes_and_is_not_falsely_certified(tmp_path):
    """The SiltSpring per-state report can carry a NaN loss (a degenerate int2
    state). Two honesty rules pinned here:

      1. The GET endpoint + listing must NOT 500 (json_safe backstop in to_dict).
      2. A state whose skill degradation is non-finite is UNMEASURED -- neither
         certified nor revoked -- so overall_certified must be False (a NaN
         compares False to <= and >, so the old `not revoked` logic falsely
         certified it; that loophole is closed).
    """
    import json
    import math
    from asea.studio.spring_jobs import SpringJob

    job = SpringJob({"module_id": "smollm2-360m", "suite_ids": ["assamese_english"],
                     "levels": ["int8", "int4", "int2"]}, tmp_path / "sp")
    # Hand-build the exact report shape _execute returns, with a NaN loss/degradation
    # for the int2 state (mirrors the real collapsed-int2 outcome).
    nan = float("nan")
    job.report = {
        "model_id": "smollm2-360m", "device": "cuda", "vram_peak_gb": 1.675,
        "tolerance": 0.05, "levels": ["int8", "int4", "int2"], "decoder_layers": 32,
        "skills": ["assamese_english_v1"],
        "reference_loss": {"assamese_english_v1": nan},   # reference also collapsed
        "states": [
            {"state": "full", "bytes_packed": None,
             "loss": {"assamese_english_v1": nan},
             "degradation": {}, "certified_skills": [], "revoked_skills": [],
             "unmeasured_skills": ["assamese_english_v1"],
             "overall_certified": True},  # full state: reference, certified by convention
            {"state": "int2", "bytes_packed": 79994880,
             "loss": {"assamese_english_v1": 19.773333},
             "degradation": {"assamese_english_v1": nan},  # non-finite -> unmeasured
             "certified_skills": [], "revoked_skills": [],
             "unmeasured_skills": ["assamese_english_v1"],
             "overall_certified": False},  # the closed loophole: not certified
        ],
    }
    # (1) serialization must not raise -- this is the line that 500'd before the fix
    served = job.to_dict()
    roundtripped = json.dumps(served)  # would raise ValueError pre-fix
    loaded = json.loads(roundtripped)
    int2 = next(s for s in loaded["report"]["states"] if s["state"] == "int2")
    assert int2["loss"]["assamese_english_v1"] == 19.773333   # finite survives
    assert int2["degradation"]["assamese_english_v1"] is None  # nan -> None
    assert int2["unmeasured_skills"] == ["assamese_english_v1"]
    assert int2["overall_certified"] is False                  # the closed loophole
    assert loaded["report"]["reference_loss"]["assamese_english_v1"] is None


# -- adversarial review 2026-08-19: transfer / skills NaN-backstop regression --
# The spring + deepapply paths were sanitized first; the transfer report,
# audit, SSE, and /api/skills/test paths were NOT -- a non-finite score from a
# metric plugin / embedding similarity would 500 the endpoint (FastAPI's
# JSONResponse serializes with allow_nan=False) or emit a literal NaN token
# the browser JSON.parse rejects (silent SSE breakage). These pin every fix.


def test_jsonsafe_coerces_numpy_scalars():
    """Numpy scalars are NOT subclasses of the matching Python scalars
    (np.int64 is not int), so without explicit coercion json_safe would let
    them fall through -> a numpy int 500s FastAPI's encoder (TypeError) and a
    numpy float's nan would stringify via default=str. json_safe must coerce
    np.integer->int, np.floating->finite-or-None, np.bool_->bool."""
    np = pytest.importorskip("numpy")
    import json
    from asea.studio._jsonsafe import json_safe

    out = json_safe({
        "i": np.int64(5), "f": np.float64(1.2345678), "nan": np.float64("nan"),
        "inf": np.float64("inf"), "b": np.bool_(True),
        "lst": [np.int32(2), np.float32(0.5)],
    })
    assert out["i"] == 5 and isinstance(out["i"], int)
    assert out["f"] == 1.234568 and isinstance(out["f"], float)
    assert out["nan"] is None and out["inf"] is None
    assert out["b"] is True
    assert out["lst"] == [2, 0.5]
    json.dumps(out)  # must round-trip through the stdlib encoder FastAPI uses


def test_classify_state_closes_empty_and_nonnumeric_loopholes():
    """_classify_state enforces three honesty rules that the old inline builder
    violated (adversarial review 2026-08-19, loophole #6):

      * empty degradation -> the state measured nothing -> NOT certified.
      * non-numeric (None/str) degradation -> UNMEASURED (the complement), not
        silently dropped into a gap that falsely certified the state.
      * non-finite (nan) degradation -> UNMEASURED, NOT certified (loophole #2).
    And the reachable cases stay correct: finite within-tol -> certified;
    finite worse-than-tol -> revoked; the full reference state -> certified.
    """
    from asea.studio.spring_jobs import _classify_state

    nan = float("nan")
    tol = 0.05
    # finite within tolerance -> certified
    s = _classify_state("int8", {"degradation": {"sk": 0.01, "sk2": 0.04}}, tol)
    assert s["certified_skills"] == ["sk", "sk2"]
    assert s["revoked_skills"] == [] and s["unmeasured_skills"] == []
    assert s["overall_certified"] is True
    # finite worse than tolerance -> revoked -> not certified
    s = _classify_state("int4", {"degradation": {"sk": 0.5}}, tol)
    assert s["revoked_skills"] == ["sk"] and s["overall_certified"] is False
    # mixed: one revoked poisons the whole state (cannot claim certified)
    s = _classify_state("int4", {"degradation": {"sk": 0.01, "bad": 0.5}}, tol)
    assert s["overall_certified"] is False
    # NaN degradation -> unmeasured -> NOT certified (the closed loophole #2)
    s = _classify_state("int2", {"degradation": {"sk": nan}}, tol)
    assert s["certified_skills"] == [] and s["revoked_skills"] == []
    assert s["unmeasured_skills"] == ["sk"]
    assert s["overall_certified"] is False
    # empty degradation -> measured nothing -> NOT certified (loophole #6)
    s = _classify_state("int8", {"degradation": {}}, tol)
    assert s["certified_skills"] == [] and s["unmeasured_skills"] == []
    assert s["overall_certified"] is False
    # non-numeric degradation -> unmeasured (complement), NOT certified (#6)
    s = _classify_state("int8", {"degradation": {"sk": None, "sk2": "n/a"}}, tol)
    assert s["certified_skills"] == [] and s["revoked_skills"] == []
    assert sorted(s["unmeasured_skills"]) == ["sk", "sk2"]
    assert s["overall_certified"] is False
    # the full reference state is certified by convention even with empty deg
    s = _classify_state("full", {"degradation": {}}, tol)
    assert s["overall_certified"] is True


def test_transfer_get_report_nan_score_does_not_500():
    """GET /api/transfers/{id} serves job.report raw; a NaN score
    (negotiation.receiver_score / evaluations[].scores) would 500 via
    FastAPI's allow_nan=False encoder. json_safe is the backstop."""
    import json
    from asea.studio.jobs import TransferJob
    from asea.studio.server import manager

    job = TransferJob({"sender": "x", "receiver": "y", "suites": ["s"]},
                      manager.workspace_root)
    nan = float("nan")
    job.status = "done"
    job.report = {
        "negotiation": {"receiver_score": nan, "sender_score": nan,
                         "headroom": nan},
        "evaluations": [{"scores": {"s": nan}, "improvement": nan,
                         "baseline": {"score": nan}, "candidate": {"score": nan}}],
    }
    manager.jobs[job.job_id] = job
    try:
        r = client.get("/api/transfers/{}".format(job.job_id))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["report"]["negotiation"]["receiver_score"] is None
        assert body["report"]["evaluations"][0]["scores"]["s"] is None
        json.dumps(body)  # must round-trip cleanly
    finally:
        del manager.jobs[job.job_id]


def test_transfer_audit_nan_entry_does_not_500(tmp_path):
    """The audit log is written with json.dumps(default=str), which does NOT
    rescue NaN (float is a recognized type), so a NaN-bearing detail
    round-trips through disk as the literal NaN token and re-parses to a nan
    float. GET /api/transfers/{id}/audit would 500. json_safe is the backstop."""
    import json
    from asea.core.pipeline import Pipeline
    from asea.studio.jobs import TransferJob
    from asea.studio.server import manager

    pipeline = Pipeline(workspace=tmp_path)
    nan = float("nan")
    pipeline.audit.append("evaluated", actor="studio",
                          detail={"score": nan, "improvement": nan})
    job = TransferJob({"sender": "x", "receiver": "y", "suites": ["s"]},
                      manager.workspace_root)
    job.pipeline = pipeline
    job.status = "done"
    manager.jobs[job.job_id] = job
    try:
        # the raw on-disk entry genuinely carries a nan float (the loophole)
        raw = pipeline.audit.entries()[-1]
        assert isinstance(raw["detail"]["score"], float)
        assert raw["detail"]["score"] != raw["detail"]["score"]  # is nan
        r = client.get("/api/transfers/{}/audit".format(job.job_id))
        assert r.status_code == 200, r.text
        body = r.json()
        served = body["entries"][-1]["detail"]
        assert served["score"] is None and served["improvement"] is None
        json.dumps(body)
    finally:
        del manager.jobs[job.job_id]


def test_transfer_sse_sanitizes_nan_token(tmp_path):
    """The transfer SSE stream serializes audit entries with json.dumps
    (allow_nan=True by default), emitting a literal ``NaN`` token the browser
    JSON.parse rejects (silent EventSource breakage). Each entry must pass
    json_safe first so the frame carries null, not NaN."""
    from asea.core.pipeline import Pipeline
    from asea.studio.jobs import TransferJob
    from asea.studio.server import manager

    job = TransferJob({"sender": "x", "receiver": "y", "suites": ["s"]},
                      tmp_path)  # tmp_path as workspace root -> no .studio litter
    # The SSE stream tails job.audit_path() (the job's OWN workspace), so the
    # pipeline's audit log must live at job.workspace for the stream to see it.
    pipeline = Pipeline(workspace=job.workspace)
    nan = float("nan")
    pipeline.audit.append("evaluated", actor="studio", detail={"score": nan})
    job.pipeline = pipeline
    job.status = "done"  # lets the generator terminate after the flush branch
    manager.jobs[job.job_id] = job
    try:
        r = client.get("/api/transfers/{}/events".format(job.job_id))
        assert r.status_code == 200
        text = r.text
        # no literal NaN token may reach the browser
        assert "NaN" not in text
        # the sanitized score must serialize as null
        assert '"score": null' in text
    finally:
        del manager.jobs[job.job_id]


def test_skills_test_nan_score_does_not_500(tmp_path, monkeypatch):
    """POST /api/skills/test returns raw round(score, 4) values; a NaN
    SuiteResult.score (a metric plugin / embedding similarity yielding nan for
    a degenerate case) would 500 via allow_nan=False. json_safe turns it to
    None (UI renders "--") rather than crashing the test."""
    from types import SimpleNamespace
    from asea.core.pipeline import Pipeline
    from asea.studio.jobs import TransferJob
    from asea.studio.server import manager
    from asea.benchmarks import harness as harness_mod

    pipeline = Pipeline(workspace=tmp_path)
    job = TransferJob({"sender": "x", "receiver": "y", "suites": ["medical_triage"]},
                      manager.workspace_root)
    job.pipeline = pipeline
    manager.jobs[job.job_id] = job
    nan = float("nan")
    fake = SimpleNamespace(score=nan, task_success=nan, case_results=[],
                          similarity_is_semantic=False)
    monkeypatch.setattr(harness_mod.BenchmarkHarness, "run",
                        lambda self, *a, **k: fake)
    try:
        r = client.post("/api/skills/test", json={
            "job_id": job.job_id, "module": "triage-corpus",
            "suite_id": "medical_triage", "similarity": "lexical",
        })
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["baseline"]["score"] is None
        assert d["candidate"]["score"] is None
        assert d["improvement"] is None
    finally:
        del manager.jobs[job.job_id]


def test_spring_rejects_unknown_quant_level():
    """`levels` is reflected into the per-state table as `state` and back to the
    client, so it is constrained to the vendor's quantization states
    (int8/int4/int2) -- an arbitrary / XSS string is a 422, not a run. Defense
    in depth beside the UI escaping the value."""
    r = client.post("/api/spring", json={
        "module_id": "smollm2-360m-hf",
        "suite_ids": ["assamese_english"],
        "levels": ["<img src=x onerror=alert(1)>", "int8"],
    })
    assert r.status_code == 422
    assert "levels" in r.text


def test_deep_apply_overrides_cannot_weaken_gate2():
    """The Studio surface may only shape TRAINING (LoRA shape, optimiser, steps,
    seed, device, quantization flag) -- never the Gate 2 thresholds. A direct
    API client passing threshold overrides must have them DROPPED, not forwarded
    (adversarial review 2026-08-18: the permissive asdict path forwarded every
    DeepApplyConfig field, letting a client disable the evaluator/safety/
    improvement/regression/movement/mock checks)."""
    from asea.studio.deepapply_jobs import _coerce_config

    cfg = _coerce_config("standard", {
        # Gate 2 / honesty knobs -- every one must be IGNORED:
        "min_evaluator_score": 0.0,
        "min_safety_score": 0.0,
        "min_improvement": -1.0,
        "max_case_regression_ratio": 99.0,
        "max_control_movement": 99.0,
        "max_synthetic_depth": 99,
        "min_trainable_params": 0,
        "strict_no_mock": False,
        "regression_tolerance": 1.0,
        "cpu_param_ceiling": 99_999_999_999,
        "max_new_tokens": 9999,
        # Training-shape knobs -- every one must be APPLIED:
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 5e-4,
        "max_steps": 32,
        "load_in_4bit": True,
    })
    # Threshold / honesty knobs kept their honest defaults (None / True).
    assert cfg.min_evaluator_score is None
    assert cfg.min_safety_score is None
    assert cfg.min_improvement is None
    assert cfg.max_case_regression_ratio is None
    assert cfg.max_control_movement is None
    assert cfg.max_synthetic_depth is None
    assert cfg.min_trainable_params == 1          # default; 0 was dropped
    assert cfg.strict_no_mock is True             # default; False was dropped
    assert cfg.regression_tolerance == 0.02      # default; 1.0 was dropped
    assert cfg.cpu_param_ceiling == 1_500_000_000  # hardware-honesty ceiling intact
    assert cfg.max_new_tokens == 48              # default; 9999 was dropped
    # Training-shape knobs were applied.
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.learning_rate == 5e-4
    assert cfg.max_steps == 32
    assert cfg.load_in_4bit is True


def test_spring_reaches_model_load_not_importerror_for_hf_receiver(monkeypatch, tmp_path):
    """A real-HF spring run must get PAST the vendor imports to the model load
    (adversarial LIVE test 2026-08-18: spring_jobs imported get_decoder_layers
    + certify_hf_states from the vendor package ROOT, which does not export
    them, so EVERY real-HF spring run crashed with ImportError -- exactly the
    blind spot the Ollama-refusal test leaves, since it returns before the
    import). With a fake HF connector (HFCausalConnector.__init__ is lazy -- it
    stores model_id and loads no weights) and a bogus model id, the run must
    fail with DeepApplyBlocked at the LOAD ("could not load ..."), NOT an
    ImportError from a wrong vendor import path. Hermetic + fast: no real
    weights load (HF_HUB_OFFLINE makes from_pretrained fail at cache lookup)."""
    from asea.modules.real import HFCausalConnector
    from asea.studio import catalog
    from asea.studio.spring_jobs import SpringJob

    fake = HFCausalConnector(
        model_id="definitely/not-a-real-model-id",
        capabilities=[],  # unused by the spring path; lazy init loads nothing
        roles=["receiver"],
    )
    monkeypatch.setattr(catalog, "build", lambda mid: fake)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # fail fast at the cache lookup

    sp = SpringJob({"module_id": "fake-hf", "suite_ids": ["assamese_english"]},
                   tmp_path / "sp")
    sp._run()
    assert sp.status == "failed"
    # The failure must be at the LOAD, not at the import -- this is the exact
    # regression: a wrong vendor import path produced error_type=="ImportError".
    assert sp.error_type == "DeepApplyBlocked", (
        "expected DeepApplyBlocked at the model load, got {}: {}".format(
            sp.error_type, sp.error)
    )
    assert "could not load" in (sp.error or "")


# ============================================================================
# Suite authoring + capability support-reject (the "add any new benchmark
# suite; reject with an explanation when a model doesn't support it" feature).
# All hermetic: BENCHMARKS is monkeypatched to a tmp dir (never writes the real
# data/benchmarks/), and catalog.build/listing are faked so no weights load and
# no ollama probe runs. The reject is a HARD server-side check before any job.
# ============================================================================


def _fr_en_suite_dict():
    """A minimal valid suite JSON dict (capability translate/text/translation/fr->en)
    with one extraction + one heldout case -- the shape POST /api/suites accepts."""
    return {
        "suite_id": "fr_en_author",
        "description": "fr->en translation authored from the Studio",
        "task_type": "translate",
        "modality": "text",
        "domain": "translation",
        "language": "fr->en",
        "cases": [
            {"case_id": "ex1", "prompt": "Bonjour", "expected": "Hello",
             "split": "extraction", "meta": {}},
            {"case_id": "ho1", "prompt": "Merci", "expected": "Thank you",
             "split": "heldout", "meta": {}},
        ],
    }


def _fake_module(module_id, capabilities, roles=("sender", "receiver")):
    """A cheap stand-in for a catalog module: manifest() returns a real
    CapabilityManifest with the given capabilities, so supports() /
    capability_set() behave exactly as the real handshake path. No weights."""
    from types import SimpleNamespace
    from asea.core.protocol import CapabilityManifest

    manifest = CapabilityManifest(
        module_id=module_id,
        display_name=module_id,
        roles=list(roles),
        capabilities=capabilities,
        is_mock=False,
    )
    return SimpleNamespace(module_id=module_id, manifest=lambda: manifest,
                           is_mock=False)


def _install_fakes(monkeypatch, fakes):
    """Point catalog.build + catalog.listing at a dict of {id: fake_module} so the
    transfer/test/train/compress endpoints see the fakes without loading weights
    or probing ollama. listing() returns the id/roles/description/requires shape
    create_transfer's existence check reads."""
    from asea.studio import catalog

    def _build(mid):
        if mid not in fakes:
            raise KeyError("unknown module '{}'".format(mid))
        return fakes[mid]

    monkeypatch.setattr(catalog, "build", _build)
    monkeypatch.setattr(catalog, "listing", lambda: [
        {"id": mid, "roles": m.manifest().roles,
         "description": mid, "requires": "fake"}
        for mid, m in fakes.items()
    ])


def test_post_suites_authors_new_suite_and_it_appears_in_listings(tmp_path, monkeypatch):
    """POST /api/suites writes <suite_id>.json atomically, the JSON suite_id ==
    the filename stem, and GET /api/suites lists it -- without touching the real
    data/benchmarks/ dir (BENCHMARKS is patched to tmp)."""
    import json
    from asea.studio import server
    from asea.benchmarks.harness import load_suite

    monkeypatch.setattr(server, "BENCHMARKS", tmp_path)
    body = _fr_en_suite_dict()
    r = client.post("/api/suites", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["suite_id"] == "fr_en_author"
    assert (tmp_path / "fr_en_author.json").exists()
    # JSON suite_id == filename stem (the Studio key convention)
    on_disk = json.loads((tmp_path / "fr_en_author.json").read_text("utf-8"))
    assert on_disk["suite_id"] == "fr_en_author"
    # round-trips through load_suite
    suite = load_suite(tmp_path / "fr_en_author.json")
    assert suite.capability().as_str() == "translate/text/translation/fr->en"
    # appears in the listing
    listed = client.get("/api/suites").json()
    assert "fr_en_author" in listed
    assert listed["fr_en_author"]["splits"]["extraction"] == 1
    assert listed["fr_en_author"]["splits"]["heldout"] == 1


def test_post_suites_requires_both_extraction_and_heldout(tmp_path, monkeypatch):
    """A suite with no heldout case is refused with 400 naming the missing split
    (a transfer reads both; a suite that can't be evaluated is not a capability)."""
    from asea.studio import server

    monkeypatch.setattr(server, "BENCHMARKS", tmp_path)
    body = _fr_en_suite_dict()
    # two cases, BOTH extraction (>=2 cases so the schema's min_length=2 passes,
    # and the handler's missing-heldout check is what fires -> 400, not 422)
    body["cases"] = [
        {"case_id": "ex1", "prompt": "Bonjour", "expected": "Hello",
         "split": "extraction", "meta": {}},
        {"case_id": "ex2", "prompt": "Oui", "expected": "Yes",
         "split": "extraction", "meta": {}},
    ]
    r = client.post("/api/suites", json=body)
    assert r.status_code == 400
    assert "heldout" in r.json()["detail"]


def test_post_suites_rejects_duplicate_id_and_unsafe_name(tmp_path, monkeypatch):
    """A duplicate stem is a 409 (refuse silent overwrite of a capability
    definition); an unsafe name is a 422 (the schema pattern blocks path
    traversal / uppercase / spaces)."""
    from asea.studio import server

    monkeypatch.setattr(server, "BENCHMARKS", tmp_path)
    # author once
    assert client.post("/api/suites", json=_fr_en_suite_dict()).status_code == 200
    # duplicate -> 409, and the original is untouched
    dup = client.post("/api/suites", json=_fr_en_suite_dict())
    assert dup.status_code == 409
    assert "already exists" in dup.json()["detail"]
    # unsafe names -> 422 (pattern ^[a-z0-9][a-z0-9_-]*$)
    bad = dict(_fr_en_suite_dict())
    for bad_id in ("../x", "UPPER", "has space", "Fr-EN"):
        bad["suite_id"] = bad_id
        assert client.post("/api/suites", json=bad).status_code == 422, bad_id


def test_support_check_reports_unsupported_capability(tmp_path, monkeypatch):
    """GET /api/suites/{id}/support: a sender that supports only as->en vs a
    fr->en suite reports supports=False with a reason naming the capability and
    listing what the model does support -- the user-facing preview of the same
    _assert_support that hard-rejects at run time."""
    from asea.core.protocol import CapabilityKey, Domain, Modality
    from asea.studio import server

    monkeypatch.setattr(server, "BENCHMARKS", tmp_path)
    client.post("/api/suites", json=_fr_en_suite_dict())

    as_en = CapabilityKey(task_type="translate", modality=Modality.TEXT, domain=Domain.TRANSLATION, language="as->en")
    fr_en = CapabilityKey(task_type="translate", modality=Modality.TEXT, domain=Domain.TRANSLATION, language="fr->en")
    _install_fakes(monkeypatch, {
        "as-sender": _fake_module("as-sender", [as_en], roles=["sender"]),
        "fr-learner": _fake_module("fr-learner", [fr_en], roles=["receiver"]),
    })
    r = client.get("/api/suites/fr_en_author/support",
                   params={"sender": "as-sender", "receiver": "fr-learner"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is False
    assert d["sender"]["supports"] is False
    assert d["receiver"]["supports"] is True
    assert any("as-sender" in x and "fr->en" in x for x in d["reasons"])


def test_create_transfer_rejects_unsupported_suite_with_explanation(tmp_path, monkeypatch):
    """POST /api/transfers with a suite whose capability the sender does NOT
    support returns 400 naming the capability + what the model supports, AND
    spawns no job (manager.jobs unchanged) -- the hard reject, not a UI hint."""
    from asea.core.protocol import CapabilityKey, Domain, Modality
    from asea.studio import server
    from asea.studio.server import manager

    monkeypatch.setattr(server, "BENCHMARKS", tmp_path)
    client.post("/api/suites", json=_fr_en_suite_dict())

    as_en = CapabilityKey(task_type="translate", modality=Modality.TEXT, domain=Domain.TRANSLATION, language="as->en")
    fr_en = CapabilityKey(task_type="translate", modality=Modality.TEXT, domain=Domain.TRANSLATION, language="fr->en")
    _install_fakes(monkeypatch, {
        "as-sender": _fake_module("as-sender", [as_en], roles=["sender"]),
        "fr-learner": _fake_module("fr-learner", [fr_en], roles=["receiver"]),
    })
    before = set(manager.jobs)
    r = client.post("/api/transfers", json={
        "sender": "as-sender", "receiver": "fr-learner",
        "suites": ["fr_en_author"],
    })
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "fr->en" in detail          # the capability the suite needs
    assert "as->en" in detail          # what the sender actually supports
    assert "sender" in detail          # the role is named
    # no job spawned
    assert set(manager.jobs) == before


def test_create_transfer_allows_supported_suite(tmp_path, monkeypatch):
    """Regression: a combo where BOTH models support the suite's capability still
    creates the job (existing behavior preserved). _preflight_model + the real
    manager.create are stubbed so no weights load and no thread spawns."""
    from types import SimpleNamespace
    from asea.core.protocol import CapabilityKey, Domain, Modality
    from asea.studio import server
    from asea.studio.server import manager

    monkeypatch.setattr(server, "BENCHMARKS", tmp_path)
    client.post("/api/suites", json=_fr_en_suite_dict())

    fr_en = CapabilityKey(task_type="translate", modality=Modality.TEXT, domain=Domain.TRANSLATION, language="fr->en")
    _install_fakes(monkeypatch, {
        "fr-sender": _fake_module("fr-sender", [fr_en], roles=["sender"]),
        "fr-learner": _fake_module("fr-learner", [fr_en], roles=["receiver"]),
    })
    monkeypatch.setattr(server, "_preflight_model", lambda mid: None)
    created = {}
    def _fake_create(config):
        created["config"] = config
        return SimpleNamespace(to_dict=lambda: {
            "job_id": "stub", "status": "queued", "errored": False,
            "config": config, "created_at": "t"})
    monkeypatch.setattr(manager, "create", _fake_create)

    r = client.post("/api/transfers", json={
        "sender": "fr-sender", "receiver": "fr-learner",
        "suites": ["fr_en_author"],
    })
    assert r.status_code == 200, r.text
    assert created["config"]["suites"] == ["fr_en_author"]
    # the stub did not inject a real job into the manager
    assert "stub" not in manager.jobs


@pytest.mark.parametrize("path,body,role", [
    ("skills/test", {"module": "as-learner", "suite_id": "fr_en_author",
                     "similarity": "lexical"}, "model"),
    ("deepapply", {"receiver_id": "as-learner", "suite_id": "fr_en_author",
                   "backend": "standard"}, "receiver"),
    ("spring", {"module_id": "as-learner", "suite_ids": ["fr_en_author"]}, "model"),
])
def test_skills_test_and_train_and_compress_reject_unsupported_suite(
    tmp_path, monkeypatch, path, body, role
):
    """The other three run paths (skills/test, deepapply, spring) hard-reject an
    unsupported suite with the SAME _assert_support explanation and spawn nothing.
    test + deepapply need a source job with a pipeline (injected); spring does
    not (its support check runs before the spring job is created)."""
    from asea.core.pipeline import Pipeline
    from asea.core.protocol import CapabilityKey, Domain, Modality
    from asea.studio import server
    from asea.studio.server import manager
    from asea.studio.jobs import TransferJob

    monkeypatch.setattr(server, "BENCHMARKS", tmp_path)
    client.post("/api/suites", json=_fr_en_suite_dict())

    as_en = CapabilityKey(task_type="translate", modality=Modality.TEXT, domain=Domain.TRANSLATION, language="as->en")
    _install_fakes(monkeypatch, {
        "as-learner": _fake_module("as-learner", [as_en], roles=["receiver"]),
    })

    # For spring, no source job is needed; ensure no spring job is spawned by
    # stubbing the manager. The 400 returns before create() is called.
    from asea.studio import spring_jobs as sj_mod
    created = {"spring": False}
    def _no_spring(config, workspace_root):
        created["spring"] = True
        raise AssertionError("spring job must not be spawned on unsupported suite")
    monkeypatch.setattr(sj_mod.SpringManager, "create", _no_spring)

    if path in ("skills/test", "deepapply"):
        # inject a source transfer job whose pipeline is initialised
        job = TransferJob({"sender": "x", "receiver": "y", "suites": ["fr_en_author"]},
                          manager.workspace_root)
        job.pipeline = Pipeline(workspace=tmp_path)
        manager.jobs[job.job_id] = job
        body = dict(body, job_id=job.job_id)
        try:
            r = client.post("/api/{}".format(path), json=body)
        finally:
            manager.jobs.pop(job.job_id, None)
    else:
        r = client.post("/api/{}".format(path), json=body)

    assert r.status_code == 400, (path, r.text)
    detail = r.json()["detail"]
    assert "fr->en" in detail
    assert "as->en" in detail
    if path == "spring":
        assert created["spring"] is False


def test_add_catalog_can_bind_model_to_new_suite_capability(tmp_path, monkeypatch):
    """Full loop: author a fr->en suite (tmp BENCHMARKS), then POST /api/catalog
    tied to it. The preflight refuses the unpulled tag (400, by design) BUT the
    entry is registered, and catalog.build(new_id) declares the fr->en capability
    -- so a model can be bound to a capability that did not exist before."""
    from asea.core.protocol import CapabilityKey, Domain, Modality
    from asea.studio import catalog, server

    monkeypatch.setattr(server, "BENCHMARKS", tmp_path)
    assert client.post("/api/suites", json=_fr_en_suite_dict()).status_code == 200

    new_id = "test-fr-learner"
    assert new_id not in catalog.CATALOG
    try:
        r = client.post("/api/catalog", json={
            "module_id": new_id,
            "ollama_tag": "definitely-not-pulled:latest",
            "role": "receiver",
            "suite_id": "fr_en_author",
        })
        # preflight refuses the unpulled tag (by design); the entry is registered
        assert r.status_code == 400
        assert new_id in catalog.CATALOG
        # the bound model declares the fr->en capability (cheap build, no network)
        module = catalog.build(new_id)
        fr_en = CapabilityKey(task_type="translate", modality=Modality.TEXT, domain=Domain.TRANSLATION, language="fr->en")
        assert module.manifest().supports(fr_en) is True
    finally:
        catalog.CATALOG.pop(new_id, None)
        catalog._cache.pop(new_id, None)
