"""Real-model stage machinery for the extraction program (Sections E-M).

Honesty contract (binding, owner's brief):

  * Hardware insufficiency is reported as ``BLOCKED_RESOURCE`` with the
    EXACT requirement and remedy -- never a substituted estimate, never a
    fixture result dressed as a real-model measurement.
  * Frequency is correlation. Only controlled interventions (verified
    mask/measure/restore) support causal language, and only for the
    components actually intervened on with the exact seeds recorded.
  * Graph roles: ``REQUIRED`` and ``NEGATIVE_OR_HARMFUL`` REQUIRE verified
    causal evidence; correlation alone can never produce them (enforced
    at schema level by ``CausalComponentGraph``).

What is implemented here:

  * Section E -- on-disk source architecture validation (config.json read
    WITHOUT torch; the GLM-5.3-Flash card is pinned field by field).
  * Section G/I -- config-arithmetic hardware admission using the SAME
    formula as the isolated worker (parameters x dtype x 2 + headroom vs
    available memory). On the development host this gate refuses with
    ``BLOCKED_RESOURCE``; that refusal IS the stage's honest output.
  * Sections L/M/N -- the causal component graph and the conservative
    extraction plan, as pure functions over RECORDED artifacts (verified
    intervention attempts + enrichment records). These run offline from
    workspace artifacts and are fixture-verifiable without the model.

What is NOT implemented here (stated, never faked): the beyond-admission
measurement orchestration (generation/judging/intervention loops against
the live worker). The CLI handlers refuse honestly naming it after the
admission gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import BlockedResource, InvalidEvidence
from .schema import (
    CausalComponentGraph,
    CausalEffectEstimate,
    ExpertRetention,
    ExtractionPlan,
    GraphNode,
    InterventionAttempt,
)

# ---------------------------------------------------------------------------
# Section E -- source architecture validation (no torch, config.json only)
# ---------------------------------------------------------------------------

#: The GLM-5.3-Flash source card this program pins (kept in sync with
#: ``asea.capability_build.adapters.glm53_flash.EXPECTED`` and the worker
#: docstring; the 2026-09-19 audit established ``scoring_func: "sigmoid"``
#: IS declared by the real config -- the old "absent" note was false).
GLM_EXPECTED_ARCHITECTURE: Dict[str, Any] = {
    "model_type": "glm5_next",
    "num_hidden_layers": 45,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "hidden_size": 4096,
    "scoring_func": "sigmoid",
    # A FLOOR, not an exact value (the adapter pins the same floor):
    # the context window must be at least 1M positions.
    "max_position_embeddings_floor": 1_000_000,
}

#: Headroom and dtype of the admission formula -- identical to the worker
#: (``workers/glm53/worker.py``): BF16 stored weights, the x2 factor for
#: the in-flight copy during load, 512 MiB headroom.
ADMISSION_DTYPE_BYTES = 2
ADMISSION_HEADROOM_MIB = 512


def load_checkpoint_config(checkpoint: Path) -> Dict[str, Any]:
    """Read config.json WITHOUT torch. GLM-5.3-Flash is a vision-language
    checkpoint: the text-model fields may live at the top level or under
    ``text_config``; both layouts are accepted, the merged text config is
    returned."""
    path = checkpoint / "config.json"
    if not path.is_file():
        raise InvalidEvidence(
            "checkpoint %s has no config.json; the source architecture "
            "cannot be validated without it" % checkpoint,
            "point --checkpoint at the unpacked GLM-5.3-Flash checkpoint "
            "directory")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise InvalidEvidence(
            "config.json is not valid JSON: %s" % exc,
            "re-download the checkpoint; a corrupt config is invalid "
            "evidence, not a partial pass")
    if not isinstance(raw, dict):
        raise InvalidEvidence(
            "config.json must be a JSON object",
            "re-download the checkpoint")
    text = raw.get("text_config")
    if isinstance(text, dict):
        merged = dict(raw)
        merged.pop("text_config", None)
        merged.update(text)
        return merged
    return dict(raw)


def validate_source_architecture(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pin the source card field by field (Section E). Returns
    ``{"valid": bool, "checks": {field: {"expected", "found", "ok"}}}``.
    A mismatch is reported, never silently forgiven -- an unexpected
    revision would make every downstream artifact a claim about a model
    that was never measured."""
    checks: Dict[str, Dict[str, Any]] = {}
    for field, expected in sorted(GLM_EXPECTED_ARCHITECTURE.items()):
        found = config.get(field.replace("_floor", ""), 0) \
            if field.endswith("_floor") else config.get(field)
        if field.endswith("_floor"):
            ok = isinstance(found, int) and found >= expected
        else:
            ok = found == expected
        checks[field] = {
            "expected": expected,
            "found": found,
            "ok": ok,
        }
    return {"valid": all(c["ok"] for c in checks.values()), "checks": checks}


# ---------------------------------------------------------------------------
# Sections G/I -- config-arithmetic hardware admission
# ---------------------------------------------------------------------------

def estimate_parameters(config: Dict[str, Any]) -> int:
    """Lower-bound parameter count from config arithmetic (no weights
    touched) -- the SAME formula the isolated worker uses, so the CLI's
    admission and the worker's preflight can never disagree."""
    hidden = int(config.get("hidden_size", 0) or 0)
    layers = int(config.get("num_hidden_layers", 0) or 0)
    dense = int(config.get("first_k_dense_replace", 0) or 0)
    routed = int(config.get("n_routed_experts", 0) or 0)
    shared = int(config.get("n_shared_experts", 0) or 0)
    expert_params = (config.get("moe_intermediate_size")
                     or config.get("intermediate_size") or 0)
    per_expert = 3 * hidden * int(expert_params)  # gate/up/down
    moe = (routed + shared) * per_expert * max(0, layers - dense)
    dense_mlp = layers * 3 * hidden * int(config.get("intermediate_size", 0) or 0)
    vocab = int(config.get("vocab_size", 0) or 0)
    return moe + dense_mlp + vocab * hidden + 4 * layers * hidden * hidden


def available_bytes() -> int:
    """Available host memory in bytes: min(MemAvailable, cgroup ceiling)
    on Linux; ``GlobalMemoryStatusEx`` on Windows; 0 when it cannot be
    measured -- admission NEVER passes on an unknown."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable"):
                    mem = int(line.split()[1]) * 1024
                    break
            else:
                mem = 0
    except (OSError, ValueError, IndexError):
        mem = 0
    if mem:
        for cgroup in ("/sys/fs/cgroup/memory.max",
                       "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                with open(cgroup, encoding="ascii") as handle:
                    limit = int(handle.read().strip())
                if 0 < limit:
                    mem = min(mem, limit)
            except (OSError, ValueError):
                continue
        return mem
    # Windows / non-proc hosts: try the ctypes memory status, then psutil.
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                ctypes.byref(status)):
            return int(status.ullAvailPhys)
    except Exception:
        pass
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def require_admission(checkpoint: Path, stage: str) -> Dict[str, Any]:
    """The admission gate every real-model stage passes through FIRST.
    Reads the checkpoint config (Section E), estimates the source size
    and refuses with :class:`BlockedResource` when the host cannot hold
    the model. The refusal message carries the exact requirement and the
    remedy -- the brief's honest outcome on an 8 GB-class host, which is
    what will happen on the development machine."""
    config = load_checkpoint_config(checkpoint)
    architecture = validate_source_architecture(config)
    if not architecture["valid"]:
        bad = sorted(field for field, check
                     in architecture["checks"].items() if not check["ok"])
        raise InvalidEvidence(
            "checkpoint %s does not match the pinned GLM-5.3-Flash source "
            "card on %s; a measurement of a different revision is not "
            "evidence about the pinned source" % (checkpoint, ", ".join(bad)),
            "use the pinned zai-org/GLM-5.3-Flash BF16 checkpoint this "
            "program's spec names")
    parameters = estimate_parameters(config)
    required = (parameters * ADMISSION_DTYPE_BYTES * 2
                + ADMISSION_HEADROOM_MIB * 1024 * 1024)
    available = available_bytes()
    if available <= 0:
        raise BlockedResource(
            requirement=(
                "%s needs the GLM-5.3-Flash source resident (~%d parameters "
                "x %d bytes x 2 + %d MiB headroom = ~%.0f GiB) and this "
                "host's available memory could not be measured"
                % (stage, parameters, ADMISSION_DTYPE_BYTES,
                   ADMISSION_HEADROOM_MIB, required / (1024 ** 3))
            ),
            remedy=(
                "run on a host where available memory is measurable and "
                "sufficient, or use the behavioural (cloud) evidence class "
                "on small hosts"
            ),
        )
    if required > available:
        raise BlockedResource(
            requirement=(
                "%s needs ~%.0f GiB resident for the GLM-5.3-Flash source "
                "(%d parameters x %d bytes x 2 + %d MiB headroom); this "
                "host reports %.2f GiB available"
                % (stage, required / (1024 ** 3), parameters,
                   ADMISSION_DTYPE_BYTES, ADMISSION_HEADROOM_MIB,
                   available / (1024 ** 3))
            ),
            remedy=(
                "run on a host with enough RAM for the ~320B-parameter BF16 "
                "teacher (the worker loads device_map='cpu', so host memory "
                "is the requirement -- a GPU alone does not change the "
                "admission), or use the behavioural (cloud) evidence class "
                "on small hosts"
            ),
        )
    return {
        "architecture_check": architecture,
        "estimated_parameters": parameters,
        "required_bytes": required,
        "available_bytes": available,
    }


# ---------------------------------------------------------------------------
# Sections L/M/N -- causal graph and conservative plan (offline builders)
# ---------------------------------------------------------------------------

#: Effect floors: a component is REQUIRED only when its verified target
#: effect is strictly positive AND exceeds its matched-control arm's
#: effect under the same protocol; a component whose verified intervention
#: regressed controls beyond this floor is NEGATIVE_OR_HARMFUL.
CONTROL_REGRESSION_FLOOR = 0.0

_GRAPH_LIMITATIONS = (
    "Roles REQUIRED/NEGATIVE_OR_HARMFUL rest on verified mask/measure/"
    "restore interventions on this workload and seed set only; they are "
    "not a general importance ranking.",
    "Correlation-only components (enrichment without a verified "
    "intervention) can never hold a causal role in this graph.",
    "The graph says nothing about components it does not name: "
    "untested is not unimportant.",
)


def load_intervention_attempts(paths: Sequence[Path]) -> List[InterventionAttempt]:
    """Load recorded InterventionAttempt artifacts (JSONL or JSON). An
    unparseable or invalid attempt is invalid evidence (exit 3), never a
    silent skip."""
    attempts: List[InterventionAttempt] = []
    for path in paths:
        if not path.is_file():
            raise InvalidEvidence(
                "intervention artifact %s does not exist" % path,
                "point --interventions at the artifacts recorded by "
                "`silt-extract intervene`")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        raw_records: List[Any]
        if path.suffix == ".jsonl":
            raw_records = [json.loads(line) for line in text.splitlines()
                           if line.strip()]
        else:
            raw_records = [json.loads(text)]
        for raw in raw_records:
            try:
                attempts.append(InterventionAttempt.model_validate(raw))
            except Exception as exc:
                raise InvalidEvidence(
                    "intervention artifact %s holds an invalid attempt: %s"
                    % (path, exc),
                    "an unrestorable or malformed attempt belongs in the "
                    "failed-attempt ledger (C7), not the causal evidence "
                    "set") from exc
    return attempts


def build_causal_effect_estimates(
    attempts: Sequence[InterventionAttempt],
) -> Dict[str, CausalEffectEstimate]:
    """Aggregate verified attempts per component into causal effect
    estimates. Only restored-and-verified attempts reach this function
    (the schema refuses the rest at load time)."""
    by_component: Dict[str, List[InterventionAttempt]] = {}
    for attempt in attempts:
        by_component.setdefault(attempt.component, []).append(attempt)
    estimates: Dict[str, CausalEffectEstimate] = {}
    for component, group in sorted(by_component.items()):
        # The matched-control arm (Section M): the minimum control drop
        # observed across the component's verified attempts. A target
        # effect that does not exceed it is not causal evidence for a
        # REQUIRED role.
        control_arm_effect = min(a.control_drop for a in group)
        estimates[component] = CausalEffectEstimate(
            capability_id=group[0].capability_id,
            component=component,
            effect=min(a.target_drop for a in group),
            control_arm_effect=control_arm_effect,
            attempts=len(group),
            seed_set=sorted({a.seed for a in group}),
            source_revision=group[0].source_revision,
        )
    return estimates


def build_causal_graph(
    *,
    capability_id: str,
    source_revision: str,
    attempts: Sequence[InterventionAttempt],
    enriched_components: Optional[Dict[str, Dict[str, Any]]] = None,
) -> CausalComponentGraph:
    """Classify components (Section L): causal roles only from verified
    interventions, correlation roles only for enrichment, matched-control
    arms decide REQUIRED vs not (Section M)."""
    enriched_components = enriched_components or {}
    estimates = build_causal_effect_estimates(attempts)
    nodes: List[GraphNode] = []
    for component, estimate in estimates.items():
        if estimate.effect > CONTROL_REGRESSION_FLOOR \
                and estimate.effect > estimate.control_arm_effect:
            role = "REQUIRED"
        elif estimate.control_arm_effect > CONTROL_REGRESSION_FLOOR:
            role = "NEGATIVE_OR_HARMFUL"
        else:
            role = "UNKNOWN"
        nodes.append(GraphNode(
            component=component,
            role=role,  # type: ignore[arg-type]
            evidence_count=estimate.attempts,
            causal_evidence=True,
        ))
    for component in sorted(set(enriched_components) - set(estimates)):
        entry = enriched_components[component] or {}
        # Correlation-only components can never be causal; the honest
        # roles are enrichment-flavoured, and unobserved-in-control stays
        # visible instead of being scored infinite.
        control_seen = entry.get("control_cases", 0)
        role = ("TARGET_ENRICHED" if control_seen == 0
                else "UNKNOWN" if entry.get("control_unobserved")
                else "SHARED_GENERAL")
        nodes.append(GraphNode(
            component=component,
            role=role,  # type: ignore[arg-type]
            evidence_count=int(entry.get("target_cases", 0)
                               + entry.get("control_cases", 0)),
            causal_evidence=False,
        ))
    if not nodes:
        raise InvalidEvidence(
            "no components to classify: the graph needs verified "
            "intervention attempts and/or enrichment records",
            "run `silt-extract intervene` (on hardware its admission gate "
            "accepts) before building a graph")
    return CausalComponentGraph(
        capability_id=capability_id,
        source_revision=source_revision,
        nodes=nodes,
        limitations=list(_GRAPH_LIMITATIONS),
    )


def build_extraction_plan(
    *,
    capability_id: str,
    source_manifest_sha256: str,
    graph: CausalComponentGraph,
) -> ExtractionPlan:
    """Derive the conservative first-extraction plan (Sections M/N):
    tokenizer, embeddings, dense layers, attention and head are retained
    (schema-enforced); the routed-expert subset per layer is the causally
    REQUIRED set ONLY -- correlation-enriched components are NOT retained
    on that basis, because correlation is not evidence of necessity."""
    retained: Dict[int, List[int]] = {}
    for node in graph.nodes:
        if node.role != "REQUIRED":
            continue
        component = node.component
        if not (component.startswith("expert:")
                and "/" in component[len("expert:"):]):
            continue
        layer_s, expert_s = component[len("expert:"):].split("/", 1)
        try:
            layer, expert = int(layer_s), int(expert_s)
        except ValueError:
            continue
        retained.setdefault(layer, []).append(expert)
    if not retained:
        raise InvalidEvidence(
            "the graph names no causally REQUIRED routed expert; a "
            "conservative causal plan retains none, and an empty "
            "extraction is not a plan",
            "record verified interventions with target effects above "
            "their matched-control arms before planning")
    graph_sha = _digest_json(graph.model_dump(mode="json", by_alias=True))
    return ExtractionPlan(
        capability_id=capability_id,
        source_manifest_sha256=source_manifest_sha256,
        graph_sha256=graph_sha,
        retained_dense=True,
        retained_tokenizer=True,
        retained_attention=True,
        retained_head=True,
        expert_retention=[
            ExpertRetention(layer_id=layer,
                            retained_experts=sorted(experts),
                            selection_basis="causal")
            for layer, experts in sorted(retained.items())
        ],
        selection_arm="causal",
    )


def _digest_json(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Section H -- power analysis (declared alongside the dataset)
# ---------------------------------------------------------------------------

def required_sample_size(
    *, baseline_pass_rate: float, minimum_detectable_effect: float,
    alpha: float = 0.05, power: float = 0.8,
) -> int:
    """Two-proportion sample size per family (normal approximation with
    continuity-free bounds): the N needed for a one-sided test to detect
    a drop of ``minimum_detectable_effect`` from the baseline pass rate
    with the stated power. Pure math, no scipy -- the dataset's power
    analysis must run in the pydantic-only core environment."""
    if not 0.0 < baseline_pass_rate < 1.0:
        raise InvalidEvidence(
            "baseline_pass_rate must be strictly inside (0, 1) for a "
            "two-proportion power analysis; a degenerate rate measures "
            "nothing", "author cases the source model can neither always "
            "pass nor always fail")
    if not 0.0 < minimum_detectable_effect < baseline_pass_rate:
        raise InvalidEvidence(
            "minimum_detectable_effect must lie inside (0, "
            "baseline_pass_rate)",
            "choose an effect the preregistered thresholds actually care "
            "about")
    p1 = baseline_pass_rate
    p2 = baseline_pass_rate - minimum_detectable_effect
    z_alpha = _z_one_sided(alpha)
    z_beta = _z_one_sided(1.0 - power)
    pooled = (p1 + p2) / 2.0
    numerator = (z_alpha * math.sqrt(2.0 * pooled * (1.0 - pooled))
                 + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2)))
    return int(math.ceil((numerator / minimum_detectable_effect) ** 2))


def _z_one_sided(alpha: float) -> float:
    """One-sided normal quantile by bisection on Phi^-1 (no scipy)."""
    if not 0.0 < alpha < 1.0:
        raise InvalidEvidence("alpha must be inside (0, 1)", "fix alpha")
    target = 1.0 - alpha
    low, high = -8.0, 8.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0