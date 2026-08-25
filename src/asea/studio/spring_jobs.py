"""SiltSpring job manager (Studio surface for "compress a big model to transfer
skills and verify the compression does not degrade them").

SiltSpring certifies a model's quantization states (int8 / int4 / int2) against
the held-out suites SILT's Gate 2 uses: it measures the full-precision reference
loss, then each quantized state's loss, and marks a state ``certified`` for a
skill when the degradation is within tolerance (directional -- only a WORSE loss
revokes, consistent with the toy SpringModel and the vendor's
``certify_hf_states``). The Studio surface runs the real vendor
``certify_hf_states`` on a model the user picks from the catalog, streaming
per-state results + peak VRAM as SSE.

HONESTY (binding):

  * The compression is REAL (vendored ``certify_hf_states``), measured against
    the SAME held-out suites the gates use -- not a fake "% smaller" number.
    ``bytes_packed`` and per-skill ``degradation`` come from the vendor.
  * Memory bound is surfaced honestly, not hidden: certifying a 7B model needs
    the full-precision reference resident (~14 GB fp16) which does NOT fit an 8 GB
    GPU -- the run loads to the chosen device and reports peak VRAM; if it OOMs
    the load fails with a typed error (never a silent pass). The UI states this
    bound so a user picks a model that fits (e.g. SmolLM2-360m / Qwen2.5-0.5B).
  * ``certified`` is directional (``degradation <= tolerance``); a FAVORABLE loss
    change is "not degraded" by design, not a loophole -- matching the toy
    SpringModel.certify and the vendor path (see memory: spring 2026-08-17).
  * The receiver must be an in-process HF model (the certifier loads its
    weights); Ollama is refused up front with a typed error, no silent fallback.
  * LOCAL ONLY; patent pending (India): loads weights on the local machine; no
    network beyond an optional one-time model download by transformers.
"""

from __future__ import annotations

import json
import math
import queue
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmarks.harness import load_suite
from ..core.pipeline import Pipeline  # noqa: F401  (type hint clarity)
from ..deepapply.errors import DeepApplyBlocked
from . import catalog
from ._jsonsafe import json_safe


def _finite_or_none(v: Any) -> Optional[float]:
    """Round a numeric value to 6 dp, returning None for non-finite (nan/inf).

    A non-finite loss/degradation becomes None so JSON can encode it (the UI
    renders "—") instead of raising ValueError. See _jsonsafe.py."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(v):
        return round(float(v), 6)
    return None


def _is_real_number(v: Any) -> bool:
    """True for a finite int/float that is NOT a bool (bool is an int subclass
    and must never be treated as a degradation measurement)."""
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
    )


def _classify_state(lv: str, r: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """Build one per-state entry from a vendor result dict, enforcing the
    SiltSpring honesty rules (binding -- "no loopholes"):

      * A skill's degradation is ``certified`` when it is a finite number
        ``<= tolerance`` (only a WORSE loss revokes; a favourable change is
        "not degraded" by design).
      * A skill's degradation is ``revoked`` when it is a finite number
        ``> tolerance``.
      * Everything else is ``unmeasured``: a non-finite (nan/inf) degradation
        AND a non-numeric one (None / str). The old code filtered unmeasured
        with a third ``isinstance`` check, which silently DROPPED non-numeric
        values -- a state whose every degradation was None/string ended up with
        empty certified/revoked/unmeasured and was falsely "certified". The
        unmeasured set is now the COMPLEMENT of certified+revoked over the
        state's skill keys, so a non-numeric degradation is counted unmeasured,
        never silently ignored (adversarial review 2026-08-19, loophole #6).
      * ``overall_certified`` for a non-full state additionally requires at
        least one certified skill: a state that measured NOTHING (empty ``deg``,
        or all values non-numeric/non-finite) is NOT certified -- the old
        ``not revoked and not unmeasured`` logic certified it because both
        sets were empty (adversarial review 2026-08-19, loophole #6). The full
        (reference) state is certified by convention.
    """
    deg = r.get("degradation") or {}
    keys = list(deg.keys())
    certified_skills = sorted(
        s for s in keys if _is_real_number(deg[s]) and deg[s] <= tolerance
    )
    revoked_skills = sorted(
        s for s in keys if _is_real_number(deg[s]) and deg[s] > tolerance
    )
    measured = set(certified_skills) | set(revoked_skills)
    # complement: anything not certified/revoked is unmeasured (catches
    # non-finite AND non-numeric values the two filters above drop).
    unmeasured_skills = sorted(s for s in keys if s not in measured)
    loss = {k: _finite_or_none(v) for k, v in (r.get("loss") or {}).items()}
    deg_safe = {
        k: _finite_or_none(v) for k, v in deg.items()
        if _is_real_number(v)
    }
    bp = r.get("bytes_packed")
    try:
        bytes_packed = int(bp) if bp is not None else None
    except (TypeError, ValueError):
        bytes_packed = None  # a non-finite bytes value is nonsensical -> None
    return {
        "state": lv,
        # The vendor sets bytes_packed=None for the "full" (reference) state --
        # it is not packed. Coerce to None (not 0) so the UI shows "full
        # precision" rather than a misleading 0 MB, and int(None) never raises.
        "bytes_packed": bytes_packed,
        "loss": loss,
        "degradation": deg_safe,
        "certified_skills": certified_skills,
        "revoked_skills": revoked_skills,
        "unmeasured_skills": unmeasured_skills,
        "overall_certified": (
            lv == "full"
            or (bool(certified_skills) and not revoked_skills and not unmeasured_skills)
        ),
    }


ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"

#: Quantization states the Studio exposes (vendor supports int8/int4/int2).
_DEFAULT_LEVELS = ("int8", "int4", "int2")


class SpringJob:
    """One SiltSpring certification run, executed on a worker thread with live
    per-state telemetry."""

    def __init__(self, config: Dict[str, Any], workspace_root: Path) -> None:
        self.job_id = "sp-{}".format((id(self) ^ int(time.time() * 1000)) & 0xFFFFFFFF)
        self.config = config
        self.workspace = Path(workspace_root) / self.job_id
        self.status = "queued"        # queued | running | done | failed
        self.error_type: Optional[str] = None
        self.error: Optional[str] = None   # server-side only, never sent to client
        self.report: Optional[Dict[str, Any]] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._telemetry: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _emit(self, event: Dict[str, Any]) -> None:
        # Sanitize at the boundary: a non-finite float in a telemetry event
        # would serialize to a literal NaN token that the browser's JSON.parse
        # rejects (silent SSE breakage). None renders honestly as "—".
        self._telemetry.put(json_safe(event))

    def _run(self) -> None:
        try:
            self.status = "running"
            self._emit({"phase": "job_started", "job_id": self.job_id})
            report = self._execute()
            self.report = report
            self.status = "done"
            self._emit({"phase": "job_done", "job_id": self.job_id})
        except DeepApplyBlocked as exc:
            self.error_type = "DeepApplyBlocked"
            self.error = str(exc)
            traceback.print_exc()
            self.status = "failed"
            self._emit({"phase": "job_failed", "job_id": self.job_id,
                        "error": "DeepApplyBlocked"})
        except Exception as exc:  # noqa: BLE001 -- typed name to client, raw to log
            self.error_type = type(exc).__name__
            self.error = str(exc)
            traceback.print_exc()
            self.status = "failed"
            self._emit({"phase": "job_failed", "job_id": self.job_id,
                        "error": self.error_type})

    def _execute(self) -> Dict[str, Any]:
        module_id = self.config.get("module_id")
        if not module_id:
            raise DeepApplyBlocked("module_id is required (an HF catalog entry)")
        suite_ids = self.config.get("suite_ids") or []
        if not suite_ids:
            raise DeepApplyBlocked("suite_ids is required (held-out suites to certify against)")
        levels = list(self.config.get("levels") or _DEFAULT_LEVELS)
        tolerance = float(self.config.get("tolerance", 0.05))
        device_pref = self.config.get("device", "auto")  # auto | cpu | cuda
        max_len = int(self.config.get("max_len", 96))

        # Cheap HF-type preflight BEFORE the heavy torch import: Ollama / non-HF
        # receivers are refused instantly (the certifier loads weights it owns).
        try:
            receiver = catalog.build(module_id)
        except KeyError as exc:
            raise DeepApplyBlocked(
                "module '{}' is not in the catalog: {}".format(module_id, exc)
            )
        from ..modules.real import HFCausalConnector

        if not isinstance(receiver, HFCausalConnector):
            raise DeepApplyBlocked(
                "module '{}' is not an in-process HF model; SiltSpring certifies a "
                "model whose weights the process loads, so it needs an HF entry "
                "(e.g. smollm2-360m-hf, qwen2.5-0.5b), not an external-weights "
                "(Ollama) receiver".format(module_id)
            )
        model_id = receiver.model_id

        # Heavy imports (torch/transformers) only past the cheap preflight.
        # Preflight the [deep] extra: a missing torch/transformers used to raise
        # a bare ImportError caught by the generic handler and surfaced to the
        # client as the raw type name "ImportError" with no install guidance --
        # a silent-degradation the project refuses. Mirror deepapply_jobs: raise
        # a typed DeepApplyBlocked naming the extra BEFORE the weight load.
        try:
            import torch  # noqa: F401  (proves the [deep] extra is installed)
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DeepApplyBlocked(
                "SiltSpring cannot run here (missing the [deep] extra: "
                "pip install -e '.[deep]'): {}".format(exc)
            )

        device = device_pref
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise DeepApplyBlocked("device 'cuda' requested but no CUDA GPU is available")

        self._emit({"phase": "model_loading", "job_id": self.job_id,
                    "model_id": model_id, "device": device})
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            dtype = torch.float16 if device != "cpu" else torch.float32
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            if device == "cuda":
                model = model.to("cuda")
            model.eval()
        except Exception as exc:  # noqa: BLE001 -- OOM / missing model / etc.
            raise DeepApplyBlocked(
                "could not load '{}': {}".format(model_id, type(exc).__name__)
            )

        from ..deepapply.backends.siltstream_vendor.hf_real import get_decoder_layers
        from ..spring.certifier import suites_from_benchmark

        layers = get_decoder_layers(model)
        suites_obj = [load_suite(BENCHMARKS / "{}.json".format(s)) for s in suite_ids]
        suites = suites_from_benchmark(suites_obj, tok, max_len=max_len)
        if not suites:
            raise DeepApplyBlocked(
                "no held-out text found in suite(s) {} to certify against".format(suite_ids)
            )
        if device == "cuda":
            suites = {k: v.to("cuda") for k, v in suites.items()}

        from ..deepapply.backends.siltstream_vendor.hf_real import certify_hf_states

        tmp = Path(tempfile.mkdtemp(prefix="silt_spring_"))
        self._emit({"phase": "certify_started", "job_id": self.job_id,
                    "levels": levels, "tolerance": tolerance,
                    "skills": sorted(suites)})
        results = certify_hf_states(model, layers, suites, levels, str(tmp),
                                    tolerance=tolerance)

        vram_peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
        )
        states: List[Dict[str, Any]] = []
        reference = results.get("full", {}).get("loss", {})
        for lv in ["full"] + levels:
            entry = _classify_state(lv, results.get(lv, {}), tolerance)
            states.append(entry)
            self._emit({"phase": "spring_state", "job_id": self.job_id, **entry})

        # Free the model before returning so the process VRAM drops.
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return {
            "model_id": model_id,
            "device": device,
            "vram_peak_gb": round(vram_peak_gb, 3) if vram_peak_gb is not None else None,
            "tolerance": tolerance,
            "levels": levels,
            "decoder_layers": len(layers),
            "skills": sorted(suites),
            "reference_loss": {k: _finite_or_none(v) for k, v in reference.items()},
            "states": states,
        }

    # -- public views -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "config": self.config,
        }
        if self.error_type is not None:
            out["error"] = self.error_type
        if self.report is not None:
            # Backstop: sanitize the served report so a non-finite float (e.g. an
            # int2 state's NaN loss) can never 500 the GET endpoint or the
            # listing. Source values are already cleaned at build time; this
            # catches anything the runner injected later.
            out["report"] = json_safe(self.report)
        return out

    def telemetry(self):
        while True:
            try:
                ev = self._telemetry.get(timeout=0.5)
            except queue.Empty:
                if self.status in ("done", "failed"):
                    break
                continue
            yield "event: telemetry\ndata: {}\n\n".format(
                json.dumps(ev, ensure_ascii=False, default=str)
            )
        while True:
            try:
                ev = self._telemetry.get_nowait()
            except queue.Empty:
                break
            yield "event: telemetry\ndata: {}\n\n".format(
                json.dumps(ev, ensure_ascii=False, default=str)
            )
        yield "event: status\ndata: {}\n\n".format(
            json.dumps({"status": self.status, "errored": self.error_type is not None})
        )


class SpringManager:
    """In-memory registry of SiltSpring jobs (session-scoped)."""

    def __init__(self) -> None:
        self._jobs: Dict[str, SpringJob] = {}
        self._lock = threading.Lock()

    def create(self, config: Dict[str, Any], workspace_root: Path) -> SpringJob:
        job = SpringJob(config, workspace_root)
        with self._lock:
            self._jobs[job.job_id] = job
        job.start()
        return job

    def get(self, job_id: str) -> SpringJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError("unknown spring job '{}'".format(job_id))
        return job

    def listing(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in
                sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]