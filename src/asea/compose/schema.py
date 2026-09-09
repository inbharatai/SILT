"""Version-one closed composition schemas. No executable extension hooks."""
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator, model_serializer


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def strict_version(cls, value):
        if isinstance(value, dict) and "schema_version" in value and type(value["schema_version"]) is not int:
            raise ValueError("schema_version must be an integer, not a boolean or coerced number")
        return value


Risk = Literal["low", "medium", "high"]
Kind = Literal["hf_text", "hf_asr", "hf_tts", "hf_vision", "fixture_text"]
RISK = {"low": 0, "medium": 1, "high": 2}


def fingerprint(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Provenance(StrictModel):
    source: StrictStr = Field(min_length=1, max_length=2048)
    risk: Risk
    license: StrictStr = Field(min_length=1, max_length=200)


class ComponentManifest(StrictModel):
    schema_version: Literal[1] = 1
    kind: Kind
    model_path: StrictStr = Field(min_length=1, max_length=4096)
    task: Literal["causal", "seq2seq", "whisper", "vits", "smolvlm", "fixture"]
    risk: Risk
    provenance: List[Provenance] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def matched_kind(self):
        supported = {"hf_text": {"causal", "seq2seq"}, "hf_asr": {"whisper"}, "hf_tts": {"vits"}, "hf_vision": {"smolvlm"}, "fixture_text": {"fixture"}}
        if self.task not in supported[self.kind]:
            raise ValueError("component kind/task mismatch")
        return self

    @property
    def effective_risk(self):
        return max([self.risk] + [p.risk for p in self.provenance], key=RISK.get)


class OutputContract(StrictModel):
    """Explicit machine-format instruction; never inferred from task prose."""
    schema_version: Literal[1] = 1
    format: Literal["raw_python", "fenced_python"]


def dump_spec(value):
    """Canonical compatible spec/node dump, retaining every legacy field."""
    return value.model_dump(mode="json")


class Node(StrictModel):
    id: StrictStr = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    component: ComponentManifest
    input: StrictStr = Field(min_length=1, max_length=64)
    prompt_prefix: StrictStr = Field(default="", max_length=8192)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    use_cache: bool = False
    output_contract: Optional[OutputContract] = None
    cache_policy: Optional[Literal["model_default", "enabled", "disabled"]] = None

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        value = handler(self)
        # Do not use exclude_none/defaults: old null/default fields remain bound.
        for key in ("output_contract", "cache_policy"):
            if getattr(self, key) is None:
                value.pop(key, None)
        return value

    @model_validator(mode="after")
    def generation_settings(self):
        from .generation_contract import preflight_node
        preflight_node(self)
        return self


class Limits(StrictModel):
    max_input_chars: int = Field(default=8192, ge=1, le=65536)
    max_output_chars: int = Field(default=16384, ge=1, le=65536)
    max_audio_seconds: int = Field(default=30, ge=1, le=120)
    max_input_bytes: int = Field(default=16777216, ge=1, le=33554432)
    max_seconds: float = Field(default=300.0, gt=0, le=3600)
    max_peak_rss_mb: float = Field(default=3800.0, gt=0, le=65536)
    threads: int = Field(default=2, ge=1, le=2)


class SelectionBudgets(StrictModel):
    max_wall_seconds: float = Field(gt=0)
    max_peak_rss_mb: float = Field(gt=0)
    max_disk_bytes: Optional[int] = Field(default=None, gt=0)


PORTS = {"hf_text": ("text", "text"), "hf_asr": ("audio", "text"), "hf_tts": ("text", "audio"), "hf_vision": ("image", "text"), "fixture_text": ("text", "text")}


class CompositionSpec(StrictModel):
    schema_version: Literal[1] = 1
    name: StrictStr = Field(min_length=1, max_length=128)
    input_type: Literal["text", "audio", "image"]
    nodes: List[Node] = Field(min_length=1, max_length=16)
    output_node: StrictStr
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def validate_graph(self):
        by_id = {n.id: n for n in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("duplicate node IDs")
        if self.output_node not in by_id:
            raise ValueError("unknown output node")
        for node in self.nodes:
            if node.input != "$input" and node.input not in by_id:
                raise ValueError("unknown edge source: " + node.input)
            source_type = self.input_type if node.input == "$input" else PORTS[by_id[node.input].component.kind][1]
            if source_type != PORTS[node.component.kind][0]:
                raise ValueError("edge type mismatch for " + node.id)
        self.topological()
        # Every node must contribute to the selected output; dead branches waste resources.
        seen = set()
        current = self.output_node
        while current != "$input":
            seen.add(current)
            current = by_id[current].input
        if seen != set(by_id):
            raise ValueError("all nodes must contribute to output (v1 single-input nodes)")
        return self

    def topological(self):
        result, pending, seen = [], list(self.nodes), {"$input"}
        while pending:
            ready = [node for node in pending if node.input in seen]
            if not ready:
                raise ValueError("cycle detected")
            for node in ready:
                result.append(node)
                seen.add(node.id)
                pending.remove(node)
        return result

    @property
    def effective_risk(self):
        return max((n.component.effective_risk for n in self.nodes), key=RISK.get)


class EvaluationCase(StrictModel):
    id: StrictStr = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    group: Literal["target", "control"]
    input: Optional[StrictStr] = Field(default=None, max_length=65536)
    input_file: Optional[StrictStr] = Field(default=None, max_length=4096)
    reference: StrictStr = Field(min_length=1, max_length=65536)
    metric: Literal["text_exact", "text_similarity_proxy", "word_error_rate", "functional_code", "function_io"]
    threshold: float = Field(ge=0, le=1)
    function_cases: Optional[List[Dict[str, Any]]] = None
    output_format: Optional[Literal["raw_python", "fenced_python"]] = None

    @model_serializer(mode="wrap")
    def compatible_case_dump(self, handler):
        value = handler(self)
        # Preserve legacy null/default fields and suite hashes; only the new
        # opt-in declaration disappears when absent.
        if self.output_format is None:
            value.pop("output_format", None)
        return value

    @model_validator(mode="before")
    @classmethod
    def validate_function_data(cls, value):
        # Validate ORIGINAL objects with the oracle's exact-builtin validator,
        # before Pydantic can normalize containers or a candidate can execute.
        if isinstance(value, dict):
            cases = value.get("function_cases")
            if value.get("metric") == "function_io":
                from asea.certification.function_oracle import _suite
                from asea.certification.sandbox import SandboxLimits
                _suite("", cases, SandboxLimits())
            elif cases is not None:
                raise ValueError("function_cases is only valid for function_io")
        return value

    @model_validator(mode="after")
    def input_present(self):
        # A question plus file is valid only for an image graph. The evaluator
        # must check the graph type; a standalone case cannot infer it from a suffix.
        if self.input is None and self.input_file is None:
            raise ValueError("provide input and/or input_file")
        if (not self.reference.strip()
                or any(value is not None and not value.strip() for value in (self.input, self.input_file))):
            raise ValueError("nonempty input and reference required")
        if self.metric in {"text_exact", "functional_code", "function_io"} and self.threshold != 1.0:
            raise ValueError(self.metric + " requires threshold 1.0")
        if self.metric == "text_similarity_proxy" and self.threshold < 0.8:
            raise ValueError("text_similarity_proxy requires threshold >= 0.8")
        if self.metric == "word_error_rate" and self.threshold > 0.5:
            raise ValueError("word_error_rate requires threshold <= 0.5")
        return self

    def validate_input_type(self, spec):
        """Preserve text/audio exclusivity; only images allow a question + file."""
        if spec.input_type == "text" and (self.input is None or self.input_file is not None):
            raise ValueError("text evaluation requires input and no input_file")
        if spec.input_type == "audio" and (self.input_file is None or self.input is not None):
            raise ValueError("audio evaluation requires input_file and no input")
        if spec.input_type == "image" and self.input_file is None:
            raise ValueError("image evaluation requires input_file and an optional question")
        if self.input is not None and len(self.input) > spec.limits.max_input_chars:
            raise ValueError("evaluation input exceeds graph character limit")
        return self


class EvaluationSuite(StrictModel):
    schema_version: Literal[1] = 1
    name: StrictStr = Field(min_length=1, max_length=128)
    reference_source: StrictStr = Field(min_length=1, max_length=2048)
    policy_version: Literal["composition-admission-v2"] = "composition-admission-v2"
    claims: List[Literal["reference_text", "coding", "safety"]] = Field(default_factory=lambda: ["reference_text"], min_length=1, max_length=3)
    cases: List[EvaluationCase] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def coverage(self):
        if not self.name.strip() or not self.reference_source.strip():
            raise ValueError("nonempty suite name and reference source required")
        if len(set(self.claims)) != len(self.claims):
            raise ValueError("duplicate claims")
        if {case.group for case in self.cases} != {"target", "control"}:
            raise ValueError("nonempty targets and controls required")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate case IDs")
        return self


def load_json(path, model):
    from asea.artifacts import safe_file
    path = safe_file(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("JSON document exceeds 2 MiB")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))
    return model.model_validate(value)
