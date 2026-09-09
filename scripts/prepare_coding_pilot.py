"""Build the frozen, adapted public coding pilot; NEVER execute upstream Python.

Only this program's own code runs. Upstream solutions/tests remain inert strings.
Use --fetch for pinned HTTPS acquisition, then --check for reproducibility.
"""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "pilot-v2"
SEED = "silt-adapted-coding-pilot-v2-2026-07-13-before-generation"
HE_PIN = "6d43fb980f9fee3c892a914eda09951f772ad10d"
MBPP_PIN = "932d4685e23f671b9e8c2abc72dd228ba5ff9252"
HF_PIN = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
RAW = "https://raw.githubusercontent.com/"
SOURCES = {
    "HumanEval.jsonl.gz": (RAW + "openai/human-eval/" + HE_PIN + "/data/HumanEval.jsonl.gz", "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"),
    "HumanEval-LICENSE.txt": (RAW + "openai/human-eval/" + HE_PIN + "/LICENSE", "bcba3de214851cce46ed5af42d6698044616eeace887c3231bc7a20474ab639e"),
    "mbpp-sanitized.json": (RAW + "google-research/google-research/" + MBPP_PIN + "/mbpp/sanitized-mbpp.json", "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"),
    "mbpp-README.md": (RAW + "google-research/google-research/" + MBPP_PIN + "/mbpp/README.md", "02d63a4ffdad806c3c87b5ea2aa91e946f6eec0c2869643b09831650883f41be"),
    "mbpp-LICENSE-evidence.md": ("https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/" + HF_PIN + "/README.md", "6377d5c76ba46b9e650daa6d5eb592e671c9b15586e39f23f50ed9bf2ac54cf6"),
    "CC-BY-4.0.txt": ("https://creativecommons.org/licenses/by/4.0/legalcode.txt", "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"),
}
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TEXT = 65536
MAX_NODES = 4096
MAX_DEPTH = 8
MAX_ELEMENTS = 128
MAX_CASES = 8
OLD_TASKS = {
    "add": "add(a, b): sum of two values",
    "square": "square(x): x multiplied by itself",
    "even": "is_even(n): even integer predicate",
    "reverse": "reverse_text(s): reverse a string",
    "maximum": "larger(a, b): larger of two values",
    "absolute": "absolute_value(n): absolute value",
}


class Unsupported(ValueError):
    """An upstream construct is outside the deliberately narrow grammar."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def parse(source):
    if not isinstance(source, str) or len(source) > MAX_TEXT:
        raise Unsupported("source_size")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise Unsupported("syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise Unsupported("ast_size")
    return tree


def literal(node, depth=0, budget=None):
    """Decode a bounded JSON literal without eval, literal_eval, or compilation."""
    if budget is None:
        budget = [0]
    budget[0] += 1
    if depth > MAX_DEPTH or budget[0] > MAX_NODES:
        raise Unsupported("literal_budget")
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or type(value) is bool:
            return value
        if type(value) is int and abs(value) <= 2**53 - 1:
            return value
        if type(value) is str and len(value) <= 4096:
            return value
        raise Unsupported("literal_type_or_size")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        if not isinstance(node.operand, ast.Constant) or type(node.operand.value) is not int:
            raise Unsupported("only_unary_integer")
        value = literal(node.operand, depth + 1, budget)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.List):
        if len(node.elts) > MAX_ELEMENTS:
            raise Unsupported("list_size")
        return [literal(x, depth + 1, budget) for x in node.elts]
    if isinstance(node, ast.Dict):
        if len(node.keys) > MAX_ELEMENTS:
            raise Unsupported("dict_size")
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise Unsupported("dict_unpacking")
            key = literal(key, depth + 1, budget)
            if type(key) is not str or key in result:
                raise Unsupported("dict_key")
            result[key] = literal(value, depth + 1, budget)
        return result
    raise Unsupported("nonliteral_" + type(node).__name__)


def assertion(node, entrypoint, alias=None):
    """Accept only assert entrypoint(JSON positional args) == JSON expected."""
    if not isinstance(node, ast.Assert) or node.msg is not None:
        raise Unsupported("not_simple_assert")
    comparison = node.test
    if (not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1
            or not isinstance(comparison.ops[0], ast.Eq) or len(comparison.comparators) != 1):
        raise Unsupported("not_single_equality")
    call = comparison.left
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != (alias or entrypoint)):
        raise Unsupported("not_direct_entrypoint_call")
    if call.keywords or len(call.args) > 16:
        raise Unsupported("kwargs_or_argument_count")
    budget = [0]
    return {"function": entrypoint, "args": [literal(x, budget=budget) for x in call.args],
            "kwargs": {}, "expected": literal(comparison.comparators[0], budget=budget)}


def parse_assertion(source, entrypoint, alias=None):
    tree = parse(source)
    if len(tree.body) != 1:
        raise Unsupported("multiple_statements")
    return assertion(tree.body[0], entrypoint, alias)


def old_family(prompt, entrypoint):
    """Conservative exclusions for the six already-exposed diagnostic families."""
    text = (entrypoint.replace("_", " ") + " " + prompt).lower()
    patterns = {
        "add": r"\b(add|sum)\b.{0,50}\b(two|both|a and b)\b|\badd\(a, b\)",
        "square": r"\bsquare\b.{0,25}\b(number|integer|x)\b|multiplied by itself",
        "even": r"\b(is even|even number|even integer|number is even)\b",
        "reverse": r"\brevers\w*\b.{0,30}\b(string|text)\b|\b(string|text)\b.{0,30}\brevers\w*\b",
        "maximum": r"\b(larger|largest|maximum|greater)\b.{0,25}\b(two|a and b)\b",
        "absolute": r"\babsolute\b",
    }
    return next((key for key, pattern in patterns.items() if re.search(pattern, text)), None)


def get_tasks(blobs):
    with gzip.GzipFile(fileobj=io.BytesIO(blobs["HumanEval.jsonl.gz"])) as handle:
        data = handle.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise Unsupported("decompressed_source_size")
    human = [json.loads(line) for line in data.decode().splitlines()]
    mbpp = json.loads(blobs["mbpp-sanitized.json"].decode())
    return [("HumanEval", row) for row in human] + [("MBPP", row) for row in mbpp]


def convert(source, row):
    """Read tests and solution signature as AST, never execute either."""
    prompt = row["prompt"]
    task_id = str(row["task_id"])
    audit = {"source": source, "source_task_id": task_id, "source_record_sha256": digest(canonical(row)),
             "rejected_test_constructs": [], "ignored_test_statements": []}
    try:
        if source == "HumanEval":
            entrypoint = row["entry_point"]
            functions = [x for x in parse(prompt).body if isinstance(x, ast.FunctionDef)]
            tree = parse(row["test"])
            checks = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "check"]
            if len(checks) != 1:
                raise Unsupported("check_function")
            # Only direct statements: loops/conditionals/assignments are not evaluated or expanded.
            nodes = checks[0].body
            alias = "candidate"
        else:
            if row.get("test_imports"):
                raise Unsupported("requires_test_imports")
            functions = [x for x in parse(row["code"]).body if isinstance(x, ast.FunctionDef)]
            if len(functions) != 1:
                raise Unsupported("requires_single_function")
            entrypoint = functions[0].name
            nodes = []
            for test in row["test_list"]:
                tree = parse(test)
                if len(tree.body) != 1:
                    raise Unsupported("multiple_test_statements")
                nodes.append(tree.body[0])
            alias = None
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", entrypoint):
            raise Unsupported("entrypoint_name")
        matches = [x for x in functions if x.name == entrypoint]
        if len(matches) != 1:
            raise Unsupported("entrypoint_signature")
        function = matches[0]
        if function.decorator_list or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
            raise Unsupported("complex_signature")
        positional = function.args.posonlyargs + function.args.args
        min_args = len(positional) - len(function.args.defaults)
        cases, seen = [], set()
        for index, node in enumerate(nodes):
            if not isinstance(node, ast.Assert):
                audit["ignored_test_statements"].append({"index": index, "type": type(node).__name__})
                continue
            try:
                case = assertion(node, entrypoint, alias)
                if not min_args <= len(case["args"]) <= len(positional):
                    raise Unsupported("arity")
                key = canonical(case)
                if key in seen:
                    audit["rejected_test_constructs"].append({"index": index, "reason": "duplicate_literal_case"})
                    continue
                seen.add(key)
                case["id"] = "io_" + str(index + 1)
                case["upstream_assertion"] = ast.get_source_segment(row["test"], node) if source == "HumanEval" else row["test_list"][index]
                cases.append(case)
            except Unsupported as exc:
                audit["rejected_test_constructs"].append({"index": index, "reason": str(exc)})
        inputs = {}
        for case in cases:
            key = canonical([case["args"], case["kwargs"]])
            expected = canonical(case["expected"])
            if key in inputs and inputs[key] != expected:
                raise Unsupported("conflicting_expected_values")
            inputs[key] = expected
        audit.update(entrypoint=entrypoint, literal_case_count=len(cases),
                     top_level_test_statement_count=len(nodes),
                     direct_assertion_count=sum(isinstance(n, ast.Assert) for n in nodes),
                     selected_case_cap=MAX_CASES)
        if len(cases) < 2:
            raise Unsupported("fewer_than_two_literal_cases")
        if len({canonical(c["expected"]) for c in cases[:MAX_CASES]}) < 2:
            raise Unsupported("retained_cases_lack_expected_value_diversity")
        old = old_family(prompt, entrypoint)
        if old:
            audit["old_diagnostic_family"] = old
            raise Unsupported("excluded_existing_development_family")
        # A group is a semantic I/O domain, never a source identity or a fake wrong answer.
        structured = any(isinstance(arg, (str, list, dict)) for case in cases for arg in case["args"])
        group = "target" if structured else "control"
        task = {"id": ("he_" + task_id.split("/")[-1]) if source == "HumanEval" else "mbpp_" + task_id,
                "source": source, "source_task_id": task_id, "entrypoint": entrypoint,
                "original_prompt": prompt, "function_parameters": [x.arg for x in positional],
                "source_record_sha256": audit["source_record_sha256"], "group": group,
                "domain": "text_and_collection_transforms" if structured else "numeric_scalar_algorithms",
                "function_cases": cases[:MAX_CASES], "available_literal_cases": len(cases),
                "license": "MIT" if source == "HumanEval" else "CC-BY-4.0"}
        audit["status"] = "eligible"
        return task, audit
    except Unsupported as exc:
        audit.update(status="excluded", reason=str(exc))
        return None, audit


STOPWORDS = set("a an the of to in on for from with given write function that which and or is are be by as this it returns return find check python def list string number numbers integer integers".split())


def prompt_tokens(task):
    # Ignore doctest examples and signatures when comparing description families.
    prompt = task["original_prompt"].split(">>>")[0]
    prompt = re.sub(r"^.*\b(def|from|import)\b.*$", "", prompt, flags=re.M)
    return set(re.findall(r"[a-z]+", prompt.lower())) - STOPWORDS


def family_clusters(tasks):
    """Connected components: matching entrypoints, normalized prompts, or close descriptions.

    This is a documented operational family definition, not a claim of semantic dedup perfection.
    No detected family may cross development/final or target/control boundaries.
    """
    parent = list(range(len(tasks)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tokens = [prompt_tokens(t) for t in tasks]
    for i, a in enumerate(tasks):
        for j in range(i):
            b = tasks[j]
            union = tokens[i] | tokens[j]
            similarity = len(tokens[i] & tokens[j]) / len(union) if union else 0
            if a["entrypoint"].lower().replace("_", "") == b["entrypoint"].lower().replace("_", "") or similarity >= 0.72:
                parent[root(i)] = root(j)
    clusters = defaultdict(list)
    for i, task in enumerate(tasks):
        clusters[root(i)].append(task)
    result = []
    for values in clusters.values():
        ids = sorted(t["id"] for t in values)
        family = "family_" + digest(canonical(ids))[:16]
        for task in values:
            task["family"] = family
        result.append(values)
    return result


def rank(task, purpose="selection"):
    return digest((SEED + ":" + purpose + ":" + task["id"]).encode())


def select(tasks):
    clusters = family_clusters(tasks)
    # One representative per detected family prevents repeats across all four cells.
    representatives = [min(values, key=rank) for values in clusters]
    duplicate_exclusions = [{"excluded": t["id"], "retained_representative": min(values, key=rank)["id"], "family": t["family"]}
                            for values in clusters for t in values if t is not min(values, key=rank)]
    selected = []
    for source in ("HumanEval", "MBPP"):
        for group, count, dev_count in (("target", 12, 3), ("control", 4, 1)):
            pool = sorted((t for t in representatives if t["source"] == source and t["group"] == group), key=rank)
            chosen = pool[:count]
            for index, task in enumerate(sorted(chosen, key=lambda t: rank(t, "split"))):
                task["split"] = "development" if index < dev_count else "final"
                selected.append(task)
    selected.sort(key=lambda t: (t["split"], t["group"], t["source"], rank(t)))
    return selected, duplicate_exclusions, representatives


def suite(tasks, split):
    cases = []
    for task in tasks:
        if task["split"] != split:
            continue
        instructions = ("Write only Python source code implementing the requested function. Do not include markdown, explanations, example calls, print statements, or tests. "
                        "Preserve the function name and parameter order. Return the requested result without external I/O or persistent state. "
                        "Include any necessary standard-library imports.\n"
                        "Required entrypoint: " + task["entrypoint"] + "(" + ", ".join(task["function_parameters"]) + ").\n\n"
                        "Original upstream prompt (verbatim):\n" + task["original_prompt"])
        cases.append({"id": task["id"], "group": task["group"], "input": instructions,
                      "reference": "Adapted literal I/O assertions from " + task["source"] + " task " + task["source_task_id"] + "; " + task["license"] + ". Descriptive reference only; see provenance.json and ATTRIBUTION.md. Not the full upstream test suite.",
                      "metric": "function_io", "threshold": 1.0,
                      "function_cases": [{k: value for k, value in case.items() if k != "upstream_assertion"} for case in task["function_cases"]]})
    return {"schema_version": 1, "name": "adapted-public-coding-pilot-v2-" + split,
            "reference_source": "Pinned OpenAI HumanEval (MIT) and Google Research sanitized MBPP (CC-BY-4.0); bounded AST-decoded literal assertions, not official full benchmark scores. See manifest.json and ATTRIBUTION.md.",
            "policy_version": "composition-admission-v2", "claims": ["coding"], "cases": cases}


ATTRIBUTION = """# Third-party attribution and modifications

## HumanEval — OpenAI
Copyright (c) OpenAI (https://openai.com). HumanEval, released with Chen et al.,
*Evaluating Large Language Models Trained on Code* (2021), https://arxiv.org/abs/2107.03374.
Repository: https://github.com/openai/human-eval . Licensed under MIT. The complete
original copyright, permission notice and warranty disclaimer are preserved in
`sources/HumanEval-LICENSE.txt`; redistribute that notice with derived tasks.

## Mostly Basic Python Problems (MBPP) — Google Research
Austin, Jacob; Odena, Augustus; Nye, Maxwell; Bosma, Maarten; Michalewski, Henryk;
Dohan, David; Jiang, Ellen; Cai, Carrie; Terry, Michael; Le, Quoc; and others,
*Program Synthesis with Large Language Models* (2021), https://arxiv.org/abs/2108.07732.
Repository: https://github.com/google-research/google-research/tree/master/mbpp .
Google Research's dataset card expressly specifies CC-BY-4.0 at
https://huggingface.co/datasets/google-research-datasets/mbpp . Its pinned card is
preserved as `sources/mbpp-LICENSE-evidence.md`; the GitHub subdirectory README
itself has no dataset license declaration. We do NOT substitute the repository's
Apache software license for the dataset's explicit CC-BY-4.0 declaration.
License and warranty disclaimer: https://creativecommons.org/licenses/by/4.0/ ;
full legal terms in `sources/CC-BY-4.0.txt`. Attribution, supplied notices,
license links, and indication of changes must travel with redistributed material.
No additional rights restrictions are imposed on these licensed data. No endorsement
by OpenAI, Google Research or Creative Commons is asserted.

## Modifications (SILT adapted public coding pilot v2)
Original source bytes, prompts, source IDs and entrypoint names are preserved.
A supported subset of direct literal assertions is converted into JSON-data
function arguments and expected values. At most eight unique literal cases per
task are retained in source order; unsupported tests are omitted and audited.
Code-only response instructions are appended around the unchanged original prompt.
Operational duplicate families, domain labels, seeded selection and new
8-development/24-final splits are added. These are NOT official HumanEval, MBPP,
EvalPlus, pass@k, or full-suite scores. No new answers or reference solutions have
been authored, substituted, executed, or model-generated by this builder.

The source snapshots may contain reference solution text; they are inert archival
data, never used as generated candidate outputs. Model inputs contain original
prompts only (including any original prompt examples), not hidden adapted cases
or reference solution bodies. Pretraining/public exposure is UNKNOWN: public
benchmark data cannot establish contamination-free held-out performance.
"""


def read_sources(output, fetch=False):
    directory = output / "sources"
    if fetch:
        directory.mkdir(parents=True, exist_ok=True)
    blobs = {}
    for name, (url, expected) in SOURCES.items():
        path = directory / name
        if not path.exists() and fetch:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SILT-licensed-pilot/2"})
            with urllib.request.urlopen(request, timeout=45) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("non-HTTPS redirect")
                content = response.read(MAX_SOURCE_BYTES + 1)
            if len(content) > MAX_SOURCE_BYTES or digest(content) != expected:
                raise ValueError("source hash/size mismatch; stop, do not substitute: " + name)
            path.write_bytes(content)
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("source size exceeded")
        content = path.read_bytes()
        if digest(content) != expected:
            raise ValueError("source hash mismatch; stop, do not substitute: " + name)
        blobs[name] = content
    if b"CC-BY-4.0" not in blobs["mbpp-LICENSE-evidence.md"] or b"Copyright (c) OpenAI" not in blobs["HumanEval-LICENSE.txt"]:
        raise ValueError("license evidence unresolved; stop and ask the operator")
    return blobs


def render(output, fetch=False):
    blobs = read_sources(output, fetch)
    converted = [convert(source, row) for source, row in get_tasks(blobs)]
    tasks = [task for task, _ in converted if task]
    audits = [audit for _, audit in converted]
    selected, duplicates, representatives = select(tasks)
    files = {"development.json": suite(selected, "development"), "final.json": suite(selected, "final"),
             "provenance.json": {"tasks": selected}, "eligibility-audit.json": {"tasks": audits, "duplicate_exclusions": duplicates},
             "code-function-metadata.json": {"note": "Entrypoint metadata for the generated Code artifact; never hardcode a generic solve name.",
                 "tasks": [{"id": t["id"], "language": "python", "function": t["entrypoint"], "parameters": t["function_parameters"], "split": t["split"]} for t in selected]}}
    selection = [{k: t[k] for k in ("id", "source", "source_task_id", "source_record_sha256", "family", "split", "group")} for t in selected]
    protocol = {"version": "adapted-public-coding-pilot-v2", "seed": SEED, "selection": selection,
                "requested_total": 32, "requested_development": 8, "requested_final": 24,
                "requested_target": 24, "requested_control": 8,
                "per_source_quota": {"target": 12, "control": 4, "development_target": 3, "development_control": 1},
                "ranking": "ascending SHA256(seed + ':' + purpose + ':' + task_id); purpose selection or split",
                "families": "connected components: identical normalized entrypoint OR description token Jaccard >= 0.72; one representative per family; heuristic, not complete semantic deduplication",
                "grouping": "target=text/collection inputs; control=numeric/scalar-only inputs; both sources in each domain; all real coding tasks with real positive oracle values",
                "frozen_before_generation": True, "model_outputs_consulted": False,
                "existing_diagnostic_tasks_excluded": OLD_TASKS,
                "upstream_split_policy": "All pinned HumanEval and sanitized MBPP rows considered; original MBPP splits NOT treated as unseen; our new splits are local pilot governance only.",
                "existing_diagnostic_source": {"path": "scripts/setup_local_validation.py", "sha256_at_protocol_definition": "fe77062c70a14dcedec0a375792b536429ba79e632fa581f21b8185fc327ee70", "lines": "37-43"},
                "literal_types": "null, bool, bounded integer (optional unary +/-), string, list, string-keyed dictionary; floats and tuples rejected",
                "expected_diversity": "at least two distinct expected JSON values among retained cases; no conflicting expected values for identical inputs",
                "maximum_literal_cases_per_task": MAX_CASES, "minimum_literal_cases_per_task": 2}
    files["selection-lock.json"] = protocol
    data = {name: json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n" for name, value in files.items()}
    data["ATTRIBUTION.md"] = ATTRIBUTION.encode()
    counts = {"upstream_total": len(converted), "upstream_by_source": dict(Counter(a["source"] for a in audits)),
              "eligible_before_duplicate_filter": len(tasks), "eligible_by_source": dict(Counter(t["source"] for t in tasks)),
              "eligible_family_representatives": len(representatives), "duplicate_excluded": len(duplicates),
              "unsupported_or_old_family_excluded": len(converted) - len(tasks),
              "eligible_not_selected": len(representatives) - len(selected), "selected": len(selected),
              "selected_by_source": dict(Counter(t["source"] for t in selected)),
              "selected_by_split_group_source": dict(Counter(t["split"] + "/" + t["group"] + "/" + t["source"] for t in selected)),
              "adapted_function_cases": sum(len(t["function_cases"]) for t in selected),
              "exclusion_reasons": dict(Counter(a.get("reason") for a in audits if a["status"] == "excluded"))}
    manifest = {"version": 2, "scope": "ADAPTED PUBLIC BENCHMARK PILOT; NOT official full HumanEval/MBPP/EvalPlus score",
                "pretraining_exposure": "unknown; no contamination-free claim", "license_status": "verified_from_preserved_primary_repository_and_publisher_dataset_card", "license_unresolved": [],
                "source_pins": {"HumanEval": HE_PIN, "MBPP_GitHub": MBPP_PIN, "MBPP_publisher_license_card": HF_PIN},
                "sources": [{"file": "sources/" + name, "url": url, "sha256": expected, "bytes": len(blobs[name])} for name, (url, expected) in SOURCES.items()],
                "selection_sha256": digest(canonical(selection)), "selection_lock_sha256": digest(data["selection-lock.json"]),
                "builder_sha256": digest(Path(__file__).read_bytes()), "counts": counts,
                "artifact_sha256": {name: digest(content) for name, content in data.items()},
                "governance": "Frozen before model generation. Do not tune on final. Development may be run first. Retain failed/time-limited results. Any fixture/selection change requires a new version and new lock; do not overwrite v2."}
    data["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode() + b"\n"
    return data, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fetch", action="store_true", help="Fetch missing pinned source bytes via HTTPS; verify hashes")
    parser.add_argument("--check", action="store_true", help="Verify source hashes and byte-for-byte fixture reproducibility; write nothing")
    args = parser.parse_args()
    if args.check and args.fetch:
        parser.error("--check is offline and read-only; cannot combine --fetch")
    data, manifest = render(args.output, args.fetch)
    for name, content in data.items():
        path = args.output / name
        if path.exists() and path.read_bytes() != content:
            raise SystemExit("Frozen fixture differs: " + name + "; stop; create a new version rather than overwrite")
        if args.check and not path.exists():
            raise SystemExit("Missing fixture: " + name)
    if not args.check:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in data.items():
            path = args.output / name
            if not path.exists():
                path.write_bytes(content)
    print(json.dumps({"output": str(args.output), "selection_sha256": manifest["selection_sha256"], "counts": manifest["counts"], "model_runs": 0}, indent=2))


if __name__ == "__main__":
    main()
