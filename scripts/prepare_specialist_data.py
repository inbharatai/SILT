"""Offline specialist-v1 data curator. Parses public reference code; NEVER executes it.

Only writes data/specialist-v1 (or an explicitly requested owned output). No model
weights, generation, benchmark runs, network acquisition, or upstream mutations.
"""
import argparse
import ast
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/specialist-v1"
DEFAULT_TOKENIZER = Path("/agent/workspace/silt-models/Qwen2.5-Coder-0.5B-Instruct")
SEED = "silt-specialist-v1-pure-sequences-20260908-before-any-generation"
SCHEMA = "silt.specialist.supervised.v1"
MAX_TOKENS = 256
MAX_RESPONSE_TOKENS = 192
PRIOR_LOCK_SHA256 = "3f9a6326dcfc417956c7db0a378f429ef3d097110e03324027d387b0d49d802a"
TOKENIZER_SHA256 = {
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
}
# Exactly eight basic scalar controls overall: two train, two validation, four final.
QUOTAS = {"validation": {"target": 6, "control": 2},
          "final": {"target": 12, "control": 4},
          "train": {"target": 62, "control": 2}}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
ADVANCED = re.compile(r"\b(prime|primes|factorial|fibonacci|tribonacci|lucas|polynomial|derivative|integral|logarithm|logarithmic|trigonometric|triangle|triangular|rectangle|rectangular|circle|circular|cylinder|sphere|spherical|parallelogram|rhombus|trapezium|ellipse|ellipsoid|area|perimeter|volume|surface|hypotenuse|pythagorean|quadratic|roots|root|divisor|divisors|gcd|lcm|binomial|coefficient|coefficients|permutation|permutations|combinations|combinatorial|subsequence|subsequences|subarray|subarrays|submatrix|matrix|matrices|graph|graphs|tree|trees|palindrome|palindromes|parentheses|brackets|balanced|anagram|anagrams|roman|binary|hexadecimal|octal|bitwise|bits|xor|narcissistic|armstrong|perfect|abundant|amicable|catalan|bell|hamming|median|variance|deviation|probability|random|heap|heaps|dynamic|recursion|recursive|sorting network|expression evaluation)\b", re.I)
ADVANCED_EXTRA = re.compile(r"\b(octagonal|tetrahedral|hexagonal|decagonal|nonagonal|polygonal|star number|parabola|directrix|inversions|contiguous|unordered pairs|pairs whose sum|sum of digits equal|number of rotations|count of rotations)\b", re.I)
ALLOWED_IMPORTS = {"typing", "re", "collections", "itertools", "functools", "string"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "input", "print", "__import__", "globals", "locals", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit"}

_spec = importlib.util.spec_from_file_location("specialist_audited_pilot", ROOT / "scripts/prepare_coding_pilot.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def packed(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"


def rank(task, purpose="selection"):
    return sha((SEED + ":" + purpose + ":" + task["id"]).encode())


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def response_for(source, row):
    # HumanEval canonical_solution is an indented body; the original prompt is
    # its verbatim signature/docstring/import prefix, not a replacement solution.
    return row["prompt"] + row["canonical_solution"] if source == "HumanEval" else row["code"]


class AlphaNormalize(ast.NodeTransformer):
    """Inert AST fingerprint: remove docs and normalize identifiers, not constants."""
    def __init__(self):
        self.names = {}

    def ident(self, name):
        if name not in self.names:
            self.names[name] = "v" + str(len(self.names))
        return self.names[name]

    def visit_Name(self, node):
        node.id = self.ident(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.ident(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.name = self.ident(node.name)
        node.returns = None
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return self.generic_visit(node)


def fingerprints(prompt, response):
    tree = pilot.parse(response)
    return {"canonical_prompt_sha256": sha(normalize(prompt).encode()),
            "canonical_source_sha256": sha(ast.dump(tree, include_attributes=False).encode()),
            "alpha_source_sha256": sha(ast.dump(AlphaNormalize().visit(tree), include_attributes=False).encode()),
            "response_sha256": sha(response.encode())}


def metadata(source, row):
    response = response_for(source, row)
    funcs = [n for n in pilot.parse(response).body if isinstance(n, ast.FunctionDef)]
    entry = row["entry_point"] if source == "HumanEval" else (funcs[0].name if funcs else "")
    task = {"id": "he_" + str(row["task_id"]).split("/")[-1] if source == "HumanEval" else "mbpp_" + str(row["task_id"]),
            "source": source, "source_task_id": str(row["task_id"]), "entrypoint": entry,
            "original_prompt": row["prompt"], "response": response,
            "source_record_sha256": sha(canonical(row))}
    task.update(fingerprints(row["prompt"], response))
    return task


def semantic_tokens(task):
    tokens = pilot.prompt_tokens(task)
    # Small morphology normalization, not a trained model or a semantic guarantee.
    return {re.sub(r"(ing|ed|s)$", "", t) if len(t) > 5 else t for t in tokens}


def operation_tags(task):
    """Conservative source-description operation groups, frozen before generation.

    Merge obvious inverse/variant operations even when surface names differ.
    These tags are intentionally broader than exact duplicate equivalence.
    """
    text = normalize(task["original_prompt"].split(">>>")[0] + " " + task["entrypoint"])
    tags = set()
    def has(pattern):
        return bool(re.search(pattern, text))
    sequence = has(r"\b(list|lists|array|arrays|sequence|elements)\b")
    if sequence and has(r"\b(odd|even)\b") and has(r"\b(remove|filter|returns|return|ones)\b") and not has(r"\b(first|index|indices|positions|sum|product)\b"):
        tags.add("sequence_parity_filter")
    if sequence and has(r"\b(duplicate|unique|distinct)\b") and has(r"\b(check|whether|contains|contain|true|false)\b"):
        tags.add("sequence_duplicate_predicate")
    if sequence and has(r"\b(common|overlapping|any value|at least one)\b") and not has(r"\b(three|3|same index)\b"):
        tags.add("sequence_overlap")
    if sequence and has(r"\b(smallest|largest|minimum|maximum)\b") and not has(r"\b(frequency|sum|difference|heterogeneous|negative|second|kth)\b"):
        tags.add("sequence_extreme")
    if has(r"\b(convert|toggle|flip)\b") and has(r"\b(upper|lower|uppercase|lowercase|case)\b") and not has(r"\b(camel|snake)\b"):
        tags.add("text_case_conversion")
    if has(r"\b(remove|strip)\b") and has(r"\b(uppercase|lowercase)\b"):
        tags.add("text_case_removal")
    if has(r"\b(replace|remove)\b") and has(r"\b(spaces|whitespaces|blank)\b"):
        tags.add("text_whitespace_replacement")
    if sequence and has(r"\b(rotate|rotation|rotations|interchange|split arr)\b"):
        tags.add("sequence_rotation_or_end_swap")
    if sequence and has(r"\b(largest|smallest)\b") and has(r"\b(sum|difference)\b") and has(r"\b(largest)\b") and has(r"\b(smallest)\b"):
        tags.add("sequence_extrema_reduction")
    if has(r"\b(three|3)\b") and sequence and has(r"\b(same position|same index|common elements|identical)\b"):
        tags.add("three_sequence_aligned_overlap")
    if has(r"\b(words|strings)\b") and has(r"\b(longer|length|size|characters)\b") and has(r"\b(find|extract|remove)\b") and not has(r"\b(longest|sum|total)\b"):
        tags.add("text_length_filter")
    if sequence and has(r"\b(sorted|monotonically|monotonic)\b") and has(r"\b(check|true|return)\b"):
        tags.add("sequence_sorted_predicate")
    if has(r"\b(split|separated|delimited)\b") and has(r"\b(string|words|strings)\b") and not has(r"\b(camel|snake)\b"):
        tags.add("text_split_words")
    if has(r"\b(strings|words|string values)\b") and has(r"\b(substring|prefix|start)\b") and has(r"\b(list|filter)\b"):
        tags.add("text_list_content_filter")
    if has(r"\ba\b.*\bfollowed\b.*\bb\b") or has(r"\btext match (one|two|three|two three)\b"):
        tags.add("regex_a_followed_by_b")
    if has(r"\b(word|words|string|strings)\b") and has(r"\bz\b") and has(r"\b(matches|containing|contains|match)\b"):
        tags.add("regex_word_contains_z")
    if sequence and has(r"\b(difference between two lists|intersection|remove all elements)\b"):
        tags.add("sequence_set_operations")
    if sequence and has(r"\b(remove)\b") and has(r"\b(duplicate|duplicates)\b"):
        tags.add("sequence_duplicate_removal")
    if sequence and has(r"\b(unique|non repeated)\b") and has(r"\b(product|sum)\b"):
        tags.add("sequence_unique_reduction")
    if sequence and has(r"\b(square|squares|cube|cubes|power)\b") and has(r"\b(each|individual)\b"):
        tags.add("sequence_power_mapping")
    if has(r"\b(string)\b") and has(r"\b(integer|decimal)\b") and has(r"\b(check|whether|represents)\b"):
        tags.add("text_numeric_validation")
    return tags


def initial_family(task):
    old = pilot.old_family(task["original_prompt"], task["entrypoint"])
    if old:
        return old
    text = normalize(task["original_prompt"].split(">>>")[0] + " " + task["entrypoint"])
    if re.search(r"\b(reverse|reverses|reversing|reversal)\b", text):
        return "reverse_expanded_all_sequences"
    if re.search(r"\b(square|squares|squared)\b", text):
        return "square_expanded_sequence_mapping"
    if re.search(r"\b(minimum|maximum|smaller|larger|largest|smallest)\b.{0,35}\btwo\b", text):
        return "two_scalar_extreme_inverse_variant"
    if re.search(r"\b(number|integer)\b.*\beven\b", text) and not re.search(r"\b(list|array|indices|positions)\b", text):
        return "even_scalar_expanded"
    return None


def families(tasks):
    """Operational transitive families across ALL sources, including consumed IDs."""
    parent = list(range(len(tasks)))
    edges = []
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tokens = [semantic_tokens(t) for t in tasks]
    names = [normalize(t["entrypoint"]).replace(" ", "") for t in tasks]
    tags = [operation_tags(t) for t in tasks]
    for i, a in enumerate(tasks):
        for j in range(i):
            b = tasks[j]
            union = tokens[i] | tokens[j]
            similarity = len(tokens[i] & tokens[j]) / len(union) if union else 0
            reason = None
            for field in ("canonical_prompt_sha256", "canonical_source_sha256", "alpha_source_sha256", "source_record_sha256"):
                if a[field] == b[field]:
                    reason = field
                    break
            if names[i] and names[i] == names[j]:
                reason = "same_function_name"
            elif min(len(names[i]), len(names[j])) >= 6 and SequenceMatcher(None, names[i], names[j]).ratio() >= 0.86:
                reason = "near_function_name"
            if similarity >= 0.60:
                reason = "description_jaccard_ge_0.60"
            if tags[i] & tags[j]:
                reason = "operation_tag:" + ",".join(sorted(tags[i] & tags[j]))
            if reason:
                parent[root(i)] = root(j)
                edges.append({"a": a["id"], "b": b["id"], "reason": reason})
    clusters = defaultdict(list)
    for i, task in enumerate(tasks):
        clusters[root(i)].append(task)
    for cluster in clusters.values():
        family = "family_" + sha(canonical(sorted(t["id"] for t in cluster)))[:16]
        for task in cluster:
            task["family"] = family
    return list(clusters.values()), edges


def shape(value):
    if isinstance(value, list):
        return all(type(v) in (str, int, bool) or v is None for v in value)
    return type(value) in (str, int, bool) or value is None


def static_contract(task):
    """Broad source-only contract; never performance- or case-difficulty ranking."""
    description = task["original_prompt"].split(">>>")[0]
    if ADVANCED.search(description + " " + task["entrypoint"].replace("_", " ")) or ADVANCED_EXTRA.search(description + " " + task["entrypoint"].replace("_", " ")):
        return "out_of_scope_advanced_or_algorithmic"
    tree = pilot.parse(task["response"])
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != task["entrypoint"]:
        return "not_single_complete_function"
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            return "module_side_effect_or_global_state"
    nodes = list(ast.walk(tree))
    if len(nodes) > 220:
        return "reference_ast_over_220_nodes"
    loops = sum(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in nodes)
    if loops > 2:
        return "reference_over_two_loops"
    for n in nodes:
        if isinstance(n, (ast.AsyncFunctionDef, ast.ClassDef, ast.Try, ast.With, ast.Raise, ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await, ast.While)):
            return "reference_outside_simple_stateless_grammar"
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            modules = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else [n.module.split(".")[0] if n.module else ""]
            if not set(modules) <= ALLOWED_IMPORTS:
                return "import_outside_pure_stdlib_allowlist"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and (n.func.id in FORBIDDEN_CALLS or n.func.id == task["entrypoint"]):
            return "io_dynamic_or_recursive_call"
        if isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            return "dunder_attribute"
    cases = task["function_cases"]
    if not all(all(shape(a) for a in c["args"]) and shape(c["expected"]) for c in cases):
        return "outside_flat_string_list_scalar_contract"
    if any(isinstance(a, (str, list)) for c in cases for a in c["args"]):
        task["group"] = "target"
        task["domain"] = "pure_python_flat_string_list_operations"
    else:
        if not all(all(type(a) is int for a in c["args"]) and type(c["expected"]) in (int, bool) for c in cases):
            return "outside_basic_integer_control_contract"
        # Numeric controls are straight-line arithmetic/predicates, not algorithms.
        if re.search(r"\b(cube|cubes|power|powers)\b", description, re.I):
            return "numeric_control_not_basic_arithmetic"
        if loops or any(isinstance(n, (ast.Pow, ast.LShift, ast.RShift, ast.BitXor, ast.BitOr, ast.BitAnd)) for n in nodes):
            return "numeric_control_not_straightline_basic"
        task["group"] = "control"
        task["domain"] = "basic_integer_scalar_arithmetic_predicates"
    return None


def make_prompt(task):
    return ("Return only complete Python code. Preserve the required entrypoint: " +
            task["entrypoint"] + "(" + ", ".join(task["function_parameters"]) + ").\n" +
            "Original task:\n" + task["original_prompt"])


def token_lengths(tokenizer, prompt, response):
    # Two documented reconstructor encodings, both complete and untruncated.
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return {"response_tokens": len(tokenizer.encode(response, add_special_tokens=False)),
            "flat_tokens_with_eos": len(tokenizer.encode(prompt + "\n" + response, add_special_tokens=False)) + 1,
            "chat_tokens": len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)),
            "generation_prompt_tokens": len(tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True))}


def load_context(prior, tokenizer_dir):
    blobs = pilot.read_sources(prior, fetch=False)
    lock_bytes = (prior / "selection-lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    consumed = {r["id"] for r in lock["selection"]}
    if len(consumed) != 32 or Counter(r["split"] for r in lock["selection"]) != {"development": 8, "final": 24}:
        raise ValueError("Expected all 32 consumed pilot-v2 IDs (8 development, 24 final)")
    if sha(lock_bytes) != PRIOR_LOCK_SHA256 or sha(lock_bytes) != json.loads((prior / "manifest.json").read_bytes())["selection_lock_sha256"]:
        raise ValueError("Prior lock digest mismatch")
    if {n: sha((tokenizer_dir / n).read_bytes()) for n in TOKENIZER_FILES} != TOKENIZER_SHA256:
        raise ValueError("Pinned offline tokenizer digest mismatch")
    # Import only the tokenizer API. No AutoModel, torch.load, model weights, or network.
    from transformers import AutoTokenizer
    import transformers
    import tokenizers
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    env = {"python": platform.python_version(), "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
           "tokenizer_class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id,
           "tokenizer_files_sha256": {n: sha((tokenizer_dir / n).read_bytes()) for n in TOKENIZER_FILES},
           "tokenizer_local_path": str(tokenizer_dir), "weights_loaded": False, "network_used": False}
    return blobs, consumed, lock, tokenizer, env


def curate(prior, tokenizer_dir):
    blobs, consumed, old_lock, tokenizer, env = load_context(prior, tokenizer_dir)
    rows = pilot.get_tasks(blobs)
    all_tasks, row_index, audit = [], {}, []
    for source, row in rows:
        task = metadata(source, row)
        all_tasks.append(task)
        row_index[task["id"]] = (source, row)
    clusters, edges = families(all_tasks)
    old_family_ids = {t["family"] for t in all_tasks if t["id"] in consumed or initial_family(t)}
    # Conservative expansion of the six exposed toy operations, including near names.
    initial_names = ["add", "square", "is_even", "reverse_text", "larger", "absolute_value"]
    for t in all_tasks:
        name = normalize(t["entrypoint"]).replace(" ", "")
        if any(name == normalize(n).replace(" ", "") or (len(name) >= 6 and len(n.replace("_", "")) >= 6 and SequenceMatcher(None, name, n.replace("_", "")).ratio() >= 0.86) for n in initial_names):
            old_family_ids.add(t["family"])
    eligible = []
    for meta in all_tasks:
        source, row = row_index[meta["id"]]
        record = {k: meta[k] for k in ("id", "source", "source_task_id", "entrypoint", "family", "source_record_sha256", "canonical_prompt_sha256", "canonical_source_sha256", "alpha_source_sha256")}
        reason = None
        if meta["id"] in consumed:
            reason = "consumed_pilot_v2_id"
        elif meta["family"] in old_family_ids:
            reason = "consumed_or_initial_six_near_family"
        task, parser_audit = pilot.convert(source, row)
        record["parser_audit"] = parser_audit
        if not reason and task is None:
            reason = parser_audit["reason"]
        if task:
            task.update(meta)
            if not reason:
                reason = static_contract(task)
            if not reason:
                task["operation_tags"] = sorted(operation_tags(task))
                task["case_statistics"] = {"retained": len(task["function_cases"]),
                    "distinct_expected": len({canonical(c["expected"]) for c in task["function_cases"]}),
                    "nonempty_expected": sum(c["expected"] is not None and c["expected"] != "" and c["expected"] != [] and c["expected"] != {} for c in task["function_cases"])}
                task["prompt"] = make_prompt(task)
                task["token_lengths"] = token_lengths(tokenizer, task["prompt"], task["response"])
                record["token_lengths"] = task["token_lengths"]
                if task["token_lengths"]["response_tokens"] > MAX_RESPONSE_TOKENS:
                    reason = "complete_response_over_192_tokens"
                elif max(task["token_lengths"]["flat_tokens_with_eos"], task["token_lengths"]["chat_tokens"]) > MAX_TOKENS:
                    reason = "complete_prompt_response_over_256_tokens"
            if not reason:
                eligible.append(task)
        record.update(status="excluded" if reason else "eligible", reason=reason)
        audit.append(record)
    family_pool = defaultdict(list)
    for t in eligible:
        family_pool[t["family"]].append(t)
    representatives = [min(v, key=rank) for v in family_pool.values()]
    duplicate_ids = {t["id"] for v in family_pool.values() for t in v if t is not min(v, key=rank)}
    for a in audit:
        if a["id"] in duplicate_ids:
            a.update(status="excluded", reason="same_eligible_family_representative")
    selected = []
    pools = {g: sorted((t for t in representatives if t["group"] == g), key=rank) for g in ("target", "control")}
    for split, quotas in QUOTAS.items():
        for group, count in quotas.items():
            chosen, pools[group] = pools[group][:count], pools[group][count:]
            for t in chosen:
                t["split"] = split
                selected.append(t)
    if sum(t["split"] == "validation" for t in selected) != 8 or sum(t["split"] == "final" for t in selected) != 16:
        raise ValueError("Insufficient independent held-out tasks: do not pad or clone")
    selected.sort(key=lambda t: (t["split"], rank(t, "ordering")))
    selected_ids = {t["id"] for t in selected}
    for a in audit:
        if a["status"] == "eligible":
            a["status"] = "selected" if a["id"] in selected_ids else "eligible_not_selected"
    counts = {"upstream_total": len(rows), "consumed_ids": len(consumed), "eligible_before_family_representatives": len(eligible),
              "eligible_independent_families": len(representatives), "eligible_by_group": dict(Counter(t["group"] for t in representatives)),
              "selected_unique_tasks": len(selected), "selected_by_split": dict(Counter(t["split"] for t in selected)),
              "selected_by_split_group_source": dict(Counter(t["split"] + "/" + t["group"] + "/" + t["source"] for t in selected)),
              "eligible_not_selected": len(representatives) - len(selected),
              "exclusions_by_reason": dict(sorted(Counter(a["reason"] for a in audit if a["status"] == "excluded").items())),
              "train_requested": 64, "train_shortfall": 64 - sum(t["split"] == "train" for t in selected),
              "function_cases": sum(len(t["function_cases"]) for t in selected)}
    return selected, audit, edges, counts, env, blobs, old_lock


def sample(task):
    keys = ("id", "prompt", "response", "source", "source_task_id", "license", "entrypoint", "function_parameters", "family", "group", "domain", "original_prompt", "source_record_sha256", "canonical_prompt_sha256", "canonical_source_sha256", "alpha_source_sha256", "response_sha256", "token_lengths", "operation_tags", "case_statistics")
    return {k: task[k] for k in keys}


def dataset(tasks, purpose):
    return {"schema": SCHEMA, "purpose": purpose, "split_policy": "standalone_LOCAL_not_official_benchmark_split", "samples": [sample(t) for t in tasks]}


def suite(tasks, split):
    return {"schema_version": 1, "name": "specialist-v1-" + split,
            "reference_source": "Pinned MIT HumanEval and CC-BY-4.0 sanitized MBPP; original literal assert subset. Standalone LOCAL split, not official benchmark scores. See ATTRIBUTION.md and manifest.json.",
            "policy_version": "composition-admission-v2", "claims": ["coding"],
            "cases": [{"id": t["id"], "group": t["group"], "input": t["prompt"],
                       "reference": t["source"] + " task " + t["source_task_id"] + " original literal assertions; " + t["license"] + ".",
                       "metric": "function_io", "threshold": 1.0, "output_format": "raw_python",
                       "function_cases": [{k: v for k, v in c.items() if k != "upstream_assertion"} for c in t["function_cases"]]} for t in tasks]}


def render(prior, tokenizer_dir):
    tasks, audits, edges, counts, env, blobs, prior_lock = curate(prior, tokenizer_dir)
    split_tasks = {s: [t for t in tasks if t["split"] == s] for s in QUOTAS}
    calibration = sorted(split_tasks["train"], key=lambda t: rank(t, "calibration"))[:32]
    counts["calibration_from_train"] = len(calibration)
    selections = [{k: t[k] for k in ("id", "source", "source_task_id", "family", "group", "domain", "split", "source_record_sha256", "canonical_prompt_sha256", "canonical_source_sha256", "alpha_source_sha256", "response_sha256", "token_lengths", "operation_tags", "case_statistics")} for t in tasks]
    lock = {"schema": "silt.specialist.selection-lock.v1", "seed": SEED, "frozen_before_model_generation": True,
            "model_outputs_consulted": False, "selection": selections, "calibration_ids": [t["id"] for t in calibration],
            "requested_quotas": QUOTAS, "prior_consumed_selection": prior_lock["selection"], "initial_six_families": pilot.OLD_TASKS,
            "ranking": "ascending SHA256(seed + ':' + purpose + ':' + source-prefixed task ID); one representative per connected family; heldouts allocated before train; purposes selection/ordering/calibration",
            "family_contract": "transitive components over all 591 source rows: equal normalized prompt/source/alpha-source/record fingerprints OR normalized function name equal/SequenceMatcher >=0.86 (minimum length6) OR morphological description-token Jaccard >=0.60 OR shared conservative operation tag; exclude entire components touching consumed IDs/initial six including reverse/power/parity/inverse-extreme expansions; heuristic NOT perfect semantic dedup",
            "scope": "pure stateless flat string/list transformations, predicates and reductions; controls basic straight-line integer arithmetic/predicates; no advanced math; <=2 loops, <=220 AST nodes, pure stdlib allowlist",
            "token_contract": {"max_combined_tokens": MAX_TOKENS, "max_response_tokens": MAX_RESPONSE_TOKENS,
                               "flat": "tokenizer.encode(prompt + '\\n' + response, add_special_tokens=False) + [eos_token_id]",
                               "chat": "apply_chat_template([user:prompt, assistant:response], tokenize=True, add_generation_prompt=False)",
                               "truncation": False, "both_encodings_must_fit": True},
            "environment": env, "teacher_capability": "UNVALIDATED: source complexity is only a preregistered hypothesis for Qwen2.5-Coder-0.5B-Instruct; no zero-error or passing guarantee"}
    files = {"train.json": dataset(split_tasks["train"], "train"), "calibration.json": dataset(calibration, "calibration_TRAIN_ONLY_deliberate_overlap"),
             "validation.json": dataset(split_tasks["validation"], "development_validation"), "final.json": dataset(split_tasks["final"], "final_locked_do_not_retune"),
             "validation-suite.json": suite(split_tasks["validation"], "validation"), "final-suite.json": suite(split_tasks["final"], "final"),
             "selection-lock.json": lock, "eligibility-audit.json": {"tasks": audits, "family_edges": edges},
             "train-cases.json": {"purpose": "train_oracle_metadata_not_heldout", "tasks": [{"id": t["id"], "function_cases": t["function_cases"]} for t in split_tasks["train"]]},
             "validation-provenance.json": {"tasks": [{"id": t["id"], "function_cases": t["function_cases"]} for t in split_tasks["validation"]]},
             "final-provenance.json": {"purpose": "final_locked", "tasks": [{"id": t["id"], "function_cases": t["function_cases"]} for t in split_tasks["final"]]}}
    data = {name: packed(value) for name, value in files.items()}
    # Preserve publisher attribution and every original notice; append truthful changes.
    attribution = (prior / "ATTRIBUTION.md").read_text() + "\n## Additional modifications: specialist-v1\n\nThe preceding pilot-v2 attribution is retained as historical provenance. This\nderivative excludes all 32 consumed pilot IDs and detected related families.\nIt preserves complete original canonical code as supervised response data\n(HumanEval prompt + canonical_solution, or MBPP code), including source imports\nand docstrings. No original code is executed or rewritten. New local train,\ntrain-only calibration, validation and final splits, narrower scope and complete\n256-token filters are added. Labels, hashes and source-only family heuristics\nare SILT metadata. References are NOT claimed model-generated candidates.\nNo new dataset, model outputs, synthesized functions or synthetic test cases are used.\nThese are NOT official benchmark splits or scores; public/pretraining exposure\nis unknown. Teacher capability remains unvalidated until a separate train/dev\nrun; do not inspect or tune against final answers. Retain these notices.\n"
    data["ATTRIBUTION.md"] = attribution.encode()
    for name, blob in blobs.items():
        data["sources/" + name] = blob
    manifest = {"schema": "silt.specialist.manifest.v1", "scope": "standalone_LOCAL_specialist_training_and_heldouts_NOT_official_benchmark",
                "primary_capability_contract": "Implement short pure Python functions over flat strings/lists: transform, filter, count, test membership or aggregate; preserve original entrypoint and return value, no external I/O. Basic integer controls measure nearby non-target support.",
                "predicted_fairness": "Groups represent real source-only input domains; both have diverse positive oracle values. Controls are simpler straight-line arithmetic, not difficulty matched. No performance parity or causal fairness claim. Source distribution and original benchmark exposure are not balanced/unknown.",
                "teacher_capability_status": "UNVALIDATED; no model weights loaded or model outputs inspected; validate later on TRAIN/development only. Never select or retune using final.",
                "official_split": False, "counts": counts, "selection_sha256": sha(canonical(selections)),
                "selection_lock_sha256": sha(data["selection-lock.json"]), "seed": SEED,
                "source_pins": {"HumanEval": pilot.HE_PIN, "MBPP_GitHub": pilot.MBPP_PIN, "MBPP_publisher_license_card": pilot.HF_PIN},
                "sources": [{"file": "sources/" + n, "url": u, "sha256": h} for n, (u, h) in pilot.SOURCES.items()],
                "prior_selection_lock_sha256": sha((prior / "selection-lock.json").read_bytes()),
                "builder_sha256": sha(Path(__file__).read_bytes()), "audited_parser_sha256": sha((ROOT / "scripts/prepare_coding_pilot.py").read_bytes()),
                "environment": env, "artifact_sha256": {n: sha(b) for n, b in sorted(data.items())},
                "governance": "Frozen before generation. Final files contain answers and are quarantined by policy, not encryption. Do not open final.json, final-suite.json or final-provenance.json for training/tuning. Dataset revisions after generation require a new version and lock. No cloning/padding for shortfalls."}
    data["manifest.json"] = packed(manifest)
    data["SHA256SUMS"] = "".join(sha(b) + "  " + n + "\n" for n, b in sorted(data.items())).encode()
    return data, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, default=ROOT / "data/pilot-v2")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--stats", action="store_true", help="Read-only aggregate eligibility report; no files or model work")
    p.add_argument("--check", action="store_true", help="Read-only byte-for-byte frozen reproducibility check")
    args = p.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(DEFAULT_OUTPUT.resolve()):
        p.error("--output must remain within the owned data/specialist-v1 subtree")
    if args.stats:
        _, _, _, counts, env, _, _ = curate(args.prior, args.tokenizer)
        print(json.dumps({"counts": counts, "environment": env, "model_runs": 0}, indent=2))
        return
    data, manifest = render(args.prior, args.tokenizer)
    for name, content in data.items():
        path = args.output / name
        if path.exists() and path.read_bytes() != content:
            raise SystemExit("Frozen fixture differs: " + name + "; create a new version rather than overwrite")
        if args.check and not path.exists():
            raise SystemExit("Missing frozen fixture: " + name)
    if not args.check:
        for name, content in data.items():
            path = args.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(content)
    print(json.dumps({"output": str(args.output.resolve()), "relative_output": str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else None,
                      "selection_sha256": manifest["selection_sha256"], "selection_lock_sha256": manifest["selection_lock_sha256"],
                      "manifest_sha256": sha(data["manifest.json"]), "counts": manifest["counts"], "model_runs": 0}, indent=2))


if __name__ == "__main__":
    main()
