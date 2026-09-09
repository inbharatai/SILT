"""Security and reproducibility checks for inert public benchmark conversion."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("coding_pilot_builder", ROOT / "scripts" / "prepare_coding_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_literal_assertion_preserves_values():
    result = pilot.parse_assertion('assert fn([-2, +3, True, None, 25], {"a": ["b"]}) == {"ok": False}', "fn")
    assert result == {"function": "fn", "args": [[-2, 3, True, None, 25], {"a": ["b"]}],
                      "kwargs": {}, "expected": {"ok": False}}
    assert pilot.parse_assertion('assert candidate(2) == 7', "actual", "candidate")["function"] == "actual"


@pytest.mark.parametrize("source", [
    'assert fn(__import__("os").system("touch /tmp/no")) == 0',
    'assert fn(2) == __import__("os").system("touch /tmp/no")',
    'assert obj.fn(2) == 3',
    'assert wrong(2) == 3',
    'assert fn(2).strip() == "3"',
    'assert fn(x=2) == 3',
    'assert fn(**{"x": 2}) == 3',
    'assert fn(*[2]) == 3',
    'assert fn(2) == 3 == 3',
    'assert fn(2) != 3',
    'assert fn(2)',
    'assert not fn(2)',
    'assert fn(2) == 3, "message"',
    'assert fn(x) == 3',
    'assert fn([x for x in []]) == []',
    'assert fn((1, 2)) == [1, 2]',
    'assert fn({1, 2}) == [1, 2]',
    'assert fn({1: 2}) == 2',
    'assert fn({"x": 1, "x": 2}) == 2',
    'assert fn({**{}}) == {}',
    'assert fn(2) == 1 + 2',
    'assert fn(-2.5) == 1',
    'assert fn(2.5) == 1',
    'assert fn(--2) == 1',
    'assert fn(-True) == 1',
    'assert fn(float("inf")) == 1',
    'assert fn(1e999) == 1',
    'assert fn(9007199254740992) == 1',
    'assert fn(b"bytes") == 1',
    'assert fn(1j) == 1',
    'assert fn(...) == 1',
    'assert fn(lambda: 1) == 1',
    'assert fn((x := 1)) == 1',
    'import os\nassert fn(2) == 3',
    'assert fn(2) == 3\nassert fn(3) == 4',
    'if True:\n    assert fn(2) == 3',
])
def test_rejects_unknown_ast_calls_kwargs_and_types(source):
    with pytest.raises(pilot.Unsupported):
        pilot.parse_assertion(source, "fn")


def test_strict_literal_resource_limits():
    for source in [
        'assert fn(' + '[' * 10 + '0' + ']' * 10 + ') == 1',
        'assert fn([' + ','.join('0' for _ in range(129)) + ']) == 1',
        'assert fn("' + 'a' * 4097 + '") == 1',
        'assert fn(' + ','.join('0' for _ in range(17)) + ') == 1',
        ' ' * (pilot.MAX_TEXT + 1),
    ]:
        with pytest.raises(pilot.Unsupported):
            pilot.parse_assertion(source, "fn")


def test_builder_never_calls_host_execution_helpers():
    tree = ast.parse((ROOT / "scripts" / "prepare_coding_pilot.py").read_text())
    forbidden = {"exec", "eval", "compile", "literal_eval", "system", "Popen", "run", "check_output"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            assert name not in forbidden


def sample_row():
    return {"task_id": 999999, "prompt": "Return a decimal digit count.", "test_imports": [],
            "code": 'def digit_count(x):\n    raise RuntimeError("must never execute reference")\n',
            "test_list": ['assert digit_count(2) == 1', 'assert digit_count(12) == 2']}


def test_expected_comes_only_from_test_literals_and_reference_is_never_run(tmp_path):
    row = sample_row()
    marker = tmp_path / "HOST_EXECUTED"
    row["code"] = f'import pathlib\npathlib.Path({str(marker)!r}).write_text("bad")\ndef digit_count(x):\n    return 999999\n'
    task, audit = pilot.convert("MBPP", row)
    assert not marker.exists()
    assert audit["status"] == "eligible"
    assert [c["expected"] for c in task["function_cases"]] == [1, 2]
    assert task["entrypoint"] == "digit_count"


def test_test_source_is_not_executed(tmp_path):
    marker = tmp_path / "HOST_EXECUTED"
    row = sample_row()
    row["test_list"] = [f'__import__("pathlib").Path({str(marker)!r}).touch(); assert digit_count(2) == 1']
    task, audit = pilot.convert("MBPP", row)
    assert task is None
    assert audit["reason"] == "multiple_test_statements"
    assert not marker.exists()


def test_unsupported_subset_and_empty_only_oracle_are_not_eligible():
    row = sample_row()
    row["test_list"][1] = 'assert digit_count(12) == 1 + 1'
    task, audit = pilot.convert("MBPP", row)
    assert task is None and audit["reason"] == "fewer_than_two_literal_cases"
    row["test_list"] = ['assert digit_count(1) == []', 'assert digit_count(2) == []']
    task, audit = pilot.convert("MBPP", row)
    assert task is None and audit["reason"] == "retained_cases_lack_expected_value_diversity"


def test_conflicting_oracle_rejected():
    row = sample_row()
    row["test_list"] = ['assert digit_count(1) == 1', 'assert digit_count(1) == 2']
    task, audit = pilot.convert("MBPP", row)
    assert task is None and audit["reason"] == "conflicting_expected_values"


def test_original_six_families_are_excluded():
    examples = [("add", "Return the sum of two numbers."), ("square", "Return the square of a number."),
                ("is_even", "Check if the number is even."), ("reverse_text", "Reverse the string."),
                ("larger", "Return the larger of two values."), ("absolute_value", "Return the absolute value.")]
    assert len(pilot.OLD_TASKS) == 6
    for name, prompt in examples:
        assert pilot.old_family(prompt, name) is not None


def test_detected_duplicate_families_cannot_split():
    tasks = [
        {"id": "a", "entrypoint": "count_digits", "original_prompt": "Count decimal digits in an input value."},
        {"id": "b", "entrypoint": "count_digits", "original_prompt": "Count digits."},
        {"id": "c", "entrypoint": "other", "original_prompt": "Reverse alphabet case."},
    ]
    groups = pilot.family_clusters(tasks)
    assert sorted(len(group) for group in groups) == [1, 2]
    assert tasks[0]["family"] == tasks[1]["family"] != tasks[2]["family"]


def test_frozen_fixtures_are_byte_reproducible_and_hash_verified():
    data, manifest = pilot.render(pilot.DEFAULT_OUTPUT)
    for name, expected in data.items():
        assert (pilot.DEFAULT_OUTPUT / name).read_bytes() == expected, name
    for name, sha in manifest["artifact_sha256"].items():
        assert pilot.digest(data[name]) == sha
    assert manifest["license_unresolved"] == []
    assert manifest["counts"]["selected"] == 32


def test_selected_splits_and_groups_and_provenance():
    data, manifest = pilot.render(pilot.DEFAULT_OUTPUT)
    provenance = json.loads(data["provenance.json"])["tasks"]
    assert len({t["family"] for t in provenance}) == len(provenance) == 32
    assert {t["source"] for t in provenance} == {"HumanEval", "MBPP"}
    ids = []
    for split, count, target_count in [("development", 8, 6), ("final", 24, 18)]:
        suite = json.loads(data[split + ".json"])
        assert len(suite["cases"]) == count <= 64
        assert sum(c["group"] == "target" for c in suite["cases"]) == target_count
        for case in suite["cases"]:
            ids.append(case["id"])
            task = next(t for t in provenance if t["id"] == case["id"])
            assert task["original_prompt"] in case["input"]
            assert case["metric"] == "function_io"
            assert 2 <= len(case["function_cases"]) <= pilot.MAX_CASES
            for function_case, original in zip(case["function_cases"], task["function_cases"]):
                assert function_case["function"] == task["entrypoint"]
                parsed = pilot.parse_assertion(original["upstream_assertion"], task["entrypoint"], "candidate" if task["source"] == "HumanEval" else None)
                assert parsed == {k: v for k, v in function_case.items() if k != "id"}
    assert len(set(ids)) == len(ids) == 32
    assert manifest["counts"]["adapted_function_cases"] == sum(len(t["function_cases"]) for t in provenance)


def test_tampered_source_fails_closed(tmp_path):
    (tmp_path / "sources").mkdir()
    for name in pilot.SOURCES:
        source = pilot.DEFAULT_OUTPUT / "sources" / name
        (tmp_path / "sources" / name).write_bytes(source.read_bytes())
    (tmp_path / "sources" / "HumanEval-LICENSE.txt").write_text("substituted")
    with pytest.raises(ValueError, match="source hash mismatch"):
        pilot.render(tmp_path)


def test_schema_compatibility_when_function_io_available():
    from asea.compose.schema import EvaluationCase, EvaluationSuite
    # The parallel runtime implementation owns the function_io schema rollout.
    if "function_io" not in str(EvaluationCase.model_fields["metric"].annotation):
        pytest.skip("function_io runtime schema not yet installed; builder must not downgrade to source execution")
    for name in ("development.json", "final.json"):
        EvaluationSuite.model_validate(json.loads((pilot.DEFAULT_OUTPUT / name).read_text()))
