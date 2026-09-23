"""E2 training-case authoring (REAL authoring, 2026-09-19).

Experiment 2 expands the KD training set: the six v1 training cases are
reused byte-for-byte (their receipted traces stay valid), and SIXTEEN NEW
authored training cases are added. This script authors those sixteen new
cases and enforces the same discipline as the v1 authoring self-check
(experiments/glm53_flash_pilot/authoring_selfcheck.py):

  * every case's oracle ``expected`` values are computed by RUNNING the
    reference implementation -- never typed by hand;
  * the reference passes EVERY check exactly, value AND type, after a
    JSON round-trip (the oracle compares JSON-round-tripped values);
  * the seeded-bug implementation fails at least one check (a real
    failure, not a decorative one);
  * each check's ``args`` arity equals the reference signature's
    parameter count;
  * check ids are unique within a case; every case has >= 2 checks;
  * the new family ids, sample ids and prompt token-shapes collide with
    NOTHING in the frozen v1 dataset (a family never crosses splits, and
    the near-duplicate shape guard is enforced at authoring time, not
    just at dataset build time).

Outputs ``cases-new-training.jsonl`` (sixteen training-split rows in the
exact v1 case format). Nothing here runs a model or grades anything.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V1_DIR = REPO / "data" / "capability_v1"
OUT = HERE / "cases-new-training.jsonl"

PROMPT_TEMPLATE = (
    "The following Python function has a bug. Fix it so it meets its "
    "specification exactly. Keep the function name and signature unchanged. "
    "Return ONLY the complete fixed function inside a single ```python code "
    "block, with no other text.\n\nSpecification: {spec}\n\n```python\n{buggy}```"
)

PROVENANCE = (
    "authored 2026-09-19 for the GLM-5.3-Flash capability pilot E2 "
    "(expanded training set); original work of the repository operator; "
    "no third-party code"
)


def _shape(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# -- the sixteen new cases --------------------------------------------------
# (sample_id, family_id, spec, reference source, buggy source, checks)
# each check: (check_id, function_name, args_list, kwargs_dict)

CASES = [
    (
        "reverse_words_v1",
        "reverse_word_order",
        "reverse_words accepts a non-empty string of words separated by "
        "single spaces and returns the same words in reverse order, joined "
        "by single spaces. Punctuation stays attached to its word. The "
        "input is not mutated.",
        'def reverse_words(sentence):\n    words = sentence.split(" ")\n'
        '    result = []\n    for i in range(len(words) - 1, -1, -1):\n'
        "        result.append(words[i])\n"
        '    return " ".join(result)\n',
        "def reverse_words(sentence):\n    return sentence[::-1]\n",
        [
            ("rev_three", "reverse_words", ["alpha beta gamma"], {}),
            ("rev_two", "reverse_words", ["one two"], {}),
            ("rev_single", "reverse_words", ["solo"], {}),
            ("rev_four", "reverse_words", ["a b c d"], {}),
        ],
    ),
    (
        "sum_digits_v1",
        "sum_digit_chars",
        "sum_digits accepts a string and returns the sum of the integer "
        "values of every ASCII digit character '0' through '9' it contains, "
        "in the order encountered. A string with no digits returns 0.",
        'def sum_digits(text):\n    total = 0\n    for ch in text:\n'
        '        if "0" <= ch <= "9":\n            total += int(ch)\n'
        "    return total\n",
        'def sum_digits(text):\n    total = 0\n    for ch in text:\n'
        '        if "0" <= ch <= "9":\n            total += 1\n'
        "    return total\n",
        [
            ("mixed", "sum_digits", ["a1b2c3"], {}),
            ("empty", "sum_digits", [""], {}),
            ("digits_letters", "sum_digits", ["9x0"], {}),
            ("no_digits", "sum_digits", ["hello"], {}),
        ],
    ),
    (
        "interleave_v1",
        "interleave_lists",
        "interleave accepts two lists of equal length and returns a new "
        "list whose elements alternate between the two inputs, starting "
        "with the first list: first[0], second[0], first[1], second[1], and "
        "so on. The inputs are not mutated.",
        "def interleave(first, second):\n    result = []\n"
        "    for i in range(len(first)):\n        result.append(first[i])\n"
        "        result.append(second[i])\n    return result\n",
        "def interleave(first, second):\n    result = []\n"
        "    for i in range(len(first)):\n        result.append(second[i])\n"
        "        result.append(first[i])\n    return result\n",
        [
            ("two_by_two", "interleave", [[1, 2], ["a", "b"]], {}),
            ("empty_pair", "interleave", [[], []], {}),
            ("three_by_three", "interleave", [[10, 20, 30], [1, 2, 3]], {}),
        ],
    ),
    (
        "adjacent_diffs_v1",
        "adjacent_differences",
        "diffs accepts a list of numbers with at least one element and "
        "returns the list of absolute differences between each pair of "
        "adjacent elements, in order. A single-element list returns an "
        "empty list.",
        "def diffs(numbers):\n    result = []\n"
        "    for i in range(len(numbers) - 1):\n"
        "        difference = numbers[i + 1] - numbers[i]\n"
        "        if difference < 0:\n            difference = -difference\n"
        "        result.append(difference)\n    return result\n",
        "def diffs(numbers):\n    result = []\n"
        "    for i in range(len(numbers) - 1):\n"
        "        result.append(numbers[i + 1] - numbers[i])\n"
        "    return result\n",
        [
            ("down_up", "diffs", [[3, 1, 4]], {}),
            ("single", "diffs", [[7]], {}),
            ("flat", "diffs", [[5, 5, 5]], {}),
            ("big_drop", "diffs", [[10, 2, 7]], {}),
        ],
    ),
    (
        "diagonal_sum_v1",
        "matrix_diagonal_sum",
        "diagonal_sum accepts a square matrix (a list of equal-length row "
        "lists of numbers) and returns the sum of the main-diagonal "
        "entries, that is matrix[i][i] for each row index i. The matrix is "
        "not mutated.",
        "def diagonal_sum(matrix):\n    total = 0\n"
        "    for i in range(len(matrix)):\n        total += matrix[i][i]\n"
        "    return total\n",
        "def diagonal_sum(matrix):\n    total = 0\n    n = len(matrix)\n"
        "    for i in range(n):\n        total += matrix[i][n - 1 - i]\n"
        "    return total\n",
        [
            ("two_by_two", "diagonal_sum", [[[1, 2], [3, 4]]], {}),
            ("one_by_one", "diagonal_sum", [[[7]]], {}),
            ("sparse_identity", "diagonal_sum",
             [[[1, 0, 0], [0, 2, 0], [0, 0, 4]]], {}),
            ("generic_2x2", "diagonal_sum", [[[9, 1], [2, 5]]], {}),
        ],
    ),
    (
        "char_counts_v1",
        "letter_frequency",
        "char_counts accepts a string and returns a dictionary that maps "
        "each lowercase letter appearing in the string to the number of "
        "times it appears, counting uppercase letters as their lowercase "
        "form. Non-letter characters are ignored. The result contains "
        "keys only for letters that appear.",
        'def char_counts(text):\n    counts = {}\n    for ch in text:\n'
        "        letter = ch.lower()\n"
        '        if "a" <= letter <= "z":\n'
        "            counts[letter] = counts.get(letter, 0) + 1\n"
        "    return counts\n",
        'def char_counts(text):\n    counts = {}\n    for ch in text:\n'
        '        if "a" <= ch <= "z" or "A" <= ch <= "Z":\n'
        "            counts[ch] = counts.get(ch, 0) + 1\n"
        "    return counts\n",
        [
            ("mixed_case", "char_counts", ["AbA"], {}),
            ("with_noise", "char_counts", ["a1!"], {}),
            ("empty", "char_counts", [""], {}),
            ("case_folding", "char_counts", ["Zz"], {}),
        ],
    ),
    (
        "cumprod_v1",
        "cumulative_products",
        "cumprod accepts a non-empty list of numbers and returns a list of "
        "the same length whose first element is the first input element and "
        "whose every following element is the product of all input elements "
        "up to and including that position. The input is not mutated.",
        "def cumprod(numbers):\n    result = [numbers[0]]\n"
        "    for n in numbers[1:]:\n"
        "        result.append(result[-1] * n)\n    return result\n",
        "def cumprod(numbers):\n    result = [numbers[0]]\n"
        "    for n in numbers[1:]:\n"
        "        result.append(result[-1] + n)\n    return result\n",
        [
            ("growing", "cumprod", [[2, 3, 4]], {}),
            ("single", "cumprod", [[5]], {}),
            ("zero_inside", "cumprod", [[1, 0, 7]], {}),
            ("negative_step", "cumprod", [[3, -1, 2]], {}),
        ],
    ),
    (
        "is_sorted_v1",
        "sorted_nondecreasing_check",
        "is_sorted accepts a list of numbers and returns True when the list "
        "is in non-decreasing order (no element is greater than the one "
        "after it) and False otherwise. Empty lists and single-element "
        "lists are sorted.",
        "def is_sorted(values):\n"
        "    for i in range(len(values) - 1):\n"
        "        if values[i] > values[i + 1]:\n            return False\n"
        "    return True\n",
        "def is_sorted(values):\n"
        "    for i in range(len(values) - 1):\n"
        "        if values[i] >= values[i + 1]:\n            return False\n"
        "    return True\n",
        [
            ("with_ties", "is_sorted", [[1, 2, 2, 3]], {}),
            ("descending", "is_sorted", [[3, 1]], {}),
            ("empty", "is_sorted", [[]], {}),
            ("single", "is_sorted", [[5]], {}),
        ],
    ),
    (
        "clamp_v1",
        "clamp_into_range",
        "clamp accepts a list of numbers and two bounds lo and hi with "
        "lo <= hi, and returns a new list in which every value below lo is "
        "replaced by lo, every value above hi is replaced by hi, and values "
        "inside the inclusive range are unchanged. The input list is not "
        "mutated.",
        "def clamp(values, lo, hi):\n    result = []\n"
        "    for v in values:\n        if v < lo:\n"
        "            result.append(lo)\n        elif v > hi:\n"
        "            result.append(hi)\n        else:\n"
        "            result.append(v)\n    return result\n",
        "def clamp(values, lo, hi):\n    result = []\n"
        "    for v in values:\n        if v < lo:\n"
        "            result.append(lo)\n        else:\n"
        "            result.append(v)\n    return result\n",
        [
            ("both_sides", "clamp", [[1, 5, 9], 2, 8], {}),
            ("inside", "clamp", [[0, 4], 0, 10], {}),
            ("negative_bounds", "clamp", [[-5, 3, 15], -1, 1], {}),
        ],
    ),
    (
        "longest_word_v1",
        "longest_word_first_tie",
        "longest_word accepts a non-empty string of words separated by "
        "single spaces and returns the longest word. When several words "
        "tie for the longest, the first of them is returned.",
        'def longest_word(sentence):\n    words = sentence.split(" ")\n'
        "    best = words[0]\n    for word in words:\n"
        "        if len(word) > len(best):\n            best = word\n"
        "    return best\n",
        'def longest_word(sentence):\n    words = sentence.split(" ")\n'
        "    best = words[0]\n    for word in words:\n"
        "        if len(word) >= len(best):\n            best = word\n"
        "    return best\n",
        [
            ("all_ties", "longest_word", ["aa bb cc dd"], {}),
            ("clear_winner", "longest_word", ["x yyy"], {}),
            ("mid_sentence", "longest_word", ["cat to bird"], {}),
        ],
    ),
    (
        "swap_case_v1",
        "swap_letter_case",
        "swap_case accepts a string and returns a new string in which every "
        "lowercase letter becomes uppercase, every uppercase letter becomes "
        "lowercase, and every non-letter character is unchanged.",
        'def swap_case(text):\n    result = ""\n    for ch in text:\n'
        '        if "a" <= ch <= "z":\n            result += ch.upper()\n'
        '        elif "A" <= ch <= "Z":\n            result += ch.lower()\n'
        "        else:\n            result += ch\n    return result\n",
        'def swap_case(text):\n    result = ""\n    for ch in text:\n'
        '        if "a" <= ch <= "z":\n            result += ch.upper()\n'
        "        else:\n            result += ch\n    return result\n",
        [
            ("mixed", "swap_case", ["aB1!"], {}),
            ("empty", "swap_case", [""], {}),
            ("pair", "swap_case", ["Zz"], {}),
            ("with_punct", "swap_case", ["no-op"], {}),
        ],
    ),
    (
        "group_parity_v1",
        "group_by_parity",
        "group_parity accepts a list of integers and returns a dictionary "
        "with exactly the keys 'even' and 'odd': under 'even' go the even "
        "integers in input order and under 'odd' go the odd integers in "
        "input order, using mathematical parity (so -3 is odd and -4 is "
        "even). The input is not mutated.",
        'def group_parity(values):\n    result = {"even": [], "odd": []}\n'
        "    for n in values:\n        if n % 2 == 0:\n"
        '            result["even"].append(n)\n        else:\n'
        '            result["odd"].append(n)\n    return result\n',
        'def group_parity(values):\n    result = {"even": [], "odd": []}\n'
        "    for n in values:\n        if n % 2 != 0 or n < 0:\n"
        '            result["odd"].append(n)\n        else:\n'
        '            result["even"].append(n)\n    return result\n',
        [
            ("positives", "group_parity", [[1, 2, 3]], {}),
            ("empty", "group_parity", [[]], {}),
            ("negatives", "group_parity", [[-4, -3]], {}),
        ],
    ),
    (
        "repeat_to_v1",
        "repeat_pattern_to_length",
        "repeat_to accepts a non-empty pattern string and a non-negative "
        "integer n, and returns the first n characters of the pattern "
        "repeated cyclically. When n is 0 the result is the empty string; "
        "the result is never longer than n characters.",
        "def repeat_to(pattern, n):\n    if n <= 0:\n        return \"\"\n"
        "    repeats = n // len(pattern) + 1\n"
        "    return (pattern * repeats)[:n]\n",
        "def repeat_to(pattern, n):\n    return pattern * n\n",
        [
            ("truncate", "repeat_to", ["ab", 5], {}),
            ("zero", "repeat_to", ["xy", 0], {}),
            ("single_char", "repeat_to", ["q", 3], {}),
            ("one_cycle_plus", "repeat_to", ["abc", 4], {}),
        ],
    ),
    (
        "dot_product_v1",
        "dot_product",
        "dot accepts two lists of numbers of equal length and returns the "
        "sum of the element-wise products of the two lists. The inputs are "
        "not mutated.",
        "def dot(u, v):\n    total = 0\n"
        "    for i in range(len(u)):\n        total += u[i] * v[i]\n"
        "    return total\n",
        "def dot(u, v):\n    total = 0\n"
        "    for i in range(len(u)):\n        total += u[i] + v[i]\n"
        "    return total\n",
        [
            ("basic", "dot", [[2, 3], [4, 5]], {}),
            ("empty", "dot", [[], []], {}),
            ("zero_element", "dot", [[1, 0], [5, 7]], {}),
        ],
    ),
    (
        "palindromes_v1",
        "filter_palindromes",
        "palindromes accepts a list of non-empty lowercase strings and "
        "returns a new list containing exactly those strings that read the "
        "same forwards and backwards, preserving input order. The input "
        "list is not mutated.",
        "def palindromes(words):\n    result = []\n    for w in words:\n"
        "        if w == w[::-1]:\n            result.append(w)\n"
        "    return result\n",
        "def palindromes(words):\n    result = []\n    for w in words:\n"
        "        if w[0] == w[-1]:\n            result.append(w)\n"
        "    return result\n",
        [
            ("mixed", "palindromes",
             [["aba", "abca", "aa", "ab"]], {}),
            ("single_char", "palindromes", [["a"]], {}),
            ("none_match", "palindromes", [["ab"]], {}),
        ],
    ),
    (
        "wrap_text_v1",
        "greedy_word_wrap",
        "wrap_text accepts a string of words separated by single spaces and "
        "a positive integer width, and greedily packs the words into lines: "
        "the first word starts a line, and each following word is appended "
        "to the current line, joined by a single space, only if the joined "
        "line would be at most width characters long; otherwise it starts a "
        "new line. Every word is longer than zero characters and at most "
        "width characters. The result is the list of lines; every word "
        "appears exactly once in order.",
        'def wrap_text(text, width):\n    words = text.split(" ")\n'
        "    lines = []\n    current = \"\"\n    for word in words:\n"
        "        if not current:\n            current = word\n"
        "        elif len(current) + 1 + len(word) <= width:\n"
        '            current = current + " " + word\n        else:\n'
        "            lines.append(current)\n            current = word\n"
        "    if current:\n        lines.append(current)\n    return lines\n",
        'def wrap_text(text, width):\n    words = text.split(" ")\n'
        "    lines = []\n    current = \"\"\n    for word in words:\n"
        "        if not current:\n            current = word\n"
        "        elif len(current) + len(word) <= width:\n"
        '            current = current + " " + word\n        else:\n'
        "            lines.append(current)\n            current = word\n"
        "    if current:\n        lines.append(current)\n    return lines\n",
        [
            ("space_counts", "wrap_text", ["aaa bb cc", 5], {}),
            ("exact_fit", "wrap_text", ["one", 3], {}),
            ("every_word_wraps", "wrap_text", ["aa bb cc", 2], {}),
        ],
    ),
]


def _load_fn(source: str, name: str):
    namespace: dict = {}
    exec(compile(source, "<case-source>", "exec"), namespace)
    fn = namespace[name]
    params = [
        p for p in __import__("inspect").signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return fn, len(params)


def _types_match(got, expected) -> bool:
    """Recursive EXACT type match (True == 1 must not pass as bool)."""
    if type(got) is not type(expected):
        return False
    if isinstance(got, dict):
        return all(
            _types_match(got[k], expected[k]) and _types_match(k, k)
            for k in got
        ) and set(got) == set(expected)
    if isinstance(got, list):
        return len(got) == len(expected) and all(
            _types_match(a, b) for a, b in zip(got, expected)
        )
    return True


def main() -> int:
    # Frozen v1 dataset: ids, families and prompt shapes the new cases must
    # not collide with (read from the BUILT dataset, not the source cases).
    v1_ids, v1_families, v1_shapes = set(), set(), set()
    for split in ("training", "development", "heldout", "final", "controls"):
        for line in (V1_DIR / ("%s.jsonl" % split)).read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            v1_ids.add(row["sample_id"])
            v1_families.add(row["family_id"])
            v1_shapes.add(_shape(row["prompt"]))

    rows = []
    seen_shapes: dict = {}
    for sample_id, family_id, spec, ref_src, buggy_src, checks in CASES:
        fn_name = checks[0][1]
        ref, arity = _load_fn(ref_src, fn_name)
        buggy, _ = _load_fn(buggy_src, fn_name)
        # ids unique within the case, >=2 checks, arity matches signature
        ids = [c[0] for c in checks]
        assert len(ids) == len(set(ids)), sample_id
        assert len(checks) >= 2, sample_id
        for cid, fname, args, kwargs in checks:
            assert fname == fn_name and len(args) == arity, (sample_id, cid)
        # expected values computed by RUNNING the reference; verify the
        # reference passes every check exactly (value + type) against the
        # JSON-round-tripped expectation the oracle will compare with
        oracle = []
        for cid, fname, args, kwargs in checks:
            got = ref(*args, **kwargs)
            expected = json.loads(json.dumps(got))
            got2 = ref(*args, **kwargs)
            assert got2 == expected and _types_match(got2, expected), (
                sample_id, cid)
            oracle.append({
                "id": cid, "function": fname, "args": args,
                "kwargs": kwargs, "expected": expected,
            })
        # the seeded bug fails at least one check (real failure)
        failures = 0
        for check in oracle:
            try:
                got = buggy(*check["args"], **check["kwargs"])
                ok = got == check["expected"] and _types_match(
                    got, check["expected"])
            except Exception:
                ok = False
            if not ok:
                failures += 1
        assert failures >= 1, (sample_id, "buggy impl passes every check")
        # no collision with the frozen v1 dataset
        assert sample_id not in v1_ids, sample_id
        assert family_id not in v1_families, family_id
        prompt = PROMPT_TEMPLATE.format(spec=spec, buggy=buggy_src)
        shape = _shape(prompt)
        assert shape and shape not in v1_shapes, sample_id
        assert shape not in seen_shapes, sample_id
        seen_shapes[shape] = sample_id
        rows.append({
            "sample_id": sample_id,
            "split": "training",
            "group": "target",
            "family_id": family_id,
            "provenance": PROVENANCE,
            "license": "CC0-1.0",
            "expected": None,
            "oracle": oracle,
            "prompt": prompt,
        })
        print("%s: %d checks, buggy fails %d" % (sample_id, len(oracle),
                                                 failures))
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print("WROTE %s (%d cases)" % (OUT, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())