#!/usr/bin/env python
"""Build the extraction program's governed functional dataset (Section H).

Writes ``data/extraction_v1/``: case files (training/development/heldout/
controls) plus ``silt.extraction.cases.v1`` manifests validated by
``silt-extract dataset validate``, a selection lock frozen before any
model run, and a POWER_ANALYSIS computed with
``asea.extraction.stages.required_sample_size``.

HONESTY (binding):

  * Every case is hand-authored synthetic CC0 content; nothing is scraped
    and nothing is a model output. The dataset is a SEED-SCALE artifact:
    the power analysis records the per-family N the preregistered
    thresholds would need, and the gap between present N and required N
    is stated -- a below-power dataset is never presented as evaluation
    material for the preregistered hypothesis.
  * The final split is NOT generated here. It does not exist yet
    ("its final split was never generated -- never traced, judged or
    used"); it will be generated and sealed only when the program
    reaches the evaluation stage, on a host the admission gate admits.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "extraction_v1"
CAPABILITY = "repo_repair_v1"

# ---------------------------------------------------------------------------
# Case table. Target cases are buggy-function repair tasks:
#   [case_id, family, group, specification, buggy_source, oracle_function,
#    [[check_id, args, expected], ...]]
# Control cases are plain function-writing tasks:
#   [case_id, family, group, task_statement, oracle_function,
#    [[check_id, args, expected], ...]]
# ---------------------------------------------------------------------------

TARGETS = [
    ["t-offby-1", "fix_off_by_one_range", "target",
     "sum_inclusive(start, end) returns the sum of all integers from "
     "start to end, INCLUSIVE of both.",
     "def sum_inclusive(start, end):\n"
     "    total = 0\n"
     "    for i in range(start, end):\n"
     "        total += i\n"
     "    return total",
     "sum_inclusive",
     [["inclusive_ends", [1, 5], 15],
      ["single_value", [3, 3], 3],
      ["negative_range", [-2, 1], -2]]],
    ["t-offby-2", "fix_off_by_one_range", "target",
     "count_steps(n) returns how many steps of size 3 are needed to reach "
     "exactly n from 0, counting the final partial step as one step.",
     "def count_steps(n):\n"
     "    steps = 0\n"
     "    pos = 0\n"
     "    while pos < n - 3:\n"
     "        pos += 3\n"
     "        steps += 1\n"
     "    return steps",
     "count_steps",
     [["exact_multiple", [9], 3],
      ["partial_step", [10], 4],
      ["zero", [0], 0]]],
    ["d-offby-1", "fix_off_by_one_range", "target",
     "last_index_of(items, value) returns the index of the LAST occurrence "
     "of value in items, or -1 when absent.",
     "def last_index_of(items, value):\n"
     "    for i in range(len(items)):\n"
     "        if items[i] == value:\n"
     "            return i\n"
     "    return -1",
     "last_index_of",
     [["last_not_first", [[1, 2, 3, 2], 2], 3],
      ["absent", [[1, 2], 9], -1],
      ["empty", [[], 1], -1]]],

    ["t-null-1", "fix_missing_null_guard", "target",
     "first_char(text) returns the first character of text, or the empty "
     "string when text is empty or None.",
     "def first_char(text):\n"
     "    return text[0]",
     "first_char",
     [["normal", ["hello"], "h"],
      ["empty", [""], ""],
      ["none", [None], ""]]],
    ["t-null-2", "fix_missing_null_guard", "target",
     "safe_get(mapping, key) returns mapping[key] when present, else "
     "None; mapping itself may be None.",
     "def safe_get(mapping, key):\n"
     "    return mapping[key]",
     "safe_get",
     [["present", [{"a": 1}, "a"], 1],
      ["absent", [{"a": 1}, "b"], None],
      ["none_mapping", [None, "a"], None]]],
    ["d-null-1", "fix_missing_null_guard", "target",
     "describe_count(items) returns 'count=<n>' with n the number of "
     "items; None input means zero items.",
     "def describe_count(items):\n"
     "    return 'count=%d' % len(items)",
     "describe_count",
     [["normal", [[1, 2]], "count=2"],
      ["none", [None], "count=0"],
      ["empty", [[]], "count=0"]]],

    ["t-mut-1", "fix_mutation_during_iteration", "target",
     "drop_zeros(items) returns a NEW list with every 0 removed; the "
     "input list must be left unmodified.",
     "def drop_zeros(items):\n"
     "    for x in items:\n"
     "        if x == 0:\n"
     "            items.remove(x)\n"
     "    return items",
     "drop_zeros",
     [["removes_zeros", [[1, 0, 2, 0, 3]], [1, 2, 3]],
      ["consecutive_zeros_skipped_by_iteration_bug", [[0, 0, 1, 0]], [1]]]],
    ["t-mut-2", "fix_mutation_during_iteration", "target",
     "unique_in_order(items) returns a new list with consecutive "
     "duplicates removed, without mutating items.",
     "def unique_in_order(items):\n"
     "    out = []\n"
     "    for x in items:\n"
     "        if not out or out[0] != x:\n"
     "            out.append(x)\n"
     "    return out",
     "unique_in_order",
     [["collapses_runs", [["a", "a", "b", "a"]], ["a", "b", "a"]],
      ["collapses_pair", [[1, 1, 2]], [1, 2]]]],
    ["d-mut-1", "fix_mutation_during_iteration", "target",
     "evens_only(items) returns a new list of the even elements; items "
     "must remain unchanged after the call.",
     "def evens_only(items):\n"
     "    for x in items:\n"
     "        if x % 2:\n"
     "            items.remove(x)\n"
     "    return items",
     "evens_only",
     [["keeps_evens", [[1, 2, 3, 4]], [2, 4]],
      ["all_odd_returns_empty", [[1, 3]], []]]],

    ["t-str-1", "fix_string_edge_case", "target",
     "initials(name) returns the first letter of each "
     "whitespace-separated word, uppercased; an empty name yields ''.",
     "def initials(name):\n"
     "    return ''.join(w[0] for w in name.split(' ')).upper()",
     "initials",
     [["two_words", ["ada lovelace"], "AL"],
      ["multiple_spaces", ["grace  hopper"], "GH"],
      ["empty", [""], ""]]],
    ["t-str-2", "fix_string_edge_case", "target",
     "slugify(text) lowercases text and replaces each run of "
     "non-alphanumeric characters with a single hyphen, with no leading "
     "or trailing hyphen.",
     "def slugify(text):\n"
     "    return text.replace(' ', '-').lower()",
     "slugify",
     [["spaces", ["Hello World"], "hello-world"],
      ["punct_runs", ["A--B!!C"], "a-b-c"],
      ["leading_trailing", ["  hi  "], "hi"]]],
    ["d-str-1", "fix_string_edge_case", "target",
     "pad_center(text, width) centers text with spaces and returns a "
     "string of exactly width characters; width shorter than text "
     "returns text unchanged, and with an odd number of padding spaces "
     "the extra space goes on the right.",
     "def pad_center(text, width):\n"
     "    return text.ljust(width)",
     "pad_center",
     [["even_pad", ["ab", 6], "  ab  "],
      ["odd_pad", ["abc", 6], " abc  "],
      ["narrow", ["abcdef", 3], "abcdef"]]],

    ["t-key-1", "fix_dict_key_error", "target",
     "tally(words) returns a dict counting occurrences of each word; "
     "words not present start at 1, never 0.",
     "def tally(words):\n"
     "    counts = {}\n"
     "    for w in words:\n"
     "        counts[w] += 1\n"
     "    return counts",
     "tally",
     [["repeats", [["a", "b", "a"]], {"a": 2, "b": 1}],
      ["empty", [[]], {}],
      ["single", [["x"]], {"x": 1}]]],
    ["t-key-2", "fix_dict_key_error", "target",
     "merged(a, b) returns a NEW dict with b's entries overriding a's; "
     "missing keys in either input are fine.",
     "def merged(a, b):\n"
     "    out = {}\n"
     "    out.update(b)\n"
     "    out.update(a)\n"
     "    return out",
     "merged",
     [["b_overrides_a", [{"x": 1}, {"x": 2}], {"x": 2}],
      ["disjoint", [{"x": 1}, {"y": 2}], {"x": 1, "y": 2}],
      ["empty_b_returns_a", [{"x": 1}, {}], {"x": 1}]]],
    ["d-key-1", "fix_dict_key_error", "target",
     "lookup_nested(mapping, outer, inner) returns mapping[outer][inner] "
     "when the full path exists, else None.",
     "def lookup_nested(mapping, outer, inner):\n"
     "    return mapping[outer][inner]",
     "lookup_nested",
     [["present", [{"a": {"b": 1}}, "a", "b"], 1],
      ["outer_missing", [{}, "a", "b"], None],
      ["inner_missing", [{"a": {}}, "a", "b"], None]]],

    ["t-div-1", "fix_integer_division_floor", "target",
     "pages_needed(items, per_page) returns the number of pages needed "
     "to list items in groups of per_page; an exact multiple needs "
     "exactly items//per_page pages.",
     "def pages_needed(items, per_page):\n"
     "    return items / per_page",
     "pages_needed",
     [["exact", [20, 10], 2],
      ["remainder", [21, 10], 3],
      ["zero", [0, 10], 0]]],
    ["t-div-2", "fix_integer_division_floor", "target",
     "midpoint(low, high) returns the integer floor midpoint between low "
     "and high.",
     "def midpoint(low, high):\n"
     "    return (low + high) / 2",
     "midpoint",
     [["odd_floor", [1, 4], 2],
      ["even", [2, 6], 4],
      ["negative", [-3, 0], -2]]],
    ["d-div-1", "fix_integer_division_floor", "target",
     "bucket_index(value, bucket_size) returns the zero-based index of "
     "the bucket value falls into; bucket_size must be positive.",
     "def bucket_index(value, bucket_size):\n"
     "    return value / bucket_size",
     "bucket_index",
     [["in_bucket", [25, 10], 2],
      ["boundary", [30, 10], 3],
      ["below_zero", [-5, 10], -1]]],

    ["t-flt-1", "fix_float_compare_tolerance", "target",
     "nearly_equal(a, b) returns True when a and b differ by less than "
     "1e-9.",
     "def nearly_equal(a, b):\n"
     "    return a == b",
     "nearly_equal",
     [["tiny_difference", [0.1 + 0.2, 0.3], True],
      ["far_apart", [1.0, 2.0], False],
      ["identical", [1.5, 1.5], True]]],
    ["t-flt-2", "fix_float_compare_tolerance", "target",
     "accumulate(values) returns the running total rounded to 6 decimal "
     "places, tolerating binary float drift.",
     "def accumulate(values):\n"
     "    total = 0.0\n"
     "    for v in values:\n"
     "        total = total + v\n"
     "    return total",
     "accumulate",
     [["drift", [[0.1] * 10], 1.0],
      ["empty", [[]], 0.0],
      ["mixed", [[0.1, 0.2, 0.3]], 0.6]]],
    ["d-flt-1", "fix_float_compare_tolerance", "target",
     "percent(part, whole) returns part/whole as a percentage rounded to "
     "4 decimals; whole == 0 returns 0.0.",
     "def percent(part, whole):\n"
     "    return (part / whole) * 100",
     "percent",
     [["normal", [1, 3], 33.3333],
      ["zero_whole", [1, 0], 0.0],
      ["full", [3, 3], 100.0]]],

    ["t-reg-1", "fix_regex_anchor", "target",
     "is_hex_color(text) returns True only when text is EXACTLY a "
     "6-digit hex color like '#a3fF00'.",
     "import re\n\n"
     "def is_hex_color(text):\n"
     "    return bool(re.search(r'#[0-9a-fA-F]{6}', text))",
     "is_hex_color",
     [["valid", ["#a3fF00"], True],
      ["extra_text", ["x #a3fF00 y"], False],
      ["too_short", ["#a3fF"], False]]],
    ["t-reg-2", "fix_regex_anchor", "target",
     "ends_with_digit(text) returns True when text ENDS with a digit.",
     "import re\n\n"
     "def ends_with_digit(text):\n"
     "    return bool(re.search(r'\\d', text))",
     "ends_with_digit",
     [["ends", ["abc1"], True],
      ["digit_inside_only", ["a1b"], False],
      ["none", ["abc"], False]]],
    ["d-reg-1", "fix_regex_anchor", "target",
     "count_whole_words(text, word) counts occurrences of word as a "
     "WHOLE word, case-sensitively.",
     "def count_whole_words(text, word):\n"
     "    return text.count(word)",
     "count_whole_words",
     [["whole_only", ["the cat the theorem", "the"], 2],
      ["no_substrings", ["cat catalog", "cat"], 1],
      ["absent", ["dog", "cat"], 0]]],

    ["t-srt-1", "fix_sort_key_stability", "target",
     "by_score_then_name(pairs) returns pairs sorted by score ascending; "
     "ties are broken by name ascending.",
     "def by_score_then_name(pairs):\n"
     "    return sorted(pairs, key=lambda p: p[1])",
     "by_score_then_name",
     [["tie_breaks", [[(2, "b"), (1, "z"), (2, "a")]],
       [(1, "z"), (2, "a"), (2, "b")]],
      ["empty", [[]], []],
      ["single", [[(1, "x")]], [(1, "x")]]]],
    ["t-srt-2", "fix_sort_key_stability", "target",
     "top_by_score(pairs, k) returns the k highest-scoring pairs, "
     "highest first; ties keep their original relative order.",
     "def top_by_score(pairs, k):\n"
     "    return sorted(pairs, key=lambda p: p[0])[:k]",
     "top_by_score",
     [["highest_first", [[(1, "a"), (3, "c"), (2, "b")], 2],
       [(3, "c"), (2, "b")]],
      ["k_over_length", [[(1, "a")], 5], [(1, "a")]],
      ["tie_order", [[(2, "x"), (1, "y"), (2, "z")], 2],
       [(2, "x"), (2, "z")]]]],
    ["d-srt-1", "fix_sort_key_stability", "target",
     "sort_case_insensitive(words) returns the words sorted "
     "case-insensitively, preserving the original spelling.",
     "def sort_case_insensitive(words):\n"
     "    return sorted(words)",
     "sort_case_insensitive",
     [["ignores_case", [["Banana", "apple", "Cherry"]],
       ["apple", "Banana", "Cherry"]],
      ["empty", [[]], []],
      ["stable_spellings", [["b", "A", "a"]], ["A", "a", "b"]]]],

    ["t-emp-1", "fix_empty_input_return", "target",
     "min_positive(numbers) returns the smallest strictly-positive "
     "number, or None when there is none.",
     "def min_positive(numbers):\n"
     "    best = numbers[0]\n"
     "    for n in numbers:\n"
     "        if n < best:\n"
     "            best = n\n"
     "    return best",
     "min_positive",
     [["mixed", [[-1, 2, 3]], 2],
      ["all_negative", [[-1, -5]], None],
      ["empty", [[]], None]]],
    ["t-emp-2", "fix_empty_input_return", "target",
     "longest_word(words) returns the longest word, or '' for an empty "
     "list; ties keep the earliest word.",
     "def longest_word(words):\n"
     "    return max(words, key=len)",
     "longest_word",
     [["normal", [["hi", "hello", "hey"]], "hello"],
      ["tie_keeps_first", [["cat", "dog", "bird"]], "cat"],
      ["empty", [[]], ""]]],
    ["d-emp-1", "fix_empty_input_return", "target",
     "mean_or_zero(numbers) returns the arithmetic mean, or 0 for an "
     "empty list.",
     "def mean_or_zero(numbers):\n"
     "    return sum(numbers) / len(numbers)",
     "mean_or_zero",
     [["normal", [[1, 2, 3]], 2.0],
      ["empty", [[]], 0],
      ["floats", [[0.5, 1.0]], 0.75]]],

    ["t-typ-1", "fix_type_coercion_bool", "target",
     "count_falsy(values) returns how many of the values are falsy "
     "(None, False, 0, '', [] all count).",
     "def count_falsy(values):\n"
     "    return sum(1 for v in values if v == False)",
     "count_falsy",
     [["mixed", [[0, False, None, "", 1]], 4],
      ["all_truthy", [[1, "x"]], 0],
      ["empty", [[]], 0]]],
    ["t-typ-2", "fix_type_coercion_bool", "target",
     "as_int(value) converts value to an int when it is an int or a "
     "numeric string, and returns None otherwise.",
     "def as_int(value):\n"
     "    return int(value)",
     "as_int",
     [["int_passthrough", [7], 7],
      ["numeric_string", ["42"], 42],
      ["non_numeric", ["x"], None],
      ["float_rejected", [1.5], None]]],
    ["d-typ-1", "fix_type_coercion_bool", "target",
     "flag_bits(flags) returns the number of True values among the "
     "flags; truthy non-bool values do not count.",
     "def flag_bits(flags):\n"
     "    return sum(1 for f in flags if f)",
     "flag_bits",
     [["bools", [[True, False, True]], 2],
      ["truthy_ignored", [[1, "x", True]], 1],
      ["empty", [[]], 0]]],

    ["t-slc-1", "fix_slice_bounds", "target",
     "last_n(items, n) returns the final n items as a new list; n larger "
     "than the length returns a copy of everything.",
     "def last_n(items, n):\n"
     "    return items[-n:]",
     "last_n",
     [["normal", [[1, 2, 3, 4], 2], [3, 4]],
      ["n_too_large", [[1, 2], 5], [1, 2]],
      ["zero", [[1, 2], 0], []]]],
    ["t-slc-2", "fix_slice_bounds", "target",
     "without_first(items, n) returns items minus its first n entries; n "
     "beyond the length yields an empty list.",
     "def without_first(items, n):\n"
     "    return items[1:n]",
     "without_first",
     [["normal", [[1, 2, 3, 4], 2], [3, 4]],
      ["n_beyond", [[1, 2], 9], []],
      ["zero", [[1, 2], 0], [1, 2]]]],
    ["d-slc-1", "fix_slice_bounds", "target",
     "middle_half(items) returns items excluding the first and last "
     "quarter (floor of each).",
     "def middle_half(items):\n"
     "    return items[len(items) // 2:]",
     "middle_half",
     [["even", [[1, 2, 3, 4, 5, 6, 7, 8]], [3, 4, 5, 6]],
      ["small", [[1, 2]], []],
      ["empty", [[]], []]]],

    ["t-acc-1", "fix_accumulator_init", "target",
     "product(numbers) returns the product of all numbers; an empty list "
     "yields 1.",
     "def product(numbers):\n"
     "    total = 0\n"
     "    for n in numbers:\n"
     "        total *= n\n"
     "    return total",
     "product",
     [["normal", [[2, 3, 4]], 24],
      ["empty", [[]], 1],
      ["zero_element", [[5, 0]], 0]]],
    ["t-acc-2", "fix_accumulator_init", "target",
     "join_nonempty(parts, sep) joins the non-empty parts with sep; "
     "all-empty input yields ''.",
     "def join_nonempty(parts, sep):\n"
     "    out = None\n"
     "    for p in parts:\n"
     "        if p:\n"
     "            out = out + p if out is not None else p\n"
     "    return out",
     "join_nonempty",
     [["normal", [["a", "", "b"], "-"], "a-b"],
      ["all_empty", [["", ""], "-"], ""],
      ["single", [["x"], "-"], "x"]]],
    ["d-acc-1", "fix_accumulator_init", "target",
     "max_gap(numbers) returns the largest difference between "
     "consecutive entries; fewer than two entries yields 0.",
     "def max_gap(numbers):\n"
     "    best = 0\n"
     "    for i in range(len(numbers)):\n"
     "        best = max(best, numbers[i + 1] - numbers[i])\n"
     "    return best",
     "max_gap",
     [["normal", [[1, 5, 3, 9]], 6],
      ["single", [[4]], 0],
      ["empty", [[]], 0],
      ["descending", [[9, 1]], -8]]],

    ["t-ear-1", "fix_early_return_path", "target",
     "classify(n) returns 'neg' for negative n, 'zero' for 0, and 'pos' "
     "otherwise.",
     "def classify(n):\n"
     "    if n < 0:\n"
     "        return 'neg'\n"
     "    return 'pos'",
     "classify",
     [["negative", [-3], "neg"],
      ["zero", [0], "zero"],
      ["positive", [7], "pos"]]],
    ["t-ear-2", "fix_early_return_path", "target",
     "find_first_divisible(items, d) returns the first item divisible by "
     "d, or None when none is.",
     "def find_first_divisible(items, d):\n"
     "    for x in items:\n"
     "        if d:\n"
     "            return x\n"
     "    return None",
     "find_first_divisible",
     [["found", [[3, 6, 9], 3], 3],
      ["skips_nonmatching", [[4, 6, 9], 3], 6],
      ["none", [[5, 7], 3], None]]],
    ["d-ear-1", "fix_early_return_path", "target",
     "sign_product(a, b) returns 'same' when a and b share a sign "
     "(treating 0 as its own class), else 'diff'.",
     "def sign_product(a, b):\n"
     "    if a * b > 0:\n"
     "        return 'same'\n"
     "    return 'diff'",
     "sign_product",
     [["both_pos", [1, 2], "same"],
      ["both_neg", [-1, -2], "same"],
      ["opposite", [-1, 2], "diff"],
      ["zero_is_own_class", [0, 0], "same"]]],

    ["h-fix-1", "fix_off_by_one_range", "target",
     "repeat_to_length(items, n) returns a list of length n built by "
     "cycling items from the start; an empty items list yields an empty "
     "list.",
     "def repeat_to_length(items, n):\n"
     "    return items * n",
     "repeat_to_length",
     [["shorter", [[1, 2], 5], [1, 2, 1, 2, 1]],
      ["exact", [[1, 2], 2], [1, 2]],
      ["empty_items", [[], 3], []]]],
    ["h-fix-2", "fix_dict_key_error", "target",
     "group_by_first(pairs) returns a dict mapping each first element to "
     "the list of second elements that followed it.",
     "def group_by_first(pairs):\n"
     "    out = {}\n"
     "    for k, v in pairs:\n"
     "        out[k] = [v]\n"
     "    return out",
     "group_by_first",
     [["groups", [[("a", 1), ("b", 2), ("a", 3)]],
       {"a": [1, 3], "b": [2]}],
      ["empty", [[]], {}]]],
    ["h-fix-3", "fix_string_edge_case", "target",
     "swap_case_first(text) returns text with ONLY its first character's "
     "case swapped; empty text stays empty.",
     "def swap_case_first(text):\n"
     "    return text[0].swapcase() + text[1:]",
     "swap_case_first",
     [["lower_first", ["hello"], "Hello"],
      ["upper_first", ["World"], "world"],
      ["empty", [""], ""]]],
    ["h-fix-4", "fix_accumulator_init", "target",
     "xor_all(bits) returns the XOR of all bits; an empty list yields "
     "False.",
     "def xor_all(bits):\n"
     "    acc = None\n"
     "    for b in bits:\n"
     "        acc = acc ^ b\n"
     "    return acc",
     "xor_all",
     [["odd_true", [[True, True, True]], True],
      ["even_false", [[True, True]], False],
      ["empty", [[]], False]]],
]

CONTROLS = [
    ["c-math-1", "math_word_problem", "control",
     "A shop sells pens at 3 for 12 rupees. Write a function pens_cost(n) "
     "that returns the cost in rupees of n pens, where partial groups of "
     "fewer than 3 pens are charged per pen at the group rate. Return "
     "ONLY the complete function.",
     "pens_cost",
     [["exact_groups", [6], 24],
      ["partial_group", [7], 28],
      ["zero", [0], 0]]],
    ["c-math-2", "math_word_problem", "control",
     "Write a function triangle_kind(a, b, c) returning 'equilateral', "
     "'isosceles' or 'scalene' for the triangle with those side "
     "lengths. Return ONLY the complete function.",
     "triangle_kind",
     [["equilateral", [3, 3, 3], "equilateral"],
      ["isosceles", [3, 3, 4], "isosceles"],
      ["scalene", [3, 4, 5], "scalene"]]],
    ["c-tr-1", "translation_phrase", "control",
     "Write a function translate_greeting(lang) that returns the "
     "standard greeting for a language: 'en' -> 'Hello', 'hi' -> "
     "'Namaste', 'bn' -> 'Nomoshkar', anything else -> the empty "
     "string. Return ONLY the complete function.",
     "translate_greeting",
     [["en", ["en"], "Hello"],
      ["hi", ["hi"], "Namaste"],
      ["bn", ["bn"], "Nomoshkar"],
      ["unknown", ["zz"], ""]]],
    ["c-tr-2", "translation_phrase", "control",
     "Write a function number_word_en(n) that returns the English word "
     "for a digit 0..9, or '' outside that range. Return ONLY the "
     "complete function.",
     "number_word_en",
     [["zero", [0], "zero"],
      ["nine", [9], "nine"],
      ["out_of_range", [10], ""]]],
    ["c-qa-1", "general_qa", "control",
     "Write a function days_in_february(year) returning 28 or 29 "
     "according to the Gregorian leap-year rule. Return ONLY the "
     "complete function.",
     "days_in_february",
     [["leap", [2024], 29],
      ["century_non_leap", [1900], 28],
      ["quad_century", [2000], 29],
      ["plain", [2023], 28]]],
    ["c-qa-2", "general_qa", "control",
     "Write a function opposite_cardinal(direction) returning the "
     "opposite compass direction from 'N', 'S', 'E' or 'W', or '' "
     "otherwise. Return ONLY the complete function.",
     "opposite_cardinal",
     [["north", ["N"], "S"],
      ["east", ["E"], "W"],
      ["invalid", ["X"], ""]]],
    ["c-nc-1", "non_target_coding", "control",
     "Write a function csv_escape(field) that returns field ready for a "
     "CSV cell: wrapping in double quotes ONLY when it contains a comma, "
     "quote or newline, and doubling internal quotes when wrapped. "
     "Return ONLY the complete function.",
     "csv_escape",
     [["plain", ["abc"], "abc"],
      ["comma", ["a,b"], '"a,b"'],
      ["internal_quotes", ["say \"hi\""], "\"say \"\"hi\"\"\""],
      ["newline", ["a\nb"], '"a\nb"']]],
    ["c-nc-2", "non_target_coding", "control",
     "Write a function roman_units(n) returning the Roman numeral for n "
     "in 1..9, or '' otherwise. Return ONLY the complete function.",
     "roman_units",
     [["one", [1], "I"],
      ["four", [4], "IV"],
      ["nine", [9], "IX"],
      ["out_of_range", [0], ""]]],
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repair_prompt(spec: str, buggy: str) -> str:
    return (
        "The following Python function has a bug. Fix it so it meets "
        "its specification exactly. Keep the function name and "
        "signature unchanged. Return ONLY the complete fixed "
        "function.\n\nSpecification: %s\n\n```python\n%s\n```"
        % (spec, buggy)
    )


def _oracle(function: str, checks: list) -> dict:
    return {
        "function": function,
        "checks": [
            {"id": cid, "args": args, "kwargs": {}, "expected": expected}
            for cid, args, expected in checks
        ],
    }


def _case(case_id: str, family: str, group: str, prompt: str,
          oracle: dict) -> dict:
    content = json.dumps({"prompt": prompt, "oracle": oracle},
                         sort_keys=True)
    return {
        "case_id": case_id,
        "family": family,
        "group": group,
        "prompt": prompt,
        "oracle": oracle,
        "license": "CC0-1.0",
        "content_sha256": _sha(content),
        "prompt_sha256": _sha(prompt),
    }


def _split_of(case_id: str) -> str:
    if case_id.startswith("t-"):
        return "training"
    if case_id.startswith("d-"):
        return "development"
    if case_id.startswith("h-"):
        return "heldout"
    if case_id.startswith("c-"):
        return "controls"
    raise SystemExit("unroutable case id: %s" % case_id)


def main() -> int:
    cases = [
        _case(cid, family, group, _repair_prompt(spec, buggy),
              _oracle(function, checks))
        for cid, family, group, spec, buggy, function, checks in TARGETS
    ] + [
        _case(cid, family, group, task, _oracle(function, checks))
        for cid, family, group, task, function, checks in CONTROLS
    ]

    # Near-duplicate discipline: prompt hashes unique across EVERY split
    # of this capability, and case ids globally unique.
    prompts = [c["prompt_sha256"] for c in cases]
    if len(set(prompts)) != len(prompts):
        raise SystemExit("duplicate prompt content across the dataset")
    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate case ids")

    DATA.mkdir(parents=True, exist_ok=True)
    by_split: dict = {}
    for case in cases:
        by_split.setdefault(_split_of(case["case_id"]), []).append(case)

    for split, split_cases in sorted(by_split.items()):
        body = "\n".join(json.dumps(c, sort_keys=True) for c in split_cases)
        (DATA / ("%s.jsonl" % split)).write_text(body + "\n", encoding="utf-8")
        manifest = {
            "schema": "silt.extraction.cases.v1",
            "capability_id": CAPABILITY,
            "split": split,
            "near_duplicate_free": True,
            "cases": [
                {"case_id": c["case_id"], "family": c["family"],
                 "group": c["group"], "content_sha256": c["content_sha256"],
                 "prompt_sha256": c["prompt_sha256"], "license": c["license"]}
                for c in split_cases
            ],
        }
        (DATA / ("%s-manifest.json" % split)).write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    # Selection lock: frozen before any model run, final split NOT generated.
    lock = {
        "schema": "silt.extraction.selection_lock.v1",
        "capability_id": CAPABILITY,
        "frozen_before_model_generation": True,
        "model_outputs_consulted": False,
        "final_generated": False,
        "note": (
            "the final split does not exist yet; it will be generated "
            "and sealed only at the evaluation stage, on a host the "
            "admission gate admits"
        ),
    }
    (DATA / "selection-lock.json").write_text(
        json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8")

    # Power analysis (Section H): the per-family N the preregistered
    # thresholds would need vs the N present. The gap is stated, never
    # papered over: this dataset is SEED-SCALE.
    from asea.extraction.stages import required_sample_size

    required = required_sample_size(
        baseline_pass_rate=0.9, minimum_detectable_effect=0.3)
    present: dict = {}
    for case in cases:
        if case["group"] == "target":
            present[case["family"]] = present.get(case["family"], 0) + 1
    power = {
        "capability_id": CAPABILITY,
        "test": "one-sided two-proportion, alpha=0.05, power=0.8",
        "baseline_pass_rate_assumed": 0.9,
        "minimum_detectable_effect": 0.3,
        "required_n_per_family": required,
        "present_n_per_family": present,
        "families_below_required_n": sorted(
            f for f, n in present.items() if n < required),
        "verdict": (
            "SEED-SCALE: every target family is below the required N; "
            "this dataset supports mechanism and pilot use only and is "
            "NOT evaluation material for the preregistered hypothesis"
        ),
    }
    (DATA / "power-analysis.json").write_text(
        json.dumps(power, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "capability_id": CAPABILITY,
        "splits": {k: len(v) for k, v in sorted(by_split.items())},
        "target_families": len({c["family"] for c in cases
                                 if c["group"] == "target"}),
        "control_families": len({c["family"] for c in cases
                                  if c["group"] == "control"}),
        "required_n_per_family": required,
        "verdict": power["verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())