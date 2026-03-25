"""Tests for the eval framework — case parsing and assertions."""

from __future__ import annotations

from pathlib import Path

import pytest
from ag2_cli.testing.assertions import check_assertion
from ag2_cli.testing.cases import EvalAssertion, load_eval_suite


class TestLoadEvalSuite:
    def test_loads_valid_yaml(self, eval_yaml_file: Path) -> None:
        suite = load_eval_suite(eval_yaml_file)
        assert suite.name == "test-suite"
        assert suite.description == "Test evaluation suite"
        assert len(suite.cases) == 2

    def test_first_case_parsed_correctly(self, eval_yaml_file: Path) -> None:
        suite = load_eval_suite(eval_yaml_file)
        case = suite.cases[0]
        assert case.name == "basic_test"
        assert "France" in case.input
        assert len(case.assertions) == 2
        assert case.assertions[0].type == "contains"
        assert case.assertions[0].value == "Paris"

    def test_second_case_parsed_correctly(self, eval_yaml_file: Path) -> None:
        suite = load_eval_suite(eval_yaml_file)
        case = suite.cases[1]
        assert case.name == "length_test"
        assert len(case.assertions) == 3

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_eval_suite(tmp_path / "missing.yaml")

    def test_raises_on_invalid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("just a string")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_eval_suite(f)

    def test_empty_cases_list(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("name: empty\ndescription: no cases\ncases: []\n")
        suite = load_eval_suite(f)
        assert len(suite.cases) == 0


class TestContainsAssertion:
    def test_passes_when_substring_present(self) -> None:
        a = EvalAssertion(type="contains", value="Paris")
        r = check_assertion(a, "The capital of France is Paris.")
        assert r.passed is True

    def test_fails_when_substring_absent(self) -> None:
        a = EvalAssertion(type="contains", value="Paris")
        r = check_assertion(a, "The capital is London.")
        assert r.passed is False


class TestContainsAllAssertion:
    def test_passes_when_all_present(self) -> None:
        a = EvalAssertion(type="contains_all", values=["Tokyo", "New York"])
        r = check_assertion(a, "Tokyo and New York are major cities.")
        assert r.passed is True

    def test_fails_when_one_missing(self) -> None:
        a = EvalAssertion(type="contains_all", values=["Tokyo", "Paris"])
        r = check_assertion(a, "Tokyo is in Japan.")
        assert r.passed is False


class TestContainsAnyAssertion:
    def test_passes_when_one_present(self) -> None:
        a = EvalAssertion(type="contains_any", values=["cat", "dog", "bird"])
        r = check_assertion(a, "I have a dog.")
        assert r.passed is True

    def test_fails_when_none_present(self) -> None:
        a = EvalAssertion(type="contains_any", values=["cat", "dog"])
        r = check_assertion(a, "I have a fish.")
        assert r.passed is False


class TestNotContainsAssertion:
    def test_passes_when_absent(self) -> None:
        a = EvalAssertion(type="not_contains", value="error")
        r = check_assertion(a, "Everything is fine.")
        assert r.passed is True

    def test_fails_when_present(self) -> None:
        a = EvalAssertion(type="not_contains", value="error")
        r = check_assertion(a, "An error occurred.")
        assert r.passed is False


class TestRegexAssertion:
    def test_passes_on_match(self) -> None:
        a = EvalAssertion(type="regex", pattern=r"\d{4}")
        r = check_assertion(a, "The year is 2026.")
        assert r.passed is True

    def test_fails_on_no_match(self) -> None:
        a = EvalAssertion(type="regex", pattern=r"\d{4}")
        r = check_assertion(a, "No numbers here.")
        assert r.passed is False

    def test_uses_value_as_fallback_pattern(self) -> None:
        a = EvalAssertion(type="regex", value=r"hello\s+world")
        r = check_assertion(a, "hello   world")
        assert r.passed is True


class TestLengthAssertions:
    def test_min_length_passes(self) -> None:
        a = EvalAssertion(type="min_length", value=5)
        r = check_assertion(a, "Hello World")
        assert r.passed is True

    def test_min_length_fails(self) -> None:
        a = EvalAssertion(type="min_length", value=100)
        r = check_assertion(a, "Short")
        assert r.passed is False

    def test_max_length_passes(self) -> None:
        a = EvalAssertion(type="max_length", value=100)
        r = check_assertion(a, "Short text")
        assert r.passed is True

    def test_max_length_fails(self) -> None:
        a = EvalAssertion(type="max_length", value=5)
        r = check_assertion(a, "This is too long")
        assert r.passed is False


class TestMaxTurnsAssertion:
    def test_passes_within_limit(self) -> None:
        a = EvalAssertion(type="max_turns", value=5)
        r = check_assertion(a, "output", turns=3)
        assert r.passed is True

    def test_fails_over_limit(self) -> None:
        a = EvalAssertion(type="max_turns", value=2)
        r = check_assertion(a, "output", turns=5)
        assert r.passed is False

    def test_exact_limit_passes(self) -> None:
        a = EvalAssertion(type="max_turns", value=3)
        r = check_assertion(a, "output", turns=3)
        assert r.passed is True


class TestNoErrorAssertion:
    def test_passes_with_no_errors(self) -> None:
        a = EvalAssertion(type="no_error")
        r = check_assertion(a, "output", errors=[])
        assert r.passed is True

    def test_fails_with_errors(self) -> None:
        a = EvalAssertion(type="no_error")
        r = check_assertion(a, "output", errors=["something went wrong"])
        assert r.passed is False


class TestUnknownAssertion:
    def test_unknown_type_fails(self) -> None:
        a = EvalAssertion(type="nonexistent_type", value="foo")
        r = check_assertion(a, "output")
        assert r.passed is False
        assert "Unknown assertion type" in r.message
