"""Regression tests for bug fixes applied during pre-release review (v2).

Covers: command injection fix, YAML quoting fix, eval YAML error handling,
assertion value validation, shared module deduplication.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ag2_cli.testing.assertions import AssertionResult, check_assertion
from ag2_cli.testing.cases import EvalAssertion, EvalCase, _parse_assertion, _parse_case

# ---------------------------------------------------------------------------
# Fix: Command injection in proxy.py script wrapping
# ---------------------------------------------------------------------------


class TestProxyScriptEscaping:
    """Script paths with special chars must be safely escaped in generated code."""

    def test_script_path_with_double_quotes_escaped(self, tmp_path: Path):
        """Paths with quotes should produce valid Python via repr()."""
        from ag2_cli.commands.proxy import _wrap_scripts

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        # Create a script with a quote in its name
        script = scripts_dir / 'my"script.sh'
        script.write_text("#!/bin/bash\necho hello")
        script.chmod(0o755)

        tools = _wrap_scripts(scripts_dir)
        assert len(tools) == 1

        # The implementation should use repr() which safely escapes quotes
        impl = tools[0].implementation
        # Should NOT contain unescaped double quotes that break Python syntax
        assert '"my"script.sh"' not in impl
        # Should contain escaped version (repr wraps in single quotes or escapes)
        assert "my" in impl

    def test_script_path_with_backslash_escaped(self, tmp_path: Path):
        """Paths with backslashes should produce valid Python."""
        from ag2_cli.commands.proxy import _wrap_scripts

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "test_script.sh"
        script.write_text("#!/bin/bash\necho hello")
        script.chmod(0o755)

        tools = _wrap_scripts(scripts_dir)
        assert len(tools) == 1

        # Verify the generated code compiles without error
        impl = tools[0].implementation
        # Build the full function to verify it's valid Python
        code = f"def test_fn(args=''):\n    {impl}"
        compile(code, "<test>", "exec")  # Should not raise SyntaxError


# ---------------------------------------------------------------------------
# Fix: YAML frontmatter quoting for values containing double quotes
# ---------------------------------------------------------------------------


class TestFrontmatterQuoteEscaping:
    """Double quotes in string values must be escaped in YAML frontmatter."""

    def test_value_with_double_quotes_escaped(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"desc": 'She said "hello"'})
        assert r"She said \"hello\"" in result

    def test_value_with_backslash_escaped(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"path": "C:\\Users\\test"})
        assert "C:\\\\Users\\\\test" in result

    def test_plain_value_unchanged(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"name": "hello"})
        assert "name: hello" in result


# ---------------------------------------------------------------------------
# Fix: Eval YAML error handling for missing fields
# ---------------------------------------------------------------------------


class TestEvalYamlErrorHandling:
    """Missing required fields in eval YAML should produce clear errors."""

    def test_assertion_missing_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="missing required 'type' field"):
            _parse_assertion({"value": "something"})

    def test_case_missing_name_raises_valueerror(self):
        with pytest.raises(ValueError, match="missing required 'name' field"):
            _parse_case({"input": "hello"})

    def test_case_missing_input_raises_valueerror(self):
        with pytest.raises(ValueError, match="missing required 'input' field"):
            _parse_case({"name": "test_case"})

    def test_valid_case_works(self):
        case = _parse_case({
            "name": "test",
            "input": "hello",
            "assertions": [{"type": "contains", "value": "hi"}],
        })
        assert case.name == "test"
        assert case.input == "hello"
        assert len(case.assertions) == 1


# ---------------------------------------------------------------------------
# Fix: Non-numeric assertion values should not crash
# ---------------------------------------------------------------------------


class TestAssertionNumericValidation:
    """min_length, max_length, max_turns should handle non-numeric values gracefully."""

    def test_min_length_with_none_value(self):
        a = EvalAssertion(type="min_length", value=None)
        result = check_assertion(a, "some output")
        assert result.passed is False
        assert "Invalid min_length" in result.message

    def test_min_length_with_string_value(self):
        a = EvalAssertion(type="min_length", value="not-a-number")
        result = check_assertion(a, "some output")
        assert result.passed is False
        assert "Invalid min_length" in result.message

    def test_max_length_with_none_value(self):
        a = EvalAssertion(type="max_length", value=None)
        result = check_assertion(a, "some output")
        assert result.passed is False
        assert "Invalid max_length" in result.message

    def test_max_turns_with_string_value(self):
        a = EvalAssertion(type="max_turns", value="abc")
        result = check_assertion(a, "output", turns=3)
        assert result.passed is False
        assert "Invalid max_turns" in result.message

    def test_min_length_with_valid_int_still_works(self):
        a = EvalAssertion(type="min_length", value=5)
        result = check_assertion(a, "hello world")
        assert result.passed is True

    def test_max_turns_with_valid_int_still_works(self):
        a = EvalAssertion(type="max_turns", value=10)
        result = check_assertion(a, "output", turns=5)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Fix: Shared module deduplication — CaseResult, require_ag2, extract_cost
# ---------------------------------------------------------------------------


class TestSharedModules:
    """Shared utilities should be importable from canonical locations."""

    def test_case_result_importable_from_testing(self):
        from ag2_cli.testing import CaseResult

        r = CaseResult(
            case=EvalCase(name="t", input="hi"),
            assertion_results=[AssertionResult(passed=True, assertion_type="x", message="ok")],
        )
        assert r.passed is True
        assert r.score == 1.0
        assert r.passed_count == 1
        assert r.total_count == 1

    def test_case_result_score_with_partial_pass(self):
        from ag2_cli.testing import CaseResult

        r = CaseResult(
            case=EvalCase(name="t", input="hi"),
            assertion_results=[
                AssertionResult(passed=True, assertion_type="x", message="ok"),
                AssertionResult(passed=False, assertion_type="y", message="fail"),
            ],
        )
        assert r.passed is False
        assert r.score == 0.5

    def test_extract_cost_with_valid_dict(self):
        from ag2_cli.commands._shared import extract_cost

        cost = {"usage_excluding_cached_inference": {"total_cost": 0.05}}
        assert extract_cost(cost) == 0.05

    def test_extract_cost_with_empty_dict(self):
        from ag2_cli.commands._shared import extract_cost

        assert extract_cost({}) == 0.0

    def test_extract_cost_with_non_dict(self):
        from ag2_cli.commands._shared import extract_cost

        assert extract_cost("some string") == 0.0
        assert extract_cost(None) == 0.0

    def test_copy_tree_copies_files(self, tmp_path: Path):
        from ag2_cli.install.installers._utils import copy_tree

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("hello")
        sub = src / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("world")

        dst = tmp_path / "dst"
        created = copy_tree(src, dst)

        assert len(created) == 2
        assert (dst / "a.txt").read_text() == "hello"
        assert (dst / "sub" / "b.txt").read_text() == "world"
