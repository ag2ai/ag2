"""Tests for the ag2 test command — eval and bench subcommands."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ag2_cli.app import app
from ag2_cli.testing import CaseResult
from ag2_cli.testing.assertions import AssertionResult
from ag2_cli.testing.cases import EvalAssertion, EvalCase
from typer.testing import CliRunner

runner = CliRunner()

# The test_eval command guards with `import autogen` which is not installed
# in the CLI test environment. We inject a fake module so the import succeeds.
_mock_autogen = MagicMock()


@pytest.fixture(autouse=True)
def _fake_autogen() -> Iterator[None]:
    """Ensure ``import autogen`` succeeds inside the test_eval command."""
    with patch.dict(sys.modules, {"autogen": _mock_autogen}):
        yield


def _make_case_result(
    case: EvalCase,
    output: str,
    assertion_results: list[AssertionResult],
    turns: int = 2,
    elapsed: float = 0.5,
) -> CaseResult:
    """Helper to build a CaseResult with known values."""
    return CaseResult(
        case=case,
        assertion_results=assertion_results,
        output=output,
        turns=turns,
        elapsed=elapsed,
    )


class TestTestEval:
    """Tests for the `ag2 test eval` subcommand."""

    def test_dry_run_shows_cases_without_running(
        self,
        eval_yaml_file: Path,
        agent_file_with_main: Path,
    ) -> None:
        """--dry-run should display case counts and exit 0 without executing."""
        # Patch _run_single_case to ensure nothing actually runs.
        # The --dry-run path exits before reaching _run_single_case,
        # but we patch it as a safety net and to verify it is never called.
        with patch("ag2_cli.commands.test._run_single_case") as mock_run:
            # _run_single_case should never be called with --dry-run
            result = runner.invoke(
                app,
                [
                    "test",
                    "eval",
                    str(agent_file_with_main),
                    "--eval",
                    str(eval_yaml_file),
                    "--dry-run",
                ],
                catch_exceptions=False,
            )

        # dry-run exits 0 and shows case count
        assert result.exit_code == 0
        assert "2 case(s)" in result.output
        assert "1 suite(s)" in result.output
        # _run_single_case must NOT have been called
        mock_run.assert_not_called()

    def test_json_output_is_valid_json(
        self,
        eval_yaml_file: Path,
        agent_file_with_main: Path,
    ) -> None:
        """--output json should produce parseable JSON with suite/case structure."""
        case = EvalCase(
            name="basic_test",
            input="What is the capital of France?",
            assertions=[
                EvalAssertion(type="contains", value="Paris"),
                EvalAssertion(type="max_turns", value=5),
            ],
        )
        passing_results = [
            AssertionResult(passed=True, assertion_type="contains", message="OK"),
            AssertionResult(passed=True, assertion_type="max_turns", message="OK"),
        ]
        case_result = _make_case_result(case, "Paris is the capital.", passing_results)

        case2 = EvalCase(
            name="length_test",
            input="Write a short paragraph.",
            assertions=[
                EvalAssertion(type="min_length", value=10),
                EvalAssertion(type="max_length", value=5000),
                EvalAssertion(type="no_error"),
            ],
        )
        passing_results2 = [
            AssertionResult(passed=True, assertion_type="min_length", message="OK"),
            AssertionResult(passed=True, assertion_type="max_length", message="OK"),
            AssertionResult(passed=True, assertion_type="no_error", message="OK"),
        ]
        case_result2 = _make_case_result(case2, "A short paragraph here.", passing_results2)

        with patch(
            "ag2_cli.commands.test._run_single_case",
            side_effect=[case_result, case_result2],
        ):
            result = runner.invoke(
                app,
                [
                    "test",
                    "eval",
                    str(agent_file_with_main),
                    "--eval",
                    str(eval_yaml_file),
                    "--output",
                    "json",
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        # Extract the JSON portion from stdout (after Rich output)
        # The JSON is printed via print(), so it appears in output
        lines = result.output.strip().split("\n")
        # Find the start of JSON array
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("["):
                json_start = i
                break
        assert json_start is not None, f"No JSON array found in output:\n{result.output}"
        json_text = "\n".join(lines[json_start:])
        data = json.loads(json_text)
        assert isinstance(data, list)
        assert len(data) == 1  # one suite
        assert data[0]["suite"] == "test-suite"
        assert len(data[0]["cases"]) == 2
        assert data[0]["cases"][0]["name"] == "basic_test"
        assert data[0]["cases"][0]["passed"] is True

    def test_exits_1_when_assertions_fail(
        self,
        eval_yaml_file: Path,
        agent_file_with_main: Path,
    ) -> None:
        """Exit code 1 when any assertion in any case fails."""
        case = EvalCase(
            name="basic_test",
            input="What is the capital of France?",
            assertions=[
                EvalAssertion(type="contains", value="Paris"),
                EvalAssertion(type="max_turns", value=5),
            ],
        )
        # First assertion fails
        mixed_results = [
            AssertionResult(
                passed=False,
                assertion_type="contains",
                message="Output does not contain 'Paris'",
            ),
            AssertionResult(passed=True, assertion_type="max_turns", message="OK"),
        ]
        failing_case_result = _make_case_result(case, "London is great.", mixed_results)

        case2 = EvalCase(
            name="length_test",
            input="Write a short paragraph.",
            assertions=[
                EvalAssertion(type="min_length", value=10),
                EvalAssertion(type="max_length", value=5000),
                EvalAssertion(type="no_error"),
            ],
        )
        passing_results = [
            AssertionResult(passed=True, assertion_type="min_length", message="OK"),
            AssertionResult(passed=True, assertion_type="max_length", message="OK"),
            AssertionResult(passed=True, assertion_type="no_error", message="OK"),
        ]
        passing_case_result = _make_case_result(case2, "Some output text here.", passing_results)

        with patch(
            "ag2_cli.commands.test._run_single_case",
            side_effect=[failing_case_result, passing_case_result],
        ):
            result = runner.invoke(
                app,
                [
                    "test",
                    "eval",
                    str(agent_file_with_main),
                    "--eval",
                    str(eval_yaml_file),
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 1

    def test_exits_0_when_all_assertions_pass(
        self,
        eval_yaml_file: Path,
        agent_file_with_main: Path,
    ) -> None:
        """Exit code 0 when every assertion in every case passes."""
        case = EvalCase(
            name="basic_test",
            input="What is the capital of France?",
            assertions=[
                EvalAssertion(type="contains", value="Paris"),
                EvalAssertion(type="max_turns", value=5),
            ],
        )
        passing_results = [
            AssertionResult(passed=True, assertion_type="contains", message="OK"),
            AssertionResult(passed=True, assertion_type="max_turns", message="OK"),
        ]
        case_result = _make_case_result(case, "Paris is the capital.", passing_results)

        case2 = EvalCase(
            name="length_test",
            input="Write a short paragraph.",
            assertions=[
                EvalAssertion(type="min_length", value=10),
                EvalAssertion(type="max_length", value=5000),
                EvalAssertion(type="no_error"),
            ],
        )
        passing_results2 = [
            AssertionResult(passed=True, assertion_type="min_length", message="OK"),
            AssertionResult(passed=True, assertion_type="max_length", message="OK"),
            AssertionResult(passed=True, assertion_type="no_error", message="OK"),
        ]
        case_result2 = _make_case_result(case2, "A short paragraph.", passing_results2)

        with patch(
            "ag2_cli.commands.test._run_single_case",
            side_effect=[case_result, case_result2],
        ):
            result = runner.invoke(
                app,
                [
                    "test",
                    "eval",
                    str(agent_file_with_main),
                    "--eval",
                    str(eval_yaml_file),
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0


class TestTestBench:
    """Tests for the `ag2 test bench` subcommand."""

    def test_bench_shows_coming_soon(self) -> None:
        """bench subcommand should print 'coming soon' and exit 0."""
        result = runner.invoke(
            app,
            ["test", "bench", "fake_agent.py", "--suite", "gaia"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "coming soon" in result.output


class TestCaseResult:
    """Tests for the CaseResult dataclass."""

    def test_passed_returns_true_when_all_assertions_pass(self) -> None:
        case = EvalCase(name="test", input="hello", assertions=[])
        cr = CaseResult(
            case=case,
            assertion_results=[
                AssertionResult(passed=True, assertion_type="contains", message="ok"),
                AssertionResult(passed=True, assertion_type="max_turns", message="ok"),
                AssertionResult(passed=True, assertion_type="no_error", message="ok"),
            ],
        )
        assert cr.passed is True

    def test_passed_returns_false_when_any_assertion_fails(self) -> None:
        case = EvalCase(name="test", input="hello", assertions=[])
        cr = CaseResult(
            case=case,
            assertion_results=[
                AssertionResult(passed=True, assertion_type="contains", message="ok"),
                AssertionResult(passed=False, assertion_type="max_turns", message="fail"),
                AssertionResult(passed=True, assertion_type="no_error", message="ok"),
            ],
        )
        assert cr.passed is False

    def test_passed_count_counts_passing_assertions(self) -> None:
        case = EvalCase(name="test", input="hello", assertions=[])
        cr = CaseResult(
            case=case,
            assertion_results=[
                AssertionResult(passed=True, assertion_type="contains", message="ok"),
                AssertionResult(passed=False, assertion_type="max_turns", message="fail"),
                AssertionResult(passed=True, assertion_type="no_error", message="ok"),
                AssertionResult(passed=False, assertion_type="min_length", message="fail"),
            ],
        )
        assert cr.passed_count == 2
        assert cr.total_count == 4
