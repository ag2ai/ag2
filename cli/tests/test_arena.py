"""Tests for ag2 arena — A/B testing agent implementations."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from ag2_cli.app import app
from ag2_cli.commands.arena import (
    ContenderResult,
    _compute_elo,
    _determine_case_winner,
    _flat_cases,
    _load_leaderboard,
    _resolve_contender_files,
    _save_leaderboard,
)
from ag2_cli.testing import CaseResult
from ag2_cli.testing.assertions import AssertionResult
from ag2_cli.testing.cases import EvalCase, EvalSuite
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_autogen():
    """Mock the `import autogen` guard so arena commands work without ag2 installed."""
    import types

    fake_autogen = types.ModuleType("autogen")
    with patch.dict("sys.modules", {"autogen": fake_autogen}):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_agent_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create two simple agent files with main() functions."""
    f1 = tmp_path / "agent_v1.py"
    f1.write_text(
        textwrap.dedent("""\
        def main(message="hello"):
            return f"V1 response to: {message}. Paris is the capital."
        """)
    )
    f2 = tmp_path / "agent_v2.py"
    f2.write_text(
        textwrap.dedent("""\
        def main(message="hello"):
            return f"V2 response to: {message}. Paris is the capital of France."
        """)
    )
    return f1, f2


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple agent files."""
    d = tmp_path / "agents"
    d.mkdir()
    (d / "alpha.py").write_text('def main(message=""): return "alpha output Paris"')
    (d / "beta.py").write_text('def main(message=""): return "beta output Paris France"')
    (d / "gamma.py").write_text('def main(message=""): return "gamma output"')
    return d


@pytest.fixture
def eval_file(tmp_path: Path) -> Path:
    """Create a simple eval YAML file."""
    f = tmp_path / "cases.yaml"
    f.write_text(
        textwrap.dedent("""\
        name: "test-arena"
        description: "Arena test cases"

        cases:
          - name: "capitals"
            input: "What is the capital of France?"
            assertions:
              - type: contains
                value: "Paris"

          - name: "detail"
            input: "Be detailed about France."
            assertions:
              - type: contains
                value: "France"
              - type: min_length
                value: 10
        """)
    )
    return f


@pytest.fixture
def leaderboard_dir(tmp_path: Path) -> Path:
    """Provide a temp leaderboard directory."""
    lb_dir = tmp_path / ".ag2" / "arena"
    lb_dir.mkdir(parents=True)
    return lb_dir


# ---------------------------------------------------------------------------
# CaseResult tests
# ---------------------------------------------------------------------------


class TestCaseResult:
    def test_passed_all_assertions_pass(self) -> None:
        r = CaseResult(
            case=EvalCase(name="test", input="hi"),
            assertion_results=[
                AssertionResult(passed=True, assertion_type="contains", message="ok"),
                AssertionResult(passed=True, assertion_type="min_length", message="ok"),
            ],
        )
        assert r.passed is True
        assert r.score == 1.0

    def test_passed_some_assertions_fail(self) -> None:
        r = CaseResult(
            case=EvalCase(name="test", input="hi"),
            assertion_results=[
                AssertionResult(passed=True, assertion_type="contains", message="ok"),
                AssertionResult(passed=False, assertion_type="min_length", message="fail"),
            ],
        )
        assert r.passed is False
        assert r.score == 0.5

    def test_score_empty_assertions(self) -> None:
        r = CaseResult(case=EvalCase(name="test", input="hi"))
        assert r.score == 0.0


class TestContenderResult:
    def test_pass_rate(self) -> None:
        cases = [
            CaseResult(
                case=EvalCase(name="c1", input=""),
                assertion_results=[AssertionResult(passed=True, assertion_type="x", message="ok")],
            ),
            CaseResult(
                case=EvalCase(name="c2", input=""),
                assertion_results=[AssertionResult(passed=False, assertion_type="x", message="fail")],
            ),
        ]
        cr = ContenderResult(name="test", source="test.py", case_results=cases)
        assert cr.pass_rate == 0.5

    def test_empty_results(self) -> None:
        cr = ContenderResult(name="test", source="test.py")
        assert cr.pass_rate == 0.0
        assert cr.avg_score == 0.0
        assert cr.avg_elapsed == 0.0
        assert cr.total_cost == 0.0


# ---------------------------------------------------------------------------
# Winner determination tests
# ---------------------------------------------------------------------------


class TestDetermineWinner:
    def test_one_passes_one_fails(self) -> None:
        r_pass = CaseResult(
            case=EvalCase(name="c", input=""),
            assertion_results=[AssertionResult(passed=True, assertion_type="x", message="ok")],
        )
        r_fail = CaseResult(
            case=EvalCase(name="c", input=""),
            assertion_results=[AssertionResult(passed=False, assertion_type="x", message="fail")],
        )
        winner = _determine_case_winner({"a": r_pass, "b": r_fail})
        assert winner == "a"

    def test_both_pass_same_score_is_tie(self) -> None:
        r = CaseResult(
            case=EvalCase(name="c", input=""),
            assertion_results=[AssertionResult(passed=True, assertion_type="x", message="ok")],
            elapsed=1.0,
        )
        winner = _determine_case_winner({"a": r, "b": r})
        assert winner is None

    def test_both_pass_different_score(self) -> None:
        r1 = CaseResult(
            case=EvalCase(name="c", input=""),
            assertion_results=[
                AssertionResult(passed=True, assertion_type="x", message="ok"),
                AssertionResult(passed=False, assertion_type="y", message="fail"),
            ],
        )
        r2 = CaseResult(
            case=EvalCase(name="c", input=""),
            assertion_results=[
                AssertionResult(passed=True, assertion_type="x", message="ok"),
                AssertionResult(passed=True, assertion_type="y", message="ok"),
            ],
        )
        # r2 has higher score (1.0 vs 0.5), but r1 is not "passed" (not all pass)
        # Actually r1.passed is False because not all assertions pass.
        # So this is "one passes, one fails" → winner is r2's key
        winner = _determine_case_winner({"a": r1, "b": r2})
        assert winner == "b"

    def test_empty_results(self) -> None:
        assert _determine_case_winner({}) is None


# ---------------------------------------------------------------------------
# ELO tests
# ---------------------------------------------------------------------------


class TestELO:
    def test_winner_gains_rating(self) -> None:
        new_a, new_b = _compute_elo(1500, 1500, "a")
        assert new_a > 1500
        assert new_b < 1500

    def test_draw_no_change_when_equal(self) -> None:
        new_a, new_b = _compute_elo(1500, 1500, None)
        assert new_a == 1500
        assert new_b == 1500

    def test_upset_gives_more_points(self) -> None:
        # Lower-rated player wins
        new_a, _ = _compute_elo(1300, 1700, "a")
        normal_a, _ = _compute_elo(1500, 1500, "a")
        # Upset should give more ELO gain
        assert (new_a - 1300) > (normal_a - 1500)

    def test_loser_loses_rating(self) -> None:
        _, new_b = _compute_elo(1500, 1500, "a")
        assert new_b < 1500


class TestLeaderboard:
    def test_save_and_load(self, tmp_path: Path) -> None:
        lb_path = tmp_path / "lb.json"
        data = {"agents": {"test": {"elo": 1500, "wins": 1, "losses": 0, "ties": 0}}}
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            _save_leaderboard(data)
            loaded = _load_leaderboard()
            assert loaded["agents"]["test"]["elo"] == 1500

    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        lb_path = tmp_path / "nonexistent.json"
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            data = _load_leaderboard()
            assert data == {"agents": {}}


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestResolveContenderFiles:
    def test_expands_directory(self, agents_dir: Path) -> None:
        files = _resolve_contender_files([agents_dir])
        assert len(files) == 3
        assert all(f.suffix == ".py" for f in files)

    def test_single_files(self, two_agent_files: tuple[Path, Path]) -> None:
        f1, f2 = two_agent_files
        files = _resolve_contender_files([f1, f2])
        assert len(files) == 2

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        with pytest.raises(typer.Exit):
            _resolve_contender_files([tmp_path / "nonexistent.py"])


class TestFlatCases:
    def test_flattens_suites(self) -> None:
        s1 = EvalSuite(
            name="s1",
            description="",
            cases=[EvalCase(name="c1", input="a"), EvalCase(name="c2", input="b")],
        )
        s2 = EvalSuite(
            name="s2",
            description="",
            cases=[EvalCase(name="c3", input="c")],
        )
        flat = _flat_cases([s1, s2])
        assert len(flat) == 3
        assert flat[0].name == "c1"
        assert flat[2].name == "c3"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestArenaCompare:
    def test_compare_needs_two_contenders(self, tmp_path: Path, eval_file: Path) -> None:
        f = tmp_path / "solo.py"
        f.write_text('def main(message=""): return "hello"')
        result = runner.invoke(app, ["arena", "compare", str(f), "--eval", str(eval_file)])
        assert result.exit_code != 0

    def test_compare_dry_run(self, two_agent_files: tuple[Path, Path], eval_file: Path) -> None:
        f1, f2 = two_agent_files
        result = runner.invoke(
            app,
            ["arena", "compare", str(f1), str(f2), "--eval", str(eval_file), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_compare_runs_eval(self, two_agent_files: tuple[Path, Path], eval_file: Path, tmp_path: Path) -> None:
        f1, f2 = two_agent_files
        lb_path = tmp_path / ".ag2" / "arena" / "leaderboard.json"
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            result = runner.invoke(
                app,
                ["arena", "compare", str(f1), str(f2), "--eval", str(eval_file)],
            )
        assert result.exit_code == 0
        assert "Arena" in result.output
        assert "Summary" in result.output

    def test_compare_json_output(self, two_agent_files: tuple[Path, Path], eval_file: Path, tmp_path: Path) -> None:
        f1, f2 = two_agent_files
        lb_path = tmp_path / ".ag2" / "arena" / "leaderboard.json"
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            result = runner.invoke(
                app,
                [
                    "arena",
                    "compare",
                    str(f1),
                    str(f2),
                    "--eval",
                    str(eval_file),
                    "--output",
                    "json",
                ],
            )
        assert result.exit_code == 0
        # JSON should be parseable somewhere in the output
        lines = result.output.strip().split("\n")
        # Find the JSON part (after the Rich output)
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break
        if json_start is not None:
            json_text = "\n".join(lines[json_start:])
            data = json.loads(json_text)
            assert "contenders" in data
            assert "wins" in data

    def test_compare_tournament(self, agents_dir: Path, eval_file: Path, tmp_path: Path) -> None:
        lb_path = tmp_path / ".ag2" / "arena" / "leaderboard.json"
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            result = runner.invoke(
                app,
                ["arena", "compare", str(agents_dir), "--eval", str(eval_file)],
            )
        assert result.exit_code == 0
        assert "Arena" in result.output


class TestArenaLeaderboard:
    def test_empty_leaderboard(self, tmp_path: Path) -> None:
        lb_path = tmp_path / "empty_lb.json"
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            result = runner.invoke(app, ["arena", "leaderboard"])
        assert result.exit_code == 0
        assert "No arena results" in result.output

    def test_leaderboard_with_data(self, tmp_path: Path) -> None:
        lb_path = tmp_path / "lb.json"
        lb_path.parent.mkdir(parents=True, exist_ok=True)
        lb_path.write_text(
            json.dumps({
                "agents": {
                    "alpha": {"elo": 1550.0, "wins": 5, "losses": 2, "ties": 1},
                    "beta": {"elo": 1450.0, "wins": 2, "losses": 5, "ties": 1},
                }
            })
        )
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            result = runner.invoke(app, ["arena", "leaderboard"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output


class TestArenaReset:
    def test_reset_existing(self, tmp_path: Path) -> None:
        lb_path = tmp_path / "lb.json"
        lb_path.write_text("{}")
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            result = runner.invoke(app, ["arena", "reset"])
        assert result.exit_code == 0
        assert "reset" in result.output.lower()
        assert not lb_path.exists()

    def test_reset_nonexistent(self, tmp_path: Path) -> None:
        lb_path = tmp_path / "nonexistent.json"
        with patch("ag2_cli.commands.arena.LEADERBOARD_PATH", lb_path):
            result = runner.invoke(app, ["arena", "reset"])
        assert result.exit_code == 0
