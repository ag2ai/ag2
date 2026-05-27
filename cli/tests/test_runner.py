"""Tests for the core runner module."""

from __future__ import annotations

import textwrap
from pathlib import Path

from ag2_cli.core.runner import CliIOStream, RunResult, execute


class TestCliIOStream:
    """Tests for the CliIOStream event capture."""

    def test_print_callback(self) -> None:
        captured: list[str] = []
        stream = CliIOStream(on_print=captured.append)
        stream.print("hello", "world")
        assert captured == ["hello world"]

    def test_print_no_callback(self) -> None:
        stream = CliIOStream()
        stream.print("hello")  # Should not raise

    def test_send_callback(self) -> None:
        captured: list[object] = []
        stream = CliIOStream(on_event=captured.append)
        stream.send("event1")
        stream.send("event2")
        assert captured == ["event1", "event2"]

    def test_input_returns_empty(self) -> None:
        stream = CliIOStream()
        assert stream.input("prompt> ") == ""

    def test_print_with_sep(self) -> None:
        captured: list[str] = []
        stream = CliIOStream(on_print=captured.append)
        stream.print("a", "b", "c", sep="-")
        assert captured == ["a-b-c"]


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_defaults(self) -> None:
        r = RunResult()
        assert r.output == ""
        assert r.turns == 0
        assert r.cost is None
        assert r.elapsed == 0.0
        assert r.errors == []
        assert r.history == []
        assert r.agent_names == []
        assert r.last_speaker is None

    def test_with_values(self) -> None:
        r = RunResult(
            output="hello",
            turns=3,
            elapsed=1.5,
            agent_names=["alice", "bob"],
            last_speaker="bob",
        )
        assert r.output == "hello"
        assert r.turns == 3
        assert r.elapsed == 1.5
        assert r.agent_names == ["alice", "bob"]
        assert r.last_speaker == "bob"


class TestExecuteMain:
    """Tests for executing main() function discoveries."""

    def test_execute_main_with_message(self, tmp_path: Path) -> None:
        f = tmp_path / "agent.py"
        f.write_text(
            textwrap.dedent("""\
            def main(message="default"):
                return f"Got: {message}"
            """)
        )
        from ag2_cli.core.discovery import discover

        discovered = discover(f)
        result = execute(discovered, "hello world")
        assert result.output == "Got: hello world"
        assert result.turns == 1
        assert result.errors == []

    def test_execute_main_without_message_param(self, tmp_path: Path) -> None:
        f = tmp_path / "agent.py"
        f.write_text(
            textwrap.dedent("""\
            def main():
                return "fixed response"
            """)
        )
        from ag2_cli.core.discovery import discover

        discovered = discover(f)
        result = execute(discovered, "ignored")
        assert result.output == "fixed response"

    def test_execute_main_no_function(self, tmp_path: Path) -> None:
        """Test that main_fn=None produces an error."""
        from ag2_cli.core.discovery import DiscoveredAgent

        discovered = DiscoveredAgent(
            kind="main",
            source_file=tmp_path / "fake.py",
            main_fn=None,
        )
        result = execute(discovered, "hello")
        assert len(result.errors) > 0
        assert "No main function" in result.errors[0]

    def test_execute_captures_print(self, tmp_path: Path) -> None:
        """Test that on_print callback receives agent output."""
        f = tmp_path / "agent.py"
        f.write_text(
            textwrap.dedent("""\
            def main(message="hi"):
                return message.upper()
            """)
        )
        from ag2_cli.core.discovery import discover

        captured_prints: list[str] = []
        discovered = discover(f)
        result = execute(discovered, "hello", on_print=captured_prints.append)
        assert result.output == "HELLO"

    def test_execute_async_main(self, tmp_path: Path) -> None:
        f = tmp_path / "agent.py"
        f.write_text(
            textwrap.dedent("""\
            async def main(message="hi"):
                return f"async: {message}"
            """)
        )
        from ag2_cli.core.discovery import discover

        discovered = discover(f)
        result = execute(discovered, "test")
        assert result.output == "async: test"

    def test_execute_records_elapsed(self, tmp_path: Path) -> None:
        f = tmp_path / "agent.py"
        f.write_text(
            textwrap.dedent("""\
            def main(message="hi"):
                return "done"
            """)
        )
        from ag2_cli.core.discovery import discover

        discovered = discover(f)
        result = execute(discovered, "hello")
        assert result.elapsed >= 0
        assert result.elapsed < 5  # Should be fast

    def test_execute_handles_exception(self, tmp_path: Path) -> None:
        f = tmp_path / "agent.py"
        f.write_text(
            textwrap.dedent("""\
            def main(message="hi"):
                raise ValueError("test error")
            """)
        )
        from ag2_cli.core.discovery import discover

        discovered = discover(f)
        result = execute(discovered, "hello")
        assert len(result.errors) > 0
        assert "test error" in result.errors[0]


class TestExecuteUnknownKind:
    """Test edge cases."""

    def test_unknown_kind_errors(self) -> None:
        from ag2_cli.core.discovery import DiscoveredAgent

        discovered = DiscoveredAgent(
            kind="unknown",
            source_file=Path("fake.py"),
        )
        result = execute(discovered, "hello")
        assert len(result.errors) > 0
        assert "Unknown discovery kind" in result.errors[0]
