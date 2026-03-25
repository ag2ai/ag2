"""Tests for ag2 replay — session recording and playback."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ag2_cli.app import app
from ag2_cli.commands.replay import (
    Session,
    SessionEvent,
    SessionMeta,
    create_session_id,
    delete_session,
    list_sessions,
    load_session,
    record_from_run_result,
    save_session,
)
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    """Provide a temp sessions directory."""
    d = tmp_path / "sessions"
    d.mkdir()
    return d


@pytest.fixture
def sample_session() -> Session:
    """Create a sample session for testing."""
    meta = SessionMeta(
        session_id="20260319-143022-abc123",
        agent_file="/tmp/my_agent.py",
        agent_names=["researcher", "writer"],
        created_at="2026-03-19T14:30:22+00:00",
        turns=4,
        duration=5.2,
        total_cost=0.0042,
        input_message="What is the capital of France?",
        final_output="Paris is the capital of France.",
    )
    events = [
        SessionEvent(
            turn=1,
            speaker="user",
            content="What is the capital of France?",
            role="user",
        ),
        SessionEvent(
            turn=2,
            speaker="researcher",
            content="The capital of France is Paris.",
            role="assistant",
            elapsed=1.5,
        ),
        SessionEvent(
            turn=3,
            speaker="writer",
            content="Paris is the capital of France, known for the Eiffel Tower.",
            role="assistant",
            elapsed=2.1,
        ),
        SessionEvent(
            turn=4,
            speaker="user",
            content="",
            role="user",
        ),
    ]
    return Session(meta=meta, events=events)


@pytest.fixture
def saved_session(sessions_dir: Path, sample_session: Session) -> Session:
    """Save and return a sample session."""
    with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
        save_session(sample_session)
    return sample_session


# ---------------------------------------------------------------------------
# SessionEvent tests
# ---------------------------------------------------------------------------


class TestSessionEvent:
    def test_creation(self) -> None:
        event = SessionEvent(turn=1, speaker="researcher", content="Hello", role="assistant")
        assert event.turn == 1
        assert event.speaker == "researcher"
        assert event.metadata == {}

    def test_with_metadata(self) -> None:
        event = SessionEvent(
            turn=1,
            speaker="researcher",
            content="Hello",
            role="assistant",
            metadata={"tool_calls": [{"name": "search"}]},
        )
        assert "tool_calls" in event.metadata


class TestSessionMeta:
    def test_creation(self) -> None:
        meta = SessionMeta(
            session_id="test-123",
            agent_file="agent.py",
            agent_names=["researcher"],
            created_at="2026-03-19T00:00:00",
            turns=3,
            duration=2.5,
        )
        assert meta.session_id == "test-123"
        assert meta.total_cost == 0.0


# ---------------------------------------------------------------------------
# Session persistence tests
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    def test_save_creates_file(self, sessions_dir: Path, sample_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            path = save_session(sample_session)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["meta"]["session_id"] == sample_session.meta.session_id

    def test_load_by_id(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            loaded = load_session(saved_session.meta.session_id)
        assert loaded.meta.session_id == saved_session.meta.session_id
        assert len(loaded.events) == len(saved_session.events)

    def test_load_by_prefix(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            loaded = load_session("20260319")
        assert loaded.meta.session_id == saved_session.meta.session_id

    def test_load_missing_exits(self, sessions_dir: Path) -> None:
        import typer

        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir), pytest.raises(typer.Exit):
            load_session("nonexistent-session-id")

    def test_roundtrip_preserves_data(self, sessions_dir: Path, sample_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            save_session(sample_session)
            loaded = load_session(sample_session.meta.session_id)
        assert loaded.meta.agent_file == sample_session.meta.agent_file
        assert loaded.events[1].content == sample_session.events[1].content
        assert loaded.meta.total_cost == sample_session.meta.total_cost


class TestListSessions:
    def test_list_empty(self, sessions_dir: Path) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            sessions = list_sessions()
        assert sessions == []

    def test_list_with_sessions(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            sessions = list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == saved_session.meta.session_id

    def test_list_multiple(self, sessions_dir: Path) -> None:
        for i in range(3):
            session = Session(
                meta=SessionMeta(
                    session_id=f"session-{i}",
                    agent_file="agent.py",
                    agent_names=[],
                    created_at=f"2026-03-{19 + i}T00:00:00",
                    turns=1,
                    duration=1.0,
                ),
                events=[],
            )
            with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
                save_session(session)

        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            sessions = list_sessions()
        assert len(sessions) == 3


class TestDeleteSession:
    def test_delete_existing(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = delete_session(saved_session.meta.session_id)
        assert result is True

    def test_delete_by_prefix(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = delete_session("20260319")
        assert result is True

    def test_delete_nonexistent(self, sessions_dir: Path) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = delete_session("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# Session ID generation tests
# ---------------------------------------------------------------------------


class TestCreateSessionId:
    def test_format(self) -> None:
        sid = create_session_id()
        parts = sid.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 6  # hex

    def test_unique(self) -> None:
        ids = {create_session_id() for _ in range(10)}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# Record from RunResult tests
# ---------------------------------------------------------------------------


class TestRecordFromRunResult:
    def test_records_from_result(self) -> None:
        class FakeResult:
            output = "Paris is the capital"
            history = [
                {"role": "user", "content": "What is the capital?", "name": "user"},
                {
                    "role": "assistant",
                    "content": "Paris is the capital",
                    "name": "researcher",
                },
            ]
            elapsed = 2.5
            cost = None
            agent_names = ["researcher"]

        session = record_from_run_result(FakeResult(), "my_agent.py", "What is the capital?")
        assert session.meta.turns == 2
        assert session.meta.agent_file == "my_agent.py"
        assert session.meta.input_message == "What is the capital?"
        assert session.events[0].speaker == "user"
        assert session.events[1].speaker == "researcher"

    def test_records_cost(self) -> None:
        class FakeResult:
            output = "test"
            history = []
            elapsed = 1.0
            cost = {"usage_excluding_cached_inference": {"total_cost": 0.005}}
            agent_names = []

        session = record_from_run_result(FakeResult(), "agent.py", "test")
        assert session.meta.total_cost == 0.005


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestReplayList:
    def test_list_empty(self, sessions_dir: Path) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(app, ["replay", "list"])
        assert result.exit_code == 0
        assert "No recorded sessions" in result.output

    def test_list_with_sessions(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(app, ["replay", "list"])
        assert result.exit_code == 0
        assert saved_session.meta.session_id in result.output


class TestReplayShow:
    def test_show_session(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(app, ["replay", "show", saved_session.meta.session_id])
        assert result.exit_code == 0
        assert "researcher" in result.output
        assert "Replay" in result.output

    def test_show_missing(self, sessions_dir: Path) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(app, ["replay", "show", "nonexistent"])
        assert result.exit_code != 0


class TestReplayExport:
    def test_export_json(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(
                app,
                ["replay", "export", saved_session.meta.session_id, "--format", "json"],
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["meta"]["session_id"] == saved_session.meta.session_id

    def test_export_markdown(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(
                app,
                ["replay", "export", saved_session.meta.session_id, "--format", "md"],
            )
        assert result.exit_code == 0
        assert "# Session:" in result.output
        assert "researcher" in result.output

    def test_export_html(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(
                app,
                ["replay", "export", saved_session.meta.session_id, "--format", "html"],
            )
        assert result.exit_code == 0
        assert "<html>" in result.output

    def test_export_to_file(self, sessions_dir: Path, saved_session: Session, tmp_path: Path) -> None:
        output = tmp_path / "export.json"
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(
                app,
                [
                    "replay",
                    "export",
                    saved_session.meta.session_id,
                    "--format",
                    "json",
                    "--output",
                    str(output),
                ],
            )
        assert result.exit_code == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["meta"]["session_id"] == saved_session.meta.session_id


class TestReplayDelete:
    def test_delete_session(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(app, ["replay", "delete", saved_session.meta.session_id])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_delete_missing(self, sessions_dir: Path) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(app, ["replay", "delete", "nonexistent"])
        assert result.exit_code != 0


class TestReplayClear:
    def test_clear_sessions(self, sessions_dir: Path, saved_session: Session) -> None:
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(app, ["replay", "clear"])
        assert result.exit_code == 0
        assert "Cleared" in result.output

    def test_clear_empty(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_sessions"
        empty_dir.mkdir()
        with patch("ag2_cli.commands.replay.SESSIONS_DIR", empty_dir):
            result = runner.invoke(app, ["replay", "clear"])
        assert result.exit_code == 0


class TestReplayCompare:
    def test_compare_two_sessions(self, sessions_dir: Path) -> None:
        # Create two sessions
        for sid in ("session-a", "session-b"):
            session = Session(
                meta=SessionMeta(
                    session_id=sid,
                    agent_file="agent.py",
                    agent_names=["agent"],
                    created_at="2026-03-19T00:00:00",
                    turns=2,
                    duration=1.0,
                ),
                events=[
                    SessionEvent(turn=1, speaker="user", content="Hello", role="user"),
                    SessionEvent(
                        turn=2,
                        speaker="agent",
                        content=f"Response from {sid}",
                        role="assistant",
                    ),
                ],
            )
            with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
                save_session(session)

        with patch("ag2_cli.commands.replay.SESSIONS_DIR", sessions_dir):
            result = runner.invoke(app, ["replay", "compare", "session-a", "session-b"])
        assert result.exit_code == 0
        assert "Comparison" in result.output
