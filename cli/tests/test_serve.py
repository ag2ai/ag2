"""Tests for the ag2 serve command and REST app builder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ag2_cli.app import app
from ag2_cli.core.discovery import DiscoveredAgent
from ag2_cli.core.runner import RunResult
from typer.testing import CliRunner

runner = CliRunner()

# ---------------------------------------------------------------------------
# Conditional imports for FastAPI / Starlette
# ---------------------------------------------------------------------------

try:
    from starlette.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

needs_fastapi = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi/starlette not installed")


def _make_discovered(
    kind: str = "agent",
    agent_names: list[str] | None = None,
) -> DiscoveredAgent:
    """Build a minimal DiscoveredAgent for testing."""
    return DiscoveredAgent(
        kind=kind,
        source_file=Path("/fake/agent.py"),
        agent_names=agent_names or ["test"],
    )


def _make_run_result(**kwargs: object) -> RunResult:
    """Build a RunResult with sensible defaults."""
    defaults: dict[str, object] = {
        "output": "Hello from the agent",
        "turns": 3,
        "elapsed": 1.23,
        "agent_names": ["test"],
    }
    defaults.update(kwargs)
    return RunResult(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# REST app tests
# ---------------------------------------------------------------------------


@needs_fastapi
class TestBuildRestApp:
    """Tests for _build_rest_app — the FastAPI app factory.

    Note: _build_rest_app does ``from ..core.runner import execute`` as a
    local import.  The closure inside the ``/chat`` endpoint captures that
    local reference, so we must patch ``ag2_cli.core.runner.execute``
    *before* calling ``_build_rest_app`` so the import picks up the mock.
    """

    @staticmethod
    def _build_app(
        discovered: DiscoveredAgent | None = None,
    ) -> TestClient:
        from ag2_cli.commands.serve import _build_rest_app

        d = discovered or _make_discovered()
        fast_app = _build_rest_app(d)
        return TestClient(fast_app)

    def test_app_has_expected_routes(self) -> None:
        client = self._build_app()
        routes = {r.path for r in client.app.routes}  # type: ignore[union-attr]
        assert "/health" in routes
        assert "/agents" in routes
        assert "/chat" in routes

    def test_health_returns_ok(self) -> None:
        client = self._build_app()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_agents_returns_agent_list(self) -> None:
        d = _make_discovered(kind="agent", agent_names=["alice", "bob"])
        client = self._build_app(d)
        resp = client.get("/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0] == {"name": "alice", "kind": "agent"}
        assert data[1] == {"name": "bob", "kind": "agent"}

    def test_chat_endpoint_calls_execute(self) -> None:
        mock_result = _make_run_result()
        with patch("ag2_cli.core.runner.execute", return_value=mock_result) as mock_execute:
            client = self._build_app()
            resp = client.post("/chat", json={"message": "Hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["output"] == "Hello from the agent"
        assert body["turns"] == 3
        assert body["elapsed"] == 1.23
        assert body["agent_names"] == ["test"]
        mock_execute.assert_called_once()

    def test_chat_endpoint_forwards_max_turns(self) -> None:
        mock_result = _make_run_result()
        with patch("ag2_cli.core.runner.execute", return_value=mock_result) as mock_execute:
            client = self._build_app()
            resp = client.post("/chat", json={"message": "Hi", "max_turns": 5})
        assert resp.status_code == 200
        call_kwargs = mock_execute.call_args
        assert call_kwargs.kwargs.get("max_turns") == 5 or call_kwargs[1].get("max_turns") == 5

    def test_chat_endpoint_rejects_missing_message(self) -> None:
        client = self._build_app()
        resp = client.post("/chat", json={})
        assert resp.status_code == 422  # FastAPI validation error


# ---------------------------------------------------------------------------
# Serve command tests (via Typer CliRunner — validation paths only)
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for serve_cmd validation logic.

    These only exercise the early-exit paths (unknown protocol, missing file,
    playground flag) to avoid actually starting a server.
    """

    def test_serve_rejects_unknown_protocol(self) -> None:
        result = runner.invoke(app, ["serve", "agent.py", "--protocol", "grpc"])
        assert result.exit_code == 1
        assert "Unknown protocol" in result.output

    @patch("ag2_cli.commands.serve._require_ag2")
    def test_serve_requires_file_to_exist(self, mock_ag2: MagicMock) -> None:
        mock_ag2.return_value = MagicMock()
        result = runner.invoke(app, ["serve", "/tmp/_no_such_agent_file_12345.py"])
        assert result.exit_code == 1
        assert "File not found" in result.output

    @patch("ag2_cli.commands.serve._require_ag2")
    def test_serve_playground_shows_coming_soon(self, mock_ag2: MagicMock, tmp_path: Path) -> None:
        mock_ag2.return_value = MagicMock()
        # Create a real file so we get past the existence check,
        # but mock discover (imported locally in serve_cmd) to avoid needing actual agents.
        agent_file = tmp_path / "agent.py"
        agent_file.write_text("x = 1\n")

        with patch("ag2_cli.core.discovery.discover") as mock_discover:
            mock_discover.side_effect = ValueError("no agent")
            result = runner.invoke(
                app,
                ["serve", str(agent_file), "--playground"],
            )
        # The --playground message is printed before discovery,
        # so it appears regardless of whether discovery succeeds.
        assert "coming soon" in result.output.lower()
