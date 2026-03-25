"""Tests for ag2 install command — CLI-level integration tests.

Covers: target resolution, search, list, uninstall, install subcommands.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from ag2_cli.commands.install import (
    _is_interactive,
    _resolve_targets,
    app,
)
from ag2_cli.install.lockfile import Lockfile
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Target resolution tests
# ---------------------------------------------------------------------------


class TestResolveTargets:
    def test_target_all_returns_all_targets(self, tmp_path: Path):
        targets = _resolve_targets("all", tmp_path)
        assert len(targets) > 0

    def test_target_claude_returns_claude(self, tmp_path: Path):
        targets = _resolve_targets("claude", tmp_path)
        assert len(targets) == 1
        assert targets[0].name == "claude"

    def test_comma_separated_targets(self, tmp_path: Path):
        targets = _resolve_targets("claude,copilot", tmp_path)
        assert len(targets) == 2
        names = {t.name for t in targets}
        assert "claude" in names
        assert "copilot" in names

    def test_unknown_target_exits(self, tmp_path: Path):
        with pytest.raises(typer.Exit):
            _resolve_targets("nonexistent-ide", tmp_path)

    def test_no_target_no_detection_exits_noninteractive(self, tmp_path: Path):
        """Non-interactive mode with no detected targets should exit."""
        with (
            patch("ag2_cli.commands.install._is_interactive", return_value=False),
            patch("ag2_cli.commands.install.detect_targets", return_value=[]),
            pytest.raises(typer.Exit),
        ):
            _resolve_targets(None, tmp_path)

    def test_no_target_with_detection_returns_detected(self, tmp_path: Path):
        """When targets are detected, they should be used."""
        mock_target = MagicMock()
        mock_target.name = "claude"
        with (
            patch("ag2_cli.commands.install._is_interactive", return_value=False),
            patch("ag2_cli.commands.install.detect_targets", return_value=[mock_target]),
        ):
            targets = _resolve_targets(None, tmp_path)
        assert len(targets) == 1


# ---------------------------------------------------------------------------
# List command tests
# ---------------------------------------------------------------------------


class TestListCmd:
    def test_list_targets(self):
        result = runner.invoke(app, ["list", "targets"])
        assert result.exit_code == 0
        assert "claude" in result.output.lower()

    def test_list_installed_empty(self, tmp_path: Path):
        result = runner.invoke(app, ["list", "installed"])
        # Should show "no artifacts installed" when running from tmp
        assert result.exit_code == 0

    def test_list_all_remote(self):
        """Listing all remote artifacts should work or fail gracefully."""
        mock_registry = {
            "artifacts": [
                {
                    "type": "skills",
                    "owner": "ag2ai",
                    "name": "fastapi",
                    "version": "1.0.0",
                    "description": "FastAPI skills",
                },
                {
                    "type": "tool",
                    "owner": "ag2ai",
                    "name": "web-search",
                    "version": "1.0.0",
                    "description": "Web search",
                },
            ]
        }
        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client:
            mock = mock_client.return_value
            mock.fetch_registry.return_value = mock_registry
            mock.list_artifacts.return_value = mock_registry["artifacts"]
            result = runner.invoke(app, ["list", "all"])
        assert result.exit_code == 0

    def test_list_specific_type(self):
        mock_registry = {
            "artifacts": [
                {
                    "type": "tool",
                    "owner": "ag2ai",
                    "name": "web-search",
                    "version": "1.0.0",
                    "description": "Web search",
                },
            ]
        }
        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client:
            mock = mock_client.return_value
            mock.fetch_registry.return_value = mock_registry
            mock.list_artifacts.return_value = mock_registry["artifacts"]
            result = runner.invoke(app, ["list", "tools"])
        assert result.exit_code == 0

    def test_list_registry_failure_handled(self):
        from ag2_cli.install.client import FetchError

        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client:
            mock = mock_client.return_value
            mock.fetch_registry.side_effect = FetchError("network error")
            result = runner.invoke(app, ["list", "all"])
        # Should handle gracefully, not crash
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Search command tests
# ---------------------------------------------------------------------------


class TestSearchCmd:
    def test_search_returns_results(self):
        mock_registry = {
            "artifacts": [
                {
                    "type": "skills",
                    "owner": "ag2ai",
                    "name": "fastapi",
                    "version": "1.0.0",
                    "description": "FastAPI skills",
                },
            ]
        }
        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client:
            mock = mock_client.return_value
            mock.fetch_registry.return_value = mock_registry
            mock.search.return_value = mock_registry["artifacts"]
            result = runner.invoke(app, ["search", "fastapi"])
        assert result.exit_code == 0
        assert "fastapi" in result.output.lower()

    def test_search_no_results(self):
        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client:
            mock = mock_client.return_value
            mock.fetch_registry.return_value = {"artifacts": []}
            mock.search.return_value = []
            result = runner.invoke(app, ["search", "nonexistent"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_registry_failure(self):
        from ag2_cli.install.client import FetchError

        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client:
            mock = mock_client.return_value
            mock.fetch_registry.side_effect = FetchError("network error")
            result = runner.invoke(app, ["search", "test"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Uninstall command tests
# ---------------------------------------------------------------------------


class TestUninstallCmd:
    def test_uninstall_with_lockfile_entry(self, tmp_path: Path):
        """Uninstall should remove tracked files and update lockfile."""
        # Create a file and record it in the lockfile
        tracked_file = tmp_path / ".claude" / "skills" / "ag2-test" / "SKILL.md"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text("test content")

        lockfile = Lockfile(tmp_path)
        lockfile.record_install(
            ref="skills/ag2ai/test",
            version="1.0.0",
            targets=["claude"],
            files=[tracked_file],
        )

        result = runner.invoke(app, ["uninstall", "skills/ag2ai/test", "--project-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert not tracked_file.exists()

    def test_uninstall_no_match_falls_back(self, tmp_path: Path):
        """Uninstall with no lockfile match falls back to target-based removal."""
        result = runner.invoke(
            app,
            ["uninstall", "skills", "--target", "claude", "--project-dir", str(tmp_path)],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Install skills command tests
# ---------------------------------------------------------------------------


class TestInstallSkillsCmd:
    def test_install_skills_default_target(self, tmp_path: Path):
        """Install skills with explicit target should work."""
        result = runner.invoke(
            app,
            ["skills", "--target", "claude", "--project-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Done" in result.output

    def test_install_skills_with_name_filter(self, tmp_path: Path):
        result = runner.invoke(
            app,
            ["skills", "--target", "claude", "--project-dir", str(tmp_path), "--name", "imports"],
        )
        assert result.exit_code == 0

    def test_install_skills_unknown_pack(self, tmp_path: Path):
        """Installing a non-existent pack should show an error."""
        result = runner.invoke(
            app,
            ["skills", "nonexistent-pack", "--target", "claude", "--project-dir", str(tmp_path)],
        )
        # Should fail because the pack doesn't exist
        assert result.exit_code != 0 or "error" in result.output.lower() or "0 files" in result.output


# ---------------------------------------------------------------------------
# Install from command tests
# ---------------------------------------------------------------------------


class TestInstallFromCmd:
    def test_install_from_nonexistent_dir(self, tmp_path: Path):
        """Installing from a local dir without artifact.json should fail."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(
            app,
            ["from", str(empty_dir), "--target", "claude", "--project-dir", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_install_from_remote_url_shows_message(self, tmp_path: Path):
        """Remote URL install should show 'coming soon' message."""
        result = runner.invoke(
            app,
            ["from", "https://github.com/example/repo", "--target", "claude", "--project-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "coming soon" in result.output.lower() or "clone" in result.output.lower()


# ---------------------------------------------------------------------------
# Update command tests
# ---------------------------------------------------------------------------


class TestUpdateCmd:
    def test_update_no_installed_artifacts(self, tmp_path: Path):
        result = runner.invoke(app, ["update", "--project-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No artifacts" in result.output

    def test_update_all_up_to_date(self, tmp_path: Path):
        lockfile = Lockfile(tmp_path)
        lockfile.record_install(
            ref="skills/ag2ai/test",
            version="1.0.0",
            targets=["claude"],
            files=[],
        )
        registry = {
            "artifacts": [
                {"type": "skills", "owner": "ag2ai", "name": "test", "version": "1.0.0"},
            ]
        }
        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client:
            mock = mock_client.return_value
            mock.fetch_registry.return_value = registry
            result = runner.invoke(app, ["update", "--project-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "up to date" in result.output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestIsInteractive:
    def test_returns_bool(self):
        result = _is_interactive()
        assert isinstance(result, bool)
