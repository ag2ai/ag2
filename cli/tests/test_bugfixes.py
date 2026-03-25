"""Regression tests for bug fixes identified during release review.

Tests are grouped by bug number and cover the specific fix applied.
"""

from __future__ import annotations

import contextlib
import sys
import textwrap
from html import escape as html_escape
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bug 1: update_cmd ref format mismatch (Critical)
# The registry lookup must use the same 3-part ref as lockfile entries.
# ---------------------------------------------------------------------------


class TestUpdateCmdRefFormat:
    """Bug 1: update_cmd was comparing 2-part keys against 3-part lockfile refs."""

    def test_registry_key_matches_lockfile_ref(self):
        """Verify _pluralize_type produces the same prefix used in Artifact.ref."""
        from ag2_cli.install.artifact import Artifact, _pluralize_type

        artifact = Artifact(name="web-search", type="tool", owner="ag2ai", version="1.0.0")
        ref = artifact.ref  # "tools/ag2ai/web-search"

        # The update command now builds keys the same way
        type_dir = _pluralize_type("tool")
        built_ref = f"{type_dir}/ag2ai/web-search"
        assert ref == built_ref

    def test_update_detects_version_mismatch(self, tmp_path: Path):
        """End-to-end: update_cmd should detect when registry has a newer version."""
        from ag2_cli.commands.install import app
        from ag2_cli.install.lockfile import Lockfile
        from typer.testing import CliRunner

        runner = CliRunner()

        # Set up a lockfile with an installed artifact
        lockfile = Lockfile(tmp_path)
        lockfile.record_install(
            ref="tools/ag2ai/web-search",
            version="1.0.0",
            targets=["claude"],
            files=[],
        )

        # Mock the registry to return a newer version
        registry = {
            "artifacts": [
                {"type": "tool", "owner": "ag2ai", "name": "web-search", "version": "2.0.0"},
            ]
        }

        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.fetch_registry.return_value = registry

            result = runner.invoke(app, ["update", "--project-dir", str(tmp_path)])

        # Should find the update (not "all up to date")
        assert "2.0.0" in result.output
        assert "up to date" not in result.output

    def test_update_all_up_to_date_when_versions_match(self, tmp_path: Path):
        """update_cmd should report up-to-date when versions match."""
        from ag2_cli.commands.install import app
        from ag2_cli.install.lockfile import Lockfile
        from typer.testing import CliRunner

        runner = CliRunner()

        lockfile = Lockfile(tmp_path)
        lockfile.record_install(
            ref="tools/ag2ai/web-search",
            version="1.0.0",
            targets=["claude"],
            files=[],
        )

        registry = {
            "artifacts": [
                {"type": "tool", "owner": "ag2ai", "name": "web-search", "version": "1.0.0"},
            ]
        }

        with patch("ag2_cli.commands.install.ArtifactClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.fetch_registry.return_value = registry

            result = runner.invoke(app, ["update", "--project-dir", str(tmp_path)])

        assert "up to date" in result.output


# ---------------------------------------------------------------------------
# Bug 2: fetch_artifact_dir ignores owner in repo path (High)
# ---------------------------------------------------------------------------


class TestFetchArtifactDirOwner:
    """Bug 2: third-party artifacts should use owner-namespaced path."""

    def test_default_owner_uses_flat_path(self, tmp_path: Path):
        """ag2ai artifacts use flat path: type_dir/name."""
        with patch.dict("os.environ", {}, clear=True):
            from ag2_cli.install.client import ArtifactClient

            client = ArtifactClient(cache_dir=tmp_path)

        files = ["tools/web-search/artifact.json"]
        with (
            patch.object(client, "_list_contents_recursive", return_value=files) as mock_list,
            patch.object(client, "_get_bytes", return_value=b"{}"),
        ):
            client.fetch_artifact_dir("tool", "web-search", owner="ag2ai")

        # Should use flat path for ag2ai
        mock_list.assert_called_once_with("tools/web-search")

    def test_third_party_owner_uses_namespaced_path(self, tmp_path: Path):
        """Third-party artifacts use owner-namespaced path: type_dir/owner/name."""
        with patch.dict("os.environ", {}, clear=True):
            from ag2_cli.install.client import ArtifactClient

            client = ArtifactClient(cache_dir=tmp_path)

        files = ["tools/myorg/custom-tool/artifact.json"]
        with (
            patch.object(client, "_list_contents_recursive", return_value=files) as mock_list,
            patch.object(client, "_get_bytes", return_value=b"{}"),
        ):
            client.fetch_artifact_dir("tool", "custom-tool", owner="myorg")

        # Should use namespaced path for non-ag2ai
        mock_list.assert_called_once_with("tools/myorg/custom-tool")


# ---------------------------------------------------------------------------
# Bug 3: async main() loses IOStream context in ThreadPool (High)
# ---------------------------------------------------------------------------


class TestAsyncMainIOStream:
    """Bug 3: async main() in thread pool should preserve IOStream context."""

    def test_async_main_returns_result(self, tmp_path: Path):
        """Basic sanity: async main() still works and returns correct output."""
        f = tmp_path / "agent.py"
        f.write_text(
            textwrap.dedent("""\
            async def main(message="hi"):
                return f"async: {message}"
            """)
        )
        from ag2_cli.core.discovery import discover
        from ag2_cli.core.runner import execute

        discovered = discover(f)
        result = execute(discovered, "test")
        assert result.output == "async: test"
        assert result.errors == []


# ---------------------------------------------------------------------------
# Bug 4: post_install runs shell commands without confirmation (High)
# ---------------------------------------------------------------------------


class TestPostInstallConfirmation:
    """Bug 4: post_install commands must prompt for user confirmation."""

    def test_post_install_skipped_when_user_declines(self, tmp_path: Path):
        """Commands should NOT run when user answers no."""
        from ag2_cli.install.installers.templates import TemplateInstaller

        installer = TemplateInstaller.__new__(TemplateInstaller)

        with (
            patch("ag2_cli.install.installers.templates.typer.confirm", return_value=False),
            patch("ag2_cli.install.installers.templates.subprocess.run") as mock_run,
        ):
            installer._run_post_install(["echo hello"], tmp_path)

        mock_run.assert_not_called()

    def test_post_install_runs_when_user_confirms(self, tmp_path: Path):
        """Commands should run when user answers yes."""
        from ag2_cli.install.installers.templates import TemplateInstaller

        installer = TemplateInstaller.__new__(TemplateInstaller)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with (
            patch("ag2_cli.install.installers.templates.typer.confirm", return_value=True),
            patch("ag2_cli.install.installers.templates.subprocess.run", return_value=mock_result) as mock_run,
        ):
            installer._run_post_install(["echo hello"], tmp_path)

        mock_run.assert_called_once()

    def test_post_install_skipped_on_eof(self, tmp_path: Path):
        """Commands should NOT run if stdin is closed (non-interactive)."""
        from ag2_cli.install.installers.templates import TemplateInstaller

        installer = TemplateInstaller.__new__(TemplateInstaller)

        with (
            patch("ag2_cli.install.installers.templates.typer.confirm", side_effect=EOFError),
            patch("ag2_cli.install.installers.templates.subprocess.run") as mock_run,
        ):
            installer._run_post_install(["rm -rf /"], tmp_path)

        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 5: format_frontmatter breaks on lists and YAML-special values (Medium)
# ---------------------------------------------------------------------------


class TestFormatFrontmatterFixes:
    """Bug 5: format_frontmatter should handle lists and YAML-special strings."""

    def test_list_values_formatted_as_yaml_array(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"globs": ["*.py", "*.js"]})
        assert "globs: [*.py, *.js]" in result

    def test_empty_list_formatted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"items": []})
        assert "items: []" in result

    def test_yaml_true_is_quoted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"value": "true"})
        assert 'value: "true"' in result

    def test_yaml_false_is_quoted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"value": "false"})
        assert 'value: "false"' in result

    def test_yaml_null_is_quoted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"value": "null"})
        assert 'value: "null"' in result

    def test_numeric_string_is_quoted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"port": "8080"})
        assert 'port: "8080"' in result

    def test_actual_bool_not_quoted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"enabled": True})
        assert "enabled: true" in result

    def test_actual_int_not_quoted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"count": 42})
        assert "count: 42" in result

    def test_empty_string_is_quoted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"name": ""})
        assert 'name: ""' in result

    def test_plain_string_not_quoted(self):
        from ag2_cli.install.targets.base import format_frontmatter

        result = format_frontmatter({"name": "hello"})
        assert "name: hello" in result


# ---------------------------------------------------------------------------
# Bug 6: parse_frontmatter can't handle YAML list values (Medium)
# ---------------------------------------------------------------------------


class TestParseFrontmatterFixes:
    """Bug 6: parse_frontmatter should parse bracket list syntax."""

    def test_parses_bracket_list(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = "---\ntags: [ag2, autogen, tools]\n---\nBody"
        fm, body = parse_frontmatter(text)
        assert fm["tags"] == ["ag2", "autogen", "tools"]
        assert body == "Body"

    def test_parses_quoted_bracket_list(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = '---\nglobs: ["*.py", "*.js"]\n---\nBody'
        fm, body = parse_frontmatter(text)
        assert fm["globs"] == ["*.py", "*.js"]

    def test_parses_empty_bracket_list(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = "---\nitems: []\n---\nBody"
        fm, body = parse_frontmatter(text)
        assert fm["items"] == []

    def test_still_parses_regular_values(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = "---\nname: my-skill\ndescription: A skill\nalwaysApply: true\n---\nBody"
        fm, body = parse_frontmatter(text)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "A skill"
        assert fm["alwaysApply"] is True


# ---------------------------------------------------------------------------
# Bug 7: serve_cmd doesn't support YAML configs (Medium)
# ---------------------------------------------------------------------------


class TestServeCmdYaml:
    """Bug 7: serve_cmd should support YAML config files like run_cmd does."""

    def test_serve_yaml_routes_to_yaml_builder(self, tmp_path: Path):
        """Verify that .yaml suffix triggers the YAML discovery path in serve_cmd."""
        # Read the serve_cmd source to verify it handles .yaml extensions
        import inspect

        from ag2_cli.commands.serve import serve_cmd

        source = inspect.getsource(serve_cmd)
        # The fix added YAML extension checking
        assert ".yaml" in source or ".yml" in source
        assert "build_agents_from_yaml" in source
        assert "load_yaml_config" in source

    def test_serve_py_file_uses_discover(self, tmp_path: Path):
        """Verify .py files still route to discover() not YAML builder."""
        py_file = tmp_path / "agent.py"
        py_file.write_text("x = 42\n")  # no agent, will fail

        with (
            patch("ag2_cli.commands.serve._require_ag2"),
            patch("ag2_cli.core.discovery.discover") as mock_discover,
        ):
            mock_discover.side_effect = ValueError("No agent found")

            from ag2_cli.commands.serve import serve_cmd

            with contextlib.suppress(SystemExit, Exception):
                serve_cmd(
                    agent_file=py_file,
                    port=8000,
                    protocol="rest",
                    ngrok=False,
                    playground=False,
                )

            mock_discover.assert_called_once()


# ---------------------------------------------------------------------------
# Bug 8: HTML export XSS vulnerability (Medium)
# ---------------------------------------------------------------------------


class TestReplayHtmlEscaping:
    """Bug 8: HTML export must escape user content to prevent XSS."""

    def test_html_export_escapes_script_tags(self, tmp_path: Path):
        from ag2_cli.commands.replay import (
            Session,
            SessionEvent,
            SessionMeta,
            app,
            save_session,
        )
        from typer.testing import CliRunner

        runner = CliRunner()

        xss_content = '<script>alert("xss")</script>'
        session = Session(
            meta=SessionMeta(
                session_id="test-xss-session",
                agent_file="agent.py",
                agent_names=["test"],
                created_at="2026-01-01T00:00:00",
                turns=1,
                duration=1.0,
            ),
            events=[
                SessionEvent(
                    turn=1,
                    speaker="assistant",
                    content=xss_content,
                    role="assistant",
                )
            ],
        )
        save_session(session)

        try:
            result = runner.invoke(app, ["export", "test-xss-session", "--format", "html"])
            # The raw script tag should NOT appear in the output
            assert "<script>" not in result.output
            # The escaped version should appear
            assert html_escape(xss_content) in result.output or "&lt;script&gt;" in result.output
        finally:
            # Clean up
            from ag2_cli.commands.replay import _session_path

            path = _session_path("test-xss-session")
            if path.exists():
                path.unlink()

    def test_html_export_escapes_speaker_name(self, tmp_path: Path):
        from ag2_cli.commands.replay import (
            Session,
            SessionEvent,
            SessionMeta,
            app,
            save_session,
        )
        from typer.testing import CliRunner

        runner = CliRunner()

        session = Session(
            meta=SessionMeta(
                session_id="test-xss-speaker",
                agent_file="agent.py",
                agent_names=["test"],
                created_at="2026-01-01T00:00:00",
                turns=1,
                duration=1.0,
            ),
            events=[
                SessionEvent(
                    turn=1,
                    speaker='<img src=x onerror="alert(1)">',
                    content="safe content",
                    role="assistant",
                )
            ],
        )
        save_session(session)

        try:
            result = runner.invoke(app, ["export", "test-xss-speaker", "--format", "html"])
            assert 'onerror="alert(1)"' not in result.output
        finally:
            from ag2_cli.commands.replay import _session_path

            path = _session_path("test-xss-speaker")
            if path.exists():
                path.unlink()


# ---------------------------------------------------------------------------
# Bug 9: lockfile stores absolute paths for out-of-project files (Medium)
# ---------------------------------------------------------------------------


class TestLockfilePathSafety:
    """Bug 9: lockfile should skip files outside the project directory."""

    def test_out_of_project_files_skipped(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lockfile = Lockfile(tmp_path)
        lockfile.record_install(
            ref="skills/ag2ai/test",
            version="1.0.0",
            targets=["claude"],
            files=[
                tmp_path / "inside" / "file.md",  # Inside project
                Path("/outside/project/file.md"),  # Outside project
            ],
        )

        info = lockfile.get_installed("skills/ag2ai/test")
        assert info is not None
        # Only the inside-project file should be recorded
        assert len(info.files) == 1
        assert "inside/file.md" in info.files[0]
        # No absolute paths
        assert not any(f.startswith("/") for f in info.files)

    def test_all_inside_project_files_kept(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lockfile = Lockfile(tmp_path)
        lockfile.record_install(
            ref="skills/ag2ai/test",
            version="1.0.0",
            targets=["claude"],
            files=[
                tmp_path / "a.md",
                tmp_path / "b.md",
                tmp_path / "sub" / "c.md",
            ],
        )

        info = lockfile.get_installed("skills/ag2ai/test")
        assert len(info.files) == 3


# ---------------------------------------------------------------------------
# Bug 10: broken module left in sys.modules on exec failure (Low)
# ---------------------------------------------------------------------------


class TestDiscoveryCleanup:
    """Bug 10: failed module import should clean up sys.modules."""

    def test_failed_exec_module_cleans_sys_modules(self, tmp_path: Path):
        f = tmp_path / "bad_agent.py"
        f.write_text("raise RuntimeError('intentional error')\n")

        from ag2_cli.core.discovery import import_agent_file

        module_name = f"_ag2_user_{f.stem}"

        with pytest.raises(RuntimeError, match="intentional error"):
            import_agent_file(f)

        # The broken module should NOT remain in sys.modules
        assert module_name not in sys.modules

    def test_successful_import_stays_in_sys_modules(self, tmp_path: Path):
        f = tmp_path / "good_agent.py"
        f.write_text("x = 42\n")

        from ag2_cli.core.discovery import import_agent_file

        module_name = f"_ag2_user_{f.stem}"
        module = import_agent_file(f)

        assert module_name in sys.modules
        assert module.x == 42

        # Clean up
        sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# Bug 11: _type_dir missing explicit "skills" entry (Low)
# ---------------------------------------------------------------------------


class TestTypeDirSkillsMapping:
    """Bug 11: _type_dir and _pluralize_type should explicitly handle 'skills'."""

    def test_type_dir_skills_plural(self):
        from ag2_cli.install.client import ArtifactClient

        assert ArtifactClient._type_dir("skills") == "skills"

    def test_type_dir_skill_singular(self):
        from ag2_cli.install.client import ArtifactClient

        assert ArtifactClient._type_dir("skill") == "skills"

    def test_pluralize_type_skills_plural(self):
        from ag2_cli.install.artifact import _pluralize_type

        assert _pluralize_type("skills") == "skills"

    def test_pluralize_type_skill_singular(self):
        from ag2_cli.install.artifact import _pluralize_type

        assert _pluralize_type("skill") == "skills"

    def test_artifact_ref_uses_skills_correctly(self):
        from ag2_cli.install.artifact import Artifact

        a = Artifact(name="ag2", type="skills", owner="ag2ai")
        assert a.ref == "skills/ag2ai/ag2"


# ---------------------------------------------------------------------------
# Bug 12: ClaudeTarget.uninstall double-counts removals (Low)
# ---------------------------------------------------------------------------


class TestClaudeTargetUninstallCount:
    """Bug 12: uninstall should only count files, not directories."""

    def test_uninstall_counts_only_files(self, tmp_path: Path):
        from ag2_cli.install.targets.claude import ClaudeTarget

        target = ClaudeTarget()

        # Set up: 1 skill dir with 2 files inside
        skills_dir = tmp_path / ".claude" / "skills"
        skill_dir = skills_dir / "ag2-test"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("content")
        (skill_dir / "extra.txt").write_text("extra")

        removed = target.uninstall(tmp_path)

        # Should be exactly 2 (the files), not 3 (files + directory)
        assert len(removed) == 2
        # And none of them should be a directory path
        assert all("SKILL.md" in str(p) or "extra.txt" in str(p) for p in removed)

    def test_uninstall_skill_and_command_counts(self, tmp_path: Path):
        from ag2_cli.install.targets.claude import ClaudeTarget

        target = ClaudeTarget()

        # 1 skill dir with 1 file + 1 command file
        skills_dir = tmp_path / ".claude" / "skills"
        skill_dir = skills_dir / "ag2-test"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("content")

        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "ag2-run.md").write_text("cmd")

        removed = target.uninstall(tmp_path)

        # 1 skill file + 1 command file = 2
        assert len(removed) == 2
