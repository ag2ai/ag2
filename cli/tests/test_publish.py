"""Tests for `ag2 publish artifact` command — validation logic only (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from ag2_cli.app import app
from ag2_cli.commands.publish import _validate_artifact
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers to build artifact dirs
# ---------------------------------------------------------------------------


def _make_artifact(tmp_path: Path, name: str, manifest: dict, extra_files: dict[str, str] | None = None) -> Path:
    """Create a minimal artifact directory for testing."""
    out = tmp_path / name
    out.mkdir(parents=True)
    (out / "artifact.json").write_text(json.dumps(manifest))
    for rel_path, content in (extra_files or {}).items():
        p = out / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return out


def _valid_template_manifest(**overrides: object) -> dict:
    base = {
        "name": "my-template",
        "type": "template",
        "description": "A test template",
        "version": "1.0.0",
        "authors": ["tester"],
        "tags": ["test"],
        "template": {"scaffold": "scaffold/", "variables": {}},
        "skills": {"dir": "skills/", "auto_install": True},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestPublishHelp:
    def test_help_lists_publish(self) -> None:
        result = runner.invoke(app, ["publish", "--help"])
        assert result.exit_code == 0
        assert "artifact" in result.output

    def test_artifact_help(self) -> None:
        result = runner.invoke(app, ["publish", "artifact", "--help"])
        assert result.exit_code == 0
        assert "dry-run" in result.output
        assert "repo" in result.output


class TestPublishDryRun:
    def test_valid_artifact_passes(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "good-tmpl",
            _valid_template_manifest(),
            {
                "scaffold/README.md.tmpl": "# hello",
                "skills/rules/arch/SKILL.md": "---\nname: arch\n---\ncontent",
            },
        )
        result = runner.invoke(app, ["publish", "artifact", str(art), "--dry-run"])
        assert result.exit_code == 0
        assert "All checks passed" in result.output
        assert "Dry run" in result.output

    def test_missing_artifact_json_fails(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(app, ["publish", "artifact", str(empty), "--dry-run"])
        assert result.exit_code == 1
        assert "Missing artifact.json" in result.output

    def test_missing_required_fields_fails(self, tmp_path: Path) -> None:
        art = _make_artifact(tmp_path, "bad", {"name": "bad", "type": "template"})
        result = runner.invoke(app, ["publish", "artifact", str(art), "--dry-run"])
        assert result.exit_code == 1
        assert "description" in result.output

    def test_not_a_directory_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        result = runner.invoke(app, ["publish", "artifact", str(f), "--dry-run"])
        assert result.exit_code == 1
        assert "Not a directory" in result.output

    def test_shows_target_path(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "my-tool",
            {
                "name": "my-tool",
                "type": "tool",
                "description": "A tool",
                "version": "1.0.0",
                "authors": ["tester"],
                "tool": {"kind": "ag2", "source": "src/"},
            },
            {"src/__init__.py": "", "skills/skills/use/SKILL.md": "---\nname: x\n---\n"},
        )
        result = runner.invoke(app, ["publish", "artifact", str(art), "--dry-run"])
        assert result.exit_code == 0
        assert "tools/my-tool/" in result.output


# ---------------------------------------------------------------------------
# Unit tests for _validate_artifact
# ---------------------------------------------------------------------------


class TestValidateArtifact:
    def test_valid_template(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "t",
            _valid_template_manifest(),
            {
                "scaffold/README.md.tmpl": "hi",
                "skills/rules/a/SKILL.md": "---\nname: a\n---\n",
            },
        )
        manifest, issues = _validate_artifact(art)
        assert manifest is not None
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0

    def test_invalid_json(self, tmp_path: Path) -> None:
        art = tmp_path / "bad-json"
        art.mkdir()
        (art / "artifact.json").write_text("{invalid json")
        manifest, issues = _validate_artifact(art)
        assert manifest is None
        assert any("Invalid JSON" in i.message for i in issues)

    def test_unknown_type(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "x",
            {
                "name": "x",
                "type": "widget",
                "description": "d",
                "version": "1.0.0",
                "authors": ["a"],
            },
        )
        _, issues = _validate_artifact(art)
        assert any("Unknown artifact type" in i.message for i in issues)

    def test_missing_type_config(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "t",
            {
                "name": "t",
                "type": "tool",
                "description": "d",
                "version": "1.0.0",
                "authors": ["a"],
            },
        )
        _, issues = _validate_artifact(art)
        errors = [i for i in issues if i.level == "error"]
        assert any("Missing 'tool' config" in i.message for i in errors)

    def test_bad_version_warns(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "t",
            _valid_template_manifest(version="abc"),
            {
                "scaffold/x": "",
                "skills/rules/a/SKILL.md": "---\nname: a\n---\n",
            },
        )
        _, issues = _validate_artifact(art)
        warnings = [i for i in issues if i.level == "warning"]
        assert any("semver" in i.message for i in warnings)

    def test_no_authors_warns(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "t",
            _valid_template_manifest(authors=[]),
            {
                "scaffold/x": "",
                "skills/rules/a/SKILL.md": "---\nname: a\n---\n",
            },
        )
        _, issues = _validate_artifact(art)
        # authors=[] triggers a required-field error AND a warning
        assert any("authors" in i.message.lower() for i in issues)

    def test_no_tags_warns(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "t",
            _valid_template_manifest(tags=[]),
            {
                "scaffold/x": "",
                "skills/rules/a/SKILL.md": "---\nname: a\n---\n",
            },
        )
        _, issues = _validate_artifact(art)
        warnings = [i for i in issues if i.level == "warning"]
        assert any("tags" in i.message.lower() for i in warnings)

    def test_missing_scaffold_dir_warns(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "t",
            _valid_template_manifest(),
            {
                "skills/rules/a/SKILL.md": "---\nname: a\n---\n",
            },
        )
        _, issues = _validate_artifact(art)
        warnings = [i for i in issues if i.level == "warning"]
        assert any("scaffold" in i.message for i in warnings)

    def test_empty_skills_dir_warns(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "t",
            _valid_template_manifest(),
            {
                "scaffold/x": "",
            },
        )
        # Create skills dir but leave it empty
        (art / "skills").mkdir(parents=True)
        _, issues = _validate_artifact(art)
        warnings = [i for i in issues if i.level == "warning"]
        assert any("no SKILL.md" in i.message for i in warnings)

    def test_skills_type_checks_root_dirs(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "s",
            {
                "name": "s",
                "type": "skills",
                "description": "d",
                "version": "1.0.0",
                "authors": ["a"],
                "tags": ["t"],
                "skills": {"dir": ".", "auto_install": True},
            },
        )
        _, issues = _validate_artifact(art)
        warnings = [i for i in issues if i.level == "warning"]
        assert any("rules/" in i.message or "skills/" in i.message for i in warnings)

    def test_bundle_skips_skills_check(self, tmp_path: Path) -> None:
        art = _make_artifact(
            tmp_path,
            "b",
            {
                "name": "b",
                "type": "bundle",
                "description": "d",
                "version": "1.0.0",
                "authors": ["a"],
                "tags": ["t"],
                "bundle": {"artifacts": [], "install_order": []},
            },
        )
        _, issues = _validate_artifact(art)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0
        # No skills-related warnings for bundles
        assert not any("skills" in i.message.lower() for i in issues)
