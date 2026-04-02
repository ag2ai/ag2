"""Tests for `ag2 create artifact` command."""

from __future__ import annotations

import json
from pathlib import Path

from ag2_cli.app import app
from typer.testing import CliRunner

runner = CliRunner()


class TestCreateArtifactHelp:
    def test_help_lists_artifact(self) -> None:
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0
        assert "artifact" in result.output

    def test_artifact_help(self) -> None:
        result = runner.invoke(app, ["create", "artifact", "--help"])
        assert result.exit_code == 0
        assert "template" in result.output
        assert "artifact.json" in result.output.lower() or "artifact" in result.output.lower()


class TestCreateArtifactValidation:
    def test_rejects_unknown_type(self) -> None:
        result = runner.invoke(app, ["create", "artifact", "bogus", "my-thing"])
        assert result.exit_code == 1
        assert "Unknown artifact type" in result.output

    def test_rejects_existing_directory(self, tmp_path: Path) -> None:
        existing = tmp_path / "my-tool"
        existing.mkdir()
        result = runner.invoke(app, ["create", "artifact", "tool", "my-tool", "-o", str(tmp_path)])
        assert result.exit_code == 1
        assert "already exists" in result.output


class TestCreateArtifactTemplate:
    def test_scaffolds_template(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["create", "artifact", "template", "my-tmpl", "-o", str(tmp_path)])
        assert result.exit_code == 0
        assert "Created" in result.output

        out = tmp_path / "my-tmpl"
        assert out.is_dir()

        # artifact.json
        manifest = json.loads((out / "artifact.json").read_text())
        assert manifest["name"] == "my-tmpl"
        assert manifest["type"] == "template"
        assert "template" in manifest
        assert manifest["template"]["scaffold"] == "scaffold/"
        assert "project_name" in manifest["template"]["variables"]

        # scaffold
        assert (out / "scaffold" / "README.md.tmpl").is_file()
        readme = (out / "scaffold" / "README.md.tmpl").read_text()
        assert "{{ project_name }}" in readme

        # skills
        assert (out / "skills" / "rules" / "my-tmpl-architecture" / "SKILL.md").is_file()
        assert (out / "skills" / "skills" / "add-feature" / "SKILL.md").is_file()

    def test_skill_frontmatter(self, tmp_path: Path) -> None:
        runner.invoke(app, ["create", "artifact", "template", "test-tmpl", "-o", str(tmp_path)])
        skill = (tmp_path / "test-tmpl" / "skills" / "rules" / "test-tmpl-architecture" / "SKILL.md").read_text()
        assert "---" in skill
        assert "name: test-tmpl-architecture" in skill
        assert "description:" in skill
        assert "license: Apache-2.0" in skill


class TestCreateArtifactTool:
    def test_scaffolds_tool(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["create", "artifact", "tool", "web-scraper", "-o", str(tmp_path)])
        assert result.exit_code == 0

        out = tmp_path / "web-scraper"
        manifest = json.loads((out / "artifact.json").read_text())
        assert manifest["type"] == "tool"
        assert manifest["tool"]["kind"] == "ag2"
        assert manifest["tool"]["source"] == "src/"

        assert (out / "src" / "__init__.py").is_file()
        assert (out / "src" / "web_scraper.py").is_file()
        assert (out / "tests" / "test_web_scraper.py").is_file()
        assert (out / "skills" / "skills" / "integrate-web-scraper" / "SKILL.md").is_file()


class TestCreateArtifactDataset:
    def test_scaffolds_dataset(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["create", "artifact", "dataset", "eval-bench", "-o", str(tmp_path)])
        assert result.exit_code == 0

        out = tmp_path / "eval-bench"
        manifest = json.loads((out / "artifact.json").read_text())
        assert manifest["type"] == "dataset"
        assert manifest["dataset"]["format"] == "jsonl"

        assert (out / "data" / "sample.jsonl").is_file()
        sample = (out / "data" / "sample.jsonl").read_text()
        assert "input" in sample

        assert (out / "skills" / "rules" / "eval-bench-schema" / "SKILL.md").is_file()


class TestCreateArtifactAgent:
    def test_scaffolds_agent(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["create", "artifact", "agent", "code-helper", "-o", str(tmp_path)])
        assert result.exit_code == 0

        out = tmp_path / "code-helper"
        manifest = json.loads((out / "artifact.json").read_text())
        assert manifest["type"] == "agent"
        assert manifest["agent"]["source"] == "agent.md"

        assert (out / "agent.md").is_file()
        agent_md = (out / "agent.md").read_text()
        assert "Code Helper" in agent_md

        assert (out / "skills" / "skills" / "use-code-helper" / "SKILL.md").is_file()


class TestCreateArtifactSkills:
    def test_scaffolds_skills(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["create", "artifact", "skills", "fastapi", "-o", str(tmp_path)])
        assert result.exit_code == 0

        out = tmp_path / "fastapi"
        manifest = json.loads((out / "artifact.json").read_text())
        assert manifest["type"] == "skills"
        assert manifest["skills"]["dir"] == "."

        assert (out / "rules" / "fastapi" / "SKILL.md").is_file()
        assert (out / "skills" / "fastapi-guide" / "SKILL.md").is_file()


class TestCreateArtifactBundle:
    def test_scaffolds_bundle(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["create", "artifact", "bundle", "starter-kit", "-o", str(tmp_path)])
        assert result.exit_code == 0

        out = tmp_path / "starter-kit"
        manifest = json.loads((out / "artifact.json").read_text())
        assert manifest["type"] == "bundle"
        assert manifest["bundle"]["artifacts"] == []
        assert "install_order" in manifest["bundle"]

        # Bundles should not have a skills directory
        assert not (out / "skills").exists()
