"""Tests for the artifact model and manifest parser."""

import json
from pathlib import Path


class TestLoadArtifactJson:
    """Test parsing artifact.json for all types."""

    def test_parse_skills_artifact(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact_json

        manifest = {
            "name": "ag2",
            "type": "skills",
            "display_name": "AG2 Skills",
            "description": "Skills for the AG2 framework",
            "version": "0.3.0",
            "authors": ["ag2ai"],
            "license": "Apache-2.0",
            "tags": ["ag2", "multi-agent"],
            "skills": {"dir": ".", "auto_install": True},
        }
        (tmp_path / "artifact.json").write_text(json.dumps(manifest))
        artifact = load_artifact_json(tmp_path / "artifact.json")

        assert artifact.name == "ag2"
        assert artifact.type == "skills"
        assert artifact.display_name == "AG2 Skills"
        assert artifact.version == "0.3.0"
        assert artifact.skills_config is not None
        assert artifact.skills_config.auto_install is True
        assert artifact.ref == "skills/ag2ai/ag2"

    def test_parse_template_artifact(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact_json

        manifest = {
            "name": "fullstack-app",
            "type": "template",
            "display_name": "Fullstack App",
            "description": "A fullstack agentic app",
            "version": "1.0.0",
            "template": {
                "scaffold": "scaffold/",
                "variables": {
                    "project_name": {"prompt": "Project name", "default": "my-app", "transform": "slug"},
                    "description": {"prompt": "Description", "default": "My app"},
                },
                "ignore": ["__pycache__", "*.pyc"],
                "post_install": ["uv sync"],
            },
            "depends": ["skills/ag2"],
        }
        (tmp_path / "artifact.json").write_text(json.dumps(manifest))
        artifact = load_artifact_json(tmp_path / "artifact.json")

        assert artifact.name == "fullstack-app"
        assert artifact.type == "template"
        assert artifact.template is not None
        assert artifact.template.scaffold == "scaffold/"
        assert "project_name" in artifact.template.variables
        assert artifact.template.variables["project_name"].transform == "slug"
        assert artifact.template.post_install == ["uv sync"]
        assert artifact.depends == ["skills/ag2"]
        assert artifact.ref == "templates/ag2ai/fullstack-app"

    def test_parse_tool_ag2(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact_json

        manifest = {
            "name": "web-search",
            "type": "tool",
            "version": "1.0.0",
            "tool": {
                "kind": "ag2",
                "source": "src/",
                "functions": [{"name": "web_search", "description": "Search the web"}],
                "requires": ["httpx>=0.27"],
            },
        }
        (tmp_path / "artifact.json").write_text(json.dumps(manifest))
        artifact = load_artifact_json(tmp_path / "artifact.json")

        assert artifact.tool is not None
        assert artifact.tool.kind == "ag2"
        assert len(artifact.tool.functions) == 1
        assert artifact.tool.requires == ["httpx>=0.27"]

    def test_parse_tool_mcp(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact_json

        manifest = {
            "name": "github-mcp",
            "type": "tool",
            "version": "1.0.0",
            "tool": {
                "kind": "mcp",
                "runtime": "python",
                "source": "src/",
                "entry_point": "server.py",
                "transport": "stdio",
                "mcp_config": {
                    "command": "uv",
                    "args": ["run", "server.py"],
                    "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
                },
                "tools_provided": [{"name": "list_repos", "description": "List repos"}],
                "requires": ["mcp[cli]>=1.0"],
            },
        }
        (tmp_path / "artifact.json").write_text(json.dumps(manifest))
        artifact = load_artifact_json(tmp_path / "artifact.json")

        assert artifact.tool.kind == "mcp"
        assert artifact.tool.runtime == "python"
        assert artifact.tool.mcp_config["command"] == "uv"
        assert len(artifact.tool.tools_provided) == 1

    def test_parse_dataset(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact_json

        manifest = {
            "name": "eval-bench",
            "type": "dataset",
            "version": "1.0.0",
            "dataset": {
                "inline": "data/",
                "remote": [
                    {"name": "full.jsonl", "url": "https://example.com/full.jsonl", "size": "150MB", "sha256": "abc"}
                ],
                "format": "jsonl",
                "schema": {"fields": [{"name": "input", "type": "string"}]},
                "eval_compatible": True,
            },
        }
        (tmp_path / "artifact.json").write_text(json.dumps(manifest))
        artifact = load_artifact_json(tmp_path / "artifact.json")

        assert artifact.dataset is not None
        assert artifact.dataset.format == "jsonl"
        assert artifact.dataset.eval_compatible is True
        assert len(artifact.dataset.remote) == 1
        assert artifact.dataset.remote[0].sha256 == "abc"

    def test_parse_agent(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact_json

        manifest = {
            "name": "research-analyst",
            "type": "agent",
            "version": "1.0.0",
            "agent": {
                "source": "agent.md",
                "model": "sonnet",
                "tools": ["Read", "Grep", "WebSearch"],
                "max_turns": 50,
                "memory": "project",
                "mcp_servers": {"bundled": ["mcp/web-crawler"]},
                "preload_skills": ["research-methodology"],
            },
        }
        (tmp_path / "artifact.json").write_text(json.dumps(manifest))
        artifact = load_artifact_json(tmp_path / "artifact.json")

        assert artifact.agent is not None
        assert artifact.agent.model == "sonnet"
        assert "Read" in artifact.agent.tools
        assert artifact.agent.max_turns == 50

    def test_parse_bundle(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact_json

        manifest = {
            "name": "starter-kit",
            "type": "bundle",
            "version": "1.0.0",
            "bundle": {
                "artifacts": [
                    {"ref": "skills/ag2", "required": True},
                    {"ref": "tools/web-search", "required": False},
                ],
                "install_order": ["skills", "tools"],
            },
        }
        (tmp_path / "artifact.json").write_text(json.dumps(manifest))
        artifact = load_artifact_json(tmp_path / "artifact.json")

        assert artifact.bundle is not None
        assert len(artifact.bundle.artifacts) == 2
        assert artifact.bundle.artifacts[0].required is True
        assert artifact.bundle.artifacts[1].required is False

    def test_parse_minimal_manifest(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact_json

        manifest = {"name": "test", "type": "skills"}
        (tmp_path / "artifact.json").write_text(json.dumps(manifest))
        artifact = load_artifact_json(tmp_path / "artifact.json")

        assert artifact.name == "test"
        assert artifact.version == "0.0.0"
        assert artifact.authors == []
        assert artifact.depends == []


class TestLoadLegacyManifest:
    """Test backward compat with existing manifest.json."""

    def test_loads_existing_skills_manifest(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_legacy_manifest

        manifest = {
            "name": "skills",
            "display_name": "AG2 Skills",
            "description": "AG2 skills",
            "version": "0.3.0",
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        artifact = load_legacy_manifest(tmp_path)

        assert artifact.name == "skills"
        assert artifact.type == "skills"
        assert artifact.display_name == "AG2 Skills"
        assert artifact.version == "0.3.0"
        assert artifact.source_dir == tmp_path


class TestLoadArtifact:
    """Test the unified load_artifact function."""

    def test_prefers_artifact_json(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact

        # Write both files
        (tmp_path / "artifact.json").write_text(json.dumps({"name": "test-new", "type": "skills"}))
        (tmp_path / "manifest.json").write_text(
            json.dumps({"name": "test-old", "display_name": "Old", "description": "", "version": "0.1.0"})
        )
        artifact = load_artifact(tmp_path)

        assert artifact is not None
        assert artifact.name == "test-new"

    def test_falls_back_to_manifest_json(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact

        (tmp_path / "manifest.json").write_text(
            json.dumps({"name": "fallback", "display_name": "Fallback", "description": "", "version": "0.2.0"})
        )
        artifact = load_artifact(tmp_path)

        assert artifact is not None
        assert artifact.name == "fallback"

    def test_returns_none_when_no_manifest(self, tmp_path: Path):
        from ag2_cli.install.artifact import load_artifact

        assert load_artifact(tmp_path) is None
