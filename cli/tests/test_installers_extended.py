"""Tests for tool, dataset, and bundle installers."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ag2_cli.install.artifact import BundleRef, InstallResult
from ag2_cli.install.client import ArtifactClient, FetchError
from ag2_cli.install.installers.bundles import BundleInstaller
from ag2_cli.install.installers.datasets import DatasetInstaller
from ag2_cli.install.installers.skills import SkillsInstaller
from ag2_cli.install.installers.tools import ToolInstaller
from ag2_cli.install.lockfile import Lockfile
from ag2_cli.install.resolver import DependencyResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ag2_tool_artifact(cache_dir: Path, name: str = "my-tool") -> Path:
    """Create an AG2 (non-MCP) tool artifact in the local cache."""
    tool_dir = cache_dir / "tools" / "ag2ai" / name / "latest"
    tool_dir.mkdir(parents=True)

    manifest = {
        "name": name,
        "type": "tool",
        "version": "1.0.0",
        "tool": {
            "kind": "ag2",
            "source": "src/",
            "install_to": "tools/",
        },
    }
    (tool_dir / "artifact.json").write_text(json.dumps(manifest))

    # Create source files
    src = tool_dir / "src"
    src.mkdir()
    (src / "__init__.py").write_text("# tool init")
    (src / "helper.py").write_text("def run(): pass")

    (tool_dir / ".fetched").touch()
    return tool_dir


def _make_mcp_tool_artifact(cache_dir: Path, name: str = "mcp-server") -> Path:
    """Create an MCP tool artifact in the local cache."""
    tool_dir = cache_dir / "tools" / "ag2ai" / name / "latest"
    tool_dir.mkdir(parents=True)

    manifest = {
        "name": name,
        "type": "tool",
        "version": "1.0.0",
        "tool": {
            "kind": "mcp",
            "source": "src/",
            "install_to": "tools/",
            "mcp_config": {
                "command": "python",
                "args": ["${toolDir}/server.py"],
            },
        },
    }
    (tool_dir / "artifact.json").write_text(json.dumps(manifest))

    src = tool_dir / "src"
    src.mkdir()
    (src / "server.py").write_text("print('mcp server')")

    (tool_dir / ".fetched").touch()
    return tool_dir


def _make_dataset_artifact(
    cache_dir: Path,
    name: str = "my-dataset",
    *,
    with_inline: bool = True,
    with_remote: bool = False,
    with_schema: bool = False,
) -> Path:
    """Create a dataset artifact in the local cache."""
    ds_dir = cache_dir / "datasets" / "ag2ai" / name / "latest"
    ds_dir.mkdir(parents=True)

    dataset_config: dict = {}
    if with_inline:
        dataset_config["inline"] = "data/"
    if with_remote:
        dataset_config["remote"] = [
            {"name": "big-file.parquet", "url": "https://example.com/big.parquet", "size": "500MB"},
        ]
    if with_schema:
        dataset_config["schema"] = {
            "type": "object",
            "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
        }

    manifest = {
        "name": name,
        "type": "dataset",
        "version": "1.0.0",
        "dataset": dataset_config,
    }
    (ds_dir / "artifact.json").write_text(json.dumps(manifest))

    # Create inline data directory
    if with_inline:
        data = ds_dir / "data"
        data.mkdir()
        (data / "train.jsonl").write_text('{"q":"hi","a":"hello"}\n')
        (data / "test.jsonl").write_text('{"q":"bye","a":"goodbye"}\n')

    (ds_dir / ".fetched").touch()
    return ds_dir


def _make_bundle_artifact(cache_dir: Path, name: str = "starter-kit") -> Path:
    """Create a bundle artifact that references a skills pack."""
    bundle_dir = cache_dir / "bundles" / "ag2ai" / name / "latest"
    bundle_dir.mkdir(parents=True)

    manifest = {
        "name": name,
        "type": "bundle",
        "version": "1.0.0",
        "bundle": {
            "artifacts": [
                {"ref": "skills/test-skills", "required": True},
            ],
            "install_order": ["skills", "tools", "templates", "datasets", "agents"],
        },
    }
    (bundle_dir / "artifact.json").write_text(json.dumps(manifest))
    (bundle_dir / ".fetched").touch()
    return bundle_dir


def _build_tool_installer(tmp_path: Path) -> tuple[ToolInstaller, Lockfile, Path]:
    """Build a ToolInstaller wired to a local cache under tmp_path."""
    cache_dir = tmp_path / "cache"
    project = tmp_path / "project"
    project.mkdir()

    client = ArtifactClient(cache_dir=cache_dir)
    lockfile = Lockfile(project)
    resolver = DependencyResolver(client, lockfile)
    skills = SkillsInstaller(client, lockfile, resolver)
    installer = ToolInstaller(client, lockfile, resolver, skills)
    return installer, lockfile, project


def _build_dataset_installer(tmp_path: Path) -> tuple[DatasetInstaller, Lockfile, Path]:
    """Build a DatasetInstaller wired to a local cache under tmp_path."""
    cache_dir = tmp_path / "cache"
    project = tmp_path / "project"
    project.mkdir()

    client = ArtifactClient(cache_dir=cache_dir)
    lockfile = Lockfile(project)
    resolver = DependencyResolver(client, lockfile)
    skills = SkillsInstaller(client, lockfile, resolver)
    installer = DatasetInstaller(client, lockfile, resolver, skills)
    return installer, lockfile, project


# ===========================================================================
# TestToolInstaller
# ===========================================================================


class TestToolInstaller:
    """Test AG2 and MCP tool installation."""

    def test_install_ag2_tool_copies_source_and_records_lockfile(self, tmp_path: Path):
        """Installing an AG2 tool copies source files and records the artifact in the lockfile."""
        installer, lockfile, project = _build_tool_installer(tmp_path)
        _make_ag2_tool_artifact(tmp_path / "cache")

        result = installer.install("my-tool", project, [])

        # Source files copied to project/tools/my-tool/
        dest = project / "tools" / "my-tool"
        assert dest.is_dir()
        assert (dest / "__init__.py").exists()
        assert (dest / "helper.py").exists()
        assert (dest / "__init__.py").read_text() == "# tool init"

        # InstallResult contains the created files
        assert len(result.files_created) == 2
        assert result.artifact.name == "my-tool"

        # Lockfile records the install
        assert lockfile.is_installed("tools/ag2ai/my-tool")

    def test_install_mcp_tool_copies_source_and_records_lockfile(self, tmp_path: Path):
        """Installing an MCP tool copies server source and records the artifact in the lockfile."""
        installer, lockfile, project = _build_tool_installer(tmp_path)
        _make_mcp_tool_artifact(tmp_path / "cache")

        # Patch configure_mcp_server to avoid writing real IDE config files
        with (
            patch(
                "ag2_cli.install.installers.tools.configure_mcp_server",
                return_value=[],
            ),
            patch(
                "ag2_cli.install.installers.tools.detect_mcp_targets",
                return_value=[],
            ),
        ):
            result = installer.install("mcp-server", project, [])

        # Source files copied
        dest = project / "tools" / "mcp-server"
        assert dest.is_dir()
        assert (dest / "server.py").exists()
        assert (dest / "server.py").read_text() == "print('mcp server')"

        assert len(result.files_created) == 1
        assert result.artifact.name == "mcp-server"
        assert result.artifact.tool.kind == "mcp"

        # Lockfile records the install
        assert lockfile.is_installed("tools/ag2ai/mcp-server")


# ===========================================================================
# TestDatasetInstaller
# ===========================================================================


class TestDatasetInstaller:
    """Test dataset installation with inline data, remote handling, and schema."""

    def test_install_with_inline_data_copies_data_directory(self, tmp_path: Path):
        """Installing a dataset with inline data copies files to project/data/<name>/."""
        installer, lockfile, project = _build_dataset_installer(tmp_path)
        _make_dataset_artifact(tmp_path / "cache", with_inline=True)

        result = installer.install("my-dataset", project, [])

        dest = project / "data" / "my-dataset"
        assert dest.is_dir()
        assert (dest / "train.jsonl").exists()
        assert (dest / "test.jsonl").exists()
        assert (dest / "train.jsonl").read_text() == '{"q":"hi","a":"hello"}\n'

        assert len(result.files_created) == 2
        assert result.artifact.name == "my-dataset"
        assert lockfile.is_installed("datasets/ag2ai/my-dataset")

    def test_install_without_full_skips_remote_files_adds_warning(self, tmp_path: Path):
        """Without --full, remote files are not downloaded and a warning is emitted."""
        installer, lockfile, project = _build_dataset_installer(tmp_path)
        _make_dataset_artifact(
            tmp_path / "cache",
            with_inline=False,
            with_remote=True,
        )

        # full=False is the default
        result = installer.install("my-dataset", project, [], full=False)

        assert len(result.warnings) == 1
        assert "Remote files not downloaded" in result.warnings[0]
        assert "big-file.parquet" in result.warnings[0]
        assert "--full" in result.warnings[0]
        # No files should have been downloaded
        dest = project / "data" / "my-dataset"
        assert not (dest / "big-file.parquet").exists()

    def test_install_writes_schema_json_when_schema_present(self, tmp_path: Path):
        """When the dataset artifact has a schema, schema.json is written to the data directory."""
        installer, lockfile, project = _build_dataset_installer(tmp_path)
        _make_dataset_artifact(
            tmp_path / "cache",
            with_inline=True,
            with_schema=True,
        )

        result = installer.install("my-dataset", project, [])

        schema_file = project / "data" / "my-dataset" / "schema.json"
        assert schema_file.exists()

        schema = json.loads(schema_file.read_text())
        assert schema["type"] == "object"
        assert "question" in schema["properties"]
        assert "answer" in schema["properties"]

        # schema.json should be among the created files
        assert schema_file in result.files_created


# ===========================================================================
# TestBundleInstaller
# ===========================================================================


class TestBundleInstaller:
    """Test bundle orchestration, artifact selection, and type dispatch."""

    def _build_bundle_installer(self, tmp_path: Path):
        """Build a BundleInstaller with mocked sub-installers."""
        cache_dir = tmp_path / "cache"
        project = tmp_path / "project"
        project.mkdir()

        client = ArtifactClient(cache_dir=cache_dir)
        lockfile = Lockfile(project)
        resolver = DependencyResolver(client, lockfile)
        skills = MagicMock(spec=SkillsInstaller)
        templates = MagicMock()
        tools = MagicMock()
        datasets = MagicMock()
        agents = MagicMock()

        installer = BundleInstaller(
            client=client,
            lockfile=lockfile,
            resolver=resolver,
            skills_installer=skills,
            template_installer=templates,
            tool_installer=tools,
            dataset_installer=datasets,
            agent_installer=agents,
        )
        return installer, lockfile, project, skills

    def test_install_bundle_installs_referenced_skills_pack(self, tmp_path: Path):
        """Installing a bundle dispatches to the skills installer for skills refs."""
        installer, lockfile, project, skills_mock = self._build_bundle_installer(tmp_path)
        _make_bundle_artifact(tmp_path / "cache")

        # Configure the mock skills installer to return a valid result
        from ag2_cli.install.artifact import Artifact

        mock_artifact = Artifact(name="test-skills", type="skills", version="1.0.0")
        mock_result = InstallResult(artifact=mock_artifact, files_created=[project / "skills" / "file.md"])
        skills_mock.install.return_value = [mock_result]

        with patch("ag2_cli.install.installers.bundles.load_skills_from_artifact", return_value=[]):
            result = installer.install("starter-kit", project, [])

        # The skills installer should have been called with the referenced pack name
        skills_mock.install.assert_called_once_with(["test-skills"], [], project)
        assert "skills/test-skills" in result.dependencies_installed
        assert lockfile.is_installed("bundles/ag2ai/starter-kit")

    def test_select_artifacts_returns_required_refs_when_no_optional(self):
        """_select_artifacts returns only required refs when there are no optional ones."""
        installer = BundleInstaller(
            client=MagicMock(),
            lockfile=MagicMock(),
            resolver=MagicMock(),
            skills_installer=MagicMock(),
            template_installer=MagicMock(),
            tool_installer=MagicMock(),
            dataset_installer=MagicMock(),
            agent_installer=MagicMock(),
        )

        refs = [
            BundleRef(ref="skills/core", required=True),
            BundleRef(ref="tools/web-search", required=True),
        ]

        selected = installer._select_artifacts(refs)
        assert selected == ["skills/core", "tools/web-search"]

    def test_install_by_type_raises_fetch_error_for_unknown_type(self, tmp_path: Path):
        """_install_by_type raises FetchError when given an unrecognised type key."""
        installer, _, project, _ = self._build_bundle_installer(tmp_path)

        with pytest.raises(FetchError, match="Unknown artifact type in bundle: widgets"):
            installer._install_by_type("widgets", "some-widget", project, [])
