"""Tool installer — installs AG2 tool functions and MCP servers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..artifact import Artifact, InstallResult, load_artifact_json, parse_artifact_id
from ..client import ArtifactClient, FetchError
from ..lockfile import Lockfile
from ..mcp_config import configure_mcp_server, detect_mcp_targets
from ..resolver import DependencyResolver
from ..targets import Target
from ._utils import copy_tree
from .skills import SkillsInstaller, load_skills_from_artifact


class ToolInstaller:
    """Installs AG2 tool functions and MCP servers."""

    def __init__(
        self,
        client: ArtifactClient,
        lockfile: Lockfile,
        resolver: DependencyResolver,
        skills_installer: SkillsInstaller,
    ):
        self.client = client
        self.lockfile = lockfile
        self.resolver = resolver
        self.skills_installer = skills_installer

    def install(
        self,
        name: str,
        project_dir: Path,
        targets: list[Target],
    ) -> InstallResult:
        """Install a tool artifact. Name can be 'owner/name' or just 'name'."""
        owner, artifact_name = parse_artifact_id(name)
        cached_dir = self.client.fetch_artifact_dir("tool", artifact_name, owner=owner)
        artifact_json = cached_dir / "artifact.json"
        if not artifact_json.exists():
            raise FetchError(f"Tool '{owner}/{artifact_name}' has no artifact.json")
        artifact = load_artifact_json(artifact_json)
        artifact.source_dir = cached_dir

        if artifact.tool is None:
            raise FetchError(f"Artifact '{name}' is not a tool (missing tool config)")

        if artifact.tool.kind == "mcp":
            return self._install_mcp_tool(artifact, project_dir, targets)
        else:
            return self._install_ag2_tool(artifact, project_dir, targets)

    def _install_ag2_tool(
        self,
        artifact: Artifact,
        project_dir: Path,
        targets: list[Target],
    ) -> InstallResult:
        """Install an AG2 tool function."""
        assert artifact.tool is not None
        assert artifact.source_dir is not None

        all_files: list[Path] = []
        tool_config = artifact.tool

        # Copy source to project_dir/tools/<name>/
        dest = project_dir / tool_config.install_to / artifact.name
        source = artifact.source_dir / tool_config.source
        if source.is_dir():
            copied = copy_tree(source, dest)
            all_files.extend(copied)

        # Install bundled skills
        skill_items = load_skills_from_artifact(artifact)
        if skill_items:
            for tgt in targets:
                paths = tgt.install(project_dir, skill_items)
                all_files.extend(paths)

        # Resolve dependencies
        deps_installed: list[str] = []
        deps = self.resolver.resolve(artifact)
        for dep in deps:
            if dep.type == "skills":
                self.skills_installer.install([dep.name], targets, project_dir)
                deps_installed.append(dep.ref)

        target_names = [t.name for t in targets]
        self.lockfile.record_install(
            ref=artifact.ref,
            version=artifact.version,
            targets=target_names,
            files=all_files,
        )

        return InstallResult(
            artifact=artifact,
            files_created=all_files,
            targets_used=target_names,
            dependencies_installed=deps_installed,
        )

    def _install_mcp_tool(
        self,
        artifact: Artifact,
        project_dir: Path,
        targets: list[Target],
    ) -> InstallResult:
        """Install an MCP server tool."""
        assert artifact.tool is not None
        assert artifact.source_dir is not None

        all_files: list[Path] = []
        tool_config = artifact.tool

        # Copy server source to project_dir/tools/<name>/
        dest = project_dir / tool_config.install_to / artifact.name
        source = artifact.source_dir / tool_config.source
        if source.is_dir():
            copied = copy_tree(source, dest)
            all_files.extend(copied)

        # Install Python/Node dependencies
        if tool_config.requires:
            _install_dependencies(tool_config.requires, dest)

        # Auto-configure MCP in detected IDEs
        if tool_config.mcp_config:
            # Resolve ${toolDir} in config values
            mcp_config = _resolve_tool_dir(tool_config.mcp_config, dest)
            mcp_targets = detect_mcp_targets(project_dir)
            config_files = configure_mcp_server(
                project_dir=project_dir,
                server_name=f"ag2-{artifact.name}",
                config=mcp_config,
                ide_targets=mcp_targets or None,
            )
            all_files.extend(config_files)

        # Install bundled skills
        skill_items = load_skills_from_artifact(artifact)
        if skill_items:
            for tgt in targets:
                paths = tgt.install(project_dir, skill_items)
                all_files.extend(paths)

        # Resolve dependencies
        deps_installed: list[str] = []
        deps = self.resolver.resolve(artifact)
        for dep in deps:
            if dep.type == "skills":
                self.skills_installer.install([dep.name], targets, project_dir)
                deps_installed.append(dep.ref)

        target_names = [t.name for t in targets]
        self.lockfile.record_install(
            ref=artifact.ref,
            version=artifact.version,
            targets=target_names,
            files=all_files,
        )

        return InstallResult(
            artifact=artifact,
            files_created=all_files,
            targets_used=target_names,
            dependencies_installed=deps_installed,
        )


def _install_dependencies(requires: list[str], tool_dir: Path) -> None:
    """Install tool dependencies (best-effort, logs warnings on failure)."""
    from ...ui import console

    for req in requires:
        try:
            result = subprocess.run(
                ["pip", "install", req],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                console.print(f"  [success]\u2713[/success] Installed {req}")
            else:
                console.print(f"  [warning]![/warning] Failed to install {req}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            console.print(f"  [warning]![/warning] Could not install {req} (pip not available)")


def _resolve_tool_dir(config: dict, tool_dir: Path) -> dict:
    """Replace ${toolDir} in MCP config values with the actual tool directory path."""
    resolved = {}
    for key, value in config.items():
        if isinstance(value, str):
            resolved[key] = value.replace("${toolDir}", str(tool_dir))
        elif isinstance(value, list):
            resolved[key] = [v.replace("${toolDir}", str(tool_dir)) if isinstance(v, str) else v for v in value]
        elif isinstance(value, dict):
            resolved[key] = _resolve_tool_dir(value, tool_dir)
        else:
            resolved[key] = value
    return resolved
