"""Agent installer — installs pre-built Claude Code subagents."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..artifact import InstallResult, load_artifact_json, parse_artifact_id
from ..client import ArtifactClient, FetchError
from ..lockfile import Lockfile
from ..resolver import DependencyResolver
from ..targets import Target
from ._utils import copy_tree
from .skills import SkillsInstaller, load_skills_from_artifact


class AgentInstaller:
    """Installs Claude Code custom subagents."""

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
        """Install an agent artifact. Name can be 'owner/name' or just 'name'."""
        owner, artifact_name = parse_artifact_id(name)
        cached_dir = self.client.fetch_artifact_dir("agent", artifact_name, owner=owner)
        artifact_json = cached_dir / "artifact.json"
        if not artifact_json.exists():
            raise FetchError(f"Agent '{owner}/{artifact_name}' has no artifact.json")
        artifact = load_artifact_json(artifact_json)
        artifact.source_dir = cached_dir

        if artifact.agent is None:
            raise FetchError(f"Artifact '{name}' is not an agent (missing agent config)")

        all_files: list[Path] = []
        agent_config = artifact.agent
        warnings: list[str] = []

        # Check Claude Code is available
        claude_dir = project_dir / ".claude"
        if not claude_dir.exists():
            warnings.append("No .claude/ directory found. Creating one. Agents currently only work with Claude Code.")

        # Install agent definition
        agent_name = f"ag2-{artifact.name}"
        agent_dest = claude_dir / "agents" / f"{agent_name}.md"
        agent_source = cached_dir / agent_config.source
        if agent_source.exists():
            agent_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(agent_source, agent_dest)
            all_files.append(agent_dest)

        # Install bundled MCP servers
        bundled_mcp = agent_config.mcp_servers.get("bundled", [])
        if bundled_mcp:
            mcp_dest_base = claude_dir / "agents" / agent_name / "mcp"
            for mcp_path_str in bundled_mcp:
                mcp_source = cached_dir / mcp_path_str
                if mcp_source.is_dir():
                    mcp_name = mcp_source.name
                    mcp_dest = mcp_dest_base / mcp_name
                    copied = copy_tree(mcp_source, mcp_dest)
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
            warnings=warnings,
        )
