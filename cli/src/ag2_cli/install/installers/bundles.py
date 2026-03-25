"""Bundle installer — installs curated collections of artifacts."""

from __future__ import annotations

from pathlib import Path

from ..artifact import InstallResult, load_artifact_json
from ..client import ArtifactClient, FetchError
from ..lockfile import Lockfile
from ..resolver import DependencyResolver
from ..targets import Target
from .agents import AgentInstaller
from .datasets import DatasetInstaller
from .skills import SkillsInstaller, load_skills_from_artifact
from .templates import TemplateInstaller
from .tools import ToolInstaller


class BundleInstaller:
    """Installs curated collections of artifacts."""

    def __init__(
        self,
        client: ArtifactClient,
        lockfile: Lockfile,
        resolver: DependencyResolver,
        skills_installer: SkillsInstaller,
        template_installer: TemplateInstaller,
        tool_installer: ToolInstaller,
        dataset_installer: DatasetInstaller,
        agent_installer: AgentInstaller,
    ):
        self.client = client
        self.lockfile = lockfile
        self.resolver = resolver
        self.skills = skills_installer
        self.templates = template_installer
        self.tools = tool_installer
        self.datasets = dataset_installer
        self.agents = agent_installer

    def install(
        self,
        name: str,
        project_dir: Path,
        targets: list[Target],
    ) -> InstallResult:
        """Install a bundle artifact — orchestrates installation of all referenced artifacts."""
        cached_dir = self.client.fetch_artifact_dir("bundle", name, version="latest")
        artifact_json = cached_dir / "artifact.json"
        if not artifact_json.exists():
            raise FetchError(f"Bundle '{name}' has no artifact.json")
        artifact = load_artifact_json(artifact_json)
        artifact.source_dir = cached_dir

        if artifact.bundle is None:
            raise FetchError(f"Artifact '{name}' is not a bundle (missing bundle config)")

        from ...ui import console

        bundle_config = artifact.bundle
        all_files: list[Path] = []
        deps_installed: list[str] = []
        warnings: list[str] = []

        # Prompt for optional artifacts
        selected_refs = self._select_artifacts(bundle_config.artifacts)

        # Group by type for ordered installation
        by_type: dict[str, list[str]] = {}
        for ref in selected_refs:
            parts = ref.split("/", 1)
            if len(parts) == 2:
                artifact_type, artifact_name = parts
                by_type.setdefault(artifact_type, []).append(artifact_name)

        # Install in order
        for type_key in bundle_config.install_order:
            type_dir = type_key if type_key != "templates" else "templates"
            names = by_type.get(type_dir, []) or by_type.get(type_key, [])
            for artifact_name in names:
                try:
                    console.print(f"\n  Installing {type_key}/{artifact_name}...")
                    result = self._install_by_type(type_key, artifact_name, project_dir, targets)
                    all_files.extend(result.files_created)
                    deps_installed.append(f"{type_key}/{artifact_name}")
                    console.print(f"  [success]\u2713[/success] {type_key}/{artifact_name}")
                except FetchError as e:
                    warnings.append(f"Failed to install {type_key}/{artifact_name}: {e}")
                    console.print(f"  [warning]![/warning] {type_key}/{artifact_name}: {e}")

        # Install bundle-level skills if any
        skill_items = load_skills_from_artifact(artifact)
        if skill_items:
            for tgt in targets:
                paths = tgt.install(project_dir, skill_items)
                all_files.extend(paths)

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

    def _select_artifacts(self, bundle_refs: list) -> list[str]:
        """Prompt for optional artifacts, auto-include required ones."""
        required = [r.ref for r in bundle_refs if r.required]
        optional = [r for r in bundle_refs if not r.required]

        if not optional:
            return required

        # Try interactive prompt for optional items
        try:
            import questionary

            if optional:
                choices = [questionary.Choice(title=r.ref, value=r.ref, checked=True) for r in optional]
                selected = questionary.checkbox(
                    "Select optional artifacts to include:",
                    choices=choices,
                ).ask()
                if selected is None:
                    selected = []
                return required + selected
        except (ImportError, EOFError):
            pass

        # Non-interactive: include everything
        return required + [r.ref for r in optional]

    def _install_by_type(
        self,
        type_key: str,
        name: str,
        project_dir: Path,
        targets: list[Target],
    ) -> InstallResult:
        """Dispatch installation to the appropriate installer."""
        if type_key == "skills":
            results = self.skills.install([name], targets, project_dir)
            if not results:
                raise FetchError(f"Skills pack '{name}' not found")
            return results[0]
        elif type_key in ("template", "templates"):
            return self.templates.install(name, project_dir, targets)
        elif type_key in ("tool", "tools"):
            return self.tools.install(name, project_dir, targets)
        elif type_key in ("dataset", "datasets"):
            return self.datasets.install(name, project_dir, targets)
        elif type_key in ("agent", "agents"):
            return self.agents.install(name, project_dir, targets)
        else:
            raise FetchError(f"Unknown artifact type in bundle: {type_key}")
