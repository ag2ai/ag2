"""Template installer — scaffolds projects from artifact templates."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import typer

from ..artifact import Artifact, InstallResult, TemplateConfig, load_artifact_json, parse_artifact_id
from ..client import ArtifactClient, FetchError
from ..lockfile import Lockfile
from ..resolver import DependencyResolver
from ..targets import Target
from .skills import SkillsInstaller, load_skills_from_artifact


class TemplateInstaller:
    """Installs project templates with scaffolding and skills."""

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
        variables: dict[str, str] | None = None,
        preview: bool = False,
    ) -> InstallResult:
        """Install a template artifact. Name can be 'owner/name' or just 'name'."""
        owner, artifact_name = parse_artifact_id(name)
        cached_dir = self.client.fetch_artifact_dir("template", artifact_name, owner=owner)
        artifact_json = cached_dir / "artifact.json"
        if not artifact_json.exists():
            raise FetchError(f"Template '{owner}/{artifact_name}' has no artifact.json")
        artifact = load_artifact_json(artifact_json)
        artifact.source_dir = cached_dir

        if artifact.template is None:
            raise FetchError(f"Artifact '{name}' is not a template (missing template config)")

        config = artifact.template

        # Resolve variables
        final_vars = self._resolve_variables(config, variables)

        # Preview mode
        if preview:
            return self._preview(artifact, config, cached_dir, final_vars)

        all_files: list[Path] = []

        # Copy scaffold
        scaffold_files = self._copy_scaffold(
            source=cached_dir / config.scaffold,
            dest=project_dir,
            variables=final_vars,
            ignore=config.ignore,
        )
        all_files.extend(scaffold_files)

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

        # Post-install commands
        if config.post_install:
            self._run_post_install(config.post_install, project_dir)

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

    def _resolve_variables(
        self,
        config: TemplateConfig,
        provided: dict[str, str] | None,
    ) -> dict[str, str]:
        """Merge provided variables with defaults, prompting for missing ones."""
        provided = provided or {}
        result: dict[str, str] = {}

        for key, spec in config.variables.items():
            if key in provided:
                result[key] = provided[key]
            elif spec.default:
                result[key] = spec.default
            else:
                # Interactive prompt
                try:
                    import questionary

                    if spec.choices:
                        answer = questionary.select(spec.prompt, choices=spec.choices).ask()
                    else:
                        answer = questionary.text(spec.prompt, default=spec.default).ask()
                    result[key] = answer or spec.default
                except (ImportError, EOFError):
                    result[key] = spec.default

        # Apply transforms
        for key, spec in config.variables.items():
            if spec.transform == "slug" and key in result:
                result[key] = _slugify(result[key])

        return result

    def _copy_scaffold(
        self,
        source: Path,
        dest: Path,
        variables: dict[str, str],
        ignore: list[str],
    ) -> list[Path]:
        """Copy scaffold files to destination, substituting variables in .tmpl files."""
        if not source.is_dir():
            return []

        created: list[Path] = []
        for src_path in sorted(source.rglob("*")):
            if not src_path.is_file():
                continue

            rel = src_path.relative_to(source)

            # Skip ignored patterns
            if any(_matches_ignore(str(rel), pat) for pat in ignore):
                continue

            # Determine destination path
            dest_rel = str(rel)
            is_template = dest_rel.endswith(".tmpl")
            if is_template:
                dest_rel = dest_rel[:-5]  # Strip .tmpl

            # Apply variable substitution to path
            dest_rel = _substitute(dest_rel, variables)
            dest_path = (dest / dest_rel).resolve()
            # Prevent path traversal via malicious variable values (e.g. "../")
            if not dest_path.is_relative_to(dest.resolve()):
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if is_template:
                # Text file with variable substitution
                content = src_path.read_text()
                content = _substitute(content, variables)
                dest_path.write_text(content)
            else:
                # Binary copy
                shutil.copy2(src_path, dest_path)

            created.append(dest_path)

        return created

    def _preview(
        self,
        artifact: Artifact,
        config: TemplateConfig,
        cached_dir: Path,
        variables: dict[str, str],
    ) -> InstallResult:
        """Preview what would be installed without writing."""
        scaffold_dir = cached_dir / config.scaffold
        preview_files: list[Path] = []
        if scaffold_dir.is_dir():
            for src_path in sorted(scaffold_dir.rglob("*")):
                if src_path.is_file():
                    rel = str(src_path.relative_to(scaffold_dir))
                    if rel.endswith(".tmpl"):
                        rel = rel[:-5]
                    rel = _substitute(rel, variables)
                    preview_files.append(Path(rel))

        return InstallResult(
            artifact=artifact,
            files_created=preview_files,
            warnings=["Preview mode — no files written"],
        )

    def _run_post_install(self, commands: list[str], cwd: Path) -> None:
        """Run post-install shell commands after user confirmation."""
        from ...ui import console

        console.print("\n  [warning]This template wants to run post-install commands:[/warning]")
        for cmd in commands:
            console.print(f"    [command]{cmd}[/command]")

        try:
            answer = typer.confirm("\n  Allow these commands to run?", default=False)
        except (EOFError, KeyboardInterrupt):
            answer = False

        if not answer:
            console.print("  [dim]Skipped post-install commands.[/dim]")
            return

        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    console.print(f"  [success]\u2713[/success] {cmd}")
                else:
                    console.print(f"  [warning]![/warning] {cmd} (exit {result.returncode})")
                    if result.stderr:
                        console.print(f"    {result.stderr.strip()}", style="dim")
            except subprocess.TimeoutExpired:
                console.print(f"  [warning]![/warning] {cmd} (timed out)")
            except FileNotFoundError:
                console.print(f"  [warning]![/warning] {cmd} (command not found)")


def _substitute(text: str, variables: dict[str, str]) -> str:
    """Replace {{ var }} placeholders in text."""
    for key, value in variables.items():
        text = text.replace(f"{{{{ {key} }}}}", value)
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def _slugify(text: str) -> str:
    """Convert text to a URL/filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _matches_ignore(path: str, pattern: str) -> bool:
    """Check if a path matches an ignore pattern (simple glob)."""
    from fnmatch import fnmatch

    return fnmatch(path, pattern) or fnmatch(path.split("/")[-1], pattern)
