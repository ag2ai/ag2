"""Dataset installer — installs inline data and downloads remote files."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..artifact import InstallResult, load_artifact_json, parse_artifact_id
from ..client import ArtifactClient, FetchError
from ..lockfile import Lockfile
from ..resolver import DependencyResolver
from ..targets import Target
from .skills import SkillsInstaller, load_skills_from_artifact


class DatasetInstaller:
    """Installs dataset artifacts with inline data and optional remote downloads."""

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
        full: bool = False,
    ) -> InstallResult:
        """Install a dataset artifact. Name can be 'owner/name' or just 'name'."""
        owner, artifact_name = parse_artifact_id(name)
        cached_dir = self.client.fetch_artifact_dir("dataset", artifact_name, owner=owner)
        artifact_json = cached_dir / "artifact.json"
        if not artifact_json.exists():
            raise FetchError(f"Dataset '{owner}/{artifact_name}' has no artifact.json")
        artifact = load_artifact_json(artifact_json)
        artifact.source_dir = cached_dir

        if artifact.dataset is None:
            raise FetchError(f"Artifact '{name}' is not a dataset (missing dataset config)")

        all_files: list[Path] = []
        dataset_config = artifact.dataset
        warnings: list[str] = []

        # Copy inline data
        dest = project_dir / "data" / artifact.name
        if dataset_config.inline:
            inline_dir = cached_dir / dataset_config.inline
            if inline_dir.is_dir():
                copied = _copy_data(inline_dir, dest)
                all_files.extend(copied)

        # Download remote files
        if full and dataset_config.remote:
            from ...ui import console

            console.print("\n  Downloading remote data files...\n")
            for remote in dataset_config.remote:
                try:
                    size_str = f" ({remote.size})" if remote.size else ""
                    console.print(f"    Downloading {remote.name}{size_str}...")
                    dest_file = dest / remote.name
                    self.client.fetch_file(
                        url=remote.url,
                        dest=dest_file,
                        sha256=remote.sha256 or None,
                    )
                    all_files.append(dest_file)
                    console.print(f"    [success]\u2713[/success] {remote.name}")
                except FetchError as e:
                    warnings.append(f"Failed to download {remote.name}: {e}")
                    console.print(f"    [warning]![/warning] {remote.name}: {e}")
        elif dataset_config.remote and not full:
            remote_names = ", ".join(r.name for r in dataset_config.remote)
            total_size = ", ".join(f"{r.size}" for r in dataset_config.remote if r.size)
            warnings.append(
                f"Remote files not downloaded: {remote_names}"
                + (f" (total: {total_size})" if total_size else "")
                + ". Use --full to download."
            )

        # Write schema info if present
        if dataset_config.schema:
            import json

            schema_file = dest / "schema.json"
            schema_file.parent.mkdir(parents=True, exist_ok=True)
            schema_file.write_text(json.dumps(dataset_config.schema, indent=2) + "\n")
            all_files.append(schema_file)

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


def _copy_data(source: Path, dest: Path) -> list[Path]:
    """Copy data files to destination."""
    dest.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for src_file in sorted(source.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(source)
        dst_file = dest / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        created.append(dst_file)
    return created
