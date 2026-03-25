"""Skills installer — installs skill packs from bundled content or remote resource hub."""

from __future__ import annotations

from pathlib import Path

from ..artifact import Artifact, InstallResult, load_artifact, parse_artifact_id
from ..client import ArtifactClient, FetchError
from ..lockfile import Lockfile
from ..registry import ContentItem, load_pack, parse_frontmatter
from ..resolver import DependencyResolver
from ..targets import Target


def load_skills_from_artifact(artifact: Artifact) -> list[ContentItem]:
    """Load ContentItems from an artifact's skill directories.

    Supports both formats:
    - Flat: rules/imports.md (current bundled format)
    - Directory: rules/imports/SKILL.md (Agent Skills standard)
    """
    if artifact.source_dir is None:
        return []

    # Determine the skills root
    skills_root = artifact.source_dir
    if artifact.skills_config and artifact.skills_config.dir != ".":
        skills_root = artifact.source_dir / artifact.skills_config.dir

    # If this is a "skills" type artifact, the categories are at the source_dir level
    if artifact.type == "skills":
        skills_root = artifact.source_dir

    items: list[ContentItem] = []
    for category, subdir in [
        ("rule", "rules"),
        ("skill", "skills"),
        ("agent", "agents"),
        ("command", "commands"),
    ]:
        cat_dir = skills_root / subdir
        if not cat_dir.is_dir():
            continue

        for entry in sorted(cat_dir.iterdir()):
            # Directory format: entry/SKILL.md (Agent Skills standard)
            if entry.is_dir():
                skill_md = entry / "SKILL.md"
                if not skill_md.exists():
                    continue
                text = skill_md.read_text()
                fm, body = parse_frontmatter(text)
                items.append(
                    ContentItem(
                        name=fm.get("name", entry.name),
                        description=fm.get("description", ""),
                        category=category,
                        frontmatter=fm,
                        body=body,
                    )
                )
            # Flat format: entry.md (current bundled format)
            elif entry.is_file() and entry.suffix == ".md":
                text = entry.read_text()
                fm, body = parse_frontmatter(text)
                items.append(
                    ContentItem(
                        name=fm.get("name", entry.stem),
                        description=fm.get("description", ""),
                        category=category,
                        frontmatter=fm,
                        body=body,
                    )
                )

    return items


class SkillsInstaller:
    """Installs skill packs to IDE targets."""

    def __init__(self, client: ArtifactClient, lockfile: Lockfile, resolver: DependencyResolver):
        self.client = client
        self.lockfile = lockfile
        self.resolver = resolver

    def install(
        self,
        names: list[str],
        targets: list[Target],
        project_dir: Path,
        name_filter: str | None = None,
    ) -> list[InstallResult]:
        """Install one or more skill packs.

        Names can be 'owner/name' or just 'name' (defaults to ag2ai/).
        For 'ag2' or 'skills': uses bundled content as primary.
        """
        results = []
        for pack_id in names:
            result = self._install_one(pack_id, targets, project_dir, name_filter)
            results.append(result)
        return results

    def _install_one(
        self,
        artifact_id: str,
        targets: list[Target],
        project_dir: Path,
        name_filter: str | None,
    ) -> InstallResult:
        """Install a single skill pack."""
        owner, name = parse_artifact_id(artifact_id)
        artifact, items = self._load_skills(owner, name)

        if name_filter:
            items = [i for i in items if i.name == name_filter]

        all_files: list[Path] = []
        target_names: list[str] = []
        for tgt in targets:
            paths = tgt.install(project_dir, items)
            all_files.extend(paths)
            target_names.append(tgt.name)

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
        )

    def _load_skills(self, owner: str, name: str) -> tuple[Artifact, list[ContentItem]]:
        """Load skills from bundled content or remote."""
        # Try bundled pack first (backward compat for ag2ai/ag2)
        if owner == "ag2ai" and name in ("ag2", "skills"):
            pack = load_pack("skills")
            if pack is not None:
                artifact = Artifact(
                    name="ag2",
                    type="skills",
                    owner="ag2ai",
                    display_name=pack.display_name,
                    description=pack.description,
                    version=pack.version,
                )
                return artifact, pack.items

        # Try remote
        try:
            cached_dir = self.client.fetch_artifact_dir("skills", name, owner=owner)
            artifact = load_artifact(cached_dir)
            if artifact is not None:
                artifact.owner = owner
                items = load_skills_from_artifact(artifact)
                return artifact, items
        except FetchError:
            pass

        # Fallback: try loading from bundled by exact name
        if owner == "ag2ai":
            pack = load_pack(name)
            if pack is not None:
                artifact = Artifact(
                    name=name,
                    type="skills",
                    owner="ag2ai",
                    display_name=pack.display_name,
                    description=pack.description,
                    version=pack.version,
                )
                return artifact, pack.items

        raise FetchError(f"Skill pack '{owner}/{name}' not found (tried bundled and remote)")
