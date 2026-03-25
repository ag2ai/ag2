"""Lockfile — tracks installed artifacts and their versions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOCKFILE_NAME = ".ag2-artifacts.lock"


@dataclass
class InstalledArtifact:
    ref: str  # e.g. "skills/ag2"
    version: str
    installed_at: str  # ISO 8601
    targets: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)  # relative paths


class Lockfile:
    """Manages the .ag2-artifacts.lock file."""

    def __init__(self, project_dir: Path):
        self.path = project_dir / LOCKFILE_NAME
        self.installed: dict[str, InstalledArtifact] = {}
        self.load()

    def load(self) -> None:
        """Load lockfile from disk."""
        if not self.path.exists():
            self.installed = {}
            return
        raw = json.loads(self.path.read_text())
        self.installed = {}
        for ref, data in raw.get("installed", {}).items():
            self.installed[ref] = InstalledArtifact(
                ref=data["ref"],
                version=data["version"],
                installed_at=data["installed_at"],
                targets=data.get("targets", []),
                files=data.get("files", []),
            )

    def save(self) -> None:
        """Write lockfile to disk."""
        data = {
            "installed": {ref: asdict(info) for ref, info in self.installed.items()},
        }
        self.path.write_text(json.dumps(data, indent=2) + "\n")

    def record_install(
        self,
        ref: str,
        version: str,
        targets: list[str],
        files: list[Path],
    ) -> None:
        """Record that an artifact was installed."""
        project_dir = self.path.parent
        relative_files = []
        for f in files:
            try:
                relative_files.append(str(f.relative_to(project_dir)))
            except ValueError:
                # Skip files outside the project directory to prevent
                # accidental deletion of external files during uninstall.
                continue

        self.installed[ref] = InstalledArtifact(
            ref=ref,
            version=version,
            installed_at=datetime.now(timezone.utc).isoformat(),
            targets=targets,
            files=relative_files,
        )
        self.save()

    def is_installed(self, ref: str, version: str | None = None) -> bool:
        """Check if an artifact is installed (optionally at a specific version)."""
        info = self.installed.get(ref)
        if info is None:
            return False
        return version is None or info.version == version

    def get_installed(self, ref: str) -> InstalledArtifact | None:
        return self.installed.get(ref)

    def record_uninstall(self, ref: str) -> InstalledArtifact | None:
        """Remove an artifact from the lockfile, returning its record for file cleanup."""
        info = self.installed.pop(ref, None)
        if info is not None:
            self.save()
        return info

    def list_installed(self) -> list[InstalledArtifact]:
        return list(self.installed.values())
