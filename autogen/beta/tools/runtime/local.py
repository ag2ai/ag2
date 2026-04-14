# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from autogen.beta.tools.local_skills.loader import SkillLoader
from autogen.beta.tools.shell.environment.base import ShellEnvironment
from autogen.beta.tools.shell.environment.local import LocalShellEnvironment
from autogen.beta.tools.skill_types import SkillMetadata


@dataclass
class LocalRuntime:
    """Local filesystem storage and subprocess execution.

    Args:
        dir:         Directory where skills are installed.
                     ``None`` → ``.agents/skills`` (default).
        cleanup:     If ``True``, the install directory is deleted at process exit.
        timeout:     Per-command timeout in seconds. Defaults to 60.
        max_output:  Maximum characters returned from a script run. Defaults to 100,000.
        blocked:     Command prefixes that are not allowed to run. Empty list → nothing blocked.
        extra_paths: Additional read-only directories to scan for skills.
                     Installed skills always go to *dir*; these paths are only
                     used for discovery.

    Example::

        # Default — installs to .agents/skills
        LocalRuntime()

        # Custom install directory
        LocalRuntime("./my-skills")

        # Full control
        LocalRuntime("./my-skills", timeout=30, cleanup=True, blocked=["rm -rf"])

        # Extra read-only search paths
        LocalRuntime("./my-skills", extra_paths=["./shared-skills"])
    """

    dir: str | Path | None = None
    cleanup: bool = False
    timeout: float = 60
    max_output: int = 100_000
    blocked: list[str] = field(default_factory=list)
    extra_paths: Sequence[str | Path] | None = None

    def __post_init__(self) -> None:
        self._install_dir = Path(self.dir) if self.dir is not None else Path(".agents/skills")
        self._extra: list[Path] = [Path(p) for p in self.extra_paths] if self.extra_paths else []
        self._loader = SkillLoader(self._install_dir, *self._extra)
        if self.cleanup:
            atexit.register(shutil.rmtree, str(self._install_dir), True)

    @property
    def install_dir(self) -> Path:
        """Resolved install directory."""
        return self._install_dir

    @property
    def lock_dir(self) -> Path:
        return self._install_dir

    def discover(self) -> list[SkillMetadata]:
        return self._loader.discover()

    def load(self, name: str) -> str:
        return self._loader.load(name)

    def get_path(self, name: str) -> Path:
        return self._loader.get_path(name)

    def invalidate(self) -> None:
        self._loader.invalidate()

    def ensure_storage(self) -> None:
        self._install_dir.mkdir(parents=True, exist_ok=True)

    def install(self, source: Path, name: str) -> None:
        dest = self._install_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

    def remove(self, name: str) -> None:
        target = (self._install_dir / name).resolve()
        if not target.is_relative_to(self._install_dir.resolve()):
            raise ValueError(f"Cannot remove '{name}': path traversal detected")
        if not target.exists():
            raise FileNotFoundError(f"Cannot remove '{name}': skill not found in {self._install_dir}")
        shutil.rmtree(target)

    def shell(self, scripts_dir: Path) -> ShellEnvironment:

        return LocalShellEnvironment(
            path=scripts_dir,
            cleanup=False,
            timeout=self.timeout,
            max_output=self.max_output,
            blocked=self.blocked or None,
        )
