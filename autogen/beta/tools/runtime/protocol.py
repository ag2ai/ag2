# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Protocol, runtime_checkable

from autogen.beta.tools.shell.environment.base import ShellEnvironment
from autogen.beta.tools.skill_types import SkillMetadata

__all__ = ("SkillRuntime",)


@runtime_checkable
class SkillRuntime(Protocol):
    """Unified runtime: storage, discovery, and execution of skills.

    A runtime is responsible for three concerns:

    1. **Storage** — where skills are installed (``install``, ``remove``).
    2. **Discovery** — scanning for installed skills (``discover``, ``load``, ``invalidate``).
    3. **Execution** — providing a shell environment to run scripts (``shell``).

    :class:`LocalRuntime` is the default implementation.
    """

    @property
    def cleanup(self) -> bool:
        """Delete runtime storage on process exit."""
        ...

    @property
    def lock_dir(self) -> Path:
        """Local directory where ``skills-lock.json`` is stored.

        Always a local path — the lock file is host metadata, not runtime storage.
        ``LocalRuntime`` returns ``_install_dir``.  A future ``DockerRuntime``
        would return a configurable local directory.
        """
        ...

    def discover(self) -> list[SkillMetadata]:
        """Return metadata for all installed skills."""
        ...

    def load(self, name: str) -> str:
        """Return the full ``SKILL.md`` text for *name*."""
        ...

    def get_path(self, name: str) -> Path:
        """Return the directory path of a skill by name.

        Raises:
            KeyError: if no skill with that name is found.
        """
        ...

    def invalidate(self) -> None:
        """Clear discovery cache (call after install / remove)."""
        ...

    def ensure_storage(self) -> None:
        """Ensure the storage backend is ready.

        ``LocalRuntime`` creates the install directory.  A ``DockerRuntime``
        might create a container volume.  A no-op for read-only runtimes.
        """
        ...

    def install(self, source: Path, name: str) -> None:
        """Move an extracted skill from a staging directory into runtime storage.

        Args:
            source: Local staging directory that contains the skill files.
            name:   Skill name (used as the sub-directory name in storage).
        """
        ...

    def remove(self, name: str) -> None:
        """Delete an installed skill from storage.

        Raises:
            ValueError:      If *name* would resolve outside the install directory.
            FileNotFoundError: If no skill with *name* is installed.
        """
        ...

    def shell(self, scripts_dir: Path) -> ShellEnvironment:
        """Return a :class:`~autogen.beta.tools.shell.ShellEnvironment` for *scripts_dir*.

        Args:
            scripts_dir: Absolute path to the skill's ``scripts/`` directory.
                         Callers resolve this path via the discovery loader so
                         that both install-dir and extra-path skills are handled
                         uniformly.
        """
        ...
