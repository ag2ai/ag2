# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ag2.tools.skills.skill_types import Skill

if TYPE_CHECKING:
    from ag2.context import ConversationContext


@runtime_checkable
class SkillRuntime(Protocol):
    """Unified runtime: storage, discovery, and IO (read + execute) of skills.

    A runtime is responsible for three concerns:

    1. **Storage** — where skills are installed (``install``, ``remove``).
    2. **Discovery** — the skills it owns (``skills``, ``invalidate``).
    3. **IO** — reading and executing those skills (``read``, ``read_resource``,
       ``execute``). The runtime owns *how* its content is reached, so a
       filesystem runtime reads from disk and an in-memory one from RAM. A
       :class:`~ag2.tools.skills.skill_types.Skill` is a pure descriptor
       and never performs IO itself.

    The read/execute methods raise
    :class:`~ag2.exceptions.SkillNotFoundError` when *this* runtime does
    not own the named skill — the signal the toolkit's multi-runtime chain uses
    to fall through to the next runtime.

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

    @property
    def skills(self) -> list[Skill]:
        """Return descriptors for all skills this runtime owns."""
        ...

    def read(self, name: str) -> str:
        """Return the model-ready content for skill *name* (wrapped SKILL.md).

        Raises:
            SkillNotFoundError: if this runtime does not own *name*.
        """
        ...

    async def read_resource(self, name: str, resource: str, context: "ConversationContext") -> str:
        """Return the content of a resource of skill *name*.

        Async because a runtime may produce the content by running a callable
        (``MemoryRuntime``), not just reading a file (``LocalRuntime``). *context*
        is the live conversation context; a runtime that runs a callable uses it
        for dependency injection (``Context`` / ``Variable`` / ``Inject``), while a
        filesystem runtime ignores it.

        Raises:
            SkillNotFoundError: if this runtime does not own *name*.
            FileNotFoundError:  if *name* has no such resource.
        """
        ...

    async def execute(
        self,
        name: str,
        script: str,
        context: "ConversationContext",
        args: dict[str, Any] | Sequence[str] | None = None,
    ) -> str:
        """Run a script of skill *name* and return its output.

        *args* is a CLI-style positional ``Sequence[str]`` for file-backed scripts
        (``LocalRuntime``) or a named-argument ``dict`` for in-process scripts
        (``MemoryRuntime``). A runtime rejects the form it does not support with a
        ``TypeError``. *context* is the live conversation context, used for
        dependency injection by runtimes that invoke a callable; a filesystem
        runtime ignores it.

        Raises:
            SkillNotFoundError: if this runtime does not own *name*.
            FileNotFoundError:  if *name* has no such script.
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
