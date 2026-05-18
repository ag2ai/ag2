# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from autogen.beta.tools.code.environment.base import CodeLanguage

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


@dataclass(slots=True)
class ExecResult:
    """Outcome of a single :meth:`Sandbox.exec` call.

    ``output`` carries the combined ``stdout + stderr`` (already trimmed
    by the backend, optionally truncated to ``max_output`` characters).
    ``exit_code`` follows POSIX conventions (``0`` = success, ``124`` =
    timeout, ``127`` = command not found).
    """

    output: str
    exit_code: int


@runtime_checkable
class Sandbox(Protocol):
    """Low-level execution backend.

    A ``Sandbox`` runs arbitrary ``argv`` lists in a ``workdir`` and returns
    combined output with an exit code. It is the shared primitive on top of
    which the higher-level adapters are built:

    - :class:`~autogen.beta.tools.shell.ShellEnvironment` — a sync
      adapter for arbitrary shell commands. Each ``run(command)`` call
      becomes ``sandbox.exec(["sh", "-c", command], shell=True)``.
    - :class:`~autogen.beta.tools.code.CodeEnvironment` — an async
      adapter for typed code snippets. Each ``run(code, language)`` call
      becomes ``sandbox.exec([interpreter, "-c", code])``.

    Implementations target local subprocesses, Docker containers, Daytona
    sandboxes, SSH, or anything else.  Adding a new backend is one
    ``Sandbox`` class — both shell and code semantics come for free via
    the adapters.
    """

    @property
    def workdir(self) -> Path:
        """Working directory in which ``argv`` is executed.

        Local backends return a host path.  Containerised backends return
        the path *as visible inside the container* (e.g. ``/workspace``).
        """
        ...

    @property
    def supported_languages(self) -> tuple[CodeLanguage, ...]:
        """Code interpreters this sandbox declares as available.

        Consumed by :class:`~autogen.beta.tools.code.CodeEnvironment` to
        validate ``run(code, language)`` requests and surface the list in
        the tool description shown to the model.  Backends with no code
        support return ``()``.
        """
        ...

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        shell: bool = False,
        context: "ConversationContext | None" = None,
    ) -> ExecResult:
        """Execute ``argv`` and return its output.

        Args:
            argv: Command and arguments.  When ``shell=False`` (default)
                  this is a normal argv list (no shell parsing on our side).
                  When ``shell=True`` the elements are joined with spaces
                  and passed to the system shell — pipelines, ``&&``,
                  redirects and other shell features are then honoured.
            env: Extra environment variables merged into the process
                 environment.  ``None`` inherits the parent environment as-is.
            timeout: Per-call timeout in seconds.  ``None`` uses the
                     backend's default.  On timeout, the implementation
                     must terminate the process and return
                     ``exit_code=124``.
            shell: If ``True``, run ``argv`` through the system shell.
            context: Active conversation context. Forwarded by the
                     :class:`~autogen.beta.tools.shell.ShellEnvironment` /
                     :class:`~autogen.beta.tools.code.CodeEnvironment`
                     adapters so backends can resolve
                     :class:`~autogen.beta.annotations.Variable` markers
                     supplied to their constructors (e.g. per-tenant
                     credentials).  Backends with no Variable-capable
                     parameters can ignore it.
        """
        ...

    async def aclose(self) -> None:
        """Release backend resources.

        Local backends typically have nothing to release; containerised
        backends stop and remove their container.  Safe to call multiple times.
        """
        ...
