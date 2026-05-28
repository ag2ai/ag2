# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from autogen.beta.tools.sandbox import Sandbox, SandboxFactory, ShellAdapter

from .base import ShellEnvironment

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


class SandboxShellEnvironment(ShellEnvironment):
    """Deprecated alias for :class:`ShellAdapter` — kept so v1 callers
    that do ``SandboxShellEnvironment(sandbox)`` keep working.
    """

    def __init__(
        self,
        sandbox: "Sandbox | SandboxFactory",
        *,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        self._adapter = ShellAdapter(
            sandbox,
            allowed=allowed,
            blocked=blocked,
            ignore=ignore,
            readonly=readonly,
            env=env,
            timeout=timeout,
        )
        self._sandbox = sandbox

    @property
    def workdir(self) -> Path | PurePosixPath:  # type: ignore[override]
        return self._adapter.workdir

    def run(self, command: str, *, context: "ConversationContext | None" = None) -> str:
        return self._adapter.run_sync(command, context=context)
