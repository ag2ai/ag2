# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from autogen.beta.tools.sandbox import LocalSandbox, ShellAdapter

from .base import ShellEnvironment

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


class LocalShellEnvironment(ShellEnvironment):
    """Deprecated façade — use :class:`ShellAdapter` directly.

    Retained as a backwards-compatible alias that builds a
    :class:`ShellAdapter` over a :class:`LocalSandbox`. All filtering
    (``allowed`` / ``blocked`` / ``ignore`` / ``readonly``) and the
    sync ``run`` API live on the adapter; this class only translates
    the v1 keyword set.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        cleanup: bool | None = None,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        env: dict[str, str] | None = None,
        timeout: float = 60,
        max_output: int = 100_000,
    ) -> None:
        self._sandbox = LocalSandbox(
            path=path,
            cleanup=cleanup,
            timeout=timeout,
            max_output=max_output,
        )
        self._adapter = ShellAdapter(
            self._sandbox,
            allowed=allowed,
            blocked=blocked,
            ignore=ignore,
            readonly=readonly,
            env=env,
            timeout=timeout,
        )

    @classmethod
    def ensure_env(cls, env: "ShellEnvironment | str | os.PathLike[str]") -> "ShellEnvironment":
        if isinstance(env, ShellEnvironment):
            return env
        return cls(env)

    @property
    def workdir(self) -> Path:
        """Host-side working directory (kept as :class:`Path` for v1 callers)."""
        return self._sandbox.host_workdir

    def run(self, command: str, *, context: "ConversationContext | None" = None) -> str:
        return self._adapter.run_sync(command, context=context)

    def __init_subclass__(cls, **kwargs: object) -> None:  # noqa: D401
        warnings.warn(
            "LocalShellEnvironment is deprecated; use ShellAdapter(LocalSandbox(...)) directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
