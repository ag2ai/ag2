# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import PurePosixPath
from typing import Any

from sprites import AsyncSprite

from ag2.tools.sandbox.factory import SingletonFactory

from .sandbox import SpritesSandbox


class SpritesEnvironment(SingletonFactory):
    """Shared Sprite backend for SandboxCodeTool and SandboxShellTool.

    The caller owns the async Sprite and SDK client. Every open returns the
    same backend; closing this environment leaves both resources untouched.
    Scope one environment to the users or agents intended to share its files.

    Args:
        sprite: An existing async Sprite handle, owned by the caller.
        workdir: Existing absolute POSIX directory inside the Sprite.
        timeout: Default per-command deadline in seconds.
        max_output: Maximum characters of combined command output to return.
        env_vars: Environment variables merged into each command.
    """

    def __init__(
        self,
        sprite: AsyncSprite,
        *,
        workdir: str | os.PathLike[str] = "/home/sprite",
        timeout: float = 60,
        max_output: int = 100_000,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            SpritesSandbox(
                sprite,
                workdir=workdir,
                timeout=timeout,
                max_output=max_output,
                env_vars=env_vars,
            )
        )

    @property
    def workdir(self) -> PurePosixPath:
        """Working directory exposed to the agent's tools."""
        return self.sandbox.workdir

    async def aclose(self) -> None:
        """Leave the caller-owned Sprite and client open."""
        await self.sandbox.aclose()

    async def __aenter__(self) -> "SpritesEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict[int, Any]) -> "SpritesEnvironment":
        return self
