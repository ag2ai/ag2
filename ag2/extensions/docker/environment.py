# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import threading
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from ag2.annotations import Variable
from ag2.tools.builtin._resolve import resolve_variable

from .sandbox import DockerSandbox

if TYPE_CHECKING:
    from ag2.context import ConversationContext


class DockerEnvironment:
    """:class:`~ag2.tools.sandbox.SandboxFactory` for
    :class:`DockerSandbox`.

    This is the backend object you hand to a tool::

        env = DockerEnvironment(image="python:3.12-slim", network_mode="none")
        shell = SandboxShellTool(env, allowed=["git"])
        code = SandboxCodeTool(env, languages=("python",))

    All :class:`~ag2.annotations.Variable`-capable parameters
    (``image``, ``env_vars``, ``network_mode``) are resolved on
    :meth:`open` against the active
    :class:`~ag2.context.ConversationContext`. The resulting
    :class:`DockerSandbox` receives only concrete values, so the backend
    never has to know about Variables or Context.

    The factory owns the sandbox lifecycle and **caches** by resolved
    parameters: the first :meth:`open` for a given set of resolved values
    starts a container; subsequent opens with the same values reuse it, so
    state (created files, installed packages) persists across tool calls.
    Different resolved values (e.g. a per-tenant ``image=Variable(...)``)
    get distinct containers. Cached containers live until :meth:`aclose`
    (or process-exit atexit, registered per container).
    """

    def __init__(
        self,
        *,
        image: "str | Variable" = "python:3.12-slim",
        env_vars: "dict[str, str] | Variable | None" = None,
        timeout: float = 60,
        network_mode: "str | Variable" = "none",
        mem_limit: str | None = "512m",
        cpu_quota: int | None = None,
        user: str | None = None,
        auto_remove: bool = True,
        host_path: str | os.PathLike[str] | None = None,
        workdir: str = "/workspace",
    ) -> None:
        self._image = image
        self._env_vars = env_vars
        self._timeout = timeout
        self._network_mode = network_mode
        self._mem_limit = mem_limit
        self._cpu_quota = cpu_quota
        self._user = user
        self._auto_remove = auto_remove
        self._host_path = host_path
        self._workdir = workdir

        self._cache: dict[Hashable, DockerSandbox] = {}
        self._cache_lock = threading.Lock()

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncIterator[DockerSandbox]:
        image = resolve_variable(self._image, context, param_name="image") if context else self._image
        env_vars = (
            resolve_variable(self._env_vars, context, param_name="env_vars") if context else self._env_vars
        ) or {}
        network_mode = (
            resolve_variable(self._network_mode, context, param_name="network_mode") if context else self._network_mode
        )

        for value, name in ((image, "image"), (network_mode, "network_mode")):
            if isinstance(value, Variable):
                raise RuntimeError(
                    f"Docker `{name}` given as Variable but no Context available to resolve it. "
                    "Variables are only resolvable when the sandbox is driven from an Agent "
                    "(SandboxCodeTool / SandboxShellTool wrappers forward the active Context)."
                )

        assert isinstance(image, str)
        assert isinstance(network_mode, str)
        assert isinstance(env_vars, dict)

        key: Hashable = (
            image,
            network_mode,
            tuple(sorted(env_vars.items())),
            self._mem_limit,
            self._cpu_quota,
            self._user,
            self._auto_remove,
            str(self._host_path) if self._host_path is not None else None,
            self._workdir,
            self._timeout,
        )

        # Reserving the cache slot is loop-agnostic (plain threading.Lock,
        # no await) so concurrent openers across throw-away event loops agree
        # on one sandbox object. The container itself starts in __aenter__,
        # which is internally guarded and idempotent.
        with self._cache_lock:
            sandbox = self._cache.get(key)
            if sandbox is None or sandbox.closed:
                sandbox = DockerSandbox(
                    image=image,
                    env_vars=dict(env_vars),
                    timeout=self._timeout,
                    network_mode=network_mode,
                    mem_limit=self._mem_limit,
                    cpu_quota=self._cpu_quota,
                    user=self._user,
                    auto_remove=self._auto_remove,
                    host_path=self._host_path,
                    workdir=self._workdir,
                )
                self._cache[key] = sandbox

        await sandbox.__aenter__()
        # Do NOT close on scope exit — the factory owns the lifecycle so the
        # container is reused by the next open() with the same key.
        yield sandbox

    async def aclose(self) -> None:
        """Tear down every cached container. Safe to call multiple times."""
        with self._cache_lock:
            sandboxes = list(self._cache.values())
            self._cache.clear()
        for sandbox in sandboxes:
            await sandbox.aclose()

    async def __aenter__(self) -> "DockerEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict) -> "DockerEnvironment":  # type: ignore[type-arg]
        # Shared resource handle: a copy is the SAME factory. The cached
        # containers (and the threading.Lock guarding them — not deepcopy-able)
        # are reused, not duplicated. This is what lets Agent.add_tool deepcopy
        # a SandboxShellTool / SandboxCodeTool backed by this environment.
        return self
