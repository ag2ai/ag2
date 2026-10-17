# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Configure reusable E2B cloud sandboxes for AG2 tools."""

import threading
from collections.abc import AsyncGenerator, Hashable
from contextlib import asynccontextmanager, suppress
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from ag2.annotations import Variable
from ag2.tools.builtin._resolve import resolve_variable

from .sandbox import E2BSandbox

if TYPE_CHECKING:
    from ag2.context import ConversationContext


class E2BEnvironment:
    """:class:`~ag2.tools.sandbox.SandboxFactory` for :class:`E2BSandbox`.

    This is the backend object you hand to a tool::

        env = E2BEnvironment()
        shell = SandboxShellTool(env)
        code = SandboxCodeTool(env, languages=("python", "bash", "javascript"))

    ``api_key``, ``template`` and ``env_vars`` accept a
    :class:`~ag2.annotations.Variable` for deferred resolution from
    ``context.variables``, which is useful for per-tenant credentials.
    ``E2B_API_KEY`` is read by the SDK when ``api_key`` is omitted.

    Sandboxes are cached by their resolved parameters: opens with the same
    values reuse one sandbox, whichever conversation makes them, so files
    survive across tool calls, and distinct values get distinct sandboxes.
    Cached sandboxes live until :meth:`aclose`, an atexit fallback, or their
    ``sandbox_timeout``.
    """

    def __init__(
        self,
        *,
        api_key: "str | Variable | None" = None,  # pragma: allowlist secret
        template: "str | Variable | None" = None,
        env_vars: "dict[str, str] | Variable | None" = None,
        timeout: float = 60,
        sandbox_timeout: int | None = None,
        workdir: str = "/home/user",
        create_options: dict[str, Any] | None = None,
    ) -> None:
        """Configure the sandboxes this factory hands out.

        Args:
            api_key: E2B API key. Falls back to ``E2B_API_KEY``.
            template: E2B template name or ID. ``None`` uses the SDK default,
                which ships ``python`` and ``node``.
            env_vars: Environment variables set for every command.
            timeout: Default per-command timeout in seconds. Must be > 0.
            sandbox_timeout: Server-side idle lifetime in seconds; each
                command extends it, to at least 30 seconds past that
                command's timeout. ``None`` uses the SDK default. It is the
                backstop that reclaims the sandbox if client cleanup never
                runs. Must be > 0.
            workdir: Sandbox-side directory commands and relative paths
                resolve against. Created on startup if it does not exist.
            create_options: Extra keyword arguments forwarded to
                ``AsyncSandbox.create`` as-is (for example ``domain``,
                ``metadata``, ``secure``, ``allow_internet_access``,
                ``network`` or ``lifecycle``).

        Raises:
            ValueError: If ``timeout`` or ``sandbox_timeout`` is not positive,
                or ``create_options`` contains ``api_key``.
        """
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than 0 seconds.")
        if sandbox_timeout is not None and sandbox_timeout <= 0:
            raise ValueError("`sandbox_timeout` must be greater than 0 seconds.")
        if create_options and "api_key" in create_options:
            raise ValueError("Pass `api_key` as an E2BEnvironment argument, not through `create_options`.")

        self._api_key = api_key
        self._template = template
        self._env_vars = env_vars
        self._timeout = timeout
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._create_options = dict(create_options or {})

        self._cache: dict[Hashable, E2BSandbox] = {}
        # Sandboxes whose kill failed, retried by the next `aclose`.
        self._unkilled: list[E2BSandbox] = []
        self._cache_lock = threading.Lock()

    @property
    def workdir(self) -> PurePosixPath:
        """Sandbox-side working directory commands and file writes resolve against."""
        return PurePosixPath(self._workdir)

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncGenerator[E2BSandbox]:
        """Yield the sandbox matching the resolved parameters.

        The factory keeps ownership: leaving this scope does not kill the
        sandbox, only :meth:`aclose` does.

        Args:
            context: Conversation context used to resolve ``Variable``
                parameters. ``None`` requires every parameter to be concrete.

        Yields:
            The ready sandbox for this parameter set.

        Raises:
            RuntimeError: If a parameter is a ``Variable`` but no ``context``
                was supplied to resolve it.
            KeyError: If a ``Variable`` names a key absent from the context.
        """
        api_key = resolve_variable(self._api_key, context, param_name="api_key") if context else self._api_key
        template = resolve_variable(self._template, context, param_name="template") if context else self._template
        env_vars = (
            resolve_variable(self._env_vars, context, param_name="env_vars") if context else self._env_vars
        ) or {}

        if any(isinstance(value, Variable) for value in (api_key, template, env_vars)):
            raise RuntimeError(
                "E2B parameters given as Variable but no Context is available to resolve them. "
                "Variables are only resolvable when a sandbox tool is invoked through an Agent."
            )
        assert api_key is None or isinstance(api_key, str)
        assert template is None or isinstance(template, str)
        assert isinstance(env_vars, dict)

        key: Hashable = (
            api_key,
            template,
            tuple(sorted(env_vars.items())),
            repr(sorted(self._create_options.items())),
            self._timeout,
            self._sandbox_timeout,
            self._workdir,
        )

        with self._cache_lock:
            sandbox = self._cache.get(key)
            if sandbox is None or sandbox.closed:
                create_options = dict(self._create_options)
                if api_key is not None:
                    create_options["api_key"] = api_key
                sandbox = E2BSandbox(
                    template=template,
                    env_vars=env_vars or None,
                    timeout=self._timeout,
                    sandbox_timeout=self._sandbox_timeout,
                    workdir=self._workdir,
                    create_options=create_options,
                )
                self._cache[key] = sandbox

        try:
            await sandbox.__aenter__()
        except BaseException:
            with suppress(BaseException):
                await sandbox.aclose()
            # The closed cache entry is replaced on the next open; if setup and
            # kill both failed, keep the sandbox for the next `aclose` to retry.
            if sandbox.sandbox_id is not None:
                with self._cache_lock:
                    self._unkilled.append(sandbox)
            raise
        # The factory owns the lifecycle so cached state survives this scope.
        yield sandbox

    async def aclose(self) -> None:
        """Kill every cached sandbox. Safe to call multiple times; a call
        retries the sandboxes a previous one failed to kill."""
        with self._cache_lock:
            sandboxes = [*self._cache.values(), *self._unkilled]
            self._cache.clear()
            self._unkilled = []
        try:
            for sandbox in sandboxes:
                await sandbox.aclose()
        finally:
            # Also on cancellation: whatever is not killed yet stays for the next call.
            with self._cache_lock:
                self._unkilled.extend(sandbox for sandbox in sandboxes if sandbox.sandbox_id is not None)

    async def __aenter__(self) -> "E2BEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict[int, Any]) -> "E2BEnvironment":
        # Shared resource handle: a copy is the SAME factory, so Agent.add_tool
        # can deepcopy a tool backed by this environment.
        return self
