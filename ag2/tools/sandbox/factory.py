# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .base import Sandbox

if TYPE_CHECKING:
    from ag2.context import ConversationContext


@runtime_checkable
class SandboxFactory(Protocol):
    """Per-call producer of a :class:`Sandbox`.

    A factory is the only place :class:`~ag2.annotations.Variable`
    parameters get resolved — backends themselves receive concrete values
    only. This isolates Variable / Context concerns from the
    execution surface.

    Implementations return an async context manager so the underlying
    backend's lifecycle (container start / stop, remote sandbox
    creation / deletion) is explicit on every call.
    """

    def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AbstractAsyncContextManager[Sandbox]:
        """Open a sandbox bound to ``context``.

        Variables registered on the factory (image, env_vars, credentials,
        …) are resolved against ``context.variables`` here. Backends that
        do not need Variables can ignore ``context``.
        """
        ...


class SingletonFactory:
    """Wrap a single :class:`Sandbox` instance as a :class:`SandboxFactory`.

    Every :meth:`open` call yields the same sandbox. The underlying
    instance's lifecycle is **not** driven by this factory — callers own
    the sandbox and close it themselves. Useful for
    :class:`~ag2.tools.sandbox.LocalSandbox`, where the workdir
    is fixed for the life of the process and there is nothing to resolve
    per-call.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    @property
    def sandbox(self) -> Sandbox:
        """The wrapped sandbox instance."""
        return self._sandbox

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncIterator[Sandbox]:
        del context
        yield self._sandbox


__all__ = ("SandboxFactory", "SingletonFactory")
