# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from inspect import Parameter, isawaitable, signature
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .base import Sandbox

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


@runtime_checkable
class SandboxFactory(Protocol):
    """Per-call producer of a :class:`Sandbox`.

    A factory is the only place :class:`~autogen.beta.annotations.Variable`
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
    :class:`~autogen.beta.tools.sandbox.LocalSandbox`, where the workdir
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


# A factory function may take the context (per-tenant resolution) or no
# argument at all. Both sync and async return values are accepted.
SandboxBuilder = (
    Callable[["ConversationContext | None"], "Sandbox | Awaitable[Sandbox]"]
    | Callable[[], "Sandbox | Awaitable[Sandbox]"]
)


class CallableFactory:
    """Adapt a plain callable into a :class:`SandboxFactory`.

    The callable produces a fresh :class:`Sandbox` per :meth:`open`; the
    factory owns its lifecycle and tears it down when the scope exits. The
    callable may accept the active
    :class:`~autogen.beta.context.ConversationContext` (to resolve
    per-tenant values itself) or take no argument, and may be sync or async.

    Useful for ad-hoc backends without writing a dedicated factory class::

        CallableFactory(lambda ctx: MySandbox(token=ctx.variables["tok"]))
    """

    def __init__(self, builder: SandboxBuilder) -> None:
        self._builder = builder
        # builders that declare a parameter receive the context; nullary ones
        # are called without it.
        self._wants_context = _takes_one_arg(builder)

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncIterator[Sandbox]:
        result = self._builder(context) if self._wants_context else self._builder()  # type: ignore[call-arg]
        sandbox = await result if isawaitable(result) else result
        async with sandbox:
            yield sandbox


def _takes_one_arg(builder: SandboxBuilder) -> bool:
    try:
        sig = signature(builder)
    except (TypeError, ValueError):
        return False
    return len([p for p in sig.parameters.values() if p.kind in _POSITIONAL_KINDS]) >= 1


_POSITIONAL_KINDS = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)


__all__ = ("CallableFactory", "SandboxFactory", "SingletonFactory")
