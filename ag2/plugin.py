# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack
from typing import Any, TypeAlias, overload

from fast_depends import Provider
from fast_depends.library.serializer import SerializerProto
from fast_depends.pydantic import PydanticSerializer
from typing_extensions import Self

from .annotations import Context
from .assembly import AssemblyPolicy
from .events import (
    ModelRequest,
)
from .events.conditions import Condition
from .hitl import HumanHook, wrap_hitl
from .middleware.base import (
    MiddlewareFactory,
    ToolMiddleware,
)
from .observers import Observer
from .observers import observer as observer_factory
from .tools.executor import ToolExecutor
from .tools.final import FunctionParameters, FunctionTool, tool
from .tools.tool import Tool
from .types import ClassInfo
from .utils import CONTEXT_OPTION_NAME, build_model

PromptHook: TypeAlias = Callable[..., str] | Callable[..., Awaitable[str]]
PromptType: TypeAlias = str | PromptHook


class PromptObserverMixin:
    """Shared prompt/observer/tool/middleware collection for :class:`Agent`,
    :class:`LiveAgent`, and :class:`Plugin`.

    Builds ``_system_prompt`` / ``_dynamic_prompt`` from a ``prompt`` argument
    and exposes the ``prompt()`` / ``observer()`` / ``tool()`` decorators plus
    the imperative ``add_middleware()`` / ``insert_middleware()`` /
    ``add_policy()`` / ``add_observer()`` collectors. Subclasses initialise
    ``_observers`` / ``_middleware`` / ``_policies`` themselves (each takes
    matching constructor arguments) and define ``add_tool()`` to choose how a
    freshly built tool is stored — eagerly (Agent/LiveAgent) or deferred
    (Plugin).
    """

    _system_prompt: list[str]
    _dynamic_prompt: list[Callable[[ModelRequest, Context], Awaitable[str]]]
    _observers: list[Observer]
    _middleware: list[MiddlewareFactory]
    _policies: list[AssemblyPolicy]

    def _init_prompts(self, prompt: PromptType | Iterable[PromptType]) -> None:
        self._system_prompt = []
        self._dynamic_prompt = []

        if isinstance(prompt, str) or callable(prompt):
            prompt = [prompt]
        for p in prompt:
            if isinstance(p, str):
                self._system_prompt.append(p)
            else:
                self._dynamic_prompt.append(_wrap_prompt_hook(p))

    @overload
    def prompt(
        self,
        func: PromptHook,
    ) -> PromptHook: ...

    @overload
    def prompt(
        self,
        func: None = None,
    ) -> Callable[[PromptHook], PromptHook]: ...

    def prompt(
        self,
        func: PromptHook | None = None,
    ) -> PromptHook | Callable[[PromptHook], PromptHook]:
        def wrapper(f: PromptHook) -> PromptHook:
            self._dynamic_prompt.append(_wrap_prompt_hook(f))
            return f

        if func:
            return wrapper(func)
        return wrapper

    @overload
    def observer(
        self,
        condition: ClassInfo | Condition | None,
        callback: Callable[..., Any],
    ) -> Callable[..., Any]: ...

    @overload
    def observer(
        self,
        condition: ClassInfo | Condition | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def observer(
        self,
        condition: ClassInfo | Condition | None = None,
        callback: Callable[..., Any] | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            obs = observer_factory(condition, func)
            self._observers.append(obs)
            return func

        if callback is not None:
            return wrapper(callback)
        return wrapper

    def add_tool(self, t: FunctionTool) -> None:
        """Store a freshly built tool. Subclasses choose eager vs deferred."""
        raise NotImplementedError

    @overload
    def tool(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool: ...

    @overload
    def tool(
        self,
        function: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> Callable[[Callable[..., Any]], FunctionTool]: ...

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
        def make_tool(f: Callable[..., Any]) -> FunctionTool:
            t = tool(
                f,
                name=name,
                description=description,
                schema=schema,
                sync_to_thread=sync_to_thread,
                middleware=middleware,
            )
            self.add_tool(t)
            return t

        if function:
            return make_tool(function)
        return make_tool

    def add_middleware(self, m: MiddlewareFactory) -> Self:
        """Append middleware as the innermost wrapper in the chain.

        The added middleware is called last on turn entry and first on turn exit,
        executing closer to the LLM call than any middleware already registered.
        """
        self._middleware.append(m)
        return self

    def insert_middleware(self, m: MiddlewareFactory) -> Self:
        """Insert middleware as the outermost wrapper in the chain.

        The inserted middleware is called first on turn entry and last on turn exit,
        executing before all middleware already registered.
        """
        self._middleware.insert(0, m)
        return self

    def add_policy(self, policy: AssemblyPolicy) -> Self:
        """Append an assembly policy to the chain.

        Policies run in order; a newly added policy runs after existing ones.
        Construction-time ordering validation (warning on suspicious sequences)
        only runs over policies passed via ``assembly=`` — late additions skip
        the check, so callers should be confident in the ordering they introduce.
        """
        self._policies.append(policy)
        return self

    def add_observer(self, observer: Observer) -> None:
        """Register an observer (before running the agent)."""
        self._observers.append(observer)


class Plugin(PromptObserverMixin):
    def __init__(
        self,
        *,
        prompt: PromptType | Iterable[PromptType] = (),
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        observers: Iterable[Observer] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
    ) -> None:
        self._tools = list(tools)
        self._middleware = list(middleware)
        self._observers = list(observers)
        self._policies = []
        self._dependencies = dependencies or {}
        self._variables = variables or {}
        self._hitl_hook = hitl_hook

        self._init_prompts(prompt)

    def hitl_hook(self, func: HumanHook) -> HumanHook:
        if self._hitl_hook is not None:
            warnings.warn(
                "You already set HITL hook, provided value overrides it",
                category=RuntimeWarning,
                stacklevel=2,
            )
        self._hitl_hook = func
        return func

    def add_tool(self, t: FunctionTool) -> None:
        """Defer the tool; it is applied to an agent later via ``_apply_plugin``."""
        self._tools.append(t)


class PluginTarget(PromptObserverMixin):
    """A :class:`PromptObserverMixin` that a :class:`Plugin` can be applied to.

    Shared by the runnable agent types (``Agent``, ``LiveAgent``). Both expose
    the agent-side surface a plugin needs — ``name``, ``_agent_dependencies``,
    ``_agent_variables``, ``_hitl_hook`` — so a plugin's collected contributions
    copy onto them identically.

    ``Plugin`` itself stays on the bare mixin: it is the *source* of
    contributions, never a target.
    """

    name: str
    tools: list[FunctionTool]
    dependency_provider: Provider
    _serializer: SerializerProto
    _tool_executor: ToolExecutor
    _hitl_hook: HumanHook | None
    _agent_dependencies: dict[Any, Any]
    _agent_variables: dict[Any, Any]

    def hitl_hook(self, func: HumanHook) -> HumanHook:
        if self._hitl_hook is not None:
            warnings.warn(
                "You already set HITL hook, provided value overrides it",
                category=RuntimeWarning,
                stacklevel=2,
            )

        self._hitl_hook = wrap_hitl(func)
        return func

    def add_tool(self, t: Callable[..., Any] | Tool) -> Self:
        """Bind a tool to this target (eager). The provider is resolved at call
        time from the live context, not stored on the tool."""
        self.tools.append(FunctionTool.ensure_tool(t))
        return self

    def _init_target(
        self,
        name: str,
        *,
        prompt: PromptType | Iterable[PromptType],
        hitl_hook: HumanHook | None,
        tools: Iterable[Callable[..., Any] | Tool],
        middleware: Iterable[MiddlewareFactory],
        observers: Iterable[Observer],
        dependencies: dict[Any, Any],
        variables: dict[Any, Any],
        plugins: Iterable["Plugin"],
    ) -> None:
        """Set up the contribution surface shared by every plugin target.

        Callers set their own type-specific state (config, stream, …) around
        this call; plugins are applied at the end, once tools, middleware, and
        the HITL hook are in place.
        """
        self.name = name

        self._agent_dependencies = dependencies or {}
        self._agent_variables = variables or {}
        self._hitl_hook = wrap_hitl(hitl_hook) if hitl_hook else None

        self._middleware = list(middleware)
        self._observers = list(observers)
        self._policies = []

        self._serializer = PydanticSerializer(
            pydantic_config={"arbitrary_types_allowed": True},
            use_fastdepends_errors=False,
        )
        self._tool_executor = ToolExecutor(self._serializer)

        self.dependency_provider = Provider()
        self.tools = []
        for t in tools:
            self.add_tool(t)

        self._init_prompts(prompt)

        for p in plugins:
            self._apply_plugin(p)

    def _apply_plugin(self, plugin: Plugin) -> None:
        """Apply a plugin's collected contributions to this target."""
        for t in plugin._tools:
            self.add_tool(t)

        for m in plugin._middleware:
            self.add_middleware(m)

        if plugin._hitl_hook is not None:
            if self._hitl_hook is not None:
                warnings.warn(
                    f"{type(self).__name__} '{self.name}' already has a HITL hook; the plugin's hook will be ignored.",
                    stacklevel=2,
                )
            else:
                self._hitl_hook = wrap_hitl(plugin._hitl_hook)

        self._agent_dependencies = plugin._dependencies | self._agent_dependencies
        self._agent_variables.update(plugin._variables)

        self._observers.extend(plugin._observers)
        self._policies.extend(plugin._policies)
        self._system_prompt.extend(plugin._system_prompt)
        self._dynamic_prompt.extend(plugin._dynamic_prompt)


def _wrap_prompt_hook(
    func: PromptHook,
) -> Callable[[ModelRequest, Context], Awaitable[str]]:
    call_model = build_model(func)

    async def wrapper(event: ModelRequest, context: Context) -> str:
        async with AsyncExitStack() as stack:
            r = await call_model.asolve(
                event,
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
                **{CONTEXT_OPTION_NAME: context},
            )
        return r

    return wrapper
