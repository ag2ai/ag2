# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ag2.annotations import Context
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.middleware import BaseMiddleware, ToolExecution, ToolMiddleware, ToolResultType
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

if TYPE_CHECKING:
    from ag2.tools.sandbox import SandboxFactory

# The name Anthropic puts on the wire for ``bash_20250124``. It is also the
# schema ``type``, so ``known_tools`` (which stores ``schema.type`` for
# non-function schemas) matches the incoming ``ToolCallEvent.name`` and the
# executor below is reached instead of the not-found guard.
ANTHROPIC_BASH_TOOL_NAME = "bash"


@dataclass(slots=True)
class AnthropicBashToolSchema(ToolSchema):
    """Anthropic's client-executed bash tool.

    Unlike the other entries under ``ag2/tools/builtin/``, this is not a
    provider-executed capability flag: Anthropic sends a plain ``tool_use``
    block and waits for a ``tool_result``.
    """

    type: str = field(default=ANTHROPIC_BASH_TOOL_NAME, init=False)
    version: Literal["bash_20250124"] = "bash_20250124"


class AnthropicBashTool(Tool):
    """Anthropic's ``bash_20250124`` tool, executed by ag2.

    Declares Anthropic's typed schema and runs the commands it asks for through
    the same :class:`~ag2.tools.sandbox.adapter.ShellAdapter` that backs
    :class:`~ag2.tools.SandboxShellTool`, so the backend choice and the
    ``allowed`` / ``blocked`` / ``readonly`` policy are unchanged.

    Anthropic-only: every other provider's mapper rejects this schema.

    Args:
        environment: Execution backend; ``None`` means a local subprocess.
        allowed / blocked / ignore / readonly: command filter set, as on
            :class:`~ag2.tools.SandboxShellTool`.
    """

    __slots__ = ("name", "_adapter", "_schema")

    def __init__(
        self,
        environment: "SandboxFactory | None" = None,
        *,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
    ) -> None:
        # Imported here, not at module scope: ``ag2/tools/builtin`` is pulled in
        # early (ag2.config -> client -> tools/__init__ -> builtin) while
        # ``ag2.tools.sandbox`` imports ``ag2.tools.code``, which imports
        # ``ag2.tools.sandbox`` back. A module-level import deadlocks that cycle.
        from ag2.tools.sandbox import LocalEnvironment
        from ag2.tools.sandbox.adapter import ShellAdapter

        backend: SandboxFactory = environment if environment is not None else LocalEnvironment()
        self._adapter = ShellAdapter(
            backend,
            allowed=allowed,
            blocked=blocked,
            ignore=ignore,
            readonly=readonly,
        )
        self._schema = AnthropicBashToolSchema()
        self.name = ANTHROPIC_BASH_TOOL_NAME

    async def schemas(self, context: "Context") -> list[ToolSchema]:
        return [self._schema]

    async def __call__(self, event: "ToolCallEvent", context: "Context") -> "ToolResultEvent":
        args = event.serialized_arguments
        try:
            if args.get("restart"):
                # The adapter opens a sandbox per ``run``, so there is no session
                # to restart. Say so rather than claim a restart that never happened.
                result = "No persistent bash session to restart; each command runs in a fresh sandbox."
            else:
                result = await self._adapter.run(args.get("command", ""), context=context)
            return ToolResultEvent.from_call(event, result=result)
        except Exception as e:
            return ToolErrorEvent.from_call(event, error=e)

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        def wrap(hook: "ToolMiddleware", inner: "ToolExecution") -> "ToolExecution":
            async def call(event: "ToolCallEvent", context: "Context") -> "ToolResultType":
                return await hook(inner, event, context)

            return call

        # This tool runs model-authored shell commands, so the agent's governance
        # and telemetry hooks have to wrap it like any other executing tool.
        execution: ToolExecution = self
        for mw in middleware:
            execution = wrap(mw.on_tool_execution, execution)

        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            await context.send(await execution(event, context))

        stack.enter_context(
            context.stream.where(ToolCallEvent.name == ANTHROPIC_BASH_TOOL_NAME).sub_scope(execute),
        )
