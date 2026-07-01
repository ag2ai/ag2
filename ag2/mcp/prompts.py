# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from mcp.types import GetPromptResult, TextContent
from mcp.types import Prompt as MCPPrompt
from mcp.types import PromptArgument as MCPPromptArgument
from mcp.types import PromptMessage as MCPPromptMessage

from ._async import call_user_fn
from .errors import MCPPromptNotFoundError

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server


@dataclass(frozen=True, slots=True)
class PromptArgument:
    """A declared argument of a :class:`Prompt`, advertised in ``prompts/list``."""

    name: str
    description: str | None = None
    required: bool = False


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """One message in a rendered prompt (``user`` or ``assistant`` role)."""

    role: Literal["user", "assistant"]
    text: str


# A prompt renderer receives the call arguments and returns either a plain string
# (rendered as a single ``user`` message) or an explicit message sequence. Sync
# or async.
RenderResult = str | Sequence[PromptMessage]
RenderFn = Callable[[dict[str, str]], Awaitable[RenderResult] | RenderResult]


@dataclass(frozen=True, slots=True)
class Prompt:
    """A reusable prompt template exposed over MCP (``prompts/list`` + ``prompts/get``).

    ``render`` receives the supplied arguments as a ``{name: value}`` dict and
    returns the messages (a bare ``str`` becomes one ``user`` message). It may be
    sync or async. ``arguments`` declares the accepted parameters for discovery.
    """

    name: str
    render: RenderFn
    description: str | None = None
    arguments: tuple[PromptArgument, ...] = field(default=())


class PromptProvider:
    """Serves a fixed set of :class:`Prompt` over MCP."""

    __slots__ = ("_prompts", "_by_name")

    def __init__(self, prompts: Sequence[Prompt]) -> None:
        self._prompts = tuple(prompts)
        self._by_name = {p.name: p for p in self._prompts}

    def register(self, server: "Server") -> None:
        provider = self

        # ``mcp``'s low-level decorators are untyped; ignore the resulting noise.
        @server.list_prompts()  # type: ignore[no-untyped-call, misc]
        async def _list_prompts() -> list[MCPPrompt]:
            return [_to_mcp_prompt(p) for p in provider._prompts]

        @server.get_prompt()  # type: ignore[no-untyped-call, misc]
        async def _get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
            return await provider.get(name, arguments or {})

    async def get(self, name: str, arguments: dict[str, str]) -> GetPromptResult:
        prompt = self._by_name.get(name)
        if prompt is None:
            raise MCPPromptNotFoundError(name)
        result = await call_user_fn(prompt.render, dict(arguments))
        return GetPromptResult(description=prompt.description, messages=_to_mcp_messages(result))


def _to_mcp_messages(result: RenderResult) -> list[MCPPromptMessage]:
    messages = [PromptMessage(role="user", text=result)] if isinstance(result, str) else list(result)
    return [MCPPromptMessage(role=m.role, content=TextContent(type="text", text=m.text)) for m in messages]


def _to_mcp_prompt(prompt: Prompt) -> MCPPrompt:
    return MCPPrompt(
        name=prompt.name,
        description=prompt.description,
        arguments=[
            MCPPromptArgument(name=a.name, description=a.description, required=a.required) for a in prompt.arguments
        ],
    )


__all__ = (
    "Prompt",
    "PromptArgument",
    "PromptMessage",
    "PromptProvider",
)
