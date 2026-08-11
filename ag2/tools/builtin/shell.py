# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass, field
from typing import Literal

from ag2.annotations import Context, Variable
from ag2.events import BuiltinToolCallEvent, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool

from ._resolve import resolve_variable


@dataclass(slots=True)
class NetworkPolicy:
    """Outbound network access policy for hosted shell containers (OpenAI)."""

    allowed_domains: list[str]


@dataclass(slots=True)
class ContainerAutoEnvironment:
    """OpenAI provisions and manages the container automatically."""

    network_policy: NetworkPolicy | None = None


@dataclass(slots=True)
class ContainerReferenceEnvironment:
    """References an existing container by ID.

    Network policy is not configurable here — it was set when the container was created
    via :class:`~ag2.config.openai.containers.ContainerManager`.
    """

    container_id: str


ShellEnvironment = ContainerAutoEnvironment | ContainerReferenceEnvironment


SHELL_TOOL_NAME = "shell"


@dataclass(slots=True)
class ShellToolSchema(ToolSchema):
    """Provider-neutral capability flag for provider-executed shell.

    Currently only OpenAI Responses API executes shell server-side.
    Anthropic's bash tool is client-side and is rejected with
    :class:`~ag2.exceptions.UnsupportedToolError` — use
    :class:`~ag2.tools.SandboxShellTool` instead.
    """

    type: str = field(default=SHELL_TOOL_NAME, init=False)
    version: Literal["bash_20250124"] = "bash_20250124"
    environment: ShellEnvironment | None = None


class ShellTool(Tool):
    """Shell execution tool — provider-executed server-side.

    Provider support:

    - **OpenAI Responses API** — maps to ``shell``. Use ``environment`` to
      control where commands execute: ``ContainerAutoEnvironment``,
      ``ContainerReferenceEnvironment``.

    - **Anthropic** — NOT supported. Claude's ``bash`` tool is a client-side
      tool (the application must execute the command and return the result).
      Using ``ShellTool`` with ``AnthropicConfig`` raises
      :class:`~ag2.exceptions.UnsupportedToolError`. Use
      :class:`~ag2.tools.SandboxShellTool` instead, which runs
      commands via subprocess and works with any provider.

    See:
    - https://developers.openai.com/api/docs/guides/tools-shell
    """

    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        *,
        environment: ShellEnvironment | Variable | None = None,
        version: Literal["bash_20250124"] = "bash_20250124",
    ) -> None:
        self._params: dict[str, object] = {"version": version}
        if environment is not None:
            self._params["environment"] = environment
        self.name = SHELL_TOOL_NAME

    async def schemas(self, context: "Context") -> list[ShellToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [ShellToolSchema(**resolved)]

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            pass

        stack.enter_context(
            context.stream.where(BuiltinToolCallEvent.name == SHELL_TOOL_NAME).sub_scope(execute),
        )
