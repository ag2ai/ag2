# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field

from autogen.beta.annotations import Context, Variable
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool

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
    via :class:`~autogen.beta.config.openai.containers.ContainerManager`.
    """

    container_id: str


@dataclass(slots=True)
class LocalEnvironment:
    """Commands execute in the local runtime (client-side, OpenAI)."""


ShellEnvironment = ContainerAutoEnvironment | ContainerReferenceEnvironment | LocalEnvironment


@dataclass(slots=True)
class ShellToolSchema(ToolSchema):
    """Provider-neutral capability flag for shell/bash execution.

    Each provider mapper converts this into the appropriate API format:

    - Anthropic: ``bash_20250124``
    - OpenAI Responses API: ``shell`` (with optional ``environment``)
    """

    type: str = field(default="shell", init=False)
    environment: ShellEnvironment | None = None


class ShellTool(Tool):
    """Shell/bash execution tool.

    Provider-specific mapping:

    - **Anthropic** — maps to ``bash_20250124``. Claude calls the tool with
      a ``command`` or ``restart`` input; the application must execute it and
      return the result (client-side tool).
      ``environment`` is ignored for Anthropic.

    - **OpenAI Responses API** — maps to ``shell`` (``gpt-5.4``).
      Use ``environment`` to control where commands execute:
      ``ContainerAutoEnvironment``, ``ContainerReferenceEnvironment``, or
      ``LocalEnvironment``. Omit ``environment`` for OpenAI's default behaviour.

    See:
    - https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
    - https://developers.openai.com/api/docs/guides/tools-shell
    """

    __slots__ = ("_params",)

    def __init__(
        self,
        *,
        environment: ShellEnvironment | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {}
        if environment is not None:
            self._params["environment"] = environment

    async def schemas(self, context: "Context") -> list[ShellToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [ShellToolSchema(**resolved)]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        pass
