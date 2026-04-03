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
class MCPServerToolSchema(ToolSchema):
    type: str = field(default="mcp_server", init=False)
    server_url: str = ""
    server_label: str = ""
    authorization_token: str | None = None
    description: str | None = None
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] | None = None
    headers: dict[str, str] | None = None


class MCPServerTool(Tool):
    __slots__ = ("_params",)

    def __init__(
        self,
        *,
        server_url: str | Variable,
        server_label: str | Variable,
        authorization_token: str | Variable | None = None,
        description: str | Variable | None = None,
        allowed_tools: list[str] | Variable | None = None,
        blocked_tools: list[str] | Variable | None = None,
        headers: dict[str, str] | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {
            "server_url": server_url,
            "server_label": server_label,
        }
        if authorization_token is not None:
            self._params["authorization_token"] = authorization_token
        if description is not None:
            self._params["description"] = description
        if allowed_tools is not None:
            self._params["allowed_tools"] = allowed_tools
        if blocked_tools is not None:
            self._params["blocked_tools"] = blocked_tools
        if headers is not None:
            self._params["headers"] = headers

    async def schemas(self, context: "Context") -> list[MCPServerToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [MCPServerToolSchema(**resolved)]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        pass
