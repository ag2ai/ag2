# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Literal

from autogen.beta.annotations import Context, Variable
from autogen.beta.events import BuiltinToolCallEvent, ToolCallEvent
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool

from ._resolve import resolve_variable

WEB_FETCH_TOOL_NAME = "web_fetch"


@dataclass(slots=True)
class WebFetchToolSchema(ToolSchema):
    type: str = field(default=WEB_FETCH_TOOL_NAME, init=False)
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations: bool | None = None
    max_content_tokens: int | None = None
    web_fetch_version: Literal["web_fetch_20250910", "web_fetch_20260209"] = "web_fetch_20250910"


class WebFetchTool(Tool):
    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        *,
        max_uses: int | Variable | None = None,
        allowed_domains: list[str] | Variable | None = None,
        blocked_domains: list[str] | Variable | None = None,
        citations: bool | Variable | None = None,
        max_content_tokens: int | Variable | None = None,
        version: Literal["web_fetch_20250910", "web_fetch_20260209"] | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {}
        if max_uses is not None:
            self._params["max_uses"] = max_uses
        if allowed_domains is not None:
            self._params["allowed_domains"] = allowed_domains
        if blocked_domains is not None:
            self._params["blocked_domains"] = blocked_domains
        if citations is not None:
            self._params["citations"] = citations
        if max_content_tokens is not None:
            self._params["max_content_tokens"] = max_content_tokens
        if version is not None:
            self._params["web_fetch_version"] = version

        self.name = WEB_FETCH_TOOL_NAME

    async def schemas(self, context: "Context") -> list[WebFetchToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [WebFetchToolSchema(**resolved)]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            pass

        stack.enter_context(
            context.stream.where(BuiltinToolCallEvent.name == WEB_FETCH_TOOL_NAME).sub_scope(execute),
        )
