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

WEB_SEARCH_TOOL_NAME = "web_search"


@dataclass(slots=True)
class UserLocation:
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UserLocation):
            return NotImplemented
        # Only compare fields that are explicitly set (not None) in either location.
        # If one location specifies a field and the other doesn't, that field is ignored.
        # UserLocation(country="DE") == UserLocation(country="DE", timezone="Europe/Berlin")
        # because the only shared explicitly-set field is country="DE".
        fields = ["city", "region", "country", "timezone"]
        self_vals = [getattr(self, f) for f in fields]
        other_vals = [getattr(other, f) for f in fields]
        for s, o in zip(self_vals, other_vals):
            if s is None and o is None:
                continue
            if s is None or o is None:
                continue
            if s != o:
                return False
        return True


@dataclass(slots=True)
class WebSearchToolSchema(ToolSchema):
    type: str = field(default=WEB_SEARCH_TOOL_NAME, init=False)
    search_context_size: Literal["low", "medium", "high"] | None = None
    max_uses: int | None = None
    user_location: UserLocation | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    web_search_version: Literal["web_search_20250305", "web_search_20260209"] = "web_search_20250305"


class WebSearchTool(Tool):
    __slots__ = (
        "_params",
        "name",
    )

    def __init__(
        self,
        *,
        search_context_size: Literal["low", "medium", "high"] | Variable | None = None,
        max_uses: int | Variable | None = None,
        user_location: UserLocation | Variable | None = None,
        allowed_domains: list[str] | Variable | None = None,
        blocked_domains: list[str] | Variable | None = None,
        version: Literal["web_search_20250305", "web_search_20260209"] | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {}
        if search_context_size is not None:
            self._params["search_context_size"] = search_context_size
        if max_uses is not None:
            self._params["max_uses"] = max_uses
        if user_location is not None:
            self._params["user_location"] = user_location
        if allowed_domains is not None:
            self._params["allowed_domains"] = allowed_domains
        if blocked_domains is not None:
            self._params["blocked_domains"] = blocked_domains
        if version is not None:
            self._params["web_search_version"] = version

        self.name = WEB_SEARCH_TOOL_NAME

    async def schemas(self, context: "Context") -> list[WebSearchToolSchema]:
        resolved = {k: resolve_variable(v, context, param_name=k) for k, v in self._params.items()}
        return [WebSearchToolSchema(**resolved)]

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
            context.stream.where(BuiltinToolCallEvent.name == WEB_SEARCH_TOOL_NAME).sub_scope(execute),
        )
