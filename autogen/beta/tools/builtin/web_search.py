# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Literal

from autogen.beta.annotations import Context, Variable
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool

from ._resolve import resolve_variable


@dataclass(slots=True)
class UserLocation:
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


@dataclass(slots=True)
class WebSearchToolSchema(ToolSchema):
    type: str = field(default="web_search", init=False)
    search_context_size: Literal["low", "medium", "high"] | None = None
    max_uses: int | None = None
    user_location: UserLocation | None = None


class WebSearchTool(Tool):
    __slots__ = ("_params",)

    def __init__(
        self,
        *,
        search_context_size: Literal["low", "medium", "high"] | Variable | None = None,
        max_uses: int | Variable | None = None,
        user_location: UserLocation | Variable | None = None,
    ) -> None:
        self._params: dict[str, object] = {}
        if search_context_size is not None:
            self._params["search_context_size"] = search_context_size
        if max_uses is not None:
            self._params["max_uses"] = max_uses
        if user_location is not None:
            self._params["user_location"] = user_location

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
        pass
