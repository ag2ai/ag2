# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from typing import Literal

from pydantic import BaseModel

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool


class UserLocation(BaseModel):
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class WebSearchToolSchema(ToolSchema):
    type: Literal["web_search"] = "web_search"
    search_context_size: Literal["low", "medium", "high"] | None = None
    max_uses: int | None = None
    user_location: UserLocation | None = None


class WebSearchTool(Tool):
    __slots__ = ("schema",)

    def __init__(
        self,
        *,
        search_context_size: Literal["low", "medium", "high"] | None = None,
        max_uses: int | None = None,
        user_location: UserLocation | None = None,
    ) -> None:
        self.schema = WebSearchToolSchema(
            search_context_size=search_context_size,
            max_uses=max_uses,
            user_location=user_location,
        )

    async def schemas(self, context: "Context") -> list[WebSearchToolSchema]:
        return [self.schema]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        pass
