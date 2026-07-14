# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Xquik tweet search extension for AG2."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

QueryType: TypeAlias = Literal["Latest", "Top"]


@dataclass(slots=True)
class XquikTweetSearchResponse:
    query: str
    tweets: list[dict[str, Any]] = field(default_factory=list)
    has_next_page: bool = False
    next_cursor: str = ""


class XquikSearchToolkit(Toolkit):
    """Toolkit that searches public X posts through the Xquik REST API.

    Passing the toolkit to an agent registers ``xquik_tweet_search``. Optional
    search defaults can be fixed when the toolkit or tool is constructed, or
    supplied through AG2 ``Variable`` values at runtime.

    An Xquik ``api_key`` is required.
    """

    __slots__ = ("_api_key", "_base_url", "_timeout")

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://xquik.com",
        timeout: float = 60.0,
        query_type: QueryType | Variable | None = None,
        cursor: str | Variable | None = None,
        since_time: str | Variable | None = None,
        until_time: str | Variable | None = None,
        limit: int | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        super().__init__(
            self.search(
                query_type=query_type,
                cursor=cursor,
                since_time=since_time,
                until_time=until_time,
                limit=limit,
            ),
            name="xquik_search_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        query_type: QueryType | Variable | None = None,
        cursor: str | Variable | None = None,
        since_time: str | Variable | None = None,
        until_time: str | Variable | None = None,
        limit: int | Variable | None = None,
        name: str = "xquik_tweet_search",
        description: str = ("Search public X posts through Xquik. Returns tweet records and pagination metadata."),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        api_key = self._api_key
        base_url = self._base_url
        timeout = self._timeout

        @tool(name=name, description=description, middleware=middleware)
        async def xquik_tweet_search(
            query: Annotated[str, Field(description="The X search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search public X posts and return a structured page of results."""
            params: dict[str, Any] = {
                "q": query,
                "queryType": resolve_variable(query_type, ctx, param_name="query_type"),
                "cursor": resolve_variable(cursor, ctx, param_name="cursor"),
                "sinceTime": resolve_variable(since_time, ctx, param_name="since_time"),
                "untilTime": resolve_variable(until_time, ctx, param_name="until_time"),
                "limit": resolve_variable(limit, ctx, param_name="limit"),
            }
            request_params = {key: value for key, value in params.items() if value is not None}

            async with httpx.AsyncClient(
                base_url=base_url,
                headers={"x-api-key": api_key},
                timeout=timeout,
            ) as client:
                response = await client.get("/api/v1/x/tweets/search", params=request_params)
                response.raise_for_status()

            raw: dict[str, Any] = response.json()
            raw_tweets = raw.get("tweets")
            tweets = raw_tweets if isinstance(raw_tweets, list) else []
            raw_cursor = raw.get("next_cursor")

            return ToolResult(
                XquikTweetSearchResponse(
                    query=query,
                    tweets=tweets,
                    has_next_page=raw.get("has_next_page") is True,
                    next_cursor=raw_cursor if isinstance(raw_cursor, str) else "",
                )
            )

        return xquik_tweet_search
