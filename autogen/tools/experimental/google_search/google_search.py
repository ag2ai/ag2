# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Annotated, Any, Optional

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ... import Depends, Tool
from ...dependency_injection import on

with optional_import_block():
    from googleapiclient.discovery import build


@require_optional_import(
    [
        "googleapiclient",
    ],
    "google-search",
)
def _google_search(
    query: str,
    search_api_key: str,
    search_engine_id: str,
    num_results: int,
) -> list[dict[str, Any]]:
    service = build("customsearch", "v1", developerKey=search_api_key)
    res = service.cse().list(q=query, cx=search_engine_id, num=num_results).execute()

    results = []
    if "items" in res:
        for item in res["items"]:
            results.append({"title": item["title"], "link": item["link"], "snippet": item.get("snippet", "")})
    return results


@export_module("autogen.tools.experimental")
class GoogleSearchTool(Tool):
    def __init__(
        self,
        *,
        search_api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
    ):
        self.search_api_key = search_api_key

        def google_search(
            query: Annotated[str, "The search query."],
            search_api_key: Annotated[str, Depends(on(search_api_key))],
            search_engine_id: Annotated[str, Depends(on(search_engine_id))],
            num_results: Annotated[int, "The number of results to return."] = 10,
        ) -> list[dict[str, Any]]:
            return _google_search(query, search_api_key, search_engine_id, num_results)

        super().__init__(
            description="Google Search",
            func_or_tool=google_search,
        )
