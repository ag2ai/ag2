# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from ....doc_utils import export_module
from ... import Tool


@export_module("autogen.tools.experimental")
class GoogleSearchTool(Tool):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

        def google_search() -> None:
            pass

        super().__init__(
            description="Google Search",
            func_or_tool=google_search,
        )
