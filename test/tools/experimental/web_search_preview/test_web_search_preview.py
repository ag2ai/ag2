# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental import WebSearchPreviewTool

from ....conftest import Credentials


@run_for_optional_imports(
    [
        "openai",
    ],
    "openai",
)
class TestWebSearchPreviewTool:
    def test_init(self, mock_credentials: Credentials) -> None:
        google_search_tool = WebSearchPreviewTool(llm_config=mock_credentials.llm_config)

        assert google_search_tool.name == "web_search_preview"
        assert google_search_tool.description.startswith("Tool used to perform a web search. It can be used as google")

    def test_web_search_preview_f(self, credentials_gpt_4o: Credentials) -> None:
        google_search_tool = WebSearchPreviewTool(llm_config=credentials_gpt_4o.llm_config)

        response = google_search_tool(
            query="Give me latest news about Alexander-Arnold",
        )

        assert isinstance(response, str)
