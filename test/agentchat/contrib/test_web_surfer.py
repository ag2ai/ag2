# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import os
import re

import pytest

from autogen import UserProxyAgent
from autogen.agentchat.contrib.web_surfer import WebSurferAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports

from ...conftest import MOCK_OPEN_AI_API_KEY, Credentials
from ...test_browser_utils import BING_QUERY, BLOG_POST_TITLE, BLOG_POST_URL

with optional_import_block() as result:
    import markdownify  # noqa: F401
    import pathvalidate  # noqa: F401
    import pdfminer  # noqa: F401
    import requests  # noqa: F401
    from bs4 import BeautifulSoup  # noqa: F401


try:
    BING_API_KEY = os.environ["BING_API_KEY"]
except KeyError:
    skip_bing = True
else:
    skip_bing = False


@run_for_optional_imports(["markdownify", "pathvalidate", "pdfminer", "requests", "bs4"], "websurfer")
@run_for_optional_imports(["openai"], "openai")
def test_web_surfer() -> None:
    with pytest.MonkeyPatch.context() as mp:
        # we mock the API key so we can register functions (llm_config must be present for this to work)
        mp.setenv("OPENAI_API_KEY", MOCK_OPEN_AI_API_KEY)
        page_size = 4096
        web_surfer = WebSurferAgent(
            "web_surfer",
            llm_config={"api_type": "openai", "model": "gpt-4o", "config_list": []},
            browser_config={"viewport_size": page_size},
        )

        # Sneak a peak at the function map, allowing us to call the functions for testing here
        function_map = web_surfer._user_proxy._function_map

        # Test some basic navigations
        response = function_map["visit_page"](BLOG_POST_URL)
        assert f"Address: {BLOG_POST_URL}".strip() in response
        assert f"Title: {BLOG_POST_TITLE}".strip() in response

        # Test scrolling
        m = re.search(r"\bViewport position: Showing page 1 of (\d+).", response)
        total_pages = int(m.group(1))  # type: ignore[union-attr]

        response = function_map["page_down"]()
        assert (
            f"Viewport position: Showing page 2 of {total_pages}." in response
        )  # Assumes the content is longer than one screen

        response = function_map["page_up"]()
        assert f"Viewport position: Showing page 1 of {total_pages}." in response

        # Try to scroll too far back up
        response = function_map["page_up"]()
        assert f"Viewport position: Showing page 1 of {total_pages}." in response

        # Try to scroll too far down
        for i in range(0, total_pages + 1):
            response = function_map["page_down"]()
        assert f"Viewport position: Showing page {total_pages} of {total_pages}." in response

        # Test web search -- we don't have a key in this case, so we expect it to raise an error (but it means the code path is correct)
        with pytest.raises(ValueError, match="Missing Bing API key."):
            response = function_map["informational_web_search"](BING_QUERY)

        with pytest.raises(ValueError, match="Missing Bing API key."):
            response = function_map["navigational_web_search"](BING_QUERY)

        # Test Q&A and summarization -- we don't have a key so we expect it to fail (but it means the code path is correct)
        with pytest.raises(IndexError):
            response = function_map["answer_from_page"]("When was it founded?")

        with pytest.raises(IndexError):
            response = function_map["summarize_page"]()


@run_for_optional_imports("openai", "openai")
@run_for_optional_imports(["markdownify", "pathvalidate", "pdfminer", "requests", "bs4"], "websurfer")
def test_web_surfer_oai(credentials_gpt_4o_mini: Credentials, credentials_gpt_4o: Credentials) -> None:
    llm_config = {"config_list": credentials_gpt_4o.config_list, "timeout": 180, "cache_seed": 42}

    summarizer_llm_config = {
        "config_list": credentials_gpt_4o_mini.config_list,
        "timeout": 180,
    }

    page_size = 4096
    web_surfer = WebSurferAgent(
        "web_surfer",
        llm_config=llm_config,
        summarizer_llm_config=summarizer_llm_config,
        browser_config={"viewport_size": page_size},
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        default_auto_reply="",
        is_termination_msg=lambda x: True,
    )

    # Make some requests that should test function calling
    user_proxy.initiate_chat(web_surfer, message="Please visit the page 'https://en.wikipedia.org/wiki/Microsoft'")

    user_proxy.initiate_chat(web_surfer, message="Please scroll down.")

    user_proxy.initiate_chat(web_surfer, message="Please scroll up.")

    user_proxy.initiate_chat(web_surfer, message="When was it founded?")

    user_proxy.initiate_chat(web_surfer, message="What's this page about?")


@pytest.mark.skipif(
    skip_bing,
    reason="do not run if bing api key is not available",
)
def test_web_surfer_bing() -> None:
    page_size = 4096
    web_surfer = WebSurferAgent(
        "web_surfer",
        llm_config={
            "config_list": [
                {
                    "model": "gpt-4o",
                    "api_key": "sk-PLACEHOLDER_KEY",
                }
            ]
        },
        browser_config={"viewport_size": page_size, "bing_api_key": BING_API_KEY},
    )

    # Sneak a peak at the function map, allowing us to call the functions for testing here
    function_map = web_surfer._user_proxy._function_map

    # Test informational queries
    response = function_map["informational_web_search"](BING_QUERY)
    assert f"Address: bing: {BING_QUERY}" in response
    assert f"Title: {BING_QUERY} - Search" in response
    assert "Viewport position: Showing page 1 of 1." in response
    assert f"A Bing search for '{BING_QUERY}' found " in response

    # Test informational queries
    response = function_map["navigational_web_search"](BING_QUERY + " Wikipedia")
    assert "Address: https://en.wikipedia.org/wiki/" in response


if __name__ == "__main__":
    """Runs this file's tests from the command line."""
    test_web_surfer()
    test_web_surfer_oai()
    test_web_surfer_bing()
