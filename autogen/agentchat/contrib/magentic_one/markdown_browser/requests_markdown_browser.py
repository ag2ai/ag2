# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Union

import requests

from .abstract_markdown_browser import AbstractMarkdownBrowser
from .markdown_search import AbstractMarkdownSearch

# TODO: Fix unfollowed import
from .mdconvert import MarkdownConverter  # type: ignore


class RequestsMarkdownBrowser(AbstractMarkdownBrowser):
    """
    (In preview) An extremely simple Python requests-powered Markdown web browser.
    This browser cannot run JavaScript, compute CSS, etc. It simply fetches the HTML document, and converts it to Markdown.
    See AbstractMarkdownBrowser for more details.
    """

    def __init__(  # type: ignore
        self,
        start_page: Union[str, None] = None,
        viewport_size: Union[int, None] = 1024 * 8,
        downloads_folder: Union[str, None] = None,
        search_engine: Union[AbstractMarkdownSearch, None] = None,
        markdown_converter: Union[MarkdownConverter, None] = None,
        requests_session: Union[requests.Session, None] = None,
        requests_get_kwargs: Union[Dict[str, Any], None] = None,
    ):
        """
        Instantiate a new RequestsMarkdownBrowser.

        Arguments:
            start_page: The page on which the browser starts (default: "about:blank")
            viewport_size: Approximately how many *characters* fit in the viewport. Viewport dimensions are adjusted dynamically to avoid cutting off words (default: 8192).
            downloads_folder: Path to where downloads are saved. If None, downloads are disabled. (default: None)
            search_engine: An instance of MarkdownSearch, which handles web searches performed by this browser (default: a new `BingMarkdownSearch()` with default parameters)
            markdown_converted: An instance of a MarkdownConverter used to convert HTML pages and downloads to Markdown (default: a new `MarkdownConerter()` with default parameters)
            request_session: The session from which to issue requests (default: a new `requests.Session()` instance with default parameters)
            request_get_kwargs: Extra parameters passed to evert `.get()` call made to requests.
        """
        pass
