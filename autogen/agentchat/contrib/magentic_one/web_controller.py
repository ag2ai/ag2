# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import os
from typing import Any

from autogen.logger import FileLogger

logger = FileLogger(config={})

from .types import SentinelMeta


# Sentinels
class DEFAULT_CHANNEL(metaclass=SentinelMeta):
    pass


class WebController:
    """
    A class to encapsulate the browser capabilities and interactions using Playwright.
    """

    def __init__(
        self,
        headless: bool = True,
        browser_channel: str | type[DEFAULT_CHANNEL] = DEFAULT_CHANNEL,
        browser_data_dir: str | None = None,
        start_page: str | None = None,
        downloads_folder: str | None = None,
        debug_dir: str = os.getcwd(),
        to_save_screenshots: bool = False,
        # navigation_allow_list=lambda url: True,
        markdown_converter: Any | None = None,  # TODO: Fixme
    ) -> None:
        """
        Initialize the WebController.

        Args:
            headless (bool): Whether to run the browser in headless mode. Defaults to True.
            browser_channel (str | type[DEFAULT_CHANNEL]): The browser channel to use. Defaults to DEFAULT_CHANNEL.
            browser_data_dir (str | None): The directory to store browser data. Defaults to None.
            start_page (str | None): The initial page to visit. Defaults to DEFAULT_START_PAGE.
            downloads_folder (str | None): The folder to save downloads. Defaults to None.
            debug_dir (str | None): The directory to save debug information. Defaults to the current working directory.
            to_save_screenshots (bool): Whether to save screenshots. Defaults to False.
            markdown_converter (Any | None): The markdown converter to use. Defaults to None.
        """

    async def init(
        self,
        headless: bool = True,
        browser_channel: str | type[DEFAULT_CHANNEL] = DEFAULT_CHANNEL,
        browser_data_dir: str | None = None,
        start_page: str | None = None,
        downloads_folder: str | None = None,
        debug_dir: str = os.getcwd(),
        to_save_screenshots: bool = False,
        markdown_converter: Any | None = None,  # TODO: Fixme
        agent_name: str = "",
    ) -> None:
        pass
