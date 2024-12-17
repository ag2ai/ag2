# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from autogen.agentchat import Agent, ConversableAgent

from .web_controller import DEFAULT_CHANNEL, WebController
from .websurfer_prompts import (
    DEFAULT_DESCRIPTION,
    SCREENSHOT_TOOL_SELECTION,
)

# Size of the image we send to the MLM
# Current values represent a 0.85 scaling to fit within the GPT-4v short-edge constraints (768px)
MLM_HEIGHT = 765
MLM_WIDTH = 1224

SCREENSHOT_TOKENS = 1105

from autogen.logger import FileLogger

logger = FileLogger(config={})


class MultimodalWebSurfer(ConversableAgent):
    """(In preview) A multimodal agent that acts as a web surfer that can search the web and visit web pages."""

    def __init__(
        self,
        name: str = "MultimodalWebSurfer",
        system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = DEFAULT_DESCRIPTION,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
        screenshot_tool_prompt: str = SCREENSHOT_TOOL_SELECTION,
    ):
        """To instantiate properly please make sure to call MultimodalWebSurfer.init"""
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
            chat_messages=chat_messages,
            silent=silent,
        )
        self.web_controller = WebController()

    async def init(
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
        Initialize the MultimodalWebSurfer.

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
        await self.web_controller.init(
            headless=headless,
            browser_channel=browser_channel,
            browser_data_dir=browser_data_dir,
            start_page=start_page,
            downloads_folder=downloads_folder,
            debug_dir=debug_dir,
            to_save_screenshots=to_save_screenshots,
            markdown_converter=markdown_converter,
            agent_name=self.name,
        )
        pass

    async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], None]:
        """Generates the actual reply. First calls the LLM to figure out which tool to use, then executes the tool.

        Returns:
            Union[str, Dict, None]: The response content which may be a string, dict or None
        """
        pass
