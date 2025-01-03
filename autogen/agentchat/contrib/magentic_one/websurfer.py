# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Any, Callable, Dict, List, Literal, Optional, Union

from autogen.agentchat import Agent, ConversableAgent

from .web_controller import WebController


class MultimodalWebSurfer(ConversableAgent):
    """A multimodal agent that acts as a web surfer, capable of interacting with web pages."""

    DEFAULT_START_PAGE = "https://www.bing.com/"

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
        description: Optional[str] = None,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
    ):
        """Initialize the web surfer agent."""
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
        self._playwright = None
        self._context = None
        self._page = None
        self._last_download = None
        self._prior_metadata_hash = None
        self.start_page = self.DEFAULT_START_PAGE
        self.downloads_folder = None
        self.to_save_screenshots = False
        self.debug_dir = None
        self._web_controller = None

    async def init(
        self,
        headless: bool = True,
        browser_channel: str | None = None,
        browser_data_dir: str | None = None,
        start_page: str | None = None,
        downloads_folder: str | None = None,
        debug_dir: str | None = None,
        to_save_screenshots: bool = False,
        markdown_converter: Any | None = None,
    ) -> None:
        """Initialize the browser and set up the environment."""
        self.start_page = start_page or self.DEFAULT_START_PAGE
        self.downloads_folder = downloads_folder
        self.to_save_screenshots = to_save_screenshots
        self.debug_dir = debug_dir
        self._markdown_converter = markdown_converter
        self._web_controller = WebController(
            headless=headless,
            browser_channel=browser_channel,
            browser_data_dir=browser_data_dir,
            start_page=start_page,
            downloads_folder=downloads_folder,
            debug_dir=debug_dir,
            to_save_screenshots=to_save_screenshots,
            markdown_converter=markdown_converter,
        )
        self._web_controller.set_agent_name(self.name)
        await self._web_controller.init()
        self._page = await self._web_controller.create_page()

    async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, Dict, None]:
        """Generate a reply based on the current state of the browser."""
        if not self._page:
            await self.init()
        return await self._summarize_page()

    async def _reset(self) -> None:
        """Reset the browser to the start page."""
        if self._web_controller:
            await self._web_controller._reset()
        self._page = await self._web_controller.create_page()

    async def _visit_page(self, url: str) -> None:
        """Visit a given URL."""
        if self._web_controller:
            await self._web_controller._visit_page(url)

    async def _back(self) -> None:
        """Go back in the browser history."""
        if self._web_controller:
            await self._web_controller._back()

    async def _page_down(self) -> None:
        """Scroll down one page."""
        if self._web_controller:
            await self._web_controller._page_down()

    async def _page_up(self) -> None:
        """Scroll up one page."""
        if self._web_controller:
            await self._web_controller._page_up()

    async def _click_id(self, identifier: str) -> None:
        """Click an element with the given ID."""
        if self._web_controller:
            await self._web_controller._click_id(identifier)

    async def _fill_id(self, identifier: str, value: str) -> None:
        """Fill an input field with the given ID with the given value."""
        if self._web_controller:
            await self._web_controller._fill_id(identifier, value)

    async def _scroll_id(self, identifier: str, direction: str) -> None:
        """Scroll an element with the given ID in the given direction."""
        if self._web_controller:
            await self._web_controller._scroll_id(identifier, direction)

    async def _summarize_page(self, question: str | None = None) -> str:
        """Summarize the current page."""
        if self._web_controller:
            return await self._web_controller._get_page_markdown()

    async def _get_ocr_text(self, image: bytes) -> str:
        """Get the OCR text from the given image."""
        if self._web_controller:
            return await self._web_controller._get_ocr_text(image)
