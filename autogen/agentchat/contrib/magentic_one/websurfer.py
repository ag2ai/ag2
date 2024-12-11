from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import random
import re
import json
import time
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)
from urllib.parse import quote_plus

import aiofiles

from PIL import Image
from autogen.agentchat import ConversableAgent, Agent
from autogen.agentchat.contrib import img_utils

from .websurfer_prompts import (
    DEFAULT_DESCRIPTION,
    SCREENSHOT_TOOL_SELECTION,
)

# TODO: Fix mdconvert (I think i saw a new pull request)
from .markdown_browser import MarkdownConverter  # type: ignore
from .web_controller import WebController, DEFAULT_CHANNEL
from .utils import clean_and_parse_json

from .set_of_mark import add_set_of_mark
from .tool_definitions import (
    TOOL_CLICK,
    TOOL_HISTORY_BACK,
    TOOL_PAGE_DOWN,
    TOOL_PAGE_UP,
    TOOL_READ_PAGE_AND_ANSWER,
    # TOOL_SCROLL_ELEMENT_DOWN,
    # TOOL_SCROLL_ELEMENT_UP,
    TOOL_SLEEP,
    TOOL_SUMMARIZE_PAGE,
    TOOL_TYPE,
    TOOL_VISIT_URL,
    TOOL_WEB_SEARCH,
    TOOL_INPUT_TEXT,
)

from .types import (
    InteractiveRegion,
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
            silent=silent
        )
        self.web_controller = WebController()
        self._chat_history: List[Dict[str,Any]] = [] 
        self.screenshot_tool_prompt = screenshot_tool_prompt


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
        self.start_page = start_page or self.DEFAULT_START_PAGE
        self.downloads_folder = downloads_folder
        self.to_save_screenshots = to_save_screenshots
        self.debug_dir = debug_dir

        if self.llm_config and self.web_controller:
            self.register_tools()
        
    def register_tools(self):
        async def visit_url(reasoning: str, url: str) -> str:
            """Navigate directly to a provided URL."""
            await self.web_controller._visit_page(url)
            return f"Visited URL: {url}"

        async def web_search(reasoning: str, query: str) -> str:
            """Performs a web search."""
            # Placeholder implementation - replace with actual web search logic using the controller
            print(f"Performing web search: {query} (Reasoning: {reasoning})")
            return f"Search results for: {query}"
        
        async def history_back(reasoning: str) -> str:
            """Navigates back one page in the browser's history."""
            await self.web_controller._back()
            return f"Navigated back."

        async def page_up(reasoning: str) -> str:
            """Scrolls the entire browser viewport one page UP towards the beginning."""
            await self.web_controller._page_up()
            return f"Scrolled page up."

        async def page_down(reasoning: str) -> str:
            """Scrolls the entire browser viewport one page DOWN towards the end."""
            await self.web_controller._page_down()
            return f"Scrolled page down."

        async def click(reasoning: str, target_id: int) -> str:
            """Clicks the mouse on the target with the given id."""
            await self.web_controller._click_id(str(target_id))
            return f"Clicked on target: {target_id}"

        async def input_text(reasoning: str, input_field_id: int, text_value: str) -> str:
            """Types the given text value into the specified field."""
            await self.web_controller._fill_id(str(input_field_id), text_value)
            return f"Inputted text into field: {input_field_id}"

        async def scroll_element_down(reasoning: str, target_id: int) -> str:
            """Scrolls a given html element (e.g., a div or a menu) DOWN."""
            await self.web_controller._scroll_id(str(target_id), "down")
            return f"Scrolled element down: {target_id}"

        async def scroll_element_up(reasoning: str, target_id: int) -> str:
            """Scrolls a given html element (e.g., a div or a menu) UP."""
            await self.web_controller._scroll_id(str(target_id), "up")
            return f"Scrolled element up: {target_id}"

        async def answer_question(reasoning: str, question: str) -> str:
            """answer a question about the current webpage's content."""
            return await self._summarize_page(question=question)

        async def summarize_page(reasoning: str) -> str:
            """summarize the entire page."""
            return await self._summarize_page()

        async def sleep(reasoning: str) -> str:
            """Wait a short period of time."""
            await self.web_controller._sleep(2)
            return f"Slept."

        # Register functions with the agent
        self.register_for_llm(name="visit_url", description=TOOL_VISIT_URL["function"]["description"])(visit_url)
        self.register_for_llm(name="web_search", description=TOOL_WEB_SEARCH["function"]["description"])(web_search)
        self.register_for_llm(name="history_back", description=TOOL_HISTORY_BACK["function"]["description"])(history_back)
        self.register_for_llm(name="page_up", description=TOOL_PAGE_UP["function"]["description"])(page_up)
        self.register_for_llm(name="page_down", description=TOOL_PAGE_DOWN["function"]["description"])(page_down)
        self.register_for_llm(name="click", description=TOOL_CLICK["function"]["description"])(click)
        self.register_for_llm(name="input_text", description=TOOL_INPUT_TEXT["function"]["description"])(input_text)
        #self.register_for_llm(name="scroll_element_down", description=TOOL_SCROLL_ELEMENT_DOWN["function"]["description"])(scroll_element_down)
        #self.register_for_llm(name="scroll_element_up", description=TOOL_SCROLL_ELEMENT_UP["function"]["description"])(scroll_element_up)
        self.register_for_llm(name="answer_question", description=TOOL_READ_PAGE_AND_ANSWER["function"]["description"])(answer_question)
        self.register_for_llm(name="summarize_page", description=TOOL_SUMMARIZE_PAGE["function"]["description"])(summarize_page)
        self.register_for_llm(name="sleep", description=TOOL_SLEEP["function"]["description"])(sleep)
        
        # Register the same functions for execution
        self.register_for_execution(name="visit_url")(visit_url)
        self.register_for_execution(name="web_search")(web_search)
        self.register_for_execution(name="history_back")(history_back)
        self.register_for_execution(name="page_up")(page_up)
        self.register_for_execution(name="page_down")(page_down)
        self.register_for_execution(name="click")(click)
        self.register_for_execution(name="input_text")(input_text)
        #self.register_for_execution(name="scroll_element_down")(scroll_element_down)
        #self.register_for_execution(name="scroll_element_up")(scroll_element_up)
        self.register_for_execution(name="answer_question")(answer_question)
        self.register_for_execution(name="summarize_page")(summarize_page)
        self.register_for_execution(name="sleep")(sleep)

        # Update tool signatures using the provided schemas
        for tool_schema in [
            TOOL_VISIT_URL, TOOL_WEB_SEARCH, TOOL_HISTORY_BACK, TOOL_PAGE_UP,
            TOOL_PAGE_DOWN, TOOL_CLICK, TOOL_INPUT_TEXT, #TOOL_SCROLL_ELEMENT_DOWN, TOOL_SCROLL_ELEMENT_UP,
            TOOL_READ_PAGE_AND_ANSWER, TOOL_SUMMARIZE_PAGE, 
            TOOL_SLEEP
        ]:
            self.update_tool_signature(tool_schema, is_remove=False)

    def _get_screenshot_selection_prompt(self, page_url, visible_targets, other_targets_str, focused_hint, tool_names) -> str:
        return self.screenshot_tool_prompt.format(page_url=page_url,
        visible_targets=visible_targets,
        other_targets_str=other_targets_str,
        focused_hint=focused_hint,
        tool_names=tool_names
        )  

    async def _reset(self) -> None: 
        self.chat_messages[self] = []
        await self.web_controller._reset()
    

    def _target_name(self, target: str, rects: Dict[str, InteractiveRegion]) -> str | None:
        try:
            return rects[target]["aria_name"].strip()
        except KeyError:
            return None

    def _format_target_list(self, ids: List[str], rects: Dict[str, InteractiveRegion]) -> List[str]:
        targets: List[str] = []
        for r in list(set(ids)):
            if r in rects:
                # Get the role
                aria_role = rects[r].get("role", "").strip()
                if len(aria_role) == 0:
                    aria_role = rects[r].get("tag_name", "").strip()

                # Get the name
                aria_name = re.sub(r"[\n\r]+", " ", rects[r].get("aria_name", "")).strip()

                # What are the actions?
                actions = ['"click"']
                if rects[r]["role"] in ["textbox", "searchbox", "search"]:
                    actions = ['"input_text"']
                actions_str = "[" + ",".join(actions) + "]"

                targets.append(f'{{"id": {r}, "name": "{aria_name}", "role": "{aria_role}", "tools": {actions_str} }}')

        return targets

    async def _execute_tool( # TODO: replace with ag2 function execution ? 
        self,
        message: Dict[str, Any],
        rects: Dict[str, InteractiveRegion],
        tool_names: str,
        use_ocr: bool = True
        )-> Tuple[bool, Union[str, Dict, None]]:
        # TODO: Handle both legacy function calls and new tool calls format 
        #if isinstance(message, dict) and "tool_responses" in message:
        #    # New tool calls format

        function = message["tool_calls"][0]["function"]
        tool_args = function["arguments"]
        if isinstance(tool_args, str):
            tool_args = clean_and_parse_json(tool_args) 

        tool_name = function["name"]
        assert tool_name is not None
        tool_id = message["tool_calls"][0]["id"]

        action_description = ""
        logger.log_event(
            source=self.name,
            name="tool_execution",
            data={
                "url": await self.web_controller.get_url(),
                "tool_name": tool_name,
                "args": tool_args
            }
        )

        if tool_name in self._function_map:
            await self._function_map[tool_name](**tool_args)
        else:
            logger.log_event(
                source=self.name,
                name="unknown_tool",
                data={
                    "tool_name": tool_name,
                    "args": tool_args
                }
            )

        await self.web_controller.wait_for_load_state()

        # Handle downloads
        if self.web_controller._last_download is not None and self.downloads_folder is not None:
            await self.web_controller.handle_download(self.downloads_folder)
        # Handle metadata
        page_metadata = json.dumps(await self.web_controller._get_page_metadata(), indent=4)
        metadata_hash = hashlib.md5(page_metadata.encode("utf-8")).hexdigest()
        if metadata_hash != self.web_controller._prior_metadata_hash:
            page_metadata = (
                "\nThe following metadata was extracted from the webpage:\n\n" + page_metadata.strip() + "\n"
            )
        else:
            page_metadata = ""
        self.web_controller._prior_metadata_hash = metadata_hash

        # Describe the viewport of the new page in words
        viewport = await self.web_controller._get_visual_viewport()
        percent_visible = int(viewport["height"] * 100 / viewport["scrollHeight"])
        percent_scrolled = int(viewport["pageTop"] * 100 / viewport["scrollHeight"])
        if percent_scrolled < 1:  # Allow some rounding error
            position_text = "at the top of the page"
        elif percent_scrolled + percent_visible >= 99:  # Allow some rounding error
            position_text = "at the bottom of the page"
        else:
            position_text = str(percent_scrolled) + "% down from the top of the page"

        new_screenshot = await self.web_controller.take_screenshot()
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            async with aiofiles.open(os.path.join(self.debug_dir, screenshot_png_name), "wb") as file:  # type: ignore
                await file.write(new_screenshot)  # type: ignore
            logger.log_event(
                source=self.name,
                name="screenshot",
                data={
                    "url": await self.web_controller.get_url(),
                    "screenshot": screenshot_png_name
                }
            )

        ocr_text = (
            await self._get_ocr_text(new_screenshot) if use_ocr is True else ""
        )

        # Return the complete observation
        message_content = ""  # message.content or ""
        page_title = await self.web_controller.get_title()
        encoded_string = base64.b64encode(new_screenshot).decode('utf-8')

        return False, {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{message_content}\n\n{action_description}\n\nHere is a screenshot of [{page_title}]({await self.web_controller.get_url()}). The viewport shows {percent_visible}% of the webpage, and is positioned {position_text}.{page_metadata}\nAutomatic OCR of the page screenshot has detected the following text:\n\n{ocr_text}".strip()
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_string}"
                    }
                }
            ]
        }

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
        assert messages is not None

        history = messages
        # Get the full conversation history
        #history = []
        #for msg in self._oai_messages.values():
        #    history.extend(msg)

        # Ask the page for interactive elements, then prepare the state-of-mark screenshot
        rects = await self.web_controller._get_interactive_rects() 
        viewport = await self.web_controller._get_visual_viewport()
        screenshot = await self.web_controller.take_screenshot()
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(screenshot, rects)

        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot_som" + current_timestamp + ".png"
            som_screenshot.save(os.path.join(self.debug_dir, screenshot_png_name))  # type: ignore
            logger.log_event(
                source=self.name,
                name="screenshot",
                data={
                    "url": await self.web_controller.get_url(),
                    "screenshot": screenshot_png_name
                }
            )
        # What tools are available?
        tools: List[Dict[str, Any]] = [
            TOOL_VISIT_URL,
            TOOL_HISTORY_BACK,
            TOOL_CLICK,
            TOOL_TYPE,
            TOOL_SUMMARIZE_PAGE,
            TOOL_READ_PAGE_AND_ANSWER,
            TOOL_SLEEP,
        ]

        # Can we reach Bing to search?
        # if self._navigation_allow_list("https://www.bing.com/"):
        tools.append(TOOL_WEB_SEARCH)

        # We can scroll up
        if viewport["pageTop"] > 5:
            tools.append(TOOL_PAGE_UP)

        # Can scroll down
        if (viewport["pageTop"] + viewport["height"] + 5) < viewport["scrollHeight"]:
            tools.append(TOOL_PAGE_DOWN)

        # Focus hint
        focused = await self.web_controller._get_focused_rect_id()
        focused_hint = ""
        if focused:
            name = self._target_name(focused, rects)
            if name:
                name = f"(and name '{name}') "

            role = "control"
            try:
                role = rects[focused]["role"]
            except KeyError:
                pass

            focused_hint = f"\nThe {role} with ID {focused} {name}currently has the input focus.\n\n"

        # Everything visible
        visible_targets = "\n".join(self._format_target_list(visible_rects, rects)) + "\n\n"

        # Everything else
        other_targets: List[str] = []
        other_targets.extend(self._format_target_list(rects_above, rects))
        other_targets.extend(self._format_target_list(rects_below, rects))

        if len(other_targets) > 0:
            other_targets_str = (
                "Additional valid interaction targets (not shown) include:\n" + "\n".join(other_targets) + "\n\n"
            )
        else:
            other_targets_str = ""

        # If there are scrollable elements, then add the corresponding tools
        # has_scrollable_elements = False
        # if has_scrollable_elements:
        #    tools.append(TOOL_SCROLL_ELEMENT_UP)
        #    tools.append(TOOL_SCROLL_ELEMENT_DOWN)

        tool_names = "\n".join([t["function"]["name"] for t in tools])

        text_prompt = self._get_screenshot_selection_prompt(await self.web_controller.get_url(), 
        visible_targets, 
        other_targets_str, 
        focused_hint, 
        tool_names)

        # Scale the screenshot for the MLM, and close the original
        scaled_screenshot = som_screenshot.resize((MLM_WIDTH, MLM_HEIGHT))
        som_screenshot.close()
        if self.to_save_screenshots:
            scaled_screenshot.save(os.path.join(self.debug_dir, "screenshot_scaled.png"))  # type: ignore

        # Add the multimodal message for the current state
        # Convert PIL image to base64
        buffer = io.BytesIO()
        scaled_screenshot.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        buffer.close()
        encoded_string = base64.b64encode(image_bytes).decode('utf-8')

        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_string}"
                    }
                }
            ]
        }

        response = await super().a_generate_reply(messages=history + [message]) # system massage

        self.web_controller._last_download = None

        if isinstance(response, str): #TODO: response format
            # Direct text response
            return response
        elif isinstance(response, dict) and "tool_calls" in response:
            # Execute the tool calls
            request_halt, tool_response = await self._execute_tool(response, rects, tool_names)
        
        if tool_response is not None:
            return tool_response
        return None

    async def _summarize_page(
        self,
        question: str | None = None,
        token_limit: int = 100000
    ) -> str:

        page_markdown: str = await self.web_controller._get_page_markdown()

        title: str = await self.web_controller.get_url()
        try:
            title = await self.web_controller.get_title()
        except Exception:
            pass

        # Take a screenshot and scale it
        screenshot = Image.open(io.BytesIO(await self.web_controller.take_screenshot()))
        scaled_screenshot = screenshot.resize((MLM_WIDTH, MLM_HEIGHT))
        screenshot.close()
        
        #ag_image = AGImage.from_pil(scaled_screenshot)
        # Calculate tokens for the image
        #image_tokens = img_utils.num_tokens_from_gpt_image(scaled_screenshot, model=self.client.)
        #token_limit = max(token_limit - image_tokens, 1000)  # Reserve space for image tokens
        
        # Convert image to data URI and clean up
        img_uri = img_utils.pil_to_data_uri(scaled_screenshot)
        scaled_screenshot.close()

        # Prepare messages for summarization
        messages: List[Dict[str,Any]] = [{
            "role": "system", 
            "content": "You are a helpful assistant that can summarize long documents to answer questions."
        }]

        # Prepare the main prompt
        prompt = f"We are visiting the webpage '{title}'. Its full-text content are pasted below, along with a screenshot of the page's current viewport."
        if question is not None:
            prompt += f" Please summarize the webpage into one or two paragraphs with respect to '{question}':\n\n"
        else:
            prompt += " Please summarize the webpage into one or two paragraphs:\n\n"

        # Grow the buffer (which is added to the prompt) until we overflow the context window or run out of lines
        buffer = ""
        for line in re.split(r"([\r\n]+)", page_markdown):
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt + buffer + line
                    },
                    #ag_image
                ]
            }

            # TODO: is something like this possible in ag2
            #remaining = self._model_client.remaining_tokens(messages + [message])
            #if remaining > SCREENSHOT_TOKENS:
            #    buffer += line
            #else:
            #    break

        # Nothing to do
        buffer = buffer.strip()
        if len(buffer) == 0:
            return "Nothing to summarize."

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt + buffer
                },
                {
                    "type": "image_url",
                    "image_url": {"url": img_uri}
                }
            ]
        })

        # Generate the response
        is_valid_response, response = await self.a_generate_oai_reply(messages=messages)
        assert is_valid_response
        assert isinstance(response, str)
        return response

    async def test_set_of_mark(self, url: str | None = None) -> Image.Image:
        """
        Test the set_of_mark functionality by visiting a URL and generating a marked screenshot.
        
        Args:
            url (str): The URL to visit and screenshot
        
        Returns:
            Image.Image: The screenshot with interactive regions marked
        """
        
        # Visit the page
        if url is not None:
            await self.web_controller._visit_page(url)
            await self.web_controller.wait_for_load_state()
            await asyncio.sleep(2)
        
        # Get interactive rects
        rects = await self.web_controller._get_interactive_rects()
        
        # Take screenshot
        screenshot = await self.web_controller.take_screenshot()
        
        # Apply set of mark
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(screenshot, rects)
        
        # Optionally save the screenshot
        if self.to_save_screenshots and self.debug_dir:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot_som" + current_timestamp + ".png"
            som_screenshot.save(os.path.join(self.debug_dir, screenshot_png_name))
            
            logger.log_event(
                source=self.name,
                name="screenshot",
                data={
                    "url": await self.web_controller.get_url(),
                    "screenshot": screenshot_png_name
                }
            )
        
        return som_screenshot

    async def manual_tool_execute(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        use_ocr: bool = True
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """
        Manually execute a tool for testing purposes.
        
        Args:
            tool_name (str): Name of the tool to execute
            tool_args (Dict[str, Any]): Arguments for the tool
            use_ocr (bool, optional): Whether to use OCR. Defaults to True.
        
        Returns:
            Tuple[bool, Union[str, Dict, None]]: Result of tool execution
        """

        # Get interactive rects for context
        rects = await self.web_controller._get_interactive_rects()

        # Prepare a tool call dictionary similar to what would be generated by the LLM
        tool_call = {
            "tool_calls": [{
                "function": {
                    "name": tool_name,
                    "arguments": tool_args
                }
            }]
        }

        # Get available tool names for validation
        tools: List[Dict[str, Any]] = [
            TOOL_VISIT_URL,
            TOOL_HISTORY_BACK,
            TOOL_CLICK,
            TOOL_TYPE,
            TOOL_SUMMARIZE_PAGE,
            TOOL_READ_PAGE_AND_ANSWER,
            TOOL_SLEEP,
            TOOL_WEB_SEARCH,
        ]
        tool_names = "\n".join([t["function"]["name"] for t in tools])

        # Execute the tool
        request_halt, tool_response = await self._execute_tool(
            tool_call, 
            rects, 
            tool_names, 
            use_ocr=use_ocr
        )

        return request_halt, tool_response

    async def _get_ocr_text(
        self, 
        image: bytes | io.BufferedIOBase | Image.Image
    ) -> str:
        scaled_screenshot = None
        if isinstance(image, Image.Image):
            scaled_screenshot = image.resize((MLM_WIDTH, MLM_HEIGHT))
        else:
            pil_image = None
            if isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image))
            else:
                pil_image = Image.open(cast(BinaryIO, image))
            scaled_screenshot = pil_image.resize((MLM_WIDTH, MLM_HEIGHT))
            pil_image.close()

        img_uri = img_utils.pil_to_data_uri(scaled_screenshot)
        await asyncio.sleep(0.1)  # Small delay
        scaled_screenshot.close()  

        messages: List[Dict[str,Any]] = [{
                "role": "system",
                "content": "You are a helpful assistant that returns text from an image.",
            }] 

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please transcribe all visible text on this page, including both main content and the labels of UI elements."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_uri
                        },
                    }
                ]
            }
        )
        

        
        is_valid_response, response = await self.a_generate_oai_reply(messages=messages)

        assert is_valid_response
        assert isinstance(response, str)
        return response
