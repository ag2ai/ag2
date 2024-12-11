from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import pathlib
import random
import re
import time
import traceback
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
    TypeAlias,
    cast,
)
from urllib.parse import quote_plus

import aiofiles

from PIL import Image
from playwright._impl._errors import Error as PlaywrightError
from playwright._impl._errors import TimeoutError
from autogen.agentchat import ConversableAgent, Agent
from autogen.agentchat.contrib import img_utils


# from playwright._impl._async_base.AsyncEventInfo
from playwright.async_api import BrowserContext, Download, Page, Playwright, async_playwright

from .websurfer_prompts import (
    DEFAULT_DESCRIPTION,
    SCREENSHOT_TOOL_SELECTION,
)

# TODO: Fix mdconvert (I think i saw a new pull request)
from .markdown_browser import MarkdownConverter  # type: ignore
from .web_controller import WebController, DEFAULT_CHANNEL
from .utils import SentinelMeta

#from ...utils import message_content_to_str

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
)

from .types import (
    InteractiveRegion,
    VisualViewport,
    interactiveregion_from_dict,
    visualviewport_from_dict,
)

# Viewport dimensions
VIEWPORT_HEIGHT = 900
VIEWPORT_WIDTH = 1440

# Size of the image we send to the MLM
# Current values represent a 0.85 scaling to fit within the GPT-4v short-edge constraints (768px)
MLM_HEIGHT = 765
MLM_WIDTH = 1224

SCREENSHOT_TOKENS = 1105

import logging

from autogen.logger import FileLogger

# Initialize logger with config
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
        args = function["arguments"]
        if isinstance(args, str):
            args = self._clean_and_parse_json(args) 

        name = function["name"]
        assert name is not None

        action_description = ""
        logger.log_event(
            source=self.name,
            name="tool_execution",
            data={
                "url": await self.web_controller.get_url(),
                "tool_name": name,
                "args": args
            }
        )

        if name == "visit_url":
            url = args["url"]
            action_description = f"I typed '{url}' into the browser address bar."
            # Check if the argument starts with a known protocol
            if url.startswith(("https://", "http://", "file://", "about:")):
                await self.web_controller._visit_page(url)
            # If the argument contains a space, treat it as a search query
            elif " " in url:
                await self.web_controller._visit_page(f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH")
            # Otherwise, prefix with https://
            else:
                await self.web_controller._visit_page("https://" + url)

        elif name == "history_back":
            action_description = "I clicked the browser back button."
            await self.web_controller._back()

        elif name == "web_search":
            query = args["query"]
            action_description = f"I typed '{query}' into the browser search bar."
            await self.web_controller._visit_page(f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH")

        elif name == "page_up":
            action_description = "I scrolled up one page in the browser."
            await self.web_controller._page_up()

        elif name == "page_down":
            action_description = "I scrolled down one page in the browser."
            await self.web_controller._page_down()

        elif name == "click":
            target_id = str(args["target_id"]) 
            target_name = self._target_name(target_id, rects)
            if target_name:
                action_description = f"I clicked '{target_name}'."
            else:
                action_description = "I clicked the control."
            await self.web_controller._click_id(target_id)

        elif name == "input_text":
            input_field_id = str(args["input_field_id"])
            text_value = str(args["text_value"])
            input_field_name = self._target_name(input_field_id, rects)
            if input_field_name:
                action_description = f"I typed '{text_value}' into '{input_field_name}'."
            else:
                action_description = f"I input '{text_value}'."
            await self.web_controller._fill_id(input_field_id, text_value)

        elif name == "scroll_element_up":
            target_id = str(args["target_id"])
            target_name = self._target_name(target_id, rects)

            if target_name:
                action_description = f"I scrolled '{target_name}' up."
            else:
                action_description = "I scrolled the control up."

            await self.web_controller._scroll_id(target_id, "up")

        elif name == "scroll_element_down":
            target_id = str(args["target_id"])
            target_name = self._target_name(target_id, rects)

            if target_name:
                action_description = f"I scrolled '{target_name}' down."
            else:
                action_description = "I scrolled the control down."

            await self.web_controller._scroll_id(target_id, "down")

        elif name == "answer_question":
            question = str(args["question"])
            # Do Q&A on the DOM. No need to take further action. Browser state does not change.
            return False, await self._summarize_page(question=question)

        elif name == "summarize_page":
            # Summarize the DOM. No need to take further action. Browser state does not change.
            return False, await self._summarize_page()

        elif name == "sleep":
            action_description = "I am waiting a short period of time before taking further action."
            await self.web_controller._sleep(3)  # There's a 2s sleep below too

        else:
            raise ValueError(f"Unknown tool '{name}'. Please choose from:\n\n{tool_names}")

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
        
        # Register the tools for this interaction
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                self.update_tool_signature({"type": "function", "function": tool["function"]}, is_remove=False)
            else:
                self.update_tool_signature({"type": "function", "function": tool}, is_remove=False)

        response = await super().a_generate_reply(messages=history + [message]) # system massage

        self.web_controller._last_download = None

        if isinstance(response, str): #TODO: response format
            # Direct text response
            return response
        elif isinstance(response, dict):
            if "tool_calls" in response:
                request_halt, tool_response = await self._execute_tool(response, rects, tool_names)
            elif "function_call" in response:
                # Legacy function call handling
                #success, func_response = await self.a_generate_function_call_reply(messages=[response])
                #if success:
                raise Exception("Legacy function call handling not implemented")
                #return await self._execute_tool(response, rects, tool_names)
        
        # Clean up registered tools
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                self.update_tool_signature({"type": "function", "function": tool["function"]}, is_remove=True)
            else:
                self.update_tool_signature({"type": "function", "function": tool}, is_remove=True)
        
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
                # TOODO: Not sure why this cast was needed, but by this point screenshot is a binary file-like object
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

    def _clean_and_parse_json(self, content: str) -> Dict[str, Any]:
        """Clean and parse JSON content from various formats."""
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")

        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            parts = content.split("```json")
            if len(parts) > 1:
                content = parts[1].split("```")[0].strip()
        elif "```" in content:  # Handle cases where json block might not be explicitly marked
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1].strip()  # Take first code block content
        
        # Find JSON-like structure if not in code block
        if not content.strip().startswith('{'):
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                raise ValueError(f"No JSON structure found in content: {content}")
            content = json_match.group(0)

        # Preserve newlines for readability in error messages
        formatted_content = content
        
        # Now clean for parsing
        try:
            # First try parsing the cleaned but formatted content
            return json.loads(formatted_content)
        except json.JSONDecodeError:
            # If that fails, try more aggressive cleaning
            cleaned_content = re.sub(r'[\n\r\t]', ' ', content)  # Replace newlines/tabs with spaces
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)  # Normalize whitespace
            cleaned_content = re.sub(r'\\(?!["\\/bfnrt])', '', cleaned_content)  # Remove invalid escapes
            cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)  # Remove trailing commas
            cleaned_content = re.sub(r'([{,]\s*)(\w+)(?=\s*:)', r'\1"\2"', cleaned_content)  # Quote unquoted keys
            cleaned_content = cleaned_content.replace("'", '"')  # Standardize quotes
            
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.log_event(
                    source=self.name,
                    name="json_error",
                    data={
                        "original_content": formatted_content,
                        "cleaned_content": cleaned_content,
                        "error": str(e)
                    }
                )
                raise ValueError(f"Failed to parse JSON after cleaning. Error: {str(e)}")