from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import pathlib
import random
import re
import time
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from PIL import Image
from playwright._impl._errors import Error as PlaywrightError
from playwright._impl._errors import TimeoutError
from playwright.async_api import BrowserContext, Download, Page, Playwright, async_playwright

from .types import InteractiveRegion, VisualViewport, interactiveregion_from_dict, visualviewport_from_dict
from autogen.logger import FileLogger

logger = FileLogger(config={})

class WebController:
    """Controls web browsing operations using Playwright."""
    
    DEFAULT_START_PAGE = "https://www.bing.com/"
    VIEWPORT_HEIGHT = 900
    VIEWPORT_WIDTH = 1440
    MLM_HEIGHT = 765
    MLM_WIDTH = 1224

    def __init__(self, downloads_folder: Optional[str] = None, debug_dir: str = os.getcwd()):
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._last_download: Optional[Download] = None
        self._prior_metadata_hash: Optional[str] = None
        self.downloads_folder = downloads_folder
        self.debug_dir = debug_dir
        
        # Read page_script
        self._page_script: str = ""
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            self._page_script = fh.read()

    async def init(
        self,
        headless: bool = True,
        browser_channel: Optional[str] = None,
        browser_data_dir: Optional[str] = None,
        start_page: Optional[str] = None
    ) -> None:
        """Initialize the web controller with browser settings."""
        self.start_page = start_page or self.DEFAULT_START_PAGE

        # Create the playwright instance
        launch_args: Dict[str, Any] = {"headless": headless}
        if browser_channel:
            launch_args["channel"] = browser_channel
        self._playwright = await async_playwright().start()

        # Create the context
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        if browser_data_dir:
            self._context = await self._playwright.chromium.launch_persistent_context(
                browser_data_dir,
                user_agent=f"{ua} {self._generate_random_string(4)}",
                **launch_args
            )
        else:
            browser = await self._playwright.chromium.launch(**launch_args)
            self._context = await browser.new_context(
                user_agent=f"{ua} {self._generate_random_string(4)}",
                locale=random.choice(['en-US', 'en-GB', 'en-CA']),
                timezone_id=random.choice(['America/New_York', 'Europe/London', 'Asia/Tokyo'])
            )

        # Set up the page
        self._context.set_default_timeout(60000)
        self._page = await self._context.new_page()
        assert self._page is not None
        self._page.on("download", self._download_handler)
        await self._page.set_viewport_size({"width": self.VIEWPORT_WIDTH, "height": self.VIEWPORT_HEIGHT})
        await self._page.add_init_script(self._page_script)
        await self._page.goto(self.start_page)
        await self._page.wait_for_load_state()

    def _download_handler(self, download: Download) -> None:
        """Handle file downloads."""
        self._last_download = download

    async def visit_page(self, url: str) -> None:
        """Visit a webpage."""
        assert self._page is not None
        try:
            await self._page.goto(url)
            await self._page.wait_for_load_state()
            self._prior_metadata_hash = None
            
        except Exception as e_outer:
            if self.downloads_folder and "net::ERR_ABORTED" in str(e_outer):
                async with self._page.expect_download() as download_info:
                    try:
                        await self._page.goto(url)
                    except Exception as e_inner:
                        if "net::ERR_ABORTED" not in str(e_inner):
                            raise e_inner
                    download = await download_info.value
                    fname = os.path.join(self.downloads_folder, download.suggested_filename)
                    await download.save_as(fname)
                    message = f"<body style=\"margin: 20px;\"><h1>Successfully downloaded '{download.suggested_filename}' to local path:<br><br>{fname}</h1></body>"
                    await self._page.goto(
                        "data:text/html;base64," + base64.b64encode(message.encode("utf-8")).decode("utf-8")
                    )
                    self._last_download = None
            else:
                raise e_outer

    async def get_page_state(self) -> Dict[str, Any]:
        """Get current page state including interactive elements and viewport info."""
        assert self._page is not None
        
        try:
            await self._page.wait_for_load_state()
        except TimeoutError:
            pass

        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass

        rects = await self._get_interactive_rects()
        viewport = await self._get_visual_viewport()
        screenshot = await self._page.screenshot()
        
        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "rects": rects,
            "viewport": viewport,
            "screenshot": screenshot
        }

    async def _get_interactive_rects(self) -> Dict[str, InteractiveRegion]:
        """Get interactive elements on the page."""
        assert self._page is not None
        result = await self._page.evaluate("MultimodalWebSurfer.getInteractiveRects();")
        typed_results: Dict[str, InteractiveRegion] = {}
        for k in result:
            typed_results[k] = interactiveregion_from_dict(result[k])
        return typed_results

    async def _get_visual_viewport(self) -> VisualViewport:
        """Get viewport information."""
        assert self._page is not None
        return visualviewport_from_dict(await self._page.evaluate("MultimodalWebSurfer.getVisualViewport();"))

    async def perform_action(self, action_type: str, params: Dict[str, Any]) -> None:
        """Perform a browser action."""
        assert self._page is not None
        
        if action_type == "click":
            await self._click_id(params["target_id"])
        elif action_type == "type":
            await self._fill_id(params["input_field_id"], params["text_value"])
        elif action_type == "scroll":
            if params["direction"] == "up":
                await self._page_up()
            else:
                await self._page_down()
        elif action_type == "back":
            await self._page.go_back()
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def _click_id(self, identifier: str) -> None:
        """Click an element by ID."""
        assert self._page is not None
        assert self._context is not None
        
        identifier = str(int(identifier.strip()))
        locator = self._page.locator(f"[__elementId='{identifier}']")
        await locator.wait_for(timeout=1000)
        await locator.scroll_into_view_if_needed()
        
        try:
            async with self._context.expect_page(timeout=6000) as page_info:
                await locator.click(delay=10)
                new_page = await page_info.value
                await new_page.wait_for_load_state('domcontentloaded')
                await self._on_new_page(new_page)
        except TimeoutError:
            await self._page.wait_for_load_state()

    async def _fill_id(self, identifier: str, value: str) -> None:
        """Fill a form field by ID."""
        assert self._page is not None
        locator = self._page.locator(f"[__elementId='{identifier}']")
        await locator.wait_for(timeout=1000)
        await locator.scroll_into_view_if_needed()
        await locator.focus()
        try:
            await locator.fill(value)
        except PlaywrightError:
            await locator.press_sequentially(value)
        await locator.press("Enter")

    async def _page_up(self) -> None:
        """Scroll page up."""
        assert self._page is not None
        await self._page.evaluate(f"window.scrollBy(0, -{self.VIEWPORT_HEIGHT-50});")

    async def _page_down(self) -> None:
        """Scroll page down."""
        assert self._page is not None
        await self._page.evaluate(f"window.scrollBy(0, {self.VIEWPORT_HEIGHT-50});")

    async def _on_new_page(self, page: Page) -> None:
        """Handle new page creation."""
        self._page = page
        self._page.on("download", self._download_handler)
        await self._page.set_viewport_size({"width": self.VIEWPORT_WIDTH, "height": self.VIEWPORT_HEIGHT})
        await self._page.add_init_script(self._page_script)
        try:
            await self._page.wait_for_load_state()
        except TimeoutError:
            pass

    def _generate_random_string(self, length: int) -> str:
        """Generate a random string."""
        import string
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    async def get_page_state(self) -> Dict[str, Any]:
        """Get current page state including title, URL and other metadata."""
        assert self._page is not None
        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "focused_id": await self._get_focused_element_id(),
            "metadata": await self._get_page_metadata()
        }

    async def get_page_content(self) -> str:
        """Get page HTML content."""
        assert self._page is not None
        return await self._page.evaluate("document.documentElement.outerHTML;")

    async def take_screenshot(self) -> bytes:
        """Take a screenshot of the current page."""
        assert self._page is not None
        return await self._page.screenshot()

    async def evaluate_script(self, script: str) -> Any:
        """Evaluate JavaScript on the page."""
        assert self._page is not None
        return await self._page.evaluate(script)

    async def scroll_element(self, identifier: str, direction: str) -> None:
        """Scroll an element up or down."""
        assert self._page is not None
        await self._page.evaluate(
            f"""
        (function() {{
            let elm = document.querySelector("[__elementId='{identifier}']");
            if (elm) {{
                if ("{direction}" == "up") {{
                    elm.scrollTop = Math.max(0, elm.scrollTop - elm.clientHeight);
                }}
                else {{
                    elm.scrollTop = Math.min(elm.scrollHeight - elm.clientHeight, elm.scrollTop + elm.clientHeight);
                }}
            }}
        }})();
        """
        )

    async def wait_for_load(self) -> None:
        """Wait for page to load, handling timeouts gracefully."""
        assert self._page is not None
        try:
            await self._page.wait_for_load_state()
        except TimeoutError:
            pass

    async def _get_focused_element_id(self) -> str:
        """Get ID of currently focused element."""
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        return str(await self._page.evaluate("MultimodalWebSurfer.getFocusedElementId();"))

    async def _get_page_metadata(self) -> Dict[str, Any]:
        """Get page metadata."""
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        result = await self._page.evaluate("MultimodalWebSurfer.getPageMetadata();")
        assert isinstance(result, dict)
        return result

    async def get_page_summary(self, question: str | None = None) -> str:
        """Get a summary of the current page content."""
        assert self._page is not None
        
        title = self._page.url
        try:
            title = await self._page.title()
        except Exception:
            pass

        # Prepare the prompt
        prompt = f"We are visiting the webpage '{title}'. Its full-text content are pasted below, along with a screenshot of the page's current viewport."
        if question is not None:
            prompt += f" Please summarize the webpage into one or two paragraphs with respect to '{question}':\n\n"
        else:
            prompt += " Please summarize the webpage into one or two paragraphs:\n\n"

        return prompt

    async def get_ocr_text(self, image: bytes | io.BufferedIOBase | Image.Image) -> str:
        """Extract text from image using OCR."""
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
        return img_uri

    async def cleanup(self) -> None:
        """Clean up browser resources."""
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()
