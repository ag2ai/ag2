import base64
import io
import os
import random
import time
import pathlib
import aiofiles
from typing import Any, Dict, Union, cast

from playwright.async_api import BrowserContext, Download, Page, Playwright, async_playwright
from playwright._impl._errors import TimeoutError, Error as PlaywrightError

# Viewport dimensions
VIEWPORT_WIDTH = 1440
VIEWPORT_HEIGHT = 900

from .types import InteractiveRegion, VisualViewport, visualviewport_from_dict, interactiveregion_from_dict
from .markdown_browser import MarkdownConverter

from autogen.logger import FileLogger

# Initialize logger with config
logger = FileLogger(config={})

from .types import SentinelMeta

# Sentinels
class DEFAULT_CHANNEL(metaclass=SentinelMeta):
    pass

class WebController:
    """
    A class to encapsulate the browser capabilities and interactions using Playwright.
    """
    DEFAULT_START_PAGE = "https://www.bing.com/"

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
        self.start_page = start_page or self.DEFAULT_START_PAGE
        self.downloads_folder = downloads_folder
        self.to_save_screenshots = to_save_screenshots
        self.debug_dir = debug_dir
        self._last_download: Download | None = None
        self._prior_metadata_hash: str | None = None

        ## Create or use the provided MarkdownConverter
        if markdown_converter is None:
            self._markdown_converter = MarkdownConverter()  # type: ignore
        else:
            self._markdown_converter = markdown_converter  # type: ignore

        # Read page_script
        self._page_script: str = ""
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            self._page_script = fh.read()

        # Define the download handler
        def _download_handler(download: Download) -> None:
            self._last_download = download

        self._download_handler = _download_handler

        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self.agent_name: str = ""

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
        agent_name: str = ""
    ) -> None:
        # Create the playwright self
        launch_args: Dict[str, Any] = {"headless": headless}
        if browser_channel is not DEFAULT_CHANNEL: 
            launch_args["channel"] = browser_channel
        self._playwright = await async_playwright().start()

        # Create the context -- are we launching persistent?
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
                #viewport={'width': 1920 + random.randint(-50, 50), 'height': 1080 + random.randint(-30, 30)},
                locale=random.choice(['en-US', 'en-GB', 'en-CA']),
                timezone_id=random.choice(['America/New_York', 'Europe/London', 'Asia/Tokyo'])
            )

        # Create the page
        self._context.set_default_timeout(60000)  # One minute
        self._page = await self._context.new_page()
        assert self._page is not None
        # self._page.route(lambda x: True, self._route_handler)
        self._page.on("download", self._download_handler)
        await self._page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
        await self._page.add_init_script(
            path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js")
        )
        await self._page.goto(self.start_page)
        await self._page.wait_for_load_state()

        # Prepare the debug directory -- which stores the screenshots generated throughout the process
        await self._set_debug_dir(debug_dir)
    
    def _generate_random_string(self, length: int) -> str:
        """Generate a random string of specified length.

        Args:
            length (int): Length of the random string to generate

        Returns:
            str: A random string containing letters and numbers
        """
        import string
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    async def _set_debug_dir(self, debug_dir: str) -> None:
        assert self._page is not None
        self.debug_dir = debug_dir
        if self.debug_dir == "":
            return

        if not os.path.isdir(self.debug_dir):
            os.mkdir(self.debug_dir)
        current_timestamp = "_" + int(time.time()).__str__()
        screenshot_png_name = "screenshot" + current_timestamp + ".png"
        debug_html = os.path.join(self.debug_dir, "screenshot" + current_timestamp + ".html")
        if self.to_save_screenshots:
            async with aiofiles.open(debug_html, "wt") as file:
                await file.write(
                    f"""
    <html style="width:100%; margin: 0px; padding: 0px;">
    <body style="width: 100%; margin: 0px; padding: 0px;">
        <img src= {screenshot_png_name} id="main_image" style="width: 100%; max-width: {VIEWPORT_WIDTH}px; margin: 0px; padding: 0px;">
        <script language="JavaScript">
    var counter = 0;
    setInterval(function() {{
    counter += 1;
    document.getElementById("main_image").src = "screenshot.png?bc=" + counter;
    }}, 300);
        </script>
    </body>
    </html>
    """.strip(),
                )
        if self.to_save_screenshots:
            await self._page.screenshot(path=os.path.join(self.debug_dir, screenshot_png_name))
            logger.log_event(
                source=self.agent_name,
                name="screenshot",
                data={
                    "url": self._page.url,
                    "screenshot": screenshot_png_name
                }
            )

            logger.log_event(
                source=self.agent_name,
                name="debug_screens",
                data={
                    "text": "Multimodal Web Surfer debug screens",
                    "url": pathlib.Path(os.path.abspath(debug_html)).as_uri()
                }
            )
    
    async def _sleep(self, duration: Union[int, float]) -> None:
        assert self._page is not None
        await self._page.wait_for_timeout(duration * 1000)
    
    async def _reset(self) -> None:
        assert self._page is not None        
        await self._visit_page(self.start_page)
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            await self._page.screenshot(path=os.path.join(self.debug_dir, screenshot_png_name))
            
            logger.log_event(
                source=self.agent_name,
                name="screenshot",
                data={
                    "url": self._page.url,
                    "screenshot": screenshot_png_name
                }
            )

        logger.log_event(
            source=self.agent_name,
            name="reset",
            data={
                "text": "Resetting browser.",
                "url": self._page.url
            }
        )

    async def _back(self) -> None:
        assert self._page is not None
        await self._page.go_back()

    async def _visit_page(self, url: str) -> None:
        assert self._page is not None
        try:
            # Regular webpage
            await self._page.goto(url)
            await self._page.wait_for_load_state()
            self._prior_metadata_hash = None
            
        except Exception as e_outer:
            # Downloaded file
            if self.downloads_folder and "net::ERR_ABORTED" in str(e_outer):
                async with self._page.expect_download() as download_info:
                    try:
                        await self._page.goto(url)
                    except Exception as e_inner:
                        if "net::ERR_ABORTED" in str(e_inner):
                            pass
                        else:
                            raise e_inner
                    download = await download_info.value
                    fname = os.path.join(self.downloads_folder, download.suggested_filename)
                    await download.save_as(fname)
                    message = f"<body style=\"margin: 20px;\"><h1>Successfully downloaded '{download.suggested_filename}' to local path:<br><br>{fname}</h1></body>"
                    await self._page.goto(
                        "data:text/html;base64," + base64.b64encode(message.encode("utf-8")).decode("utf-8")
                    )
                    self._last_download = None  # Since we already handled it
            else:
                raise e_outer

    async def _page_down(self) -> None:
        assert self._page is not None
        await self._page.evaluate(f"window.scrollBy(0, {VIEWPORT_HEIGHT-50});")

    async def _page_up(self) -> None:
        assert self._page is not None
        await self._page.evaluate(f"window.scrollBy(0, -{VIEWPORT_HEIGHT-50});")

    async def _click_id(self, identifier: str) -> None:
        assert self._page is not None
        assert self._context is not None
        
        # Ensure identifier is a valid integer string
        try:
            identifier = str(int(identifier.strip()))
        except ValueError:
            raise ValueError(f"Invalid element identifier: {identifier}")
        
        if not identifier:
            raise ValueError("Empty element identifier")
        
        # Try multiple locator strategies
        locator_strategies = [
            f"[__elementId='{identifier}']",
            f"[__elementId={identifier}]",
        ]
        
        for strategy in locator_strategies:
            try:
                target = self._page.locator(strategy)
                await target.wait_for(timeout=1000)
                
                # If found, proceed with click
                await target.scroll_into_view_if_needed()
                box = cast(Dict[str, Union[int, float]], await target.bounding_box())
                
                # Track current page state before click
                current_url = self._page.url
                current_pages = len(self._context.pages)
                
                # Click with a short timeout to catch immediate popups
                popup_detected = False
                try:
                    async with self._context.expect_page(timeout=6000) as page_info:
                        await target.click(delay=10)
                        popup_detected = True
                        try:
                            new_page = await page_info.value
                            await new_page.wait_for_load_state('domcontentloaded', timeout=4000)
                            await self._on_new_page(new_page)
                            
                            logger.log_event(
                                source=self.agent_name,
                                name="popup",
                                data={
                                    "url": new_page.url,
                                    "text": "New tab or window opened"
                                }
                            )
                            return
                        except TimeoutError:
                            logger.log_event(
                                source=self.agent_name,
                                name="popup_timeout",
                                data={
                                    "text": "Popup detected but timed out waiting for load"
                                }
                            )
                            # Continue to check for delayed popups
                except TimeoutError:
                    # No immediate popup, check for navigation or delayed popups
                    try:
                        await self._page.wait_for_load_state()
                    except TimeoutError:
                        pass
                    
                    # Check for any new pages that appeared
                    if len(self._context.pages) > current_pages:
                        new_page = self._context.pages[-1]
                        await new_page.wait_for_load_state('domcontentloaded')
                        await self._on_new_page(new_page)
                        
                        logger.log_event(
                            source=self.agent_name,
                            name="delayed_popup",
                            data={
                                "url": new_page.url,
                                "text": "New tab or window opened after delay"
                            }
                        )
                        return
                    
                    # Check if current page navigated
                    new_url = self._page.url
                    if new_url != current_url:
                        logger.log_event(
                            source=self.agent_name,
                            name="page_navigation",
                            data={
                                "old_url": current_url,
                                "new_url": new_url,
                                "text": "Page navigated after click"
                            }
                        )
                    
                    # Update page state
                    await self._get_interactive_rects()
                    return
            
            except TimeoutError:
                # Try next strategy
                continue
        
        raise ValueError(f"Could not find element with ID {identifier}. Page may have changed.")
    
    async def _fill_id(self, identifier: str, value: str) -> None:
        assert self._page is not None
        
        # Try multiple locator strategies
        locator_strategies = [
            f"[__elementId='{identifier}']",
            f"[__elementId={identifier}]",
        ]
        
        for strategy in locator_strategies:
            try:
                target = self._page.locator(strategy)
                await target.wait_for(timeout=1000)
                
                # Fill it
                await target.scroll_into_view_if_needed()
                await target.focus()
                try:
                    await target.fill(value)
                except PlaywrightError:
                    await target.press_sequentially(value)
                await target.press("Enter")
                return
            
            except TimeoutError:
                # Try next strategy
                continue
        
        raise ValueError(f"Could not find element with ID {identifier}. Page may have changed.")

    async def _scroll_id(self, identifier: str, direction: str) -> None:
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

    async def _get_interactive_rects(self) -> Dict[str, InteractiveRegion]:
        assert self._page is not None

        # Ensure page is fully loaded
        try:
            await self._page.wait_for_load_state()
        except TimeoutError:
            pass

        # Read the regions from the DOM
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass

        result = cast(
            Dict[str, Dict[str, Any]], await self._page.evaluate("MultimodalWebSurfer.getInteractiveRects();")
        )

        # Convert the results into appropriate types
        assert isinstance(result, dict)
        typed_results: Dict[str, InteractiveRegion] = {}
        for k in result:
            assert isinstance(k, str)
            typed_results[k] = interactiveregion_from_dict(result[k])

        return typed_results

    async def _get_visual_viewport(self) -> VisualViewport:
        assert self._page is not None
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        return visualviewport_from_dict(await self._page.evaluate("MultimodalWebSurfer.getVisualViewport();"))

    async def _get_focused_rect_id(self) -> str:
        assert self._page is not None
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        result = await self._page.evaluate("MultimodalWebSurfer.getFocusedElementId();")
        return str(result)

    async def _get_page_metadata(self) -> Dict[str, Any]:
        assert self._page is not None
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        result = await self._page.evaluate("MultimodalWebSurfer.getPageMetadata();")
        assert isinstance(result, dict)
        return cast(Dict[str, Any], result)

    async def _get_page_markdown(self) -> str:
        assert self._page is not None
        html = await self._page.evaluate("document.documentElement.outerHTML;")
        # TOODO: fix types
        res = self._markdown_converter.convert_stream(io.StringIO(html), file_extension=".html", url=self._page.url)  # type: ignore
        return res.text_content  # type: ignore

    async def _on_new_page(self, page: Page) -> None:
        self._page = page
        assert self._page is not None
        # self._page.route(lambda x: True, self._route_handler)
        self._page.on("download", self._download_handler)
        await self._page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
        await self._sleep(0.2)
        self._prior_metadata_hash = None
        await self._page.add_init_script(
            path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js")
        )
        try:
            await self._page.wait_for_load_state()
        except TimeoutError:
            pass
    
    async def take_screenshot(self) -> bytes:
        """Takes a screenshot of the current page."""
        assert self._page is not None
        return await self._page.screenshot()

    async def get_url(self) -> str:
        """Returns the current URL of the page."""
        assert self._page is not None
        return self._page.url
    
    async def get_title(self) -> str:
        """Returns the title of the current page."""
        assert self._page is not None
        return await self._page.title()

    async def wait_for_load_state(self) -> None:
        assert self._page is not None
        try:
            await self._page.wait_for_load_state()
        except TimeoutError:
            logger.log_event(
                source=self.agent_name,
                name="page_load_timeout",
                data={
                    "url": self._page.url
                }
            )

    async def handle_download(self, downloads_folder: str) -> None:
        """Handles the download of a file initiated by the browser."""
        assert self._page is not None
        if self._last_download is not None:
            fname = os.path.join(downloads_folder, self._last_download.suggested_filename)
            await self._last_download.save_as(fname)  # type: ignore
            page_body = f"<html><head><title>Download Successful</title></head><body style=\"margin: 20px;\"><h1>Successfully downloaded '{self._last_download.suggested_filename}' to local path:<br><br>{fname}</h1></body></html>"
            await self._page.goto(
                "data:text/html;base64," + base64.b64encode(page_body.encode("utf-8")).decode("utf-8")
            )
            await self._page.wait_for_load_state()
            self._last_download = None # Reset last download