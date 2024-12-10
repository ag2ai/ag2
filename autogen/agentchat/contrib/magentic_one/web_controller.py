import base64
import io
import os
import random
from typing import Any, Dict, Optional, Union, cast, TYPE_CHECKING

from playwright.async_api import BrowserContext, Download, Page, Playwright, async_playwright
from playwright._impl._errors import TimeoutError, Error as PlaywrightError


# Viewport dimensions
VIEWPORT_WIDTH = 1440
VIEWPORT_HEIGHT = 900

from .types import InteractiveRegion, VisualViewport, visualviewport_from_dict, interactiveregion_from_dict
from .markdown_browser import MarkdownConverter
from .set_of_mark import add_set_of_mark

from autogen.logger import FileLogger

# Initialize logger with config
logger = FileLogger(config={})

from .utils import SentinelMeta

# Sentinels
class DEFAULT_CHANNEL(metaclass=SentinelMeta):
    pass


class BaseBrowserController:
    def __init__(self):
        self._page = None
        self._browser = None

    async def init(
        self,
        headless: bool = True,
        browser_channel: str | type["DEFAULT_CHANNEL"] = "chrome",
        browser_data_dir: str | None = None,
        start_page: str | None = None
    ) -> None:
        raise NotImplementedError("Subclass must implement init method")

    async def visit_page(self, url: str):
        raise NotImplementedError("Subclass must implement visit_page method")

    async def click_element(self, selector: str):
        raise NotImplementedError("Subclass must implement click_element method")

    async def fill_input(self, selector: str, text: str):
        raise NotImplementedError("Subclass must implement fill_input method")

    async def scroll_page(self, direction: str, amount: int):
        raise NotImplementedError("Subclass must implement scroll_page method")

    async def take_screenshot(self):
        raise NotImplementedError("Subclass must implement take_screenshot method")

    async def get_page_metadata(self):
        raise NotImplementedError("Subclass must implement get_page_metadata method")

    async def get_interactive_rects(self):
        raise NotImplementedError("Subclass must implement get_interactive_rects method")

class PlaywrightBrowserController(BaseBrowserController):
    def __init__(
        self, 
        downloads_folder: Optional[str] = None, 
        markdown_converter: Optional[Any] = None,
        debug_dir: str = os.getcwd(),
        to_save_screenshots: bool = False,
        agent_name: str = ""
    ):
        super().__init__()
        self.downloads_folder = downloads_folder
        self.debug_dir = debug_dir
        self.to_save_screenshots = to_save_screenshots
        self._markdown_converter = markdown_converter or MarkdownConverter()
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._last_download: Optional[Download] = None
        self._prior_metadata_hash: Optional[str] = None
        self.name = agent_name # TODO: should this just be passed with the method ? 
        
        # Read page_script
        self._page_script: str = ""
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            self._page_script = fh.read()

        # Define the download handler
        def _download_handler(download: Download) -> None:
            self._last_download = download

        self._download_handler = _download_handler

    async def init(
        self,
        headless: bool = True,
        browser_channel: str | type[DEFAULT_CHANNEL] = DEFAULT_CHANNEL,
        browser_data_dir: str | None = None,
        start_page: str | None = None
    ) -> None:
        """
        Initialize the Playwright browser context.

        Args:
            headless (bool): Whether to run the browser in headless mode. Defaults to True.
            browser_channel (str | type[DEFAULT_CHANNEL]): The browser channel to use. Defaults to DEFAULT_CHANNEL.
            browser_data_dir (str | None): The directory to store browser data. Defaults to None.
            start_page (str | None): The initial page to visit. Defaults to None.
        """
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
                locale=random.choice(['en-US', 'en-GB', 'en-CA']),
                timezone_id=random.choice(['America/New_York', 'Europe/London', 'Asia/Tokyo'])
            )

        # Create the page
        self._context.set_default_timeout(60000)  # One minute
        self._page = await self._context.new_page()
        assert self._page is not None
        self._page.on("download", self._download_handler)
        await self._page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
        await self._page.add_init_script(
            path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js")
        )
        
        if start_page:
            await self._page.goto(start_page)
            await self._page.wait_for_load_state()

    def _generate_random_string(self, length: int) -> str:
        """Generate a random string of specified length."""
        import string
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    #async def close(self) -> None:
    #    """Close the browser and clean up resources."""
    #    await self._browser_controller.close()

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

    async def _back(self) -> None:
        assert self._page is not None
        await self._page.go_back()

    async def _show_download_success_page(self, filename: str, filepath: str) -> None:
        """Show a success page after download completion."""
        assert self._page is not None
        page_body = f"""<html>
            <head><title>Download Successful</title></head>
            <body style="margin: 20px;">
                <h1>Successfully downloaded '{filename}' to local path:<br><br>{filepath}</h1>
            </body>
        </html>"""
        await self._page.goto(
            "data:text/html;base64," + base64.b64encode(page_body.encode("utf-8")).decode("utf-8")
        )

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
                    await self._show_download_success_page(download.suggested_filename, fname)
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
                                source=self.name,
                                name="popup",
                                data={
                                    "url": new_page.url,
                                    "text": "New tab or window opened"
                                }
                            )
                            return
                        except TimeoutError:
                            logger.log_event(
                                source=self.name,
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
                            source=self.name,
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
                            source=self.name,
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

    async def _sleep(self, duration: Union[int, float]) -> None:
        assert self._page is not None
        await self._page.wait_for_timeout(duration * 1000)


    async def _get_interactive_rects(self) -> Dict[str, InteractiveRegion]:
        """Get all interactive regions from the current page."""
        assert self._page is not None

        try:
            # Ensure page is loaded and script is injected
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

    async def take_screenshot(self, path=None):
        assert self._page is not None
        return await self._page.screenshot(path=path)

    def get_url(self) -> str:
        """Get the current page URL synchronously."""
        assert self._page is not None
        return self._page.url

    async def get_title(self):
        assert self._page is not None
        return await self._page.title()

    async def wait_for_load_state(self):
        assert self._page is not None
        await self._page.wait_for_load_state()
