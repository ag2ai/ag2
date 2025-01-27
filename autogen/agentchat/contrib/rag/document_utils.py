# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager

_logger = logging.getLogger(__name__)


def is_url(url: str) -> bool:
    """Check if the string is a valid URL.

    It checks whether the URL has a valid scheme and network location.
    """
    try:
        result = urlparse(url)
        # urlparse will not raise an exception for invalid URLs, so we need to check the components
        return_bool = bool(result.scheme and result.netloc)
        if not return_bool:
            _logger.error(f"Error when checking if {url} is a valid URL: Invalid URL.")
        return return_bool
    except Exception as e:
        _logger.error(f"Error when checking if {url} is a valid URL: {e}")
        return False


@require_optional_import(["selenium", "webdriver_manager"], "rag")
def _download_rendered_html(url: str) -> str:
    """Downloads a rendered HTML page of a given URL using headless ChromeDriver.

    Args:
        url (str): URL of the page to download.

    Returns:
        str: The rendered HTML content of the page.
    """
    # Set up Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Enable headless mode
    options.add_argument("--disable-gpu")  # Disabling GPU hardware acceleration
    options.add_argument("--no-sandbox")  # Bypass OS security model
    options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

    # Set the location of the ChromeDriver
    service = ChromeService(ChromeDriverManager().install())

    # Create a new instance of the Chrome driver with specified options
    driver = webdriver.Chrome(service=service, options=options)

    # Open a page
    driver.get(url)

    # Get the rendered HTML
    html_content = driver.page_source

    # Close the browser
    driver.quit()

    return html_content  # type: ignore[no-any-return]


def download_url(url: Any, output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Download the content of a URL and save it as an HTML file."""
    url = str(url)
    rendered_html = _download_rendered_html(url)
    url_path = Path(urlparse(url).path)
    if url_path.suffix and url_path.suffix != ".html":
        raise ValueError("Only HTML files can be downloaded directly.")

    filename = url_path.name or "downloaded_content.html"
    if len(filename) < 5 or filename[-5:] != ".html":
        filename += ".html"
    output_dir = Path(output_dir) if output_dir else Path()
    filepath = output_dir / filename
    with filepath.open("w", encoding="utf-8") as f:
        f.write(rendered_html)

    return filepath


def list_files(directory: Union[Path, str]) -> list[Path]:
    """Recursively list all files in a directory.

    This function will raise an exception if the directory does not exist.
    """
    path = Path(directory)

    if not path.is_dir():
        raise ValueError(f"The directory {directory} does not exist.")

    return [f for f in path.rglob("*") if f.is_file()]


@export_module("autogen.agentchat.contrib.rag")
def handle_input(input_path: Union[Path, str], output_dir: Optional[Union[Path, str]] = None) -> list[Path]:
    """Process the input string and return the appropriate file paths"""
    input_path = Path(input_path)

    if is_url(str(input_path)):
        _logger.info("Detected URL. Downloading content...")
        return [download_url(url=input_path, output_dir=output_dir)]
    elif input_path.is_dir():
        _logger.info("Detected directory. Listing files...")
        return list_files(directory=input_path)
    elif input_path.is_file():
        _logger.info("Detected file. Returning file path...")
        return [Path(input_path)]
    else:
        raise ValueError("The input provided is neither a URL, directory, nor a file path.")
