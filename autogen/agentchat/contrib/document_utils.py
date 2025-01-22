# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

_logger = logging.getLogger(__name__)


def is_url(url: str) -> bool:
    """
    Check if the string is a valid URL.

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


def _download_rendered_html(url: str) -> str:
    """
    Downloads a rendered HTML page of a given URL using headless ChromeDriver.

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
    service = Service(ChromeDriverManager().install())

    # Create a new instance of the Chrome driver with specified options
    driver = webdriver.Chrome(service=service, options=options)

    # Open a page
    driver.get(url)

    # Get the rendered HTML
    html_content = driver.page_source

    # Close the browser
    driver.quit()

    return html_content


def download_url(url: str, output_dir: str = None) -> Path:
    """Download the content of a URL and save it as an HTML file."""
    rendered_html = _download_rendered_html(url)
    url_path = Path(urlparse(url).path)
    if url_path.suffix and url_path.suffix != ".html":
        raise ValueError("Only HTML files can be downloaded directly.")

    filename = url_path.name or "downloaded_content.html"
    if len(filename) < 5 or filename[-5:] != ".html":
        filename += ".html"
    filepath = os.path.join(output_dir, filename) if output_dir else os.path.join(os.getcwd(), filename)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(rendered_html)

    return Path(filepath)


def list_files(directory: str) -> list[Path]:
    """
    Recursively list all files in a directory.

    This function will raise an exception if the directory does not exist.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist.")

    file_list = []
    for root, _, files in os.walk(directory):
        if root is None:
            raise RuntimeError("The root directory is None.")
        if files is None:
            raise RuntimeError("The list of files is None.")
        for file in files:
            if file is None:
                raise RuntimeError("One of the files is None.")
            file_list.append(Path(os.path.join(root, file)))
    return file_list


def handle_input(input_file_string: str, output_dir: str = None) -> list[Path]:
    """Process the input string and return the appropriate file paths"""
    if is_url(input_file_string):
        _logger.info("Detected URL. Downloading content...")
        return [download_url(url=input_file_string, output_dir=output_dir)]
    elif os.path.isdir(input_file_string):
        _logger.info("Detected directory. Listing files...")
        return list_files(directory=input_file_string)
    elif os.path.isfile(input_file_string):
        _logger.info("Detected file. Returning file path...")
        return [Path(input_file_string)]
    else:
        raise ValueError("The input provided is neither a URL, directory, nor a file path.")
