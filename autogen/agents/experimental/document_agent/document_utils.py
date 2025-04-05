# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from .url_utils import ExtensionToFormat, InputFormat, URLAnalyzer

with optional_import_block():
    import requests
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager

__all__ = ["handle_input", "preprocess_path"]

_logger = logging.getLogger(__name__)


class QueryType(Enum):
    RAG_QUERY = "RAG_QUERY"
    # COMMON_QUESTION = "COMMON_QUESTION"


class Ingest(BaseModel):
    path_or_url: str = Field(description="The path or URL of the documents to ingest.")


class Query(BaseModel):
    query_type: QueryType = Field(description="The type of query to perform for the Document Agent.")
    query: str = Field(description="The query to perform for the Document Agent.")


def is_url(url: str) -> bool:
    """Check if the string is a valid URL.

    It checks whether the URL has a valid scheme and network location.
    """
    try:
        url = url.strip()
        result = urlparse(url)
        # urlparse will not raise an exception for invalid URLs, so we need to check the components
        return_bool = bool(result.scheme and result.netloc)
        return return_bool
    except Exception:
        return False


@require_optional_import(["selenium", "webdriver_manager", "requests"], "rag")
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

    try:
        # Open a page
        driver.get(url)

        # Get the rendered HTML
        html_content = driver.page_source
        return str(html_content)

    finally:
        # Close the browser
        driver.quit()


@require_optional_import(["requests", "selenium", "webdriver_manager"], "rag")
def _download_binary_file(url: str, output_dir: Path) -> Path:
    """Downloads a file directly from the given URL.

    Uses appropriate mode (binary/text) based on file extension or content type.

    Args:
        url (str): URL of the file to download.
        output_dir (Path): Directory to save the file.

    Returns:
        Path: Path to the saved file.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use URLAnalyzer to get information about the URL
    analyzer = URLAnalyzer(url)
    analysis = analyzer.analyze(test_url=True, follow_redirects=True)

    # Get file info
    final_url = analysis.get("final_url", url)
    file_type = analysis.get("file_type")
    content_type = analysis.get("mime_type", "")

    _logger.info(f"Original URL: {url}")
    _logger.info(f"Final URL after redirects: {final_url}")
    _logger.info(f"Detected content type: {content_type}")
    _logger.info(f"Detected file type: {file_type}")

    # Check if the file type is supported
    if file_type == InputFormat.INVALID:
        raise ValueError(f"File type is not supported: {analysis}")

    # Parse URL components from the final URL
    parsed_url = urlparse(final_url)
    path = Path(parsed_url.path)

    # Extract filename and extension from URL
    filename = path.name
    suffix = path.suffix.lower()

    # For URLs without proper filename/extension, or with generic content types
    if not filename or not suffix:
        # Create a unique filename
        unique_id = abs(hash(url)) % 10000

        # Determine extension from file type
        if file_type is not None and isinstance(file_type, InputFormat):
            ext = _get_extension_from_file_type(file_type, content_type)
        else:
            ext = None

        # Create filename
        prefix = "image" if file_type == InputFormat.IMAGE else "download"
        filename = f"{prefix}_{unique_id}{ext}"

    # Ensure the filename has the correct extension
    if suffix:
        # Check if the extension is valid for the file type
        current_ext = suffix[1:] if suffix.startswith(".") else suffix
        if file_type is not None and isinstance(file_type, InputFormat):
            if not _is_valid_extension_for_file_type(current_ext, file_type):
                # If not, add the correct extension
                ext = _get_extension_from_file_type(file_type, content_type)
                filename = f"{Path(filename).stem}{ext}"
        else:
            ext = _get_extension_from_file_type(InputFormat.INVALID, content_type)
            filename = f"{Path(filename).stem}{ext}"
    else:
        # No extension, add one based on file type
        if file_type is not None and isinstance(file_type, InputFormat):
            ext = _get_extension_from_file_type(file_type, content_type)
        else:
            ext = _get_extension_from_file_type(InputFormat.INVALID, content_type)
        filename = f"{filename}{ext}"

    _logger.info(f"Using filename: {filename} for URL: {url}")

    # Create final filepath
    filepath = output_dir / filename

    # Determine if this is binary or text based on extension
    suffix = Path(filename).suffix.lower()
    text_extensions = [".md", ".txt", ".csv", ".html", ".htm", ".xml", ".json", ".adoc"]
    is_binary = suffix not in text_extensions

    # Download with appropriate mode
    try:
        if not is_binary:
            _logger.info(f"Downloading as text file: {final_url}")
            response = requests.get(final_url, timeout=30)
            response.raise_for_status()

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            _logger.info(f"Downloading as binary file: {final_url}")
            response = requests.get(final_url, stream=True, timeout=30)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
    except Exception as e:
        _logger.error(f"Download failed: {e}")
        raise

    return filepath


def _get_extension_from_file_type(file_type: InputFormat, content_type: str = "") -> str:
    """Get a file extension based on the file type and content type."""
    # Create a reverse mapping from InputFormat to a default extension
    # We choose the first extension found for each format
    format_to_extension = {}
    for ext, fmt in ExtensionToFormat.items():
        if fmt not in format_to_extension:
            format_to_extension[fmt] = ext

    # Special case for images: use content type to determine exact image format
    if file_type == InputFormat.IMAGE:
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpeg"
        elif "png" in content_type:
            return ".png"
        elif "tiff" in content_type:
            return ".tiff"
        elif "bmp" in content_type:
            return ".bmp"
        # Fallback to default image extension
        ext = format_to_extension.get(InputFormat.IMAGE, "png")
        return f".{ext}"

    # For all other formats, use the default extension
    if file_type in format_to_extension:
        return f".{format_to_extension[file_type]}"

    return ".bin"  # Default for unknown types


def _is_valid_extension_for_file_type(extension: str, file_type: InputFormat) -> bool:
    """Check if the extension is valid for the given file type."""
    # Remove leading dot if present
    if extension.startswith("."):
        extension = extension[1:]

    # Check if the extension is in URLAnalyzer.ExtensionToFormat
    # and if it maps to the given file type
    return extension in ExtensionToFormat and ExtensionToFormat[extension] == file_type


@require_optional_import(["selenium", "webdriver_manager", "requests"], "rag")
def download_url(url: Any, output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Download the content of a URL and save it as a file.

    For direct file URLs (.md, .pdf, .docx, etc.), downloads the raw file.
    For web pages without file extensions or .html/.htm extensions, uses Selenium to render the content.
    """
    url = str(url)
    output_dir = Path(output_dir) if output_dir else Path()

    # Use URLAnalyzer to determine what type of file the URL is
    analyzer = URLAnalyzer(url)
    analysis = analyzer.analyze(test_url=True, follow_redirects=True)

    # Log the analysis result
    _logger.info(f"URL analysis result: {analysis}")

    # Get the final URL after redirects
    final_url = analysis.get("final_url", url)

    # Determine the file type
    is_file = analysis.get("is_file", False)
    file_type = analysis.get("file_type")

    # If it's a direct file URL (not HTML), download it directly
    if is_file and file_type != InputFormat.HTML and file_type != InputFormat.INVALID:
        _logger.info("Detected direct file URL. Downloading...")
        return _download_binary_file(url=final_url, output_dir=output_dir)

    # If it's a web page, use Selenium to render it
    if file_type == InputFormat.HTML or not is_file:
        _logger.info("Detected web page. Rendering...")
        rendered_html = _download_rendered_html(final_url)

        # Determine filename
        parsed_url = urlparse(final_url)
        path = Path(parsed_url.path)
        filename = path.name or "downloaded_content.html"
        if not filename.endswith(".html"):
            filename += ".html"

        # Save the rendered HTML
        filepath = output_dir / filename
        with open(file=filepath, mode="w", encoding="utf-8") as f:
            f.write(rendered_html)

        return filepath

    # Otherwise, try to download as a binary file
    _logger.info("Unknown URL type. Trying to download as binary file...")
    return _download_binary_file(url=final_url, output_dir=output_dir)


def list_files(directory: Union[Path, str]) -> list[Path]:
    """Recursively list all files in a directory.

    This function will raise an exception if the directory does not exist.
    """
    path = Path(directory)

    if not path.is_dir():
        raise ValueError(f"The directory {directory} does not exist.")

    return [f for f in path.rglob("*") if f.is_file()]


@export_module("autogen.agents.experimental.document_agent")
def handle_input(input_path: Union[Path, str], output_dir: Union[Path, str] = "./output") -> list[Path]:
    """Process the input string and return the appropriate file paths"""

    output_dir = preprocess_path(str_or_path=output_dir, is_dir=True, mk_path=True)
    if isinstance(input_path, str) and is_url(input_path):
        _logger.info("Detected URL. Downloading content...")
        try:
            return [download_url(url=input_path, output_dir=output_dir)]
        except Exception as e:
            raise e

    if isinstance(input_path, str):
        input_path = Path(input_path)
    if not input_path.exists():
        raise ValueError("The input provided does not exist.")
    elif input_path.is_dir():
        _logger.info("Detected directory. Listing files...")
        return list_files(directory=input_path)
    elif input_path.is_file():
        _logger.info("Detected file. Returning file path...")
        return [input_path]
    else:
        raise ValueError("The input provided is neither a URL, directory, nor a file path.")


@export_module("autogen.agents.experimental.document_agent")
def preprocess_path(
    str_or_path: Union[Path, str], mk_path: bool = False, is_file: bool = False, is_dir: bool = True
) -> Path:
    """Preprocess the path for file operations.

    Args:
        str_or_path (Union[Path, str]): The path to be processed.
        mk_path (bool, optional): Whether to create the path if it doesn't exist. Default is True.
        is_file (bool, optional): Whether the path is a file. Default is False.
        is_dir (bool, optional): Whether the path is a directory. Default is True.

    Returns:
        Path: The preprocessed path.
    """

    # Convert the input to a Path object if it's a string
    temp_path = Path(str_or_path)

    # Ensure the path is absolute
    absolute_path = temp_path.absolute()
    absolute_path = absolute_path.resolve()
    if absolute_path.exists():
        return absolute_path

    # Check if the path should be a file or directory
    if is_file and is_dir:
        raise ValueError("Path cannot be both a file and a directory.")

    # If mk_path is True, create the directory or parent directory
    if mk_path:
        if is_file and not absolute_path.parent.exists():
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
        elif is_dir and not absolute_path.exists():
            absolute_path.mkdir(parents=True, exist_ok=True)

    # Perform checks based on is_file and is_dir flags
    if is_file and not absolute_path.is_file():
        raise FileNotFoundError(f"File not found: {absolute_path}")
    elif is_dir and not absolute_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {absolute_path}")

    return absolute_path
