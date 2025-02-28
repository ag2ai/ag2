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
    import requests
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager

_logger = logging.getLogger(__name__)

# The supported file extensions in a URL, align with docling ingestion
SUPPORTED_URLFILE_EXTENSIONS = [
    ".html",
    ".htm",
    ".md",
    ".pdf",
    ".docx",
    ".csv",
    ".pptx",
    ".png",
    ".jpeg",
    ".tiff",
    ".bmp",
    ".jpg",
    ".xlsx",
]


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

    try:
        # Open a page
        driver.get(url)

        # Get the rendered HTML
        html_content = driver.page_source
        return str(html_content)

    finally:
        # Close the browser
        driver.quit()


@require_optional_import(["requests"], "rag")
def _download_binary_file(url: str, output_dir: Path) -> Path:
    """Downloads a file directly from the given URL.

    Works only for file types defined in SUPPORTED_URLFILE_EXTENSIONS.
    Uses appropriate mode (binary/text) based on file extension or content type.

    Args:
        url (str): URL of the file to download.
        output_dir (Path): Directory to save the file.

    Returns:
        Path: Path to the saved file.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Follow redirects and get final URL and headers
    try:
        # Set a longer timeout for link services that might be slow
        head_response = requests.head(url, allow_redirects=True, timeout=30)
        content_type = head_response.headers.get("Content-Type", "").lower()

        # Get the final URL after following redirects
        final_url = head_response.url
        _logger.info(f"Original URL: {url}")
        _logger.info(f"Final URL after redirects: {final_url}")
        _logger.info(f"Detected content type: {content_type}")

        # If the final URL is different, use it for further processing
        if final_url != url:
            url = final_url
    except Exception as e:
        _logger.warning(f"Failed to follow redirects: {e}")
        content_type = ""
        final_url = url

    # Parse URL components from the final URL
    parsed_url = urlparse(final_url)
    url_path = Path(parsed_url.path)

    # Extract filename and extension from URL
    filename = url_path.name
    suffix = url_path.suffix.lower()

    # Check if the extension is directly supported
    if suffix and suffix in SUPPORTED_URLFILE_EXTENSIONS:
        _logger.info(f"Found supported extension in URL: {suffix}")
    elif suffix:
        # We have an extension but it's not supported
        raise ValueError(f"File extension {suffix} is not in the supported list: {SUPPORTED_URLFILE_EXTENSIONS}")

    # For URLs without proper filename/extension, or with generic content types like application/octet-stream
    # we need to do more investigation
    if not suffix or "." not in filename or content_type == "application/octet-stream":
        # Map content types to our supported extensions
        mime_to_ext_mapping = {
            "image/jpeg": ".jpeg",
            "image/jpg": ".jpeg",
            "image/png": ".png",
            "image/tiff": ".tiff",
            "image/bmp": ".bmp",
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "text/csv": ".csv",
            "text/markdown": ".md",
            "text/html": ".html",
        }

        # For application/octet-stream, try to get more information
        if content_type == "application/octet-stream":
            # Check if there's a Content-Disposition header which might have filename info
            content_disp = head_response.headers.get("Content-Disposition", "")
            if "filename=" in content_disp:
                try:
                    # Extract filename from Content-Disposition
                    import re

                    filename_match = re.search(r'filename=["\']?([^"\';]+)', content_disp)
                    if filename_match:
                        detected_filename = filename_match.group(1)
                        detected_suffix = Path(detected_filename).suffix.lower()
                        if detected_suffix in SUPPORTED_URLFILE_EXTENSIONS:
                            suffix = detected_suffix
                            filename = detected_filename
                            _logger.info(f"Found filename in Content-Disposition: {filename} with extension {suffix}")
                except Exception as e:
                    _logger.warning(f"Error parsing Content-Disposition: {e}")

            # If we still don't have a supported extension, try to sample the content
            if not suffix or suffix not in SUPPORTED_URLFILE_EXTENSIONS:
                try:
                    # Download first few bytes to try to detect file type
                    sample_response = requests.get(url, stream=True, timeout=30)
                    sample_bytes = b""
                    for chunk in sample_response.iter_content(chunk_size=1024):
                        sample_bytes += chunk
                        if len(sample_bytes) >= 1024:
                            break

                    # Check for common file signatures
                    if sample_bytes.startswith(b"%PDF"):
                        suffix = ".pdf"
                        _logger.info("Detected PDF from file signature")
                    elif sample_bytes[0:4] == b"\x50\x4b\x03\x04":  # PK magic number for ZIP files
                        # This could be docx, xlsx, pptx (all Office Open XML formats)
                        # Default to PDF for now, later we can add more specific detection
                        suffix = ".pdf"  # Default for unknown
                        _logger.info("Detected ZIP-based format (possibly Office document)")

                    # Close the sample response to avoid keeping the connection open
                    sample_response.close()
                except Exception as e:
                    _logger.warning(f"Error sampling file content: {e}")

        # Get extension based on content type
        ext = None
        if suffix and suffix in SUPPORTED_URLFILE_EXTENSIONS:
            ext = suffix
        else:
            for mime, extension in mime_to_ext_mapping.items():
                if mime in content_type:
                    ext = extension
                    break

        # If we still couldn't determine a supported extension, raise an error
        if not ext:
            raise ValueError(
                f"Content type '{content_type}' does not map to any supported file type: {SUPPORTED_URLFILE_EXTENSIONS}"
            )

        # If we didn't get a filename from the URL or Content-Disposition
        if not filename or filename == "":
            # Create filename using URL hash for uniqueness
            unique_id = abs(hash(url)) % 10000

            # Prefix based on content type
            prefix = "image" if ext in [".jpeg", ".png", ".tiff", ".bmp"] else "download"
            filename = f"{prefix}_{unique_id}{ext}"
        elif not filename.endswith(ext):
            # Ensure filename has the correct extension
            filename = f"{Path(filename).stem}{ext}"

        _logger.info(f"Created filename: {filename} for URL: {url}")

    # Create final filepath
    filepath = output_dir / filename
    _logger.info(f"Saving to: {filepath}")

    # Determine if this is binary or text based on extension
    text_extensions = [".md", ".txt", ".csv", ".html", ".htm"]
    is_binary = suffix not in text_extensions

    # Download with appropriate mode
    try:
        if not is_binary:
            _logger.info(f"Downloading as text file: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            _logger.info(f"Downloading as binary file: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
    except Exception as e:
        _logger.error(f"Download failed: {e}")
        raise

    return filepath


def download_url(url: Any, output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Download the content of a URL and save it as a file.

    For direct file URLs (.md, .pdf, .docx, etc.), downloads the raw file.
    For web pages without file extensions or .html/.htm extensions, uses Selenium to render the content.
    """
    url = str(url)
    url_path = Path(urlparse(url).path)
    output_dir = Path(output_dir) if output_dir else Path()

    # Get the file extension (if any)
    suffix = url_path.suffix.lower()

    # Determine if we should use direct download (vs. Selenium)
    needs_rendering = False

    # No file extension or specifically .html/.htm - might need rendering
    if not suffix or suffix in [".html", ".htm"]:
        # Check content type to confirm
        try:
            head_response = requests.head(url, allow_redirects=True, timeout=10)
            content_type = head_response.headers.get("Content-Type", "").lower()
            # Only use rendering for HTML content types
            needs_rendering = "text/html" in content_type
        except Exception:
            # If HEAD request fails, assume it's a webpage that needs rendering
            needs_rendering = True

    # If it has a supported extension that's not .html/.htm, download directly
    if suffix and suffix in SUPPORTED_URLFILE_EXTENSIONS and suffix not in [".html", ".htm"]:
        return _download_binary_file(url=url, output_dir=output_dir)

    # If it needs rendering, use Selenium
    if needs_rendering:
        rendered_html = _download_rendered_html(url)

        # Determine filename
        filename = url_path.name or "downloaded_content.html"
        if not suffix:
            filename += ".html"

        filepath = output_dir / filename
        with open(file=filepath, mode="w", encoding="utf-8") as f:
            f.write(rendered_html)
        return filepath

    # Otherwise, download directly
    return _download_binary_file(url=url, output_dir=output_dir)


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
        return [download_url(url=input_path, output_dir=output_dir)]

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
