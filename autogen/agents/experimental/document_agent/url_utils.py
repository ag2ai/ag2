# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

import requests


class InputFormat(Enum):
    """Enum representing supported input file formats."""

    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    XML = "xml"
    IMAGE = "image"
    PDF = "pdf"
    ASCIIDOC = "asciidoc"
    MD = "md"
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    INVALID = "invalid"  # Server errors or not a URL


# Map common file extensions to InputFormat
# INVALID means it's not supported
# See: https://github.com/DS4SD/docling/blob/e25d557c06afd77f1bb2c1ac4d2ece4dffcd52bd/docling/datamodel/base_models.py#L56
ExtensionToFormat = {
    # DOCX formats
    "docx": InputFormat.DOCX,
    "dotx": InputFormat.DOCX,
    "docm": InputFormat.DOCX,
    "dotm": InputFormat.DOCX,
    # PPTX formats
    "pptx": InputFormat.PPTX,
    "potx": InputFormat.PPTX,
    "ppsx": InputFormat.PPTX,
    "pptm": InputFormat.PPTX,
    "potm": InputFormat.PPTX,
    "ppsm": InputFormat.PPTX,
    # Excel formats
    "xlsx": InputFormat.XLSX,
    # HTML formats
    "html": InputFormat.HTML,
    "htm": InputFormat.HTML,
    "xhtml": InputFormat.HTML,
    # XML formats
    "xml": InputFormat.XML,
    "nxml": InputFormat.XML,
    "txt": InputFormat.XML,  # Note: .txt could be many formats, XML is just one possibility
    # Image formats
    "png": InputFormat.IMAGE,
    "jpg": InputFormat.IMAGE,
    "jpeg": InputFormat.IMAGE,
    "tiff": InputFormat.IMAGE,
    "tif": InputFormat.IMAGE,
    "bmp": InputFormat.IMAGE,
    # PDF format
    "pdf": InputFormat.PDF,
    # AsciiDoc formats
    "adoc": InputFormat.ASCIIDOC,
    "asciidoc": InputFormat.ASCIIDOC,
    "asc": InputFormat.ASCIIDOC,
    # Markdown formats
    "md": InputFormat.MD,
    "markdown": InputFormat.MD,
    # CSV format
    "csv": InputFormat.CSV,
    # JSON format
    "json": InputFormat.JSON,
    # Unsupported formats
    "doc": InputFormat.INVALID,
    "ppt": InputFormat.INVALID,
    "xls": InputFormat.INVALID,
    "gif": InputFormat.INVALID,
}


class URLAnalyzer:
    """
    A class that analyzes URLs to determine if they point to web pages or files.
    """

    # Mapping of input formats to their corresponding MIME types
    FormatToMimeType: dict[InputFormat, list[str]] = {
        InputFormat.DOCX: [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
        ],
        InputFormat.PPTX: [
            "application/vnd.openxmlformats-officedocument.presentationml.template",
            "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ],
        InputFormat.HTML: ["text/html", "application/xhtml+xml"],
        InputFormat.XML: ["application/xml", "text/xml", "text/plain"],
        InputFormat.IMAGE: [
            "image/png",
            "image/jpeg",
            "image/tiff",
            "image/gif",
            "image/bmp",
        ],
        InputFormat.PDF: ["application/pdf"],
        InputFormat.ASCIIDOC: ["text/asciidoc"],
        InputFormat.MD: ["text/markdown", "text/x-markdown"],
        InputFormat.CSV: ["text/csv"],
        InputFormat.XLSX: ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
        InputFormat.JSON: ["application/json"],
    }

    # Create a reverse mapping from MIME types to formats
    # Note: For ambiguous MIME types (like "application/xml"), we'll favor the first format found
    MimeTypeToFormat = {}
    for format_type, mime_types in FormatToMimeType.items():
        for mime_type in mime_types:
            if mime_type not in MimeTypeToFormat:
                MimeTypeToFormat[mime_type] = format_type

    def __init__(self, url: str):
        """
        Initialize the URLAnalyzer with a URL.

        Args:
            url (str): The URL to analyze
        """
        self.url = url
        self.analysis_result: Optional[dict[str, Any]] = None
        self.final_url: Optional[str] = None
        self.redirect_chain: list[str] = []

    def analyze(
        self, test_url: bool = False, follow_redirects: bool = True, prioritize_extension: bool = True
    ) -> dict[str, Any]:
        """
        Analyze the URL to determine if it points to a web page or a file.

        Args:
            test_url (bool): Whether to test the URL by making a request
            follow_redirects (bool): Whether to follow redirects when testing the URL
            prioritize_extension (bool): Whether to prioritize file extension over MIME type

        Returns:
            dict: A dictionary containing the analysis results
        """
        result = {
            "url": self.url,
            "is_file": False,
            "file_type": None,
            "mime_type": None,
            "method": "extension_analysis",
            "redirects": False,
            "redirect_count": 0,
            "final_url": self.url,
        }

        # First try to analyze based on the URL extension
        extension_analysis = self._analyze_by_extension(self.url)
        if extension_analysis["is_file"]:
            result.update(extension_analysis)

        # If test_url is True, make a request
        if test_url:
            request_analysis = self._analyze_by_request(follow_redirects)
            if request_analysis:
                # Update the redirect information
                if self.final_url and self.final_url != self.url:
                    result["redirects"] = True
                    result["redirect_count"] = len(self.redirect_chain)
                    result["redirect_chain"] = self.redirect_chain
                    result["final_url"] = self.final_url

                    # Re-analyze based on the final URL extension
                    if self.final_url != self.url:
                        final_extension_analysis = self._analyze_by_extension(self.final_url)
                        if final_extension_analysis["is_file"]:
                            # If prioritizing extension and we have a file extension match
                            if prioritize_extension:
                                # Keep the MIME type from the request but use file type from extension
                                mime_type = request_analysis.get("mime_type")
                                request_analysis.update(final_extension_analysis)
                                if mime_type:
                                    request_analysis["mime_type"] = mime_type
                            else:
                                # Only use extension analysis if request didn't identify a file type
                                if not request_analysis.get("file_type"):
                                    request_analysis.update(final_extension_analysis)

                # If prioritize_extension is True and we have both extension and MIME type analyses
                if (
                    prioritize_extension
                    and result.get("extension")
                    and result.get("file_type")
                    and request_analysis.get("mime_type")
                ):
                    # Keep the extension-based file type but add the MIME type from the request
                    request_analysis["file_type"] = result["file_type"]
                    request_analysis["is_file"] = True

                result.update(request_analysis)
                result["method"] = "request_analysis"

        # Store the result for later access
        self.analysis_result = result

        return result

    def _analyze_by_extension(self, url: str) -> dict[str, Any]:
        """
        Analyze URL based on its file extension.

        Args:
            url (str): The URL to analyze

        Returns:
            dict: Analysis results based on the file extension
        """
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        # Check if the URL has a file extension
        if "." in path:
            extension = path.split(".")[-1]

            # Check if it's a known file extension
            if extension in ExtensionToFormat:
                format_type = ExtensionToFormat[extension]
                return {
                    "is_file": True,
                    "file_type": format_type,
                    "extension": extension,
                }

        # If no file extension was found or it's not recognized,
        # assume it's a web page (but this could be confirmed with a request)
        return {
            "is_file": False,
            "file_type": None,
            "extension": None,
        }

    def _analyze_by_request(self, follow_redirects: bool = True) -> Optional[dict[str, Any]]:
        """
        Analyze URL by making a HEAD request to check Content-Type.

        Args:
            follow_redirects (bool): Whether to follow redirects

        Returns:
            Optional[dict]: Analysis results based on the HTTP response or None if the request failed
        """
        try:
            # Store redirect history
            self.redirect_chain = []

            # First try a HEAD request (faster but some servers don't handle it well)
            response = requests.head(self.url, allow_redirects=follow_redirects, timeout=5)

            # If the server returns a 405 (Method Not Allowed) for HEAD, try GET
            if response.status_code == 405:
                response = requests.get(self.url, allow_redirects=follow_redirects, timeout=5, stream=True)
                # Close the connection without downloading the content
                response.close()

            # Check for non-success status codes
            if response.status_code >= 400:
                return {
                    "is_file": False,
                    "file_type": InputFormat.INVALID,
                    "mime_type": None,
                    "error": f"HTTP error: {response.status_code}",
                    "status_code": response.status_code,
                }

            # Store information about redirects
            if hasattr(response, "history") and response.history:
                self.redirect_chain = [r.url for r in response.history]
                self.final_url = response.url
            else:
                self.final_url = self.url

            # Get the Content-Type header
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()

            # Check if it matches any of our known MIME types
            format_type = self.MimeTypeToFormat.get(content_type)

            # Handle different content types
            if format_type:
                return {
                    "is_file": True,
                    "file_type": format_type,
                    "mime_type": content_type,
                }
            elif content_type in ["text/html", "application/xhtml+xml"]:
                return {
                    "is_file": False,
                    "file_type": None,
                    "mime_type": content_type,
                }
            else:
                # Content type was found but not in our mapping
                return {
                    "is_file": True,
                    "file_type": "unknown",
                    "mime_type": content_type,
                }

        except requests.exceptions.TooManyRedirects:
            # Handle redirect loops or too many redirects
            return {
                "is_file": False,
                "file_type": InputFormat.INVALID,
                "mime_type": None,
                "error": "Too many redirects",
                "redirects": True,
            }
        except requests.exceptions.ConnectionError:
            # Handle connection errors (e.g., DNS failure, refused connection)
            return {
                "is_file": False,
                "file_type": InputFormat.INVALID,
                "mime_type": None,
                "error": "Connection error - URL may be invalid or server unavailable",
            }
        except requests.exceptions.Timeout:
            # Handle timeout
            return {"is_file": False, "file_type": InputFormat.INVALID, "mime_type": None, "error": "Request timed out"}
        except requests.exceptions.InvalidURL:
            # Handle invalid URL
            return {
                "is_file": False,
                "file_type": InputFormat.INVALID,
                "mime_type": None,
                "error": "Invalid URL format",
            }
        except Exception as e:
            # If the request fails for any other reason
            return {"is_file": False, "file_type": InputFormat.INVALID, "mime_type": None, "error": str(e)}

    def get_result(self) -> Optional[dict[str, Any]]:
        """
        Get the last analysis result, or None if the URL hasn't been analyzed yet.

        Returns:
            Optional[dict]: The analysis result or None
        """
        return self.analysis_result

    def get_redirect_info(self) -> dict[str, Any]:
        """
        Get information about redirects that occurred during the last request.

        Returns:
            dict: Information about redirects
        """
        if not self.final_url:
            return {
                "redirects": False,
                "redirect_count": 0,
                "original_url": self.url,
                "final_url": self.url,
                "redirect_chain": [],
            }

        return {
            "redirects": self.url != self.final_url,
            "redirect_count": len(self.redirect_chain),
            "original_url": self.url,
            "final_url": self.final_url,
            "redirect_chain": self.redirect_chain,
        }

    def follow_redirects(self) -> Tuple[str, list[str]]:
        """
        Follow redirects for the URL without analyzing content types.

        Returns:
            Tuple[str, list[str]]: The final URL and the redirect chain
        """
        try:
            response = requests.head(self.url, allow_redirects=True, timeout=5)

            # If the server returns a 405 (Method Not Allowed) for HEAD, try GET
            if response.status_code == 405:
                response = requests.get(self.url, allow_redirects=True, timeout=5, stream=True)
                # Close the connection without downloading the content
                response.close()

            # Update redirect information
            if hasattr(response, "history") and response.history:
                self.redirect_chain = [r.url for r in response.history]
                self.final_url = response.url
            else:
                self.final_url = self.url
                self.redirect_chain = []

            return self.final_url, self.redirect_chain

        except Exception:
            # If the request fails, return the original URL
            return self.url, []

    @classmethod
    def get_supported_formats(cls) -> list[InputFormat]:
        """Return a list of supported file formats."""
        return list(cls.FormatToMimeType.keys())

    @classmethod
    def get_supported_mime_types(cls) -> list[str]:
        """Return a list of all supported MIME types."""
        return list(cls.MimeTypeToFormat.keys())

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Return a list of supported file extensions."""
        return list(ExtensionToFormat.keys())
