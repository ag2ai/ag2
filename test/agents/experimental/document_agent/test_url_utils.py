# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import requests

from autogen.agents.experimental.document_agent.url_utils import ExtensionToFormat, InputFormat, URLAnalyzer


# Tests for InputFormat enum and ExtensionToFormat mapping
class TestFormatMapping:
    def test_input_format_enum(self) -> None:
        """Test that InputFormat enum has all expected values"""
        assert InputFormat.DOCX.value == "docx"
        assert InputFormat.PPTX.value == "pptx"
        assert InputFormat.HTML.value == "html"
        assert InputFormat.XML.value == "xml"
        assert InputFormat.IMAGE.value == "image"
        assert InputFormat.PDF.value == "pdf"
        assert InputFormat.ASCIIDOC.value == "asciidoc"
        assert InputFormat.MD.value == "md"
        assert InputFormat.CSV.value == "csv"
        assert InputFormat.XLSX.value == "xlsx"
        assert InputFormat.JSON.value == "json"
        assert InputFormat.INVALID.value == "invalid"

    def test_extension_to_format_mapping(self) -> None:
        """Test that ExtensionToFormat mapping works correctly for various extensions"""
        # Common formats
        assert ExtensionToFormat["pdf"] == InputFormat.PDF
        assert ExtensionToFormat["docx"] == InputFormat.DOCX
        assert ExtensionToFormat["pptx"] == InputFormat.PPTX
        assert ExtensionToFormat["html"] == InputFormat.HTML
        assert ExtensionToFormat["md"] == InputFormat.MD
        assert ExtensionToFormat["json"] == InputFormat.JSON
        assert ExtensionToFormat["csv"] == InputFormat.CSV

        # Image formats
        assert ExtensionToFormat["jpg"] == InputFormat.IMAGE
        assert ExtensionToFormat["jpeg"] == InputFormat.IMAGE
        assert ExtensionToFormat["png"] == InputFormat.IMAGE
        assert ExtensionToFormat["tiff"] == InputFormat.IMAGE

        # Invalid/unsupported formats
        assert ExtensionToFormat["doc"] == InputFormat.INVALID
        assert ExtensionToFormat["ppt"] == InputFormat.INVALID
        assert ExtensionToFormat["xls"] == InputFormat.INVALID


# Tests for URLAnalyzer class
class TestURLAnalyzer:
    @pytest.fixture
    def pdf_url(self) -> str:
        return "https://example.com/document.pdf"

    @pytest.fixture
    def html_url(self) -> str:
        return "https://example.com/webpage.html"

    @pytest.fixture
    def no_extension_url(self) -> str:
        return "https://example.com/page"

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        mock = MagicMock()
        mock.status_code = 200
        mock.headers = {"Content-Type": "application/pdf"}
        mock.history = []
        mock.url = "https://example.com/document.pdf"
        return mock

    def test_init(self, pdf_url: str) -> None:
        """Test URLAnalyzer initialization"""
        analyzer = URLAnalyzer(pdf_url)
        assert analyzer.url == pdf_url
        assert analyzer.analysis_result is None
        assert analyzer.final_url is None
        assert analyzer.redirect_chain == []

    def test_analyze_by_extension_pdf(self, pdf_url: str) -> None:
        """Test URL analysis by extension for PDF files"""
        analyzer = URLAnalyzer(pdf_url)
        result = analyzer._analyze_by_extension(pdf_url)
        assert result["is_file"] is True
        assert result["file_type"] == InputFormat.PDF
        assert result["extension"] == "pdf"

    def test_analyze_by_extension_html(self, html_url: str) -> None:
        """Test URL analysis by extension for HTML files"""
        analyzer = URLAnalyzer(html_url)
        result = analyzer._analyze_by_extension(html_url)
        assert result["is_file"] is True
        assert result["file_type"] == InputFormat.HTML
        assert result["extension"] == "html"

    def test_analyze_by_extension_no_extension(self, no_extension_url: str) -> None:
        """Test URL analysis by extension for URLs without extensions"""
        analyzer = URLAnalyzer(no_extension_url)
        result = analyzer._analyze_by_extension(no_extension_url)
        assert result["is_file"] is False
        assert result["file_type"] is None
        assert result["extension"] is None

    @patch("requests.head")
    def test_analyze_by_request_pdf(self, mock_head: MagicMock, pdf_url: str, mock_response: MagicMock) -> None:
        """Test URL analysis by making an HTTP request for PDF"""
        mock_head.return_value = mock_response
        analyzer = URLAnalyzer(pdf_url)
        result = analyzer._analyze_by_request()
        assert result is not None
        assert result["is_file"] is True
        assert result["file_type"] == InputFormat.PDF
        assert result["mime_type"] == "application/pdf"

    @patch("requests.head")
    def test_analyze_by_request_method_not_allowed(
        self, mock_head: MagicMock, pdf_url: str, mock_response: MagicMock
    ) -> None:
        """Test fallback to GET when HEAD method is not allowed (405)"""
        # Set up mock for 405 response
        mock_head.return_value = MagicMock()
        mock_head.return_value.status_code = 405

        # Set up mock for GET request after 405
        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_response
            analyzer = URLAnalyzer(pdf_url)
            result = analyzer._analyze_by_request()

            assert result is not None
            assert result["is_file"] is True
            assert result["file_type"] == InputFormat.PDF
            assert result["mime_type"] == "application/pdf"

            # Verify GET was called after HEAD returned 405
            mock_get.assert_called_once()

    @patch("requests.head")
    def test_analyze_by_request_connection_error(self, mock_head: MagicMock, pdf_url: str) -> None:
        """Test handling of connection errors during URL analysis"""
        mock_head.side_effect = requests.exceptions.ConnectionError("Connection refused")
        analyzer = URLAnalyzer(pdf_url)
        result = analyzer._analyze_by_request()
        assert result is not None
        assert result["is_file"] is False
        assert result["file_type"] == InputFormat.INVALID
        assert "Connection error" in result["error"]

    @patch("requests.head")
    def test_analyze_by_request_timeout(self, mock_head: MagicMock, pdf_url: str) -> None:
        """Test handling of timeout errors during URL analysis"""
        mock_head.side_effect = requests.exceptions.Timeout("Request timed out")
        analyzer = URLAnalyzer(pdf_url)
        result = analyzer._analyze_by_request()
        assert result is not None
        assert result["is_file"] is False
        assert result["file_type"] == InputFormat.INVALID
        assert "timed out" in result["error"]

    @patch("requests.head")
    def test_analyze_by_request_too_many_redirects(self, mock_head: MagicMock, pdf_url: str) -> None:
        """Test handling of too many redirects during URL analysis"""
        mock_head.side_effect = requests.exceptions.TooManyRedirects("Too many redirects")
        analyzer = URLAnalyzer(pdf_url)
        result = analyzer._analyze_by_request()
        assert result is not None
        assert result["is_file"] is False
        assert result["file_type"] == InputFormat.INVALID
        assert "Too many redirects" in result["error"]
        assert result["redirects"] is True

    @patch("requests.head")
    def test_analyze_by_request_with_redirects(self, mock_head: MagicMock, pdf_url: str) -> None:
        """Test URL analysis with redirects"""
        # Create a response with redirect history
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}

        redirect1 = MagicMock()
        redirect1.url = "https://example.com/redirect1"
        redirect2 = MagicMock()
        redirect2.url = "https://example.com/redirect2"

        mock_response.history = [redirect1, redirect2]
        mock_response.url = "https://example.com/final.pdf"

        mock_head.return_value = mock_response

        analyzer = URLAnalyzer(pdf_url)
        result = analyzer._analyze_by_request(follow_redirects=True)

        assert result is not None
        assert result["is_file"] is True
        assert result["file_type"] == InputFormat.PDF
        assert analyzer.redirect_chain == ["https://example.com/redirect1", "https://example.com/redirect2"]
        assert analyzer.final_url == "https://example.com/final.pdf"

    @patch("autogen.agents.experimental.document_agent.url_utils.URLAnalyzer._analyze_by_extension")
    @patch("autogen.agents.experimental.document_agent.url_utils.URLAnalyzer._analyze_by_request")
    def test_analyze_prioritize_extension(
        self, mock_request: MagicMock, mock_extension: MagicMock, pdf_url: str
    ) -> None:
        """Test that analyze() prioritizes extension over MIME type when specified"""
        # Extension analysis returns PDF
        mock_extension.return_value = {"is_file": True, "file_type": InputFormat.PDF, "extension": "pdf"}

        # Request analysis returns different MIME type
        mock_request.return_value = {
            "is_file": True,
            "file_type": InputFormat.HTML,  # Different from extension
            "mime_type": "text/html",
        }

        analyzer = URLAnalyzer(pdf_url)
        result = analyzer.analyze(test_url=True, prioritize_extension=True)

        # Should use file type from extension but keep MIME type from request
        assert result is not None
        assert result["is_file"] is True
        assert result["file_type"] == InputFormat.PDF  # From extension
        assert result["mime_type"] == "text/html"  # From request

    @patch("autogen.agents.experimental.document_agent.url_utils.URLAnalyzer._analyze_by_extension")
    @patch("autogen.agents.experimental.document_agent.url_utils.URLAnalyzer._analyze_by_request")
    def test_analyze_prioritize_request(self, mock_request: MagicMock, mock_extension: MagicMock, pdf_url: str) -> None:
        """Test that analyze() prioritizes MIME type over extension when specified"""
        # Extension analysis returns PDF
        mock_extension.return_value = {"is_file": True, "file_type": InputFormat.PDF, "extension": "pdf"}

        # Request analysis returns different MIME type
        mock_request.return_value = {"is_file": True, "file_type": InputFormat.HTML, "mime_type": "text/html"}

        analyzer = URLAnalyzer(pdf_url)
        result = analyzer.analyze(test_url=True, prioritize_extension=False)

        # Should use file type from request
        assert result is not None
        assert result["is_file"] is True
        assert result["file_type"] == InputFormat.HTML  # From request
        assert result["mime_type"] == "text/html"  # From request

    def test_get_result_before_analyze(self, pdf_url: str) -> None:
        """Test get_result() before analyze() is called"""
        analyzer = URLAnalyzer(pdf_url)
        assert analyzer.get_result() is None

    @patch("autogen.agents.experimental.document_agent.url_utils.URLAnalyzer._analyze_by_extension")
    @patch("autogen.agents.experimental.document_agent.url_utils.URLAnalyzer._analyze_by_request")
    def test_get_result_after_analyze(self, mock_request: MagicMock, mock_extension: MagicMock, pdf_url: str) -> None:
        """Test get_result() after analyze() is called"""
        mock_extension.return_value = {"is_file": True, "file_type": InputFormat.PDF, "extension": "pdf"}
        mock_request.return_value = None

        analyzer = URLAnalyzer(pdf_url)
        analyzer.analyze()

        result = analyzer.get_result()
        assert result is not None
        assert result is not None
        assert result["is_file"] is True
        assert result["file_type"] == InputFormat.PDF

    def test_get_redirect_info_no_redirects(self, pdf_url: str) -> None:
        """Test get_redirect_info() when no redirects have occurred"""
        analyzer = URLAnalyzer(pdf_url)

        redirect_info = analyzer.get_redirect_info()
        assert redirect_info["redirects"] is False
        assert redirect_info["redirect_count"] == 0
        assert redirect_info["original_url"] == pdf_url
        assert redirect_info["final_url"] == pdf_url
        assert redirect_info["redirect_chain"] == []

    @patch("requests.head")
    def test_follow_redirects(self, mock_head: MagicMock, pdf_url: str) -> None:
        """Test follow_redirects() method"""
        # Create a response with redirect history
        mock_response = MagicMock()
        mock_response.status_code = 200

        redirect1 = MagicMock()
        redirect1.url = "https://example.com/redirect1"
        redirect2 = MagicMock()
        redirect2.url = "https://example.com/redirect2"

        mock_response.history = [redirect1, redirect2]
        mock_response.url = "https://example.com/final.pdf"

        mock_head.return_value = mock_response

        analyzer = URLAnalyzer(pdf_url)
        final_url, redirect_chain = analyzer.follow_redirects()

        assert final_url == "https://example.com/final.pdf"
        assert redirect_chain == ["https://example.com/redirect1", "https://example.com/redirect2"]
        assert analyzer.redirect_chain == ["https://example.com/redirect1", "https://example.com/redirect2"]
        assert analyzer.final_url == "https://example.com/final.pdf"

    @patch("requests.head")
    def test_follow_redirects_error(self, mock_head: MagicMock, pdf_url: str) -> None:
        """Test follow_redirects() when an error occurs"""
        mock_head.side_effect = Exception("Connection error")

        analyzer = URLAnalyzer(pdf_url)
        final_url, redirect_chain = analyzer.follow_redirects()

        # Should return original URL on error
        assert final_url == pdf_url
        assert redirect_chain == []

    def test_class_methods_for_formats(self) -> None:
        """Test class methods for retrieving supported formats, MIME types, and extensions"""
        formats = URLAnalyzer.get_supported_formats()
        mime_types = URLAnalyzer.get_supported_mime_types()
        extensions = URLAnalyzer.get_supported_extensions()

        # Check formats
        assert InputFormat.PDF in formats
        assert InputFormat.DOCX in formats
        assert InputFormat.HTML in formats
        assert InputFormat.IMAGE in formats
        assert InputFormat.INVALID not in formats  # Invalid is not a supported format

        # Check MIME types
        assert "application/pdf" in mime_types
        assert "text/html" in mime_types
        assert "image/png" in mime_types
        assert "text/markdown" in mime_types

        # Check extensions
        assert "pdf" in extensions
        assert "docx" in extensions
        assert "html" in extensions
        assert "jpg" in extensions
        assert "md" in extensions
        assert "doc" in extensions  # Invalid extensions are also in the mapping
