# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from autogen.agents.experimental.document_agent.document_utils import (
    _download_rendered_html,
    download_url,
    handle_input,
    is_url,
    list_files,
)
from autogen.import_utils import optional_import_block, skip_on_missing_imports

with optional_import_block():
    import selenium


class TestIsUrl:
    def test_valid_url(self) -> None:
        url = "https://www.example.com"
        assert is_url(url)

    def test_invalid_url_without_scheme(self) -> None:
        url = "www.example.com"
        assert not is_url(url)

    def test_invalid_url_without_network_location(self) -> None:
        url = "https://"
        assert not is_url(url)

    def test_url_with_invalid_scheme(self) -> None:
        url = "invalid_scheme://www.example.com"
        assert not is_url(url)

    def test_empty_url_string(self) -> None:
        url = ""
        assert not is_url(url)

    def test_url_string_with_whitespace(self) -> None:
        url = " https://www.example.com "
        assert is_url(url)

    def test_url_string_with_special_characters(self) -> None:
        url = "https://www.example.com/path?query=param#fragment"
        assert is_url(url)

    def test_attribute_error(self) -> None:
        url = None
        assert not is_url(url)  # type: ignore[arg-type]


class TestDownloadUrl:
    @pytest.fixture
    def mock_chrome(self) -> Iterable[MagicMock]:
        with patch("selenium.webdriver.Chrome") as mock:
            yield mock

    @pytest.fixture
    def mock_chrome_driver_manager(self) -> Iterable[MagicMock]:
        with patch("webdriver_manager.chrome.ChromeDriverManager.install") as mock:
            yield mock

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_non_string_input(self) -> None:
        url = 123
        with pytest.raises(selenium.common.exceptions.InvalidArgumentException, match="invalid argument"):
            download_url(url)  # type: ignore[arg-type]

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_download_with_valid_url(self, mock_chrome: MagicMock) -> None:
        url = "https://www.google.com"
        mock_chrome.return_value.get.return_value = None
        mock_chrome.return_value.page_source = "<html>Test HTML</html>"
        html_content = _download_rendered_html(url)
        assert isinstance(html_content, str)
        assert html_content != ""

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_download_with_invalid_url(self, mock_chrome: MagicMock) -> None:
        url = "invalid_url"
        mock_chrome.return_value.get.side_effect = ValueError("Invalid URL")
        with pytest.raises(ValueError, match="Invalid URL"):
            _download_rendered_html(url)

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_chrome_driver_not_installed(self, mock_chrome_driver_manager: MagicMock) -> None:
        url = "https://www.google.com"
        mock_chrome_driver_manager.side_effect = ValueError("Chrome driver not installed")
        with pytest.raises(ValueError, match="Chrome driver not installed"):
            _download_rendered_html(url)

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_chrome_driver_connection_error(self, mock_chrome: MagicMock) -> None:
        url = "https://www.google.com"
        mock_chrome.return_value.get.side_effect = ValueError("Connection error")
        with pytest.raises(ValueError, match="Connection error"):
            _download_rendered_html(url)

    @pytest.fixture
    def mock_html_value(self) -> str:
        return "<html>Example</html>"

    @pytest.fixture
    def mock_download(self, mock_html_value: str) -> Iterable[MagicMock]:
        with patch("autogen.agents.experimental.document_agent.document_utils._download_rendered_html") as mock:
            mock.return_value = mock_html_value
            yield mock

    @pytest.fixture
    def mock_open_file(self) -> Iterable[MagicMock]:
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            yield mock_file

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_download_url_valid_html(self, mock_html_value: str, mock_download: str, tmp_path: Path) -> None:
        url = "https://www.example.com/index.html"
        filepath = download_url(url, tmp_path.resolve())
        assert filepath.suffix == ".html"
        with open(file=filepath, mode="r") as html_file:
            content = html_file.read()
            assert content == mock_html_value

    @skip_on_missing_imports(["selenium", "webdriver_manager", "requests"], "rag")
    def test_download_url_non_html(self, tmp_path: Path) -> None:
        # Image URL
        url = "https://picsum.photos/id/237/200/300.jpg"
        file_path = download_url(url, tmp_path.resolve())
        assert file_path.suffix == ".jpg"

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_download_url_no_extension(self, mock_html_value: str, mock_download: str, tmp_path: Path) -> None:
        url = "https://www.example.com/path"
        filepath = download_url(url, str(tmp_path))
        assert filepath.suffix == ".html"
        with open(file=filepath, mode="r") as html_file:
            content = html_file.read()
            assert content == mock_html_value

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_download_url_no_output_dir(
        self, mock_html_value: str, mock_download: str, mock_open_file: MagicMock
    ) -> None:
        url = "https://www.example.com"
        filepath = download_url(url)
        assert filepath.suffix == ".html"
        mock_open_file.assert_called_with(file=filepath, mode="w", encoding="utf-8")
        m_file_handle = mock_open_file()
        m_file_handle.write.assert_called_with(mock_html_value)

    @skip_on_missing_imports(["selenium", "webdriver_manager"], "rag")
    def test_download_url_invalid_url(self) -> None:
        url = "invalid url"
        with patch(
            "autogen.agents.experimental.document_agent.document_utils._download_rendered_html"
        ) as mock_download:
            mock_download.side_effect = Exception("Invalid URL")
            with pytest.raises(Exception):
                download_url(url)

    @pytest.fixture
    def path_with_two_files(self, tmp_path: Path) -> Path:
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        with file1.open("w") as f1, file2.open("w") as f2:
            f1.write("File 1 content")
            f2.write("File 2 content")
        return tmp_path

    def test_list_files(self, path_with_two_files: Path) -> None:
        file_list = list_files(str(path_with_two_files))
        assert set(str(f) for f in file_list) == {str(path_with_two_files / f) for f in ["file1.txt", "file2.txt"]}

    def test_handle_input_directory(self, path_with_two_files: Path) -> None:
        file_list = handle_input(str(path_with_two_files))
        assert set(str(f) for f in file_list) == {str(path_with_two_files / f) for f in ["file1.txt", "file2.txt"]}

    def test_handle_input_file(self, tmp_path: Path) -> None:
        file = tmp_path / "file.txt"
        with file.open("w") as f:
            f.write("File content")
        file_list = handle_input(str(file))
        assert file_list == [file]

    def test_handle_input_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            handle_input("invalid input")
