# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from autogen.agentchat.contrib.rag.document_utils import (
    _download_rendered_html,
    download_url,
    handle_input,
    is_url,
    list_files,
)


def test_valid_url():
    url = "https://www.example.com"
    assert is_url(url)


def test_invalid_url_without_scheme():
    url = "www.example.com"
    assert not is_url(url)


def test_invalid_url_without_network_location():
    url = "https://"
    assert not is_url(url)


def test_url_with_invalid_scheme():
    url = "invalid_scheme://www.example.com"
    assert not is_url(url)


def test_empty_url_string():
    url = ""
    assert not is_url(url)


def test_url_string_with_whitespace():
    url = " https://www.example.com "
    assert is_url(url)


def test_url_string_with_special_characters():
    url = "https://www.example.com/path?query=param#fragment"
    assert is_url(url)


def test_attribute_error():
    url = None
    assert not is_url(url)


def test_logger_error(caplog):
    url = None
    is_url(url)
    assert "Error when checking if None is a valid URL" in caplog.text


def test_non_string_input():
    url = 123
    with pytest.raises(Exception):
        download_url(url)


@pytest.fixture
def mock_chrome():
    with patch("selenium.webdriver.Chrome") as mock:
        yield mock


@pytest.fixture
def mock_chrome_driver_manager():
    with patch("webdriver_manager.chrome.ChromeDriverManager.install") as mock:
        yield mock


def test_valid_url(mock_chrome):
    url = "https://www.google.com"
    mock_chrome.return_value.get.return_value = None
    mock_chrome.return_value.page_source = "<html>Test HTML</html>"
    html_content = _download_rendered_html(url)
    assert isinstance(html_content, str)
    assert html_content != ""


def test_invalid_url(mock_chrome):
    url = "invalid_url"
    mock_chrome.return_value.get.side_effect = Exception("Invalid URL")
    with pytest.raises(Exception):
        _download_rendered_html(url)


def test_chrome_driver_not_installed(mock_chrome_driver_manager):
    url = "https://www.google.com"
    mock_chrome_driver_manager.side_effect = Exception("Chrome driver not installed")
    with pytest.raises(Exception):
        _download_rendered_html(url)


def test_chrome_driver_connection_error(mock_chrome):
    url = "https://www.google.com"
    mock_chrome.return_value.get.side_effect = Exception("Connection error")
    with pytest.raises(Exception):
        _download_rendered_html(url)


mock_html_value = "<html>Example</html>"


@pytest.fixture
def mock_download():
    with patch("autogen.agentchat.contrib.rag.document_utils._download_rendered_html") as mock:
        mock.return_value = mock_html_value
        yield mock


@pytest.fixture
def mock_open_file():
    with patch("builtins.open", new_callable=mock_open) as mock_file:
        yield mock_file


def test_download_url_valid_html(mock_download, mock_open_file, tmp_path):
    url = "https://www.example.com/index.html"
    filepath = download_url(url, tmp_path.resolve())
    assert filepath.suffix == ".html"
    mock_open_file.assert_called_with(str(filepath), "w", encoding="utf-8")
    m_file_handle = mock_open_file()
    m_file_handle.write.assert_called_with(mock_html_value)


def test_download_url_non_html(mock_download, mock_open_file, tmp_path):
    url = "https://www.example.com/image.jpg"
    with pytest.raises(ValueError):
        download_url(url, tmp_path.resolve())


def test_download_url_no_extension(mock_download, mock_open_file, tmp_path):
    url = "https://www.example.com/path"
    filepath = download_url(url, str(tmp_path))
    assert filepath.suffix == ".html"
    mock_open_file.assert_called_with(str(filepath), "w", encoding="utf-8")
    m_file_handle = mock_open_file()
    m_file_handle.write.assert_called_with(mock_html_value)


def test_download_url_no_output_dir(mock_download, mock_open_file):
    url = "https://www.example.com"
    with patch("os.getcwd") as mock_getcwd:
        mock_getcwd.return_value = "/fake/cwd"
        filepath = download_url(url)
        assert filepath.parent == Path("/fake/cwd")
        assert filepath.suffix == ".html"
        mock_open_file.assert_called_with(str(filepath), "w", encoding="utf-8")
        m_file_handle = mock_open_file()
        m_file_handle.write.assert_called_with(mock_html_value)


def test_download_url_invalid_url():
    url = "invalid url"
    with patch("autogen.agentchat.contrib.rag.document_utils._download_rendered_html") as mock_download:
        mock_download.side_effect = Exception("Invalid URL")
        with pytest.raises(Exception):
            download_url(url)


def test_list_files(tmpdir):
    directory = tmpdir
    file1 = directory.join("file1.txt")
    file2 = directory.join("file2.txt")
    file1.write("File 1 content")
    file2.write("File 2 content")
    file_list = list_files(str(directory))
    assert len(file_list) == 2
    assert str(file_list[0]) == str(file1)
    assert str(file_list[1]) == str(file2)


def test_handle_input_directory(tmpdir):
    directory = tmpdir
    file1 = directory.join("file1.txt")
    file2 = directory.join("file2.txt")
    file1.write("File 1 content")
    file2.write("File 2 content")
    file_list = handle_input(str(directory))
    assert len(file_list) == 2
    assert str(file_list[0]) == str(file1)
    assert str(file_list[1]) == str(file2)


def test_handle_input_file(tmpdir):
    file = tmpdir.join("file.txt")
    file.write("File content")
    file_list = handle_input(str(file))
    assert len(file_list) == 1
    assert str(file_list[0]) == str(file)


def test_handle_input_invalid_input():
    with pytest.raises(ValueError):
        handle_input("invalid input")
