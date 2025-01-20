# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from autogen.agentchat.contrib.rag.document_utils import download_url, handle_input, is_url, list_files


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


@pytest.fixture
def mock_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Example</html>"
    return mock_response


def test_valid_url(mock_response):
    url = "https://www.example.com"
    with patch("requests.get", return_value=mock_response):
        filepath = download_url(url)
        assert isinstance(filepath, Path)
        assert os.path.exists(filepath)


def test_valid_url_with_output_dir(mock_response):
    url = "https://www.example.com"
    output_dir = "/tmp"
    with patch("requests.get", return_value=mock_response):
        filepath = download_url(url, output_dir)
        assert isinstance(filepath, Path)
        assert os.path.exists(filepath)
        assert str(filepath).startswith(output_dir)


def test_non_existent_output_dir(mock_response):
    url = "https://www.example.com"
    output_dir = "/non/existent/dir"
    with patch("requests.get", return_value=mock_response):
        with pytest.raises(FileNotFoundError):
            download_url(url, output_dir)


def test_invalid_url():
    url = "https://www.example.com"
    with patch("requests.get", side_effect=requests.exceptions.HTTPError):
        with pytest.raises(requests.exceptions.HTTPError):
            download_url(url)


def test_non_string_input():
    url = 123
    with pytest.raises(Exception):
        download_url(url)


def test_download_url(mock_response, tmpdir):
    url = "https://www.example.com"
    with patch("requests.get", return_value=mock_response):
        filepath = download_url(url, output_dir=tmpdir)
        assert filepath.exists()
        with open(filepath, "r", encoding="utf-8") as file:
            assert file.read() == "<html>Example</html>"


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


def test_handle_input_url(mock_response, tmpdir):
    url = "https://www.example.com"
    with patch("requests.get", return_value=mock_response):
        filepath = handle_input(url, output_dir=tmpdir)
        assert filepath[0].exists()
        with open(filepath[0], "r", encoding="utf-8") as file:
            assert file.read() == "<html>Example</html>"


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
