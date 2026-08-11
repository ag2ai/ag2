# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ag2.config import XAIConfig
from ag2.config.xai import XAIFilesClient
from ag2.files.types import FileContent, FileProvider, UploadedFile


@patch("ag2.config.xai.files.AsyncClient")
def test_files_api_can_be_created_for_xai(mock_async_client: MagicMock) -> None:
    config = XAIConfig(model="grok-4-fast", api_key="test-key")

    client = config.create_files_client()

    assert isinstance(client, XAIFilesClient)
    mock_async_client.assert_called_once_with(
        api_key="test-key",
        api_host="api.x.ai",
        timeout=None,
        metadata=None,
        channel_options=None,
    )


@pytest.mark.asyncio
class TestXAIFilesClient:
    @patch("ag2.config.xai.files.AsyncClient")
    async def test_upload(self, mock_async_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_async_client.return_value = mock_client
        mock_client.files.upload.return_value = SimpleNamespace(
            id="file_123",
            filename="hello.txt",
            size=5,
            created_at=123,
        )

        result = await XAIFilesClient(XAIConfig(model="grok-4-fast")).upload(
            b"hello",
            "hello.txt",
            "assistants",
        )

        assert result == UploadedFile(
            file_id="file_123",
            filename="hello.txt",
            provider=FileProvider.XAI,
            bytes_count=5,
            purpose="assistants",
            created_at=123.0,
        )
        mock_client.files.upload.assert_awaited_once_with(b"hello", filename="hello.txt")

    @patch("ag2.config.xai.files.AsyncClient")
    async def test_read(self, mock_async_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_async_client.return_value = mock_client
        mock_client.files.get.return_value = SimpleNamespace(filename="hello.txt")
        mock_client.files.content.return_value = b"file-bytes"

        result = await XAIFilesClient(XAIConfig(model="grok-4-fast")).read("file_123")

        assert result == FileContent(name="hello.txt", data=b"file-bytes")

    @patch("ag2.config.xai.files.AsyncClient")
    async def test_list(self, mock_async_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_async_client.return_value = mock_client
        mock_client.files.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="file_1",
                    filename="a.txt",
                    size=100,
                    created_at=123,
                ),
            ]
        )

        result = await XAIFilesClient(XAIConfig(model="grok-4-fast")).list()

        assert result == [
            UploadedFile(
                file_id="file_1",
                filename="a.txt",
                provider=FileProvider.XAI,
                bytes_count=100,
                created_at=123.0,
            ),
        ]

    @patch("ag2.config.xai.files.AsyncClient")
    async def test_delete(self, mock_async_client: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_async_client.return_value = mock_client

        await XAIFilesClient(XAIConfig(model="grok-4-fast")).delete("file_123")

        mock_client.files.delete.assert_awaited_once_with("file_123")
