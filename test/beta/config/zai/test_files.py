# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autogen.beta.config.zai.files import ZAIFilesClient
from autogen.beta.files.types import FileContent, FileProvider, UploadedFile


@patch("autogen.beta.config.zai.files.ZaiClient")
def test_files_client_construction_omits_none_options(mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
    ZAIFilesClient(zai_config)

    mock_zai_client.assert_called_once_with(
        api_key="test-key",
        max_retries=3,
        disable_token_cache=True,
    )


@pytest.mark.asyncio
class TestZAIFilesClient:
    # The zai SDK is synchronous (wrapped via asyncio.to_thread), so the inner client is a
    # MagicMock, not an AsyncMock.

    @patch("autogen.beta.config.zai.files.ZaiClient")
    async def test_upload_defaults_to_retrieval(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.create.return_value = SimpleNamespace(
            id="file_123",
            filename="hello.txt",
            bytes=5,
            purpose="retrieval",
            created_at=123,
        )

        result = await ZAIFilesClient(zai_config).upload(b"hello", "hello.txt")

        assert result == UploadedFile(
            file_id="file_123",
            filename="hello.txt",
            provider=FileProvider.ZAI,
            bytes_count=5,
            purpose="retrieval",
            created_at=123.0,
        )
        kwargs = mock_client.files.create.call_args.kwargs
        assert kwargs["purpose"] == "retrieval"
        name, buffer = kwargs["file"]
        assert name == "hello.txt"
        assert buffer.read() == b"hello"

    @patch("autogen.beta.config.zai.files.ZaiClient")
    async def test_upload_with_explicit_purpose(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.create.return_value = SimpleNamespace(
            id="file_123",
            filename="data.jsonl",
            bytes=10,
            purpose="batch",
            created_at=1,
        )

        result = await ZAIFilesClient(zai_config).upload(b"payload---", "data.jsonl", "batch")

        assert result == UploadedFile(
            file_id="file_123",
            filename="data.jsonl",
            provider=FileProvider.ZAI,
            bytes_count=10,
            purpose="batch",
            created_at=1.0,
        )
        assert mock_client.files.create.call_args.kwargs["purpose"] == "batch"

    @patch("autogen.beta.config.zai.files.ZaiClient")
    async def test_read_returns_bytes_without_metadata(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.content.return_value = SimpleNamespace(content=b"file-bytes")

        result = await ZAIFilesClient(zai_config).read("file_123")

        assert result == FileContent(name=None, data=b"file-bytes", media_type=None)
        mock_client.files.content.assert_called_once_with("file_123")

    @patch("autogen.beta.config.zai.files.ZaiClient")
    async def test_list(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="file_1",
                    filename="a.jsonl",
                    bytes=100,
                    purpose="retrieval",
                    created_at=123,
                ),
            ]
        )

        result = await ZAIFilesClient(zai_config).list()

        assert result == [
            UploadedFile(
                file_id="file_1",
                filename="a.jsonl",
                provider=FileProvider.ZAI,
                bytes_count=100,
                purpose="retrieval",
                created_at=123.0,
            ),
        ]

    @patch("autogen.beta.config.zai.files.ZaiClient")
    async def test_delete(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client

        await ZAIFilesClient(zai_config).delete("file_123")

        mock_client.files.delete.assert_called_once_with("file_123")
