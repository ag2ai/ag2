# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dirty_equals import IsFloat

from autogen.beta.config.openai.files import OpenAIFilesClient
from autogen.beta.files.types import FileContent, FileProvider, UploadedFile


@pytest.mark.asyncio
class TestOpenAIFilesClient:
    @patch("autogen.beta.config.openai.files.AsyncOpenAI")
    async def test_upload(self, mock_openai_cls: MagicMock, openai_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.files.create.return_value = SimpleNamespace(
            id="file-abc",
            filename="test.pdf",
            bytes=2048,
            purpose="assistants",
            created_at=1700000000,
        )

        result = await OpenAIFilesClient(openai_config).upload(b"pdf-data", "test.pdf", "assistants")

        assert result == UploadedFile(
            file_id="file-abc",
            filename="test.pdf",
            provider=FileProvider.OPENAI,
            bytes_count=2048,
            purpose="assistants",
            created_at=1700000000.0,
        )
        assert result.created_at == 1700000000.0

    @patch("autogen.beta.config.openai.files.AsyncOpenAI")
    async def test_read(self, mock_openai_cls: MagicMock, openai_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.files.content.return_value = SimpleNamespace(content=b"file-bytes")
        mock_client.files.retrieve.return_value = SimpleNamespace(filename="doc.pdf")

        result = await OpenAIFilesClient(openai_config).read("file-abc")

        assert result == FileContent(name="doc.pdf", data=b"file-bytes")

    @patch("autogen.beta.config.openai.files.AsyncOpenAI")
    async def test_list(self, mock_openai_cls: MagicMock, openai_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.files.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="file-1",
                    filename="a.txt",
                    bytes=100,
                    purpose="assistants",
                    created_at=None,
                ),
                SimpleNamespace(
                    id="file-2",
                    filename="b.csv",
                    bytes=200,
                    purpose="fine-tune",
                    created_at=1700000000,
                ),
            ]
        )

        result = await OpenAIFilesClient(openai_config).list()

        assert result == [
            UploadedFile(
                file_id="file-1",
                filename="a.txt",
                provider=FileProvider.OPENAI,
                bytes_count=100,
                purpose="assistants",
            ),
            UploadedFile(
                file_id="file-2",
                filename="b.csv",
                provider=FileProvider.OPENAI,
                bytes_count=200,
                purpose="fine-tune",
                created_at=1700000000.0,
            ),
        ]
        assert result[0].created_at == IsFloat()
        assert result[1].created_at == 1700000000.0

    @patch("autogen.beta.config.openai.files.AsyncOpenAI")
    async def test_delete(self, mock_openai_cls: MagicMock, openai_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client

        await OpenAIFilesClient(openai_config).delete("file-abc")

        mock_client.files.delete.assert_awaited_once_with("file-abc")
