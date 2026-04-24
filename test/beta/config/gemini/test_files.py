# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta.config.gemini.files import GeminiFilesClient
from autogen.beta.files.types import FileContent, FileProvider, UploadedFile


@pytest.mark.asyncio
class TestGeminiFilesClient:
    @patch("autogen.beta.config.gemini.files.genai")
    async def test_upload(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_client.aio.files.upload.return_value = SimpleNamespace(
            name="files/abc123",
            size_bytes=512,
            create_time="2025-01-01T00:00:00Z",
        )

        result = await GeminiFilesClient(gemini_config).upload(b"audio-data", "recording.mp3")

        assert result == UploadedFile(
            file_id="files/abc123",
            filename="recording.mp3",
            provider=FileProvider.GEMINI,
            bytes_count=512,
            purpose=None,
            created_at="2025-01-01T00:00:00Z",
        )
        call_kwargs = mock_client.aio.files.upload.await_args.kwargs
        assert call_kwargs["config"] == {"display_name": "recording.mp3", "mime_type": "audio/mpeg"}

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_upload_unknown_mime_falls_back_to_octet_stream(
        self, mock_genai: MagicMock, gemini_config: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_client.aio.files.upload.return_value = SimpleNamespace(name="files/x", size_bytes=3)

        await GeminiFilesClient(gemini_config).upload(b"abc", "blob.unknownext")

        call_kwargs = mock_client.aio.files.upload.await_args.kwargs
        assert call_kwargs["config"] == {
            "display_name": "blob.unknownext",
            "mime_type": "application/octet-stream",
        }

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_read_downloadable(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_client.aio.files.get.return_value = SimpleNamespace(
            name="files/gen1",
            display_name="video.mp4",
            mime_type="video/mp4",
            download_uri="https://example/download",
        )
        mock_client.aio.files.download.return_value = b"video-bytes"

        result = await GeminiFilesClient(gemini_config).read("files/gen1")

        assert result == FileContent(name="video.mp4", data=b"video-bytes", media_type="video/mp4")

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_read_uploaded_raises(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_client.aio.files.get.return_value = SimpleNamespace(
            name="files/abc123",
            display_name="doc.pdf",
            mime_type="application/pdf",
            download_uri=None,
        )

        with pytest.raises(NotImplementedError, match="user-uploaded files"):
            await GeminiFilesClient(gemini_config).read("files/abc123")

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_list(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_client.aio.files.list.return_value = SimpleNamespace(
            page=[
                SimpleNamespace(
                    name="files/1",
                    display_name="doc.pdf",
                    size_bytes=100,
                    create_time="2025-01-01",
                ),
            ]
        )

        result = await GeminiFilesClient(gemini_config).list()

        assert result == [
            UploadedFile(
                file_id="files/1",
                filename="doc.pdf",
                provider=FileProvider.GEMINI,
                bytes_count=100,
                purpose=None,
                created_at="2025-01-01",
            ),
        ]

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_delete(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client

        await GeminiFilesClient(gemini_config).delete("files/abc123")

        mock_client.aio.files.delete.assert_awaited_once_with(name="files/abc123")
