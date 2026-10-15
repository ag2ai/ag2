# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google import genai
from google.genai import types

from ag2.config import GeminiConfig
from ag2.config.gemini.files import GeminiFilesClient
from ag2.files import FilesAPI
from ag2.files.types import FileContent, FileProvider, UploadedFile


@pytest.mark.asyncio
@pytest.mark.parametrize("page_count", [0, 1, 3])
async def test_list_follows_sdk_pages(page_count: int) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        page = len(requests)
        payload: dict[str, Any] = {
            "files": [
                {
                    "name": f"files/{page}",
                    "displayName": f"{page}.txt",
                    "sizeBytes": str(page),
                    "createTime": "2025-01-01T00:00:00Z",
                }
            ]
            if page_count
            else []
        }
        if page < page_count:
            payload["nextPageToken"] = f"page-{page + 1}"
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        sdk = genai.Client(
            api_key="test", vertexai=False, http_options=types.HttpOptions(httpx_async_client=http_client)
        )
        # Supply the real SDK at its public constructor seam, without a live API request.
        with patch("ag2.config.gemini.files.genai.Client", return_value=sdk):
            files = await FilesAPI(GeminiConfig(model="test", api_key="test")).list()

    assert files == [
        UploadedFile(
            file_id=f"files/{page}",
            filename=f"{page}.txt",
            provider=FileProvider.GEMINI,
            bytes_count=page,
            purpose=None,
            created_at=1735689600.0,
        )
        for page in range(1, page_count + 1)
    ]
    assert [dict(request.url.params) for request in requests] == [
        {},
        *[{"pageToken": f"page-{page + 1}"} for page in range(1, page_count)],
    ]


@pytest.mark.asyncio
class TestGeminiFilesClient:
    @patch("ag2.config.gemini.files.genai")
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
            created_at=1735689600.0,
        )
        assert result.created_at == 1735689600.0
        call_kwargs = mock_client.aio.files.upload.await_args.kwargs
        assert call_kwargs["config"] == {"display_name": "recording.mp3", "mime_type": "audio/mpeg"}

    @patch("ag2.config.gemini.files.genai")
    async def test_upload_unknown_mime_falls_back_to_octet_stream(
        self, mock_genai: MagicMock, gemini_config: MagicMock
    ) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_client.aio.files.upload.return_value = types.File(name="files/x", size_bytes=3)

        await GeminiFilesClient(gemini_config).upload(b"abc", "blob.unknownext")

        call_kwargs = mock_client.aio.files.upload.await_args.kwargs
        assert call_kwargs["config"] == {
            "display_name": "blob.unknownext",
            "mime_type": "application/octet-stream",
        }

    @patch("ag2.config.gemini.files.genai")
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

    @patch("ag2.config.gemini.files.genai")
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

    @patch("ag2.config.gemini.files.genai")
    async def test_list(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_client.aio.files.list.return_value = MagicMock(
            page=[
                SimpleNamespace(
                    name="files/1",
                    display_name="doc.pdf",
                    size_bytes=100,
                    create_time="2025-01-01",
                ),
            ]
        )

        mock_client.aio.files.list.return_value.__aiter__.return_value = mock_client.aio.files.list.return_value.page
        result = await GeminiFilesClient(gemini_config).list()

        assert result == [
            UploadedFile(
                file_id="files/1",
                filename="doc.pdf",
                provider=FileProvider.GEMINI,
                bytes_count=100,
                purpose=None,
                created_at=1735689600.0,
            ),
        ]
        assert result[0].created_at == 1735689600.0

    @patch("ag2.config.gemini.files.genai")
    async def test_delete(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client

        await GeminiFilesClient(gemini_config).delete("files/abc123")

        mock_client.aio.files.delete.assert_awaited_once_with(name="files/abc123")

    @patch("ag2.config.gemini.files.genai")
    async def test_upload_rejects_a_file_without_a_name(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_genai.Client.return_value = mock_client
        mock_client.aio.files.upload.return_value = SimpleNamespace(name=None, size_bytes=512, create_time=None)

        with pytest.raises(ValueError, match="no resource name"):
            await GeminiFilesClient(gemini_config).upload(b"audio-data", "recording.mp3")
