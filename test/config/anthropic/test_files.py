# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from ag2.config.anthropic import AnthropicConfig
from ag2.config.anthropic.files import AnthropicFilesClient
from ag2.files.types import FileContent, FileProvider, UploadedFile

_METADATA = {
    "id": "file-011CNha8",
    "type": "file",
    "filename": "output.csv",
    "mime_type": "text/csv",
    "size_bytes": 9,
    "created_at": "2025-01-01T00:00:00Z",
}


def _transport_config() -> AnthropicConfig:
    """A config whose transport answers the two calls `read` makes.

    The SDK builds its own response objects from the wire, so the shapes under test
    are the SDK's rather than a double's.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/content"):
            return httpx2.Response(200, content=b"file-data", headers={"content-type": "text/csv"})
        return httpx2.Response(200, json=_METADATA)

    return AnthropicConfig(
        model="claude-haiku-4-5",
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_read_takes_the_bytes_off_the_sdks_binary_response() -> None:
    """`download` answers an `AsyncBinaryAPIResponse`, whose bytes are behind `read()`."""
    result = await AnthropicFilesClient(_transport_config()).read("file-011CNha8")

    assert result == FileContent(name="output.csv", data=b"file-data", media_type="text/csv")


@pytest.mark.asyncio
class TestAnthropicFilesClient:
    @patch("ag2.config.anthropic.files.AsyncAnthropic")
    async def test_upload(self, mock_anthropic_cls: MagicMock, anthropic_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.beta.files.upload.return_value = SimpleNamespace(
            id="file-011CNha8",
            filename="document.pdf",
            size_bytes=1024000,
            created_at="2025-01-01T00:00:00Z",
        )

        result = await AnthropicFilesClient(anthropic_config).upload(b"pdf-data", "document.pdf")

        assert result == UploadedFile(
            file_id="file-011CNha8",
            filename="document.pdf",
            provider=FileProvider.ANTHROPIC,
            bytes_count=1024000,
            purpose=None,
            created_at=1735689600.0,
        )
        assert result.created_at == 1735689600.0

    @patch("ag2.config.anthropic.files.AsyncAnthropic")
    async def test_list(self, mock_anthropic_cls: MagicMock, anthropic_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.beta.files.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="file-1",
                    filename="a.pdf",
                    size_bytes=100,
                    created_at="2025-01-01T00:00:00Z",
                ),
            ]
        )

        result = await AnthropicFilesClient(anthropic_config).list()

        assert result == [
            UploadedFile(
                file_id="file-1",
                filename="a.pdf",
                provider=FileProvider.ANTHROPIC,
                bytes_count=100,
                purpose=None,
                created_at=1735689600.0,
            ),
        ]
        assert result[0].created_at == 1735689600.0

    @patch("ag2.config.anthropic.files.AsyncAnthropic")
    async def test_delete(self, mock_anthropic_cls: MagicMock, anthropic_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client

        await AnthropicFilesClient(anthropic_config).delete("file-011CNha8")

        mock_client.beta.files.delete.assert_awaited_once_with("file-011CNha8")
