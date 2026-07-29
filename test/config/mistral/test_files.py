# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mistralai.client.types import UNSET

from ag2.config.mistral.files import MistralFilesClient
from ag2.files.types import FileContent, FileProvider, UploadedFile


class _StreamingResponse:
    """The unconsumed httpx.Response ``download_async`` returns.

    ``.content`` before ``aread()`` raises, as httpx does.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._read = False

    async def aread(self) -> bytes:
        self._read = True
        return self._data

    @property
    def content(self) -> bytes:
        if not self._read:
            raise RuntimeError("Attempted to access streaming response content, without having called `read()`.")
        return self._data


def _streaming_response(data: bytes) -> _StreamingResponse:
    return _StreamingResponse(data)


def _file(file_id: str = "file_123", filename: str = "hello.jsonl") -> SimpleNamespace:
    return SimpleNamespace(
        id=file_id,
        filename=filename,
        size_bytes=5,
        purpose="ocr",
        created_at=123,
        mimetype="application/jsonl",
    )


@patch("ag2.config.mistral.files.Mistral")
def test_construction_omits_unset_options(mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
    MistralFilesClient(mistral_config)

    mock_mistral.assert_called_once_with(api_key="test-key")


@pytest.mark.asyncio
class TestMistralFilesClient:
    @patch("ag2.config.mistral.files.Mistral")
    async def test_upload_defaults_to_ocr_purpose(self, mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
        client = MagicMock()
        mock_mistral.return_value = client
        client.files.upload_async = AsyncMock(return_value=_file())

        result = await MistralFilesClient(mistral_config).upload(b"hello", "hello.jsonl")

        assert result == UploadedFile(
            file_id="file_123",
            filename="hello.jsonl",
            provider=FileProvider.MISTRAL,
            bytes_count=5,
            purpose="ocr",
            created_at=123.0,
        )
        kwargs = client.files.upload_async.call_args.kwargs
        assert kwargs["purpose"] == "ocr"
        assert (kwargs["file"].file_name, kwargs["file"].content) == ("hello.jsonl", b"hello")

    @patch("ag2.config.mistral.files.Mistral")
    async def test_upload_honours_explicit_purpose(self, mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
        client = MagicMock()
        mock_mistral.return_value = client
        client.files.upload_async = AsyncMock(return_value=_file())

        await MistralFilesClient(mistral_config).upload(b"x", "train.jsonl", purpose="fine-tune")

        assert client.files.upload_async.call_args.kwargs["purpose"] == "fine-tune"

    @patch("ag2.config.mistral.files.Mistral")
    async def test_read_combines_metadata_and_content(self, mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
        client = MagicMock()
        mock_mistral.return_value = client
        client.files.retrieve_async = AsyncMock(return_value=_file())
        client.files.download_async = AsyncMock(return_value=_streaming_response(b"hello"))

        result = await MistralFilesClient(mistral_config).read("file_123")

        assert result == FileContent(name="hello.jsonl", data=b"hello", media_type="application/jsonl")

    @patch("ag2.config.mistral.files.Mistral")
    async def test_read_normalises_unset_mimetype(self, mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
        """Omitted SDK fields arrive as the ``Unset()`` sentinel, not ``None``."""
        metadata = _file()
        metadata.mimetype = UNSET
        client = MagicMock()
        mock_mistral.return_value = client
        client.files.retrieve_async = AsyncMock(return_value=metadata)
        client.files.download_async = AsyncMock(return_value=_streaming_response(b"hello"))

        result = await MistralFilesClient(mistral_config).read("file_123")

        assert result.media_type is None

    @patch("ag2.config.mistral.files.Mistral")
    async def test_list_single_page(self, mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
        client = MagicMock()
        mock_mistral.return_value = client
        client.files.list_async = AsyncMock(return_value=SimpleNamespace(data=[_file()]))

        result = await MistralFilesClient(mistral_config).list()

        assert result == [
            UploadedFile(
                file_id="file_123",
                filename="hello.jsonl",
                provider=FileProvider.MISTRAL,
                bytes_count=5,
                purpose="ocr",
                created_at=123.0,
            )
        ]
        assert client.files.list_async.await_count == 1

    @patch("ag2.config.mistral.files.Mistral")
    async def test_list_follows_pagination(self, mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
        client = MagicMock()
        mock_mistral.return_value = client
        full_page = [_file(f"file_{i}") for i in range(100)]
        client.files.list_async = AsyncMock(
            side_effect=[
                SimpleNamespace(data=full_page),
                SimpleNamespace(data=[_file("file_100")]),
            ]
        )

        result = await MistralFilesClient(mistral_config).list()

        assert len(result) == 101
        assert [c.kwargs["page"] for c in client.files.list_async.call_args_list] == [0, 1]

    @patch("ag2.config.mistral.files.Mistral")
    async def test_list_handles_empty_response(self, mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
        client = MagicMock()
        mock_mistral.return_value = client
        client.files.list_async = AsyncMock(return_value=SimpleNamespace(data=None))

        assert await MistralFilesClient(mistral_config).list() == []

    @patch("ag2.config.mistral.files.Mistral")
    async def test_delete(self, mock_mistral: MagicMock, mistral_config: MagicMock) -> None:
        client = MagicMock()
        mock_mistral.return_value = client
        client.files.delete_async = AsyncMock()

        await MistralFilesClient(mistral_config).delete("file_123")

        client.files.delete_async.assert_awaited_once_with(file_id="file_123")
