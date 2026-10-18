# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from xai_sdk.proto import files_pb2, files_pb2_grpc

from ag2.config import XAIConfig
from ag2.config.xai import XAIFilesClient
from ag2.files import FilesAPI
from ag2.files.types import FileContent, FileProvider, UploadedFile


class PaginatedFilesServer(files_pb2_grpc.FilesServicer):
    def __init__(self, page_count: int) -> None:
        self.page_count = page_count
        self.requests: list[files_pb2.ListFilesRequest] = []

    async def ListFiles(  # noqa: N802
        self, request: files_pb2.ListFilesRequest, context: object
    ) -> files_pb2.ListFilesResponse:
        self.requests.append(request)
        page = len(self.requests)
        return files_pb2.ListFilesResponse(
            data=[files_pb2.File(id=f"file-{page}", filename=f"{page}.txt", size=page, created_at={"seconds": 123})]
            if self.page_count
            else [],
            pagination_token=f"page-{page + 1}" if page < self.page_count else "",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("page_count", [0, 1, 3])
async def test_list_follows_sdk_pages(page_count: int) -> None:
    servicer = PaginatedFilesServer(page_count)
    server = grpc.aio.server()
    files_pb2_grpc.add_FilesServicer_to_server(servicer, server)  # type: ignore[no-untyped-call]
    port = server.add_secure_port("localhost:0", grpc.local_server_credentials())
    await server.start()
    try:
        files = await FilesAPI(XAIConfig(model="test", api_key="test", api_host=f"localhost:{port}")).list()
    finally:
        await server.stop(None)

    assert files == [
        UploadedFile(
            file_id=f"file-{page}",
            filename=f"{page}.txt",
            provider=FileProvider.XAI,
            bytes_count=page,
            created_at=123.0,
        )
        for page in range(1, page_count + 1)
    ]
    assert servicer.requests == [
        files_pb2.ListFilesRequest(),
        *[files_pb2.ListFilesRequest(pagination_token=f"page-{page + 1}") for page in range(1, page_count)],
    ]


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
            created_at=Timestamp(seconds=123),
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
            pagination_token="",
            data=[
                SimpleNamespace(
                    id="file_1",
                    filename="a.txt",
                    size=100,
                    created_at=Timestamp(seconds=123),
                ),
            ],
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
