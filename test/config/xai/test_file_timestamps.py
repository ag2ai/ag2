# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator

import grpc
import pytest
from xai_sdk.proto import files_pb2, files_pb2_grpc

from ag2.config import XAIConfig
from ag2.files import FilesAPI
from ag2.files.types import FileProvider, UploadedFile


class TimestampFilesServer(files_pb2_grpc.FilesServicer):
    def __init__(self, seconds: int, nanos: int) -> None:
        self.file = files_pb2.File(
            id="file-1", filename="hello.txt", size=5, created_at={"seconds": seconds, "nanos": nanos}
        )
        self.uploads: list[files_pb2.UploadFileChunk] = []

    async def UploadFile(  # noqa: N802
        self, requests: AsyncIterator[files_pb2.UploadFileChunk], context: object
    ) -> files_pb2.File:
        self.uploads = [request async for request in requests]
        return self.file

    async def ListFiles(  # noqa: N802
        self, request: files_pb2.ListFilesRequest, context: object
    ) -> files_pb2.ListFilesResponse:
        return files_pb2.ListFilesResponse(data=[self.file])


@pytest.mark.asyncio
@pytest.mark.parametrize(("seconds", "nanos"), [(0, 0), (123, 0), (1_700_000_000, 125_000_000)])
@pytest.mark.parametrize("operation", ["upload", "list"])
async def test_preserves_server_creation_time(seconds: int, nanos: int, operation: str) -> None:
    servicer = TimestampFilesServer(seconds, nanos)
    server = grpc.aio.server()
    files_pb2_grpc.add_FilesServicer_to_server(servicer, server)  # type: ignore[no-untyped-call]
    port = server.add_secure_port("localhost:0", grpc.local_server_credentials())
    await server.start()
    try:
        api = FilesAPI(XAIConfig(model="test", api_key="test", api_host=f"localhost:{port}"))
        if operation == "upload":
            result = await api.upload(data=b"hello", filename="hello.txt")
        else:
            [result] = await api.list()
    finally:
        await server.stop(None)

    assert result == UploadedFile(file_id="file-1", filename="hello.txt", provider=FileProvider.XAI, bytes_count=5)
    # BaseEvent equality deliberately excludes created_at.
    assert result.created_at == seconds + nanos / 1_000_000_000
    assert servicer.uploads == (
        [
            files_pb2.UploadFileChunk(init=files_pb2.UploadFileInit(name="hello.txt")),
            files_pb2.UploadFileChunk(data=b"hello"),
        ]
        if operation == "upload"
        else []
    )
