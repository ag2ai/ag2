# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator

import pytest
from xai_sdk.proto import files_pb2, files_pb2_grpc

from ag2.config import XAIConfig
from ag2.files import FilesAPI
from ag2.files.types import FileProvider, UploadedFile

from ._helpers import serve_files


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


TIMESTAMPS = pytest.mark.parametrize(("seconds", "nanos"), [(0, 0), (123, 0), (1_700_000_000, 125_000_000)])


def _expected(seconds: int, nanos: int) -> UploadedFile:
    return UploadedFile(file_id="file-1", filename="hello.txt", provider=FileProvider.XAI, bytes_count=5)


@pytest.mark.asyncio
@TIMESTAMPS
async def test_upload_preserves_server_creation_time(seconds: int, nanos: int) -> None:
    servicer = TimestampFilesServer(seconds, nanos)
    async with serve_files(servicer) as address:
        result = await FilesAPI(XAIConfig(model="test", api_key="test", api_host=address)).upload(
            data=b"hello", filename="hello.txt"
        )

    assert result == _expected(seconds, nanos)
    # BaseEvent equality deliberately excludes created_at.
    assert result.created_at == seconds + nanos / 1_000_000_000
    assert servicer.uploads == [
        files_pb2.UploadFileChunk(init=files_pb2.UploadFileInit(name="hello.txt")),
        files_pb2.UploadFileChunk(data=b"hello"),
    ]


@pytest.mark.asyncio
@TIMESTAMPS
async def test_list_preserves_server_creation_time(seconds: int, nanos: int) -> None:
    async with serve_files(TimestampFilesServer(seconds, nanos)) as address:
        [result] = await FilesAPI(XAIConfig(model="test", api_key="test", api_host=address)).list()

    assert result == _expected(seconds, nanos)
    assert result.created_at == seconds + nanos / 1_000_000_000
