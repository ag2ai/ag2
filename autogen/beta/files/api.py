# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from os import PathLike
from pathlib import Path

from autogen.beta.config.config import ModelConfig

from .protocol import FilesClient
from .types import FileContent, UploadedFile


class FilesAPI(FilesClient):
    __slots__ = ("_client",)

    def __init__(self, config: ModelConfig) -> None:
        self._client = config.create_files_client()

    async def upload(
        self,
        *,
        path: str | PathLike[str] | None = None,
        data: bytes | None = None,
        filename: str | None = None,
        purpose: str | None = None,
    ) -> UploadedFile:
        """Upload a file to the provider."""
        if path is not None:
            p = Path(path)
            data = p.read_bytes()
            filename = filename or p.name
        if data is None:
            raise ValueError("Either 'path' or 'data' must be provided.")
        if filename is None:
            raise ValueError("'filename' is required when using 'data' without 'path'.")
        return await self._client.upload(data, filename, purpose)

    async def read(self, file_id: str) -> FileContent:
        """Download file content by ID."""
        return await self._client.read(file_id)

    async def list(self) -> list[UploadedFile]:
        """List all files in the provider."""
        return await self._client.list()

    async def delete(self, file_id: str) -> None:
        """Delete a file by ID."""
        await self._client.delete(file_id)
