# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from xai_sdk import AsyncClient

from ag2.files.types import FileContent, FileProvider, UploadedFile, _created_at_to_float

if TYPE_CHECKING:
    from ag2.config.xai.config import XAIConfig


class XAIFilesClient:
    """Files API client for xAI."""

    __slots__ = ("_client",)

    def __init__(self, config: "XAIConfig") -> None:
        self._client = AsyncClient(
            api_key=config.api_key,
            api_host=config.api_host,
            timeout=config.timeout,
            metadata=config.metadata,
            channel_options=config.channel_options,
        )

    async def upload(self, data: bytes, filename: str, purpose: str | None = None) -> UploadedFile:
        result = await self._client.files.upload(data, filename=filename)
        return UploadedFile(
            file_id=result.id,
            filename=result.filename,
            provider=FileProvider.XAI,
            bytes_count=result.size,
            purpose=purpose,
            created_at=_created_at_to_float(result.created_at),
        )

    async def read(self, file_id: str) -> FileContent:
        metadata = await self._client.files.get(file_id)
        data = await self._client.files.content(file_id)
        return FileContent(
            name=metadata.filename,
            data=data,
        )

    async def list(self) -> list[UploadedFile]:
        result = await self._client.files.list()
        return [
            UploadedFile(
                file_id=f.id,
                filename=f.filename,
                provider=FileProvider.XAI,
                bytes_count=f.size,
                created_at=_created_at_to_float(f.created_at),
            )
            for f in result.data
        ]

    async def delete(self, file_id: str) -> None:
        await self._client.files.delete(file_id)
