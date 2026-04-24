# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from io import BytesIO
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

from autogen.beta.files.types import FileContent, FileProvider, UploadedFile

if TYPE_CHECKING:
    from autogen.beta.config.anthropic.config import AnthropicConfig


class AnthropicFilesClient:
    """Files API client for Anthropic."""

    __slots__ = ("_client",)

    def __init__(self, config: "AnthropicConfig") -> None:
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout if config.timeout is not None else 600.0,
            max_retries=config.max_retries,
            default_headers=config.default_headers,
            http_client=config.http_client,
        )

    async def upload(self, data: bytes, filename: str, purpose: str | None = None) -> UploadedFile:
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        result = await self._client.beta.files.upload(
            file=(filename, BytesIO(data), mime_type),
        )
        return UploadedFile(
            file_id=result.id,
            filename=result.filename if hasattr(result, "filename") else filename,
            provider=FileProvider.ANTHROPIC,
            bytes_count=result.size_bytes if hasattr(result, "size_bytes") else None,
            purpose=purpose,
            created_at=result.created_at if hasattr(result, "created_at") else None,
        )

    async def read(self, file_id: str) -> FileContent:
        response = await self._client.beta.files.download(file_id)
        metadata = await self._client.beta.files.retrieve_metadata(file_id)
        return FileContent(
            name=metadata.filename if hasattr(metadata, "filename") else None,
            data=response.content if hasattr(response, "content") else bytes(response),
            media_type=metadata.mime_type if hasattr(metadata, "mime_type") else None,
        )

    async def list(self) -> list[UploadedFile]:
        result = await self._client.beta.files.list()
        return [
            UploadedFile(
                file_id=f.id,
                filename=f.filename if hasattr(f, "filename") else None,
                provider=FileProvider.ANTHROPIC,
                bytes_count=f.size_bytes if hasattr(f, "size_bytes") else None,
                purpose=None,
                created_at=f.created_at if hasattr(f, "created_at") else None,
            )
            for f in result.data
        ]

    async def delete(self, file_id: str) -> None:
        await self._client.beta.files.delete(file_id)
