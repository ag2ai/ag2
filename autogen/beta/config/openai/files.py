# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from autogen.beta.files.types import FileContent, FileProvider, UploadedFile, _created_at_to_float

if TYPE_CHECKING:
    from autogen.beta.config.openai.config import OpenAIConfig, OpenAIResponsesConfig


class OpenAIFilesClient:
    """Files API client for OpenAI."""

    __slots__ = ("_client",)

    def __init__(self, config: "OpenAIConfig | OpenAIResponsesConfig") -> None:
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            organization=config.organization,
            project=config.project,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            default_headers=config.default_headers,
            default_query=config.default_query,
            http_client=config.http_client,
        )

    async def upload(self, data: bytes, filename: str, purpose: str | None = None) -> UploadedFile:
        result = await self._client.files.create(
            file=(filename, BytesIO(data)),
            purpose=purpose or "assistants",
        )
        return UploadedFile(
            file_id=result.id,
            filename=result.filename,
            provider=FileProvider.OPENAI,
            bytes_count=result.bytes,
            purpose=result.purpose,
            created_at=_created_at_to_float(result.created_at),
        )

    async def read(self, file_id: str) -> FileContent:
        response = await self._client.files.content(file_id)
        metadata = await self._client.files.retrieve(file_id)
        return FileContent(
            name=metadata.filename,
            data=response.content,
        )

    async def list(self) -> list[UploadedFile]:
        result = await self._client.files.list()
        return [
            UploadedFile(
                file_id=f.id,
                filename=f.filename,
                provider=FileProvider.OPENAI,
                bytes_count=f.bytes,
                purpose=f.purpose,
                created_at=_created_at_to_float(f.created_at),
            )
            for f in result.data
        ]

    async def delete(self, file_id: str) -> None:
        await self._client.files.delete(file_id)
