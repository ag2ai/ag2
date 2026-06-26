# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from io import BytesIO
from typing import TYPE_CHECKING, Any

from zai import ZaiClient

from autogen.beta.files.types import FileContent, FileProvider, UploadedFile, _created_at_to_float

if TYPE_CHECKING:
    from autogen.beta.config.zai.config import ZAIConfig

_DEFAULT_PURPOSE = "retrieval"


class ZAIFilesClient:
    """Files API client for Z.AI."""

    __slots__ = ("_client",)

    def __init__(self, config: "ZAIConfig") -> None:
        kwargs: dict[str, Any] = {}
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        if config.base_url is not None:
            kwargs["base_url"] = config.base_url
        if config.timeout is not None:
            kwargs["timeout"] = config.timeout
        kwargs["max_retries"] = config.max_retries
        if config.http_client is not None:
            kwargs["http_client"] = config.http_client
        if config.custom_headers is not None:
            kwargs["custom_headers"] = config.custom_headers
        kwargs["disable_token_cache"] = config.disable_token_cache
        if config.source_channel is not None:
            kwargs["source_channel"] = config.source_channel
        self._client = ZaiClient(**kwargs)

    async def upload(self, data: bytes, filename: str, purpose: str | None = None) -> UploadedFile:
        result = await asyncio.to_thread(
            self._client.files.create,
            file=(filename, BytesIO(data)),
            purpose=purpose or _DEFAULT_PURPOSE,
        )
        return UploadedFile(
            file_id=result.id,
            filename=result.filename,
            provider=FileProvider.ZAI,
            bytes_count=result.bytes,
            purpose=result.purpose,
            created_at=_created_at_to_float(result.created_at),
        )

    async def read(self, file_id: str) -> FileContent:
        # The zai SDK exposes only raw content (no metadata-by-id endpoint), so the
        # filename and media type are unavailable here.
        response = await asyncio.to_thread(self._client.files.content, file_id)
        return FileContent(name=None, data=response.content, media_type=None)

    async def list(self) -> list[UploadedFile]:
        result = await asyncio.to_thread(self._client.files.list)
        return [
            UploadedFile(
                file_id=f.id,
                filename=f.filename,
                provider=FileProvider.ZAI,
                bytes_count=f.bytes,
                purpose=f.purpose,
                created_at=_created_at_to_float(f.created_at),
            )
            for f in result.data
        ]

    async def delete(self, file_id: str) -> None:
        await asyncio.to_thread(self._client.files.delete, file_id)
