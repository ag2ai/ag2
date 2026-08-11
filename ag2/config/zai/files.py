# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from io import BytesIO
from typing import TYPE_CHECKING, Any

from zai import ZaiClient

from ag2.files.types import FileContent, FileProvider, UploadedFile, _created_at_to_float

if TYPE_CHECKING:
    from ag2.config.zai.config import ZAIConfig

# Default upload purpose; see ZAIFilesClient.upload for why "batch".
_DEFAULT_PURPOSE = "batch"

# Purposes that GET /files can enumerate without extra params; see ZAIFilesClient.list.
_LISTABLE_PURPOSES = ("batch", "fine-tune", "voice-clone-input")


class ZAIFilesClient:
    """Files API client for Z.AI.

    Unlike OpenAI/xAI, Z.AI's Files API is *purpose-bound* — it is not a general blob
    store, and each ``purpose`` has its own rules:

    - ``"batch"``     — a ``.jsonl``/``.xlsx`` batch-input file; the only purpose that
      supports the full upload/list/read(download)/delete round-trip.
    - ``"fine-tune"`` — a conversational ``.jsonl``; upload/list/delete work, but the
      server rejects content download.
    - ``"retrieval"`` — Doc/Docx/PDF/Xlsx/URL, but it REQUIRES a ``knowledge_id`` (a
      pre-existing knowledge base) or the server returns 400 "知识库id不能为空".
    """

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

    async def upload(
        self,
        data: bytes,
        filename: str,
        purpose: str | None = None,
        *,
        knowledge_id: str | None = None,
        sentence_size: int | None = None,
        custom_separator: list[str] | None = None,
    ) -> UploadedFile:
        """Upload a file. Defaults to the "batch" purpose so a plain upload works out of
        the box; pass another ``purpose`` (and ``knowledge_id``/``sentence_size``/
        ``custom_separator`` for "retrieval") to opt into the other modes.
        """
        # Forward only the extras the caller actually set; the SDK rejects unexpected None.
        extra: dict[str, Any] = {}
        if knowledge_id is not None:
            extra["knowledge_id"] = knowledge_id
        if sentence_size is not None:
            extra["sentence_size"] = sentence_size
        if custom_separator is not None:
            extra["custom_separator"] = custom_separator
        result = await asyncio.to_thread(
            self._client.files.create,
            file=(filename, BytesIO(data)),
            purpose=purpose or _DEFAULT_PURPOSE,
            **extra,
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
        """List uploaded files.

        Z.AI's GET /files requires a ``purpose`` filter (an unfiltered list returns
        nothing), and "retrieval" additionally needs a ``knowledge_id``. So we query
        each purpose that is listable without extra params and merge the results;
        "retrieval" files are not enumerable here.
        """
        results = await asyncio.gather(
            *(asyncio.to_thread(self._client.files.list, purpose=p) for p in _LISTABLE_PURPOSES),
            return_exceptions=True,
        )
        return [
            UploadedFile(
                file_id=f.id,
                filename=f.filename,
                provider=FileProvider.ZAI,
                bytes_count=f.bytes,
                purpose=f.purpose,
                created_at=_created_at_to_float(f.created_at),
            )
            for result in results
            if not isinstance(result, BaseException)
            for f in result.data
        ]

    async def delete(self, file_id: str) -> None:
        await asyncio.to_thread(self._client.files.delete, file_id)
