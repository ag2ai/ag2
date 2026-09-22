# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import TYPE_CHECKING, Final, get_args

from openai import AsyncOpenAI
from openai.types import FilePurpose

from ag2.files.types import FileContent, FileProvider, UploadedFile, _created_at_to_float

if TYPE_CHECKING:
    from ag2.config.openai.config import OpenAIConfig, OpenAIResponsesConfig

# The purposes OpenAI's Files API accepts, taken from the SDK rather than restated, so
# the set follows the SDK when it grows. ``get_args`` hands back ``Any``; the annotation
# is what states the shape, and it is what narrows ``purpose`` below.
_PURPOSES: Final[tuple[FilePurpose, ...]] = get_args(FilePurpose)
_DEFAULT_PURPOSE: Final[FilePurpose] = "assistants"


def _resolve_purpose(purpose: str | None) -> FilePurpose:
    """Narrow a requested purpose to the closed set OpenAI accepts.

    An unsupported one is named here rather than sent and answered with a 400.
    """
    if purpose is None:
        return _DEFAULT_PURPOSE
    if purpose not in _PURPOSES:
        raise ValueError(f"OpenAI does not accept the file purpose {purpose!r}; expected one of {_PURPOSES}.")
    return purpose


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
            purpose=_resolve_purpose(purpose),
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
