# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import io
import mimetypes
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from ag2.files.types import FileContent, FileProvider, UploadedFile, _created_at_to_float

if TYPE_CHECKING:
    from ag2.config.gemini.config import GeminiConfig


def _file_id(file: types.File) -> str:
    """Read the resource name off a file, which the SDK declares optional on every field.

    A file with no name cannot be read, sent or deleted, so it is named here rather than
    carried on as `None` through an `UploadedFile` that promises a `str`.
    """
    if file.name is None:
        raise ValueError("Gemini answered with a file that has no resource name.")
    return file.name


class GeminiFilesClient:
    """Files API client for Google Gemini."""

    __slots__ = ("_client",)

    def __init__(self, config: "GeminiConfig") -> None:
        self._client = genai.Client(api_key=config.api_key)

    async def upload(self, data: bytes, filename: str, purpose: str | None = None) -> UploadedFile:
        mime_type, _ = mimetypes.guess_type(filename)
        # The SDK's own TypedDict for the mapping form it accepts, so the literal below
        # stays a plain dict and is still checked against what `upload` takes.
        config: types.UploadFileConfigDict = {
            "display_name": filename,
            "mime_type": mime_type or "application/octet-stream",
        }
        result = await self._client.aio.files.upload(file=io.BytesIO(data), config=config)

        return UploadedFile(
            file_id=_file_id(result),
            filename=filename,
            provider=FileProvider.GEMINI,
            bytes_count=result.size_bytes if result.size_bytes is not None else len(data),
            purpose=purpose,
            created_at=_created_at_to_float(result.create_time),
        )

    async def read(self, file_id: str) -> FileContent:
        file_info = await self._client.aio.files.get(name=file_id)
        if file_info.download_uri is None:
            raise NotImplementedError(
                "Gemini does not allow downloading user-uploaded files. "
                "Only model-generated files with a download_uri can be downloaded."
            )
        data = await self._client.aio.files.download(file=file_info)
        return FileContent(
            name=file_info.display_name if file_info.display_name else None,
            data=data,
            media_type=file_info.mime_type if file_info.mime_type else None,
        )

    async def list(self) -> list[UploadedFile]:
        pager = await self._client.aio.files.list()
        return [
            UploadedFile(
                file_id=_file_id(f),
                filename=f.display_name if f.display_name else None,
                provider=FileProvider.GEMINI,
                bytes_count=f.size_bytes if f.size_bytes else None,
                purpose=None,
                created_at=_created_at_to_float(f.create_time),
            )
            for f in pager.page
        ]

    async def delete(self, file_id: str) -> None:
        await self._client.aio.files.delete(name=file_id)
