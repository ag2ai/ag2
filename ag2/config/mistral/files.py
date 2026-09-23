# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Final, get_args

from mistralai.client import Mistral
from mistralai.client.models import File, FilePurpose
from mistralai.client.types import UnrecognizedStr

from ag2.files.types import FileContent, FileProvider, UploadedFile, _created_at_to_float

if TYPE_CHECKING:
    from ag2.config.mistral.config import MistralConfig

# The only purpose accepting arbitrary bytes; "batch" and "fine-tune" require
# schema-conforming JSONL. Also the only one referenceable by id from a message.
_DEFAULT_PURPOSE: Final[FilePurpose] = "ocr"

# The purposes the SDK names, taken from its own literal rather than restated.
# `FilePurpose` is `Union[Literal[...], UnrecognizedStr]`; the literal is its first arm.
_KNOWN_PURPOSES: Final[tuple[FilePurpose, ...]] = get_args(get_args(FilePurpose)[0])

# The list endpoint paginates; 100 is the SDK's own page ceiling.
_PAGE_SIZE = 100


def _resolve_purpose(purpose: str | None) -> FilePurpose:
    """Mistral's purposes are an open enum, so one the SDK does not name is sent as ``UnrecognizedStr``."""
    if not purpose:
        return _DEFAULT_PURPOSE
    # A loop rather than `in`, so the match narrows to the SDK's literal.
    for known in _KNOWN_PURPOSES:
        if purpose == known:
            return known
    return UnrecognizedStr(purpose)


def _text(value: Any) -> str | None:
    """Normalise an omitted SDK field, which arrives as ``Unset()`` not ``None``."""
    return value if isinstance(value, str) else None


class MistralFilesClient:
    """Files API client for Mistral."""

    __slots__ = ("_client",)

    def __init__(self, config: "MistralConfig") -> None:
        kwargs: dict[str, Any] = {}
        if config.api_key is not None:
            kwargs["api_key"] = config.api_key
        if config.server_url is not None:
            kwargs["server_url"] = config.server_url
        if config.timeout_ms is not None:
            kwargs["timeout_ms"] = config.timeout_ms
        if config.async_client is not None:
            kwargs["async_client"] = config.async_client
        self._client = Mistral(**kwargs)

    async def upload(self, data: bytes, filename: str, purpose: str | None = None) -> UploadedFile:
        result = await self._client.files.upload_async(
            file=File(file_name=filename, content=data),
            purpose=_resolve_purpose(purpose),
        )
        return UploadedFile(
            file_id=result.id,
            filename=result.filename,
            provider=FileProvider.MISTRAL,
            bytes_count=result.size_bytes,
            purpose=result.purpose,
            created_at=_created_at_to_float(result.created_at),
        )

    async def read(self, file_id: str) -> FileContent:
        metadata = await self._client.files.retrieve_async(file_id=file_id)
        response = await self._client.files.download_async(file_id=file_id)
        # Response is an unconsumed stream; .content before aread() raises.
        data = await response.aread()
        return FileContent(
            name=metadata.filename,
            data=data,
            media_type=_text(metadata.mimetype),
        )

    async def list(self) -> list[UploadedFile]:
        files: list[UploadedFile] = []
        page = 0
        while True:
            result = await self._client.files.list_async(page=page, page_size=_PAGE_SIZE)
            batch = result.data or []
            files.extend(
                UploadedFile(
                    file_id=f.id,
                    filename=f.filename,
                    provider=FileProvider.MISTRAL,
                    bytes_count=f.size_bytes,
                    purpose=f.purpose,
                    created_at=_created_at_to_float(f.created_at),
                )
                for f in batch
            )
            if len(batch) < _PAGE_SIZE:
                return files
            page += 1

    async def delete(self, file_id: str) -> None:
        await self._client.files.delete_async(file_id=file_id)
