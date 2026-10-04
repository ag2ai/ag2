# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from typing_extensions import TypeVar

from .types import FileContent, UploadedFile

# What a provider accepts as an upload purpose: a plain string by default, or the closed
# set its SDK types (e.g. OpenAI's `FilePurpose`). It only appears as a parameter, so a
# client taking `str` is also usable wherever a narrower purpose is expected.
P = TypeVar("P", contravariant=True, default=str)


@runtime_checkable
class FilesClient(Protocol[P]):
    async def upload(self, data: bytes, filename: str, purpose: P | None = None) -> UploadedFile: ...

    async def read(self, file_id: str) -> FileContent: ...

    async def list(self) -> list[UploadedFile]: ...

    async def delete(self, file_id: str) -> None: ...
