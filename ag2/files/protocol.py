# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from .types import FileContent, UploadedFile


@runtime_checkable
class FilesClient(Protocol):
    async def upload(self, data: bytes, filename: str, purpose: str | None = None) -> UploadedFile: ...

    async def read(self, file_id: str) -> FileContent: ...

    async def list(self) -> list[UploadedFile]: ...

    async def delete(self, file_id: str) -> None: ...
