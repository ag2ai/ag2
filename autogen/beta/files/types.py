# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from autogen.beta.events.input_events import FileIdInput

if TYPE_CHECKING:
    from .api import FilesAPI


@dataclass(frozen=True, slots=True)
class FileContent:
    name: str | None
    data: bytes
    media_type: str | None = None


class FileProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class UploadedFile(FileIdInput):
    provider: FileProvider | None = None
    bytes_count: int | None = None
    purpose: str | None = None
    created_at: str | None = None

    async def read(self, client: "FilesAPI") -> FileContent:
        """Download the file content using the given FilesAPI client."""
        return await client.read(self.file_id)
