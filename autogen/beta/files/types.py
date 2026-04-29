# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass
from datetime import datetime, timezone
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

    async def read(self, client: "FilesAPI") -> FileContent:
        """Download the file content using the given FilesAPI client."""
        return await client.read(self.file_id)


def _datetime_to_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _created_at_to_float(value: object | None) -> float:
    if value is None:
        return time.time()

    if isinstance(value, int | float):
        return float(value)

    if isinstance(value, datetime):
        return _datetime_to_timestamp(value)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return time.time()
        try:
            return float(raw)
        except ValueError:
            normalized = f"{raw[:-1]}+00:00" if raw.endswith("Z") else raw
            return _datetime_to_timestamp(datetime.fromisoformat(normalized))

    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        return float(timestamp())

    return time.time()
