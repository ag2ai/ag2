# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsFloat

from autogen.beta.events import BinaryResult
from autogen.beta.events.input_events import FileIdInput, Input
from autogen.beta.files import FileContent, FileProvider, UploadedFile


class TestUploadedFile:
    def test_is_file_id_input(self) -> None:
        assert isinstance(UploadedFile(file_id="file-123", filename="test.pdf"), FileIdInput)

    def test_is_input(self) -> None:
        assert isinstance(UploadedFile(file_id="file-123"), Input)

    def test_fields_preserved(self) -> None:
        f = UploadedFile(
            file_id="file-123",
            filename="doc.pdf",
            provider=FileProvider.OPENAI,
            bytes_count=1024,
            purpose="assistants",
            created_at=1767225600.0,
        )

        assert f == UploadedFile(
            file_id="file-123",
            filename="doc.pdf",
            provider=FileProvider.OPENAI,
            bytes_count=1024,
            purpose="assistants",
            created_at=1767225600.0,
        )
        assert f.created_at == 1767225600.0

    def test_minimal_construction_defaults(self) -> None:
        assert UploadedFile(file_id="file-456") == UploadedFile(
            file_id="file-456",
            filename=None,
            provider=None,
            bytes_count=None,
            purpose=None,
            created_at=IsFloat(),
        )


class TestFileContent:
    def test_frozen(self) -> None:
        fc = FileContent(name="test.txt", data=b"hello")
        with pytest.raises(AttributeError):
            fc.name = "other.txt"  # type: ignore[misc]

    def test_fields_preserved(self) -> None:
        assert FileContent(name="doc.pdf", data=b"pdf-bytes", media_type="application/pdf") == FileContent(
            name="doc.pdf",
            data=b"pdf-bytes",
            media_type="application/pdf",
        )


class TestBinaryResult:
    def test_name_from_metadata(self) -> None:
        br = BinaryResult(data=b"image-bytes", metadata={"filename": "cat.png", "size": "512x512"})
        assert br.name == "cat.png"

    def test_name_default(self) -> None:
        assert BinaryResult(data=b"data").name == "generated_file"


@pytest.mark.asyncio
async def test_binary_result_content_returns_bytes() -> None:
    br = BinaryResult(data=b"raw-data", metadata={"filename": "out.bin"})
    assert await br.content() == b"raw-data"
