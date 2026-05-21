# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from autogen.beta.events import BinaryInput, DocumentInput, FileIdInput, UrlInput


def test_url_returns_document_url_input() -> None:
    result = DocumentInput("https://example.com/doc.pdf")

    assert isinstance(result, UrlInput)
    assert result.url == "https://example.com/doc.pdf"


class TestFileId:
    def test_returns_document_file_id_input(self) -> None:
        result = DocumentInput(file_id="file-abc123")

        assert isinstance(result, FileIdInput)
        assert result.file_id == "file-abc123"
        assert result.filename is None

    def test_with_filename(self) -> None:
        result = DocumentInput(file_id="file-abc123", filename="report.pdf")

        assert isinstance(result, FileIdInput)
        assert result.file_id == "file-abc123"
        assert result.filename == "report.pdf"


class TestData:
    def test_returns_binary_input(self) -> None:
        result = DocumentInput(data=b"raw", media_type="application/pdf")

        assert isinstance(result, BinaryInput)
        assert result.data == b"raw"
        assert result.media_type == "application/pdf"

    def test_missing_media_type_raises(self) -> None:
        with pytest.raises(ValueError, match="media_type"):
            DocumentInput(data=b"raw")


class TestPath:
    def test_infers_pdf(self, tmp_path: Path) -> None:
        f = tmp_path / "report.pdf"
        f.write_bytes(b"pdf-data")

        result = DocumentInput(path=f)

        assert isinstance(result, BinaryInput)
        assert result.data == b"pdf-data"
        assert result.media_type == "application/pdf"
        assert result.vendor_metadata == {"filename": "report.pdf"}

    def test_infers_csv(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b,c")

        result = DocumentInput(path=f)

        assert result.media_type == "text/csv"

    def test_infers_docx(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.docx"
        f.write_bytes(b"docx-data")

        result = DocumentInput(path=f)

        assert result.media_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.xyz"
        f.write_bytes(b"data")

        with pytest.raises(ValueError, match="Cannot infer"):
            DocumentInput(path=f)

    def test_unknown_extension_with_explicit_media_type(self, tmp_path: Path) -> None:
        f = tmp_path / "file.xyz"
        f.write_bytes(b"data")

        result = DocumentInput(path=f, media_type="application/pdf")

        assert isinstance(result, BinaryInput)
        assert result.media_type == "application/pdf"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.md"
        f.write_bytes(b"# Hello")

        result = DocumentInput(path=str(f))

        assert result.media_type == "text/markdown"


def test_no_args_raises() -> None:
    with pytest.raises(ValueError, match="requires one of"):
        DocumentInput()
