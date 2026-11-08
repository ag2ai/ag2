# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from ag2.files import FileContent, FilesAPI, UploadedFile
from ag2.files.protocol import FilesClient
from ag2.testing import TestConfig, TestFilesClient


async def upload_report(client: FilesClient) -> UploadedFile:
    return await client.upload(b"report", "report.txt", "assistants")


@pytest.mark.asyncio
class TestFilesAPIUpload:
    async def test_works_where_a_files_client_is_expected(self) -> None:
        files = FilesAPI(TestConfig(files=TestFilesClient()))

        uploaded = await upload_report(files)

        assert await files.list() == [uploaded]
        assert await files.read(uploaded.file_id) == FileContent(name="report.txt", data=b"report")

    async def test_reads_the_file_from_a_path(self, tmp_path: Path) -> None:
        files = FilesAPI(TestConfig(files=TestFilesClient()))
        path = tmp_path / "notes.md"
        path.write_bytes(b"# notes")

        uploaded = await files.upload(path=path, purpose="retrieval")

        assert uploaded.filename == "notes.md"
        assert uploaded.purpose == "retrieval"
        assert await files.read(uploaded.file_id) == FileContent(name="notes.md", data=b"# notes")

    async def test_rejects_a_path_passed_as_data(self) -> None:
        files = FilesAPI(TestConfig(files=TestFilesClient()))

        with pytest.raises(TypeError, match="path="):
            await files.upload("report.pdf")

        assert await files.list() == []

    async def test_keeps_surviving_files_after_a_deletion(self) -> None:
        files = FilesAPI(TestConfig(files=TestFilesClient()))
        first = await files.upload(b"a", "a.txt")
        second = await files.upload(b"b", "b.txt")
        await files.delete(first.file_id)

        third = await files.upload(b"c", "c.txt")

        assert await files.list() == [second, third]
        assert await files.read(second.file_id) == FileContent(name="b.txt", data=b"b")
        assert await files.read(third.file_id) == FileContent(name="c.txt", data=b"c")
