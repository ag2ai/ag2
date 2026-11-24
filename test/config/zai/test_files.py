# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from zai.core import (
    APIAuthenticationError,
    APIInternalError,
    APIReachLimitError,
    APIServerFlowExceedError,
    APIStatusError,
    APITimeoutError,
)

from ag2 import FilesAPI
from ag2.config import ZAIConfig
from ag2.config.zai.files import ZAIFilesClient
from ag2.files.types import FileContent, FileProvider, UploadedFile


@patch("ag2.config.zai.files.ZaiClient")
def test_files_client_construction_omits_none_options(mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
    ZAIFilesClient(zai_config)

    mock_zai_client.assert_called_once_with(
        api_key="test-key",
        max_retries=3,
        disable_token_cache=True,
    )


@pytest.mark.asyncio
class TestZAIFilesClient:
    # The zai SDK is synchronous (wrapped via asyncio.to_thread), so the inner client is a
    # MagicMock, not an AsyncMock.

    @patch("ag2.config.zai.files.ZaiClient")
    async def test_upload_defaults_to_batch(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.create.return_value = SimpleNamespace(
            id="file_123",
            filename="hello.jsonl",
            bytes=5,
            purpose="batch",
            created_at=123,
        )

        result = await ZAIFilesClient(zai_config).upload(b"hello", "hello.jsonl")

        assert result == UploadedFile(
            file_id="file_123",
            filename="hello.jsonl",
            provider=FileProvider.ZAI,
            bytes_count=5,
            purpose="batch",
            created_at=123.0,
        )
        kwargs = mock_client.files.create.call_args.kwargs
        assert kwargs["purpose"] == "batch"
        # Provider-specific extras are omitted unless the caller sets them.
        assert "knowledge_id" not in kwargs
        name, buffer = kwargs["file"]
        assert name == "hello.jsonl"
        assert buffer.read() == b"hello"

    @patch("ag2.config.zai.files.ZaiClient")
    async def test_upload_forwards_retrieval_options(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.create.return_value = SimpleNamespace(
            id="file_123", filename="doc.pdf", bytes=5, purpose="retrieval", created_at=1
        )

        await ZAIFilesClient(zai_config).upload(
            b"hello",
            "doc.pdf",
            "retrieval",
            knowledge_id="kb-123",
            sentence_size=256,
            custom_separator=["\n"],
        )

        kwargs = mock_client.files.create.call_args.kwargs
        assert kwargs["purpose"] == "retrieval"
        assert kwargs["knowledge_id"] == "kb-123"
        assert kwargs["sentence_size"] == 256
        assert kwargs["custom_separator"] == ["\n"]

    @patch("ag2.config.zai.files.ZaiClient")
    async def test_upload_with_explicit_purpose(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.create.return_value = SimpleNamespace(
            id="file_123",
            filename="data.jsonl",
            bytes=10,
            purpose="batch",
            created_at=1,
        )

        result = await ZAIFilesClient(zai_config).upload(b"payload---", "data.jsonl", "batch")

        assert result == UploadedFile(
            file_id="file_123",
            filename="data.jsonl",
            provider=FileProvider.ZAI,
            bytes_count=10,
            purpose="batch",
            created_at=1.0,
        )
        assert mock_client.files.create.call_args.kwargs["purpose"] == "batch"

    @patch("ag2.config.zai.files.ZaiClient")
    async def test_read_returns_bytes_without_metadata(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.content.return_value = SimpleNamespace(content=b"file-bytes")

        result = await ZAIFilesClient(zai_config).read("file_123")

        assert result == FileContent(name=None, data=b"file-bytes", media_type=None)
        mock_client.files.content.assert_called_once_with("file_123")

    @patch("ag2.config.zai.files.ZaiClient")
    async def test_list_aggregates_across_purposes(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client

        # GET /files needs a purpose filter, so the client queries each listable purpose
        # and merges the results.
        per_purpose = {
            "batch": [SimpleNamespace(id="file_1", filename="a.jsonl", bytes=100, purpose="batch", created_at=1)],
            "fine-tune": [
                SimpleNamespace(id="file_2", filename="b.jsonl", bytes=200, purpose="fine-tune", created_at=2)
            ],
        }
        mock_client.files.list.side_effect = lambda *, purpose: SimpleNamespace(data=per_purpose.get(purpose, []))

        result = await ZAIFilesClient(zai_config).list()

        assert result == [
            UploadedFile(
                file_id="file_1",
                filename="a.jsonl",
                provider=FileProvider.ZAI,
                bytes_count=100,
                purpose="batch",
                created_at=1.0,
            ),
            UploadedFile(
                file_id="file_2",
                filename="b.jsonl",
                provider=FileProvider.ZAI,
                bytes_count=200,
                purpose="fine-tune",
                created_at=2.0,
            ),
        ]

    @patch("ag2.config.zai.files.ZaiClient")
    async def test_delete(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client

        await ZAIFilesClient(zai_config).delete("file_123")

        mock_client.files.delete.assert_called_once_with("file_123")

    @patch("ag2.config.zai.files.ZaiClient")
    async def test_upload_rejects_a_file_without_an_id(self, mock_zai_client: MagicMock, zai_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_zai_client.return_value = mock_client
        mock_client.files.create.return_value = SimpleNamespace(
            id=None, filename="hello.jsonl", bytes=5, purpose="batch", created_at=123
        )

        with pytest.raises(ValueError, match="no id"):
            await ZAIFilesClient(zai_config).upload(b"hello", "hello.jsonl")


@pytest.fixture
def file_list_transport() -> MagicMock:
    return MagicMock()


@pytest.fixture
def files_api(file_list_transport: MagicMock) -> Generator[FilesAPI]:
    with httpx.Client(transport=httpx.MockTransport(file_list_transport)) as http_client:
        yield FilesAPI(ZAIConfig(model="test", api_key="test-key", max_retries=0, http_client=http_client))


@pytest.mark.asyncio
class TestZAIFileListingErrors:
    @pytest.mark.parametrize(
        ("status_code", "error_type"),
        [
            (401, APIAuthenticationError),
            (429, APIReachLimitError),
            (500, APIInternalError),
            (503, APIServerFlowExceedError),
        ],
    )
    async def test_all_failed_requests_raise_sdk_error(
        self, files_api: FilesAPI, file_list_transport: MagicMock, status_code: int, error_type: type[APIStatusError]
    ) -> None:
        file_list_transport.side_effect = lambda request: httpx.Response(
            status_code,
            json={"error": {"code": str(status_code), "message": f"Cannot list {request.url.params['purpose']}"}},
        )

        with pytest.raises(error_type, match="Cannot list batch") as exc:
            await files_api.list()

        assert exc.value.status_code == status_code
        assert exc.value.response.json() == {"error": {"code": str(status_code), "message": "Cannot list batch"}}
        assert file_list_transport.call_count == 3

    async def test_all_timed_out_requests_raise_sdk_error(
        self, files_api: FilesAPI, file_list_transport: MagicMock
    ) -> None:
        timeout = httpx.ReadTimeout("Timed out listing files")
        file_list_transport.side_effect = timeout

        with pytest.raises(APITimeoutError) as exc:
            await files_api.list()

        assert exc.value.__cause__ is timeout
        assert file_list_transport.call_count == 3

    @pytest.mark.parametrize(
        "successful_purposes",
        [("batch",), ("fine-tune",), ("voice-clone-input",), ("batch", "fine-tune", "voice-clone-input")],
    )
    @pytest.mark.parametrize("has_files", [False, True])
    async def test_successful_requests_still_return_files_or_empty_list(
        self, files_api: FilesAPI, file_list_transport: MagicMock, successful_purposes: tuple[str, ...], has_files: bool
    ) -> None:
        file_list_transport.side_effect = lambda request: httpx.Response(
            200 if request.url.params["purpose"] in successful_purposes else 503,
            json={
                "data": [
                    {
                        "id": f"file-{request.url.params['purpose']}",
                        "filename": "jobs.jsonl",
                        "bytes": 12,
                        "purpose": request.url.params["purpose"],
                        "created_at": 123,
                    }
                ]
                if has_files
                else [],
            },
        )

        result = await files_api.list()

        assert result == [
            UploadedFile(
                file_id=f"file-{purpose}",
                filename="jobs.jsonl",
                provider=FileProvider.ZAI,
                bytes_count=12,
                purpose=purpose,
                created_at=123.0,
            )
            for purpose in successful_purposes
            if has_files
        ]
        assert file_list_transport.call_count == 3
