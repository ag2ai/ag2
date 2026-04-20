# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import logging

import pytest
from dirty_equals import IsPartialDict

from autogen.beta import ToolResult
from autogen.beta.config.anthropic.mappers import convert_messages
from autogen.beta.events import (
    AudioInput,
    AudioUrlInput,
    BinaryInput,
    BinaryType,
    DocumentInput,
    DocumentUrlInput,
    FileIdInput,
    ImageInput,
    ImageUrlInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
    VideoInput,
    VideoUrlInput,
)
from autogen.beta.exceptions import UnsupportedInputError


def _model_response_with_tool_call(arguments: str | None) -> ModelResponse:
    """Helper to build a ModelResponse containing a single tool call."""
    return ModelResponse(
        message=None,
        tool_calls=ToolCallsEvent(
            calls=[ToolCallEvent(id="tc_1", name="list_items", arguments=arguments)],
        ),
    )


def _matching_tool_result(content: str = "ok") -> ToolResultsEvent:
    """Companion ToolResultsEvent so the tool_use above isn't dropped as an orphan."""
    return ToolResultsEvent(
        results=[
            ToolResultEvent(
                parent_id="tc_1",
                name="list_items",
                result=ToolResult(content=content),
            )
        ],
    )


class TestConvertMessagesEmptyArguments:
    """json.loads must not crash on empty or None tool call arguments."""

    @pytest.mark.parametrize("arguments", ["", None])
    def test_empty_arguments_produce_empty_dict(self, arguments: str | None) -> None:
        response = _model_response_with_tool_call(arguments)
        result = convert_messages([response, _matching_tool_result()])

        assert result[0] == IsPartialDict({
            "role": "assistant",
            "content": [IsPartialDict({"type": "tool_use", "id": "tc_1", "name": "list_items", "input": {}})],
        })

    def test_valid_arguments_are_preserved(self) -> None:
        response = _model_response_with_tool_call('{"category": "books"}')
        result = convert_messages([response, _matching_tool_result()])

        assert result[0] == IsPartialDict({
            "content": [IsPartialDict({"type": "tool_use", "input": {"category": "books"}})],
        })

    def test_empty_object_arguments(self) -> None:
        response = _model_response_with_tool_call("{}")
        result = convert_messages([response, _matching_tool_result()])

        assert result[0] == IsPartialDict({
            "content": [IsPartialDict({"type": "tool_use", "input": {}})],
        })


def test_full_sequence_with_empty_args() -> None:
    """A request -> response-with-tool-call -> tool-result sequence should convert cleanly."""
    events = [
        ModelRequest([TextInput("What items do we have?")]),
        _model_response_with_tool_call(""),
        ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="list_items",
                    result=ToolResult(content="apple, banana"),
                )
            ],
        ),
    ]
    result = convert_messages(events)

    assert result[0] == IsPartialDict({"role": "user"})
    assert result[1] == IsPartialDict({
        "role": "assistant",
        "content": [IsPartialDict({"input": {}})],
    })
    assert result[2] == IsPartialDict({
        "role": "user",
        "content": [IsPartialDict({"type": "tool_result"})],
    })


class TestImageUrlInput:
    IMAGE_URL = "https://example.com/image.png"

    def test_converts_to_image_url_block(self) -> None:
        result = convert_messages([ModelRequest([ImageUrlInput(url=self.IMAGE_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "url", "url": self.IMAGE_URL}}],
            }
        ]


class TestImageBinaryInput:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_converts_to_image_base64_block(self) -> None:
        result = convert_messages([ModelRequest([ImageInput(data=self.SAMPLE_BYTES, media_type="image/png")])])

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": expected_b64},
                    }
                ],
            }
        ]

    def test_vendor_metadata_cache_control_merges(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="image/png",
                    vendor_metadata={"cache_control": {"type": "ephemeral"}},
                    kind=BinaryType.IMAGE,
                )
            ])
        ])

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "image", "cache_control": {"type": "ephemeral"}})],
            })
        ]

    def test_vendor_metadata_filename_filtered_out(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="image/png",
                    vendor_metadata={"filename": "photo.png"},
                    kind=BinaryType.IMAGE,
                )
            ])
        ])

        content = result[0]["content"][0]
        assert "filename" not in content


class TestDocumentUrlInput:
    DOC_URL = "https://example.com/doc.pdf"

    def test_converts_to_document_url_block(self) -> None:
        result = convert_messages([ModelRequest([DocumentUrlInput(url=self.DOC_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "url", "url": self.DOC_URL}}],
            }
        ]


class TestDocumentBinaryInput:
    SAMPLE_BYTES = b"%PDF-1.4"

    def test_converts_to_document_base64_block(self) -> None:
        result = convert_messages([ModelRequest([DocumentInput(data=self.SAMPLE_BYTES, media_type="application/pdf")])])

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": expected_b64},
                    }
                ],
            }
        ]

    def test_vendor_metadata_merges(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="application/pdf",
                    vendor_metadata={"cache_control": {"type": "ephemeral"}},
                    kind=BinaryType.DOCUMENT,
                )
            ])
        ])

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "document", "cache_control": {"type": "ephemeral"}})],
            })
        ]


class TestFileIdInput:
    FILE_ID = "file_011CNha8iCJcU1wXNR6q4V8w"

    def test_no_filename_defaults_to_document(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_image_filename_uses_image_block(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="photo.jpg")])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_pdf_filename_uses_document_block(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="report.pdf")])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]


class TestMultipleInputs:
    def test_multiple_inputs_grouped_into_one_message(self) -> None:
        result = convert_messages([
            ModelRequest([
                TextInput("Describe these images."),
                ImageUrlInput(url="https://example.com/a.png"),
                ImageUrlInput(url="https://example.com/b.jpg"),
            ])
        ])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 3
        assert result[0]["content"][0] == {"type": "text", "text": "Describe these images."}
        assert result[0]["content"][1] == IsPartialDict({"type": "image"})
        assert result[0]["content"][2] == IsPartialDict({"type": "image"})


class TestUnsupportedInputs:
    def test_audio_url_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="AudioUrlInput.*anthropic"):
            convert_messages([ModelRequest([AudioUrlInput(url="https://example.com/audio.wav")])])

    def test_video_url_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="VideoUrlInput.*anthropic"):
            convert_messages([ModelRequest([VideoUrlInput(url="https://example.com/video.mp4")])])

    def test_audio_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*audio.*anthropic"):
            convert_messages([ModelRequest([AudioInput(data=b"\x00audio", media_type="audio/wav")])])

    def test_video_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*video.*anthropic"):
            convert_messages([ModelRequest([VideoInput(data=b"\x00video", media_type="video/mp4")])])

    def test_generic_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*binary.*anthropic"):
            convert_messages([
                ModelRequest([BinaryInput(data=b"\x00", media_type="application/octet-stream", kind=BinaryType.BINARY)])
            ])


class TestOrphanedToolResults:
    """Tool results whose matching tool_use was trimmed should be filtered out."""

    def test_orphaned_tool_result_is_dropped(self) -> None:
        """A ToolResultsEvent with no matching ModelResponse tool_use should be omitted."""
        events = [
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="orphan_id",
                        name="missing_tool",
                        result=ToolResult(content="stale"),
                    )
                ],
            ),
            ModelRequest([TextInput("hello")]),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "text", "text": "hello"})],
            })
        ]

    def test_mixed_orphaned_and_valid_results(self) -> None:
        """Only valid tool results are kept; orphaned ones are filtered from the same event."""
        events = [
            _model_response_with_tool_call('{"q": "test"}'),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="list_items",
                        result=ToolResult(content="valid"),
                    ),
                    ToolResultEvent(
                        parent_id="orphan_id",
                        name="gone_tool",
                        result=ToolResult(content="stale"),
                    ),
                ],
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({"role": "assistant"}),
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "tool_result", "tool_use_id": "tc_1"})],
            }),
        ]

    def test_all_results_orphaned_skips_entire_block(self) -> None:
        """If every result in a ToolResultsEvent is orphaned, no user message is emitted."""
        events = [
            ModelRequest([TextInput("hi")]),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="gone_1",
                        name="a",
                        result=ToolResult(content="x"),
                    ),
                    ToolResultEvent(
                        parent_id="gone_2",
                        name="b",
                        result=ToolResult(content="y"),
                    ),
                ],
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "text", "text": "hi"})],
            })
        ]

    def test_valid_tool_results_are_preserved(self) -> None:
        """Normal flow: tool results with matching tool_use IDs pass through."""
        events = [
            ModelRequest([TextInput("go")]),
            _model_response_with_tool_call("{}"),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="list_items",
                        result=ToolResult(content="items"),
                    )
                ],
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({"role": "user"}),
            IsPartialDict({"role": "assistant"}),
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "tool_result", "tool_use_id": "tc_1"})],
            }),
        ]


class TestOrphanedToolUse:
    """tool_use blocks with no matching tool_result must be dropped; an
    Anthropic request with orphan tool_use is rejected with "messages.N:
    tool_use ids were found without tool_result blocks immediately after".
    """

    def test_orphan_tool_use_is_dropped(self) -> None:
        """Assistant tool_use without any ToolResultsEvent → drop it."""
        events = [
            ModelRequest([TextInput("hello")]),
            _model_response_with_tool_call("{}"),
        ]
        result = convert_messages(events)

        # Assistant message is completely omitted (only tool_use, no text)
        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "text", "text": "hello"})],
            })
        ]

    def test_orphan_tool_use_preserves_assistant_text(self) -> None:
        """If the assistant has text alongside an orphan tool_use, keep the text."""
        events = [
            ModelRequest([TextInput("hi")]),
            ModelResponse(
                message=ModelMessage(content="Let me check that."),
                tool_calls=ToolCallsEvent(
                    calls=[ToolCallEvent(id="tc_1", name="lookup", arguments="{}")],
                ),
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({"role": "user"}),
            IsPartialDict({
                "role": "assistant",
                "content": [{"type": "text", "text": "Let me check that."}],
            }),
        ]

    def test_orphan_tool_use_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Dropping an orphan tool_use should leave a debugging breadcrumb."""
        events = [_model_response_with_tool_call("{}")]
        with caplog.at_level(logging.WARNING, logger="autogen.beta.config.anthropic.mappers"):
            convert_messages(events)

        assert any("orphan tool_use" in rec.message for rec in caplog.records)
        assert any("tc_1" in rec.message for rec in caplog.records)

    def test_resolved_tool_use_is_not_dropped(self) -> None:
        """Sanity check: tool_use with a matching tool_result passes through."""
        events = [
            _model_response_with_tool_call("{}"),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="list_items",
                        result=ToolResult(content="ok"),
                    )
                ],
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({
                "role": "assistant",
                "content": [IsPartialDict({"type": "tool_use", "id": "tc_1"})],
            }),
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "tool_result", "tool_use_id": "tc_1"})],
            }),
        ]

    def test_mixed_resolved_and_orphan_tool_uses(self) -> None:
        """With two tool_use calls, drop only the one without a matching result."""
        events = [
            ModelResponse(
                message=None,
                tool_calls=ToolCallsEvent(
                    calls=[
                        ToolCallEvent(id="tc_1", name="resolved", arguments="{}"),
                        ToolCallEvent(id="tc_2", name="orphan", arguments="{}"),
                    ],
                ),
            ),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="resolved",
                        result=ToolResult(content="ok"),
                    )
                ],
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({
                "role": "assistant",
                "content": [IsPartialDict({"type": "tool_use", "id": "tc_1"})],
            }),
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "tool_result", "tool_use_id": "tc_1"})],
            }),
        ]


class TestIndividualToolResultFallback:
    """If the ``ToolResultsEvent`` wrapper fails to persist (e.g., a turn
    crash on a shared stream), loose ``ToolResultEvent`` /
    ``ToolErrorEvent`` entries should still resolve their parent tool_use.
    """

    def test_individual_tool_result_resolves_tool_use(self) -> None:
        events = [
            _model_response_with_tool_call("{}"),
            ToolResultEvent(
                parent_id="tc_1",
                name="list_items",
                result=ToolResult(content="fallback-result"),
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({
                "role": "assistant",
                "content": [IsPartialDict({"type": "tool_use", "id": "tc_1"})],
            }),
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({
                    "type": "tool_result",
                    "tool_use_id": "tc_1",
                })],
            }),
        ]
        # ToolResult wraps content as JSON — assert the value is present.
        assert "fallback-result" in result[1]["content"][0]["content"]

    def test_individual_tool_error_resolves_tool_use(self) -> None:
        events = [
            _model_response_with_tool_call("{}"),
            ToolErrorEvent(
                parent_id="tc_1",
                name="list_items",
                error=ValueError("boom"),
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({"role": "assistant"}),
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({
                    "type": "tool_result",
                    "tool_use_id": "tc_1",
                })],
            }),
        ]

    def test_wrapper_takes_precedence_over_individual(self) -> None:
        """If both ToolResultsEvent and its individual events are present
        in the stream (the race-survivor scenario), we must not double-emit.
        """
        events = [
            _model_response_with_tool_call("{}"),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="list_items",
                        result=ToolResult(content="wrapped"),
                    )
                ],
            ),
            ToolResultEvent(
                parent_id="tc_1",
                name="list_items",
                result=ToolResult(content="loose"),
            ),
        ]
        result = convert_messages(events)

        tool_result_msgs = [m for m in result if m["role"] == "user"]
        assert len(tool_result_msgs) == 1
        # Wrapped result was emitted; loose event was skipped (no double-emit).
        assert "wrapped" in tool_result_msgs[0]["content"][0]["content"]
        assert "loose" not in tool_result_msgs[0]["content"][0]["content"]

    def test_individual_event_without_parent_in_stream_is_skipped(self) -> None:
        """Loose tool_result with no matching tool_use is filtered (mirror of orphan tool_result)."""
        events = [
            ModelRequest([TextInput("hi")]),
            ToolResultEvent(
                parent_id="ghost_id",
                name="nope",
                result=ToolResult(content="stale"),
            ),
        ]
        result = convert_messages(events)

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "text", "text": "hi"})],
            })
        ]
