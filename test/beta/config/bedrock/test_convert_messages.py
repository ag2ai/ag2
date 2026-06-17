# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from autogen.beta import ToolResult
from autogen.beta.compact import CompactionSummary
from autogen.beta.config.bedrock.mappers import convert_messages
from autogen.beta.events import (
    BinaryInput,
    BinaryType,
    DataInput,
    DocumentInput,
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.exceptions import UnsupportedInputError


def _model_response_with_tool_call(arguments: str | None, text: str | None = None) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(text) if text else None,
        tool_calls=ToolCallsEvent(
            calls=[ToolCallEvent(id="tc_1", name="list_items", arguments=arguments)],
        ),
    )


def _matching_tool_result(content: str = "ok") -> ToolResultsEvent:
    """Companion ToolResultsEvent so the toolUse above isn't dropped as an orphan."""
    return ToolResultsEvent(
        results=[
            ToolResultEvent(
                parent_id="tc_1",
                name="list_items",
                result=ToolResult(content),
            )
        ],
    )


class TestModelRequest:
    def test_text_input(self) -> None:
        result = convert_messages([ModelRequest([TextInput("hello")])], SerializerCls)

        assert result == [{"role": "user", "content": [{"text": "hello"}]}]

    def test_data_input_is_serialized(self) -> None:
        data = {"category": "books", "limit": 3}
        result = convert_messages([ModelRequest([DataInput(data)])], SerializerCls)

        assert result == [{"role": "user", "content": [{"text": SerializerCls.encode(data).decode()}]}]

    def test_image_bytes(self) -> None:
        result = convert_messages(
            [ModelRequest([ImageInput(data=b"img-bytes", media_type="image/png")])],
            SerializerCls,
        )

        assert result == [
            {
                "role": "user",
                "content": [{"image": {"format": "png", "source": {"bytes": b"img-bytes"}}}],
            }
        ]

    def test_document_bytes_default_name(self) -> None:
        result = convert_messages(
            [ModelRequest([DocumentInput(data=b"pdf-bytes", media_type="application/pdf")])],
            SerializerCls,
        )

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": "pdf",
                            "name": "document",
                            "source": {"bytes": b"pdf-bytes"},
                        },
                    }
                ],
            }
        ]

    def test_document_name_is_sanitized(self) -> None:
        doc = BinaryInput(
            b"pdf-bytes",
            media_type="application/pdf",
            vendor_metadata={"filename": "Q3 report!.pdf"},
            kind=BinaryType.DOCUMENT,
        )
        result = convert_messages([ModelRequest([doc])], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": "pdf",
                            "name": "Q3 report pdf",
                            "source": {"bytes": b"pdf-bytes"},
                        },
                    }
                ],
            }
        ]

    def test_url_input_raises(self) -> None:
        with pytest.raises(UnsupportedInputError):
            convert_messages([ModelRequest([ImageInput("https://example.com/image.png")])], SerializerCls)

    def test_file_id_input_raises(self) -> None:
        with pytest.raises(UnsupportedInputError):
            convert_messages([ModelRequest([ImageInput(file_id="file-abc123")])], SerializerCls)

    def test_unknown_image_media_type_raises(self) -> None:
        image = BinaryInput(b"img", media_type="image/tiff", kind=BinaryType.IMAGE)

        with pytest.raises(UnsupportedInputError):
            convert_messages([ModelRequest([image])], SerializerCls)


class TestModelResponse:
    def test_text_only(self) -> None:
        response = ModelResponse(message=ModelMessage("hi there"))
        result = convert_messages([ModelRequest([TextInput("hi")]), response], SerializerCls)

        assert result[1] == {"role": "assistant", "content": [{"text": "hi there"}]}

    def test_text_and_tool_use(self) -> None:
        response = _model_response_with_tool_call('{"category": "books"}', text="Let me check.")
        result = convert_messages([response, _matching_tool_result()], SerializerCls)

        assert result[-2] == {
            "role": "assistant",
            "content": [
                {"text": "Let me check."},
                {
                    "toolUse": {
                        "toolUseId": "tc_1",
                        "name": "list_items",
                        "input": {"category": "books"},
                    },
                },
            ],
        }

    @pytest.mark.parametrize("arguments", ["", None])
    def test_empty_arguments_produce_empty_dict(self, arguments: str | None) -> None:
        response = _model_response_with_tool_call(arguments)
        result = convert_messages([response, _matching_tool_result()], SerializerCls)

        assert result[-2]["content"] == [
            {"toolUse": {"toolUseId": "tc_1", "name": "list_items", "input": {}}},
        ]


class TestToolResults:
    def test_text_result(self) -> None:
        events = [_model_response_with_tool_call("{}"), _matching_tool_result("apple, banana")]
        result = convert_messages(events, SerializerCls)

        assert result[-1] == {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tc_1",
                        "content": [{"text": "apple, banana"}],
                    },
                }
            ],
        }

    def test_error_result_sets_status(self) -> None:
        events = [
            _model_response_with_tool_call("{}"),
            ToolResultsEvent(
                results=[
                    ToolErrorEvent(
                        parent_id="tc_1",
                        name="list_items",
                        error=ValueError("boom"),
                        result=ToolResult("boom"),
                    )
                ],
            ),
        ]
        result = convert_messages(events, SerializerCls)

        assert result[-1] == {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tc_1",
                        "content": [{"text": "boom"}],
                        "status": "error",
                    },
                }
            ],
        }

    def test_image_result_part(self) -> None:
        events = [
            _model_response_with_tool_call("{}"),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(
                        parent_id="tc_1",
                        name="list_items",
                        result=ToolResult(ImageInput(data=b"img", media_type="image/jpeg")),
                    )
                ],
            ),
        ]
        result = convert_messages(events, SerializerCls)

        assert result[-1] == {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tc_1",
                        "content": [{"image": {"format": "jpeg", "source": {"bytes": b"img"}}}],
                    },
                }
            ],
        }


class TestRoleAlternation:
    def test_consecutive_user_messages_are_merged(self) -> None:
        events = [
            ModelRequest([TextInput("first")]),
            ModelRequest([TextInput("second")]),
        ]
        result = convert_messages(events, SerializerCls)

        assert result == [{"role": "user", "content": [{"text": "first"}, {"text": "second"}]}]

    def test_tool_result_and_request_share_a_user_turn(self) -> None:
        events = [
            ModelRequest([TextInput("what do we have?")]),
            _model_response_with_tool_call("{}"),
            _matching_tool_result("apple"),
            ModelRequest([TextInput("now summarize")]),
        ]
        result = convert_messages(events, SerializerCls)

        assert [m["role"] for m in result] == ["user", "assistant", "user"]
        assert result[-1]["content"] == [
            {"toolResult": {"toolUseId": "tc_1", "content": [{"text": "apple"}]}},
            {"text": "now summarize"},
        ]

    def test_assistant_first_history_gets_user_filler(self) -> None:
        events = [ModelResponse(message=ModelMessage("continuing from before"))]
        result = convert_messages(events, SerializerCls)

        assert result == [
            {"role": "user", "content": [{"text": "Please continue."}]},
            {"role": "assistant", "content": [{"text": "continuing from before"}]},
        ]


class TestOrphanFiltering:
    def test_orphan_tool_use_is_dropped(self) -> None:
        events = [
            ModelRequest([TextInput("question")]),
            _model_response_with_tool_call("{}", text="thinking"),
        ]
        result = convert_messages(events, SerializerCls)

        assert result == [
            {"role": "user", "content": [{"text": "question"}]},
            {"role": "assistant", "content": [{"text": "thinking"}]},
        ]

    def test_orphan_tool_result_is_dropped(self) -> None:
        events = [
            ModelRequest([TextInput("question")]),
            _model_response_with_tool_call("{}", text="thinking"),
            ToolResultsEvent(
                results=[
                    ToolResultEvent(parent_id="tc_unknown", name="other", result=ToolResult("stale")),
                ],
            ),
        ]
        result = convert_messages(events, SerializerCls)

        assert [m["role"] for m in result] == ["user", "assistant"]

    def test_compacted_history_drops_unmatched_tool_results(self) -> None:
        """Compaction can leave tool results with no tool-bearing ModelResponse — still dropped."""
        events = [
            ToolCallEvent(id="tc_lost", name="remember", arguments="{}"),
            ToolResultEvent(parent_id="tc_lost", name="remember", result=ToolResult("stored")),
            ToolResultsEvent(
                results=[ToolResultEvent(parent_id="tc_lost", name="remember", result=ToolResult("stored"))],
            ),
            ModelResponse(message=ModelMessage("noted")),
            ModelRequest([TextInput("next question")]),
        ]
        result = convert_messages(events, SerializerCls)

        assert result == [
            {"role": "user", "content": [{"text": "Please continue."}]},
            {"role": "assistant", "content": [{"text": "noted"}]},
            {"role": "user", "content": [{"text": "next question"}]},
        ]

    def test_loose_tool_result_event_fallback(self) -> None:
        """An individual ToolResultEvent without its wrapper still renders a toolResult turn."""
        events = [
            _model_response_with_tool_call("{}"),
            ToolResultEvent(parent_id="tc_1", name="list_items", result=ToolResult("ok")),
        ]
        result = convert_messages(events, SerializerCls)

        assert result[-1] == {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "tc_1", "content": [{"text": "ok"}]}}],
        }


def test_compaction_summary_renders_as_user_turn() -> None:
    summary = CompactionSummary(summary="Looked up Paris and Tokyo.", event_count=6)

    result = convert_messages([summary], SerializerCls)

    assert result == [
        {"role": "user", "content": [{"text": "[Summary of earlier conversation]\nLooked up Paris and Tokyo."}]}
    ]
