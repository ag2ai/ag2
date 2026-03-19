# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

from autogen.beta.events.tool_events import (
    ClientToolCallEvent,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResult,
)


class TestToolCallEventProviderData:
    """provider_data must default to dict, not list."""

    def test_default_is_dict(self) -> None:
        tc = ToolCallEvent(name="search")
        assert isinstance(tc.provider_data, dict)

    def test_dict_operations_work(self) -> None:
        tc = ToolCallEvent(name="search")
        tc.provider_data["key"] = "value"
        assert tc.provider_data.get("key") == "value"

    def test_each_instance_gets_own_dict(self) -> None:
        tc1 = ToolCallEvent(name="a")
        tc2 = ToolCallEvent(name="b")
        tc1.provider_data["x"] = 1
        assert "x" not in tc2.provider_data


class TestClientToolCallEventFromCall:
    """from_call must link back to the original call via parent_id."""

    def test_parent_id_matches_original_id(self) -> None:
        original = ToolCallEvent(name="search", arguments='{"q": "test"}')
        client_call = ClientToolCallEvent.from_call(original)
        assert client_call.parent_id == original.id

    def test_name_and_arguments_preserved(self) -> None:
        original = ToolCallEvent(name="calc", arguments='{"x": 1}')
        client_call = ClientToolCallEvent.from_call(original)
        assert client_call.name == original.name
        assert client_call.arguments == original.arguments


class TestToolErrorEventContent:
    """content must capture the traceback from self.error, not sys.exc_info()."""

    def test_content_contains_original_error(self) -> None:
        try:
            raise ValueError("test error message")
        except Exception as e:
            event = ToolErrorEvent(
                parent_id="call-123",
                name="test_tool",
                error=e,
                result=ToolResult(),
            )

        # Access content OUTSIDE the except block
        assert "ValueError" in event.content
        assert "test error message" in event.content

    def test_content_does_not_return_none_type(self) -> None:
        try:
            raise RuntimeError("something broke")
        except Exception as e:
            event = ToolErrorEvent(
                parent_id="x",
                name="t",
                error=e,
                result=ToolResult(),
            )

        assert "NoneType" not in event.content


class TestSerializedArgumentsCache:
    """serialized_arguments must cache even when the result is an empty dict."""

    def test_empty_dict_is_cached(self) -> None:
        tc = ToolCallEvent(name="tool", arguments="{}")
        with patch.object(json, "loads", wraps=json.loads) as mock_loads:
            _ = tc.serialized_arguments
            _ = tc.serialized_arguments
            _ = tc.serialized_arguments
            assert mock_loads.call_count == 1

    def test_non_empty_dict_is_cached(self) -> None:
        tc = ToolCallEvent(name="tool", arguments='{"key": "value"}')
        with patch.object(json, "loads", wraps=json.loads) as mock_loads:
            result = tc.serialized_arguments
            _ = tc.serialized_arguments
            assert mock_loads.call_count == 1
            assert result == {"key": "value"}

    def test_setter_updates_cache(self) -> None:
        tc = ToolCallEvent(name="tool", arguments='{"a": 1}')
        tc.serialized_arguments = {"b": 2}
        assert tc.serialized_arguments == {"b": 2}
