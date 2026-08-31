# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A failed hosted MCP call reports *how* it failed.

The Responses API answers a failed `mcp_call` with one of three arms, each
carrying different fields. A caller branches on the discriminator rather than
matching on prose, so these tests assert the discriminator survives into the
result event's metadata alongside that arm's own fields.
"""

from typing import Any

import pytest
from dirty_equals import IsPartialDict

from ag2.events import BuiltinToolResultEvent
from ag2.tools.builtin.mcp_server import MCP_SERVER_TOOL_NAME

from ._helpers import MCP_CALL, events_of, results


def _failed_call(error: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "mcp_err",
        "type": "mcp_call",
        "name": "ask_question",
        "server_label": "deepwiki",
        "arguments": "{}",
        "status": "failed",
        "error": error,
    }


async def _error_metadata(error: dict[str, Any]) -> Any:
    [result] = results(await events_of(_failed_call(error)))
    return result.result.metadata["error"]


@pytest.mark.asyncio
class TestEachArmIsReportedAsItself:
    async def test_protocol_error_carries_its_code_and_message(self) -> None:
        error = await _error_metadata({
            "type": "mcp_protocol_error",
            "code": -32601,
            "message": "Method not found",
        })

        assert error == {"type": "mcp_protocol_error", "code": -32601, "message": "Method not found"}

    async def test_tool_execution_error_carries_the_tools_content(self) -> None:
        error = await _error_metadata({
            "type": "mcp_tool_execution_error",
            "content": [{"type": "text", "text": "the repository does not exist"}],
        })

        assert error == {
            "type": "mcp_tool_execution_error",
            "content": [{"type": "text", "text": "the repository does not exist"}],
        }

    async def test_http_error_carries_its_code_and_message(self) -> None:
        error = await _error_metadata({"type": "http_error", "code": 503, "message": "Service Unavailable"})

        assert error == {"type": "http_error", "code": 503, "message": "Service Unavailable"}


@pytest.mark.asyncio
class TestTheThreeAreDistinguishable:
    async def test_without_parsing_message_text(self) -> None:
        kinds = [
            await _error_metadata({"type": "mcp_protocol_error", "code": -32601, "message": "same words"}),
            await _error_metadata({"type": "mcp_tool_execution_error", "content": "same words"}),
            await _error_metadata({"type": "http_error", "code": 503, "message": "same words"}),
        ]

        assert [e["type"] for e in kinds] == ["mcp_protocol_error", "mcp_tool_execution_error", "http_error"]

    async def test_arms_are_not_flattened_into_one_shape(self) -> None:
        protocol = await _error_metadata({"type": "mcp_protocol_error", "code": -32601, "message": "m"})
        execution = await _error_metadata({"type": "mcp_tool_execution_error", "content": "boom"})

        assert set(protocol) - set(execution) == {"code", "message"}
        assert set(execution) - set(protocol) == {"content"}


@pytest.mark.asyncio
class TestTicket12sContractIsUnchanged:
    async def test_a_successful_call_carries_no_error(self) -> None:
        [result] = results(await events_of(MCP_CALL))

        assert "error" not in result.result.metadata

    async def test_a_failed_call_keeps_its_event_name_and_shape(self) -> None:
        [result] = results(await events_of(_failed_call({"type": "http_error", "code": 503, "message": "m"})))

        assert isinstance(result, BuiltinToolResultEvent)
        assert (result.parent_id, result.name) == ("mcp_err", MCP_SERVER_TOOL_NAME)
        assert result.result.metadata == IsPartialDict({
            "server_label": "deepwiki",
            "tool": "ask_question",
            "status": "failed",
        })
