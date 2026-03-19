# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from autogen.beta.debug.client import DebugClient, _serialize_value, get_server, serialize_event
from autogen.beta.events.types import ModelRequest

# ── _serialize_value edge cases ───────────────────────────────────────────


def test_serialize_value_primitives() -> None:
    assert _serialize_value("hello") == "hello"
    assert _serialize_value(42) == 42
    assert _serialize_value(3.14) == 3.14
    assert _serialize_value(True) is True
    assert _serialize_value(None) is None


def test_serialize_value_list() -> None:
    assert _serialize_value([1, "a", None]) == [1, "a", None]


def test_serialize_value_dict() -> None:
    assert _serialize_value({"key": "val", "num": 2}) == {"key": "val", "num": 2}


def test_serialize_value_exception() -> None:
    result = _serialize_value(ValueError("bad input"))
    assert result == {"type": "ValueError", "message": "bad input"}


def test_serialize_value_dataclass() -> None:
    @dataclass
    class Point:
        x: int
        y: int

    result = _serialize_value(Point(1, 2))
    assert result == {"x": 1, "y": 2}


def test_serialize_value_fallback_repr() -> None:
    """Unknown types fall back to repr()."""
    result = _serialize_value(object.__new__(type("Custom", (), {"__repr__": lambda s: "Custom()"})))
    assert result == "Custom()"


def test_serialize_value_nested_list_of_events() -> None:
    events = [ModelRequest(content="a"), ModelRequest(content="b")]
    result = _serialize_value(events)
    assert len(result) == 2
    assert result[0]["type"] == "ModelRequest"
    assert result[0]["content"] == "a"


def test_serialize_value_nested_dict_with_event() -> None:
    result = _serialize_value({"event": ModelRequest(content="hi")})
    assert result["event"]["type"] == "ModelRequest"


def test_serialize_value_base_event() -> None:
    event = ModelRequest(content="test")
    result = _serialize_value(event)
    assert result["type"] == "ModelRequest"
    assert result["content"] == "test"


# ── serialize_event ───────────────────────────────────────────────────────


def test_serialize_event_returns_type_and_data() -> None:
    event = ModelRequest(content="hello")
    result = serialize_event(event)
    assert result["type"] == "ModelRequest"
    assert "data" in result
    assert result["data"]["content"] == "hello"


# ── DebugClient ───────────────────────────────────────────────────────────


def test_debug_client_strips_trailing_slash() -> None:
    client = DebugClient("http://localhost:8765/")
    assert client._base_url == "http://localhost:8765"


def test_debug_client_no_trailing_slash() -> None:
    client = DebugClient("http://localhost:8765")
    assert client._base_url == "http://localhost:8765"


@pytest.mark.asyncio()
async def test_register_session_posts_to_server() -> None:
    client = DebugClient("http://localhost:8765")
    mock_post = AsyncMock()
    mock_http_client = AsyncMock()
    mock_http_client.post = mock_post
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with patch("autogen.beta.debug.client.httpx.AsyncClient", return_value=mock_http_client):
        await client.register_session("sess-1", ["Be helpful."])

    mock_post.assert_awaited_once_with(
        "http://localhost:8765/sessions/sess-1",
        json={"prompt": ["Be helpful."]},
    )


@pytest.mark.asyncio()
async def test_send_event_posts_to_server() -> None:
    client = DebugClient("http://localhost:8765")
    mock_post = AsyncMock()
    mock_http_client = AsyncMock()
    mock_http_client.post = mock_post
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with patch("autogen.beta.debug.client.httpx.AsyncClient", return_value=mock_http_client):
        await client.send_event("sess-1", "ModelRequest", {"content": "hi"})

    mock_post.assert_awaited_once_with(
        "http://localhost:8765/sessions/sess-1/events",
        json={"event_type": "ModelRequest", "event_data": {"content": "hi"}},
    )


@pytest.mark.asyncio()
async def test_send_event_swallows_exceptions() -> None:
    """send_event must never raise — it's fire-and-forget."""
    client = DebugClient("http://localhost:8765")
    mock_http_client = AsyncMock()
    mock_http_client.post = AsyncMock(side_effect=ConnectionError("fail"))
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)

    with patch("autogen.beta.debug.client.httpx.AsyncClient", return_value=mock_http_client):
        await client.send_event("sess-1", "ModelRequest", {"content": "hi"})
    # No exception raised


# ── get_server factory ────────────────────────────────────────────────────


def test_get_server_returns_debug_client() -> None:
    client = get_server("http://localhost:8765")
    assert isinstance(client, DebugClient)
    assert client._base_url == "http://localhost:8765"
