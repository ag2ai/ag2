# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from a2a.types import AgentCard
from dirty_equals import IsPartialDict, IsStr

from ag2 import Agent, AudioInput, DocumentInput, ImageInput
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.extension import MIME_HISTORY, MIME_TOOL_CALL, MIME_TOOL_RESULT, MIME_TOOL_SCHEMAS
from ag2.a2a.testing import make_test_client_factory, make_test_rest_client_factory, pick_free_port
from ag2.events import BinaryInput, BinaryType, ModelRequest, ToolCallEvent
from ag2.testing import TestConfig, TrackingConfig

from ._helpers import StatelessScript

CARD_PATHS = ("/.well-known/agent-card.json", "/.well-known/agent.json")


def get_weather(city: str) -> str:
    return f"It is sunny in {city}"


def _server() -> A2AServer:
    return A2AServer(Agent("srv", config=TestConfig("ok")))


@pytest.mark.asyncio
class TestCardCacheControl:
    @pytest.mark.parametrize("build", ["build_jsonrpc", "build_rest"])
    @pytest.mark.parametrize("path", CARD_PATHS)
    async def test_header_set_on_card_and_legacy_alias(self, build: str, path: str) -> None:
        app = getattr(_server(), build)(url="http://test", card_cache_control="public, max-age=300")

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.get(path)

        assert response.status_code == 200
        assert response.headers["cache-control"] == "public, max-age=300"

    @pytest.mark.parametrize("build", ["build_jsonrpc", "build_rest"])
    async def test_header_absent_by_default(self, build: str) -> None:
        app = getattr(_server(), build)(url="http://test")

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.get(CARD_PATHS[0])

        assert "cache-control" not in response.headers


class TestCardInputModes:
    def test_default_card_declares_ag2_media_types(self) -> None:
        card = build_card(Agent("srv", config=TestConfig("ok")), url="http://test")

        assert {MIME_HISTORY, MIME_TOOL_CALL, MIME_TOOL_RESULT, MIME_TOOL_SCHEMAS} <= set(card.default_input_modes)

    def test_untyped_config_declares_every_typed_media_type(self) -> None:
        card = build_card(Agent("srv", config=TestConfig("ok")), url="http://test")

        assert {"image/png", "application/pdf", "audio/wav", "video/mp4"} <= set(card.default_input_modes)

    def test_explicit_input_modes_replace_the_derived_list(self) -> None:
        card = build_card(Agent("srv", config=TestConfig("ok")), url="http://test", input_modes=["image/png"])

        assert list(card.default_input_modes) == ["image/png"]

    @pytest.mark.parametrize("build", ["build_jsonrpc", "build_rest", "build_grpc"])
    def test_custom_card_without_ag2_types_rejected_when_validating(self, build: str) -> None:
        card = build_card(Agent("srv", config=TestConfig("ok")), url="http://test")
        del card.default_input_modes[:]
        card.default_input_modes.append("text/plain")
        server = A2AServer(Agent("srv", config=TestConfig("ok")), validate_input_modes=True)
        kwargs = {"bind": "127.0.0.1:0", "grpc_url": "localhost:1"} if build == "build_grpc" else {"url": "http://test"}

        with pytest.raises(ValueError, match="vnd.ag2"):
            getattr(server, build)(card=card, **kwargs)

    def test_custom_card_without_ag2_types_fine_when_not_validating(self) -> None:
        card = build_card(Agent("srv", config=TestConfig("ok")), url="http://test")
        del card.default_input_modes[:]
        card.default_input_modes.append("text/plain")

        _server().build_jsonrpc(url="http://test", card=card)


@pytest.mark.asyncio
class TestValidateInputModes:
    async def test_ag2_client_round_trip_with_validation_on(self) -> None:
        server_agent = Agent(
            "srv",
            config=StatelessScript(ToolCallEvent(name="get_weather", arguments='{"city": "Paris"}'), "done"),
        )
        server = A2AServer(server_agent, validate_input_modes=True)
        client = Agent(
            "cli",
            config=A2AConfig(
                card_url="http://test",
                httpx_client_factory=make_test_client_factory(server, url="http://test"),
                streaming=False,
            ),
        )
        client.tool(get_weather)

        reply = await client.ask("paris?")

        assert reply.response.content == "done"

    async def test_undeclared_media_type_rejected(self) -> None:
        app = A2AServer(_server().agent, validate_input_modes=True).build_jsonrpc(url="http://test")
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {
                "message": {
                    "messageId": "m1",
                    "role": "ROLE_USER",
                    "parts": [{"raw": "AAAA", "mediaType": "application/x-unknown"}],
                },
            },
        }
        headers = {"A2A-Version": "1.0"}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post("/", content=json.dumps(body), headers=headers)

        # -32005 is ContentTypeNotSupportedError
        assert response.json()["error"] == IsPartialDict({
            "code": -32005,
            "message": IsStr(regex=".*application/x-unknown.*"),
        })


def _media(kind: str) -> BinaryInput:
    return {
        "image": ImageInput(data=b"\x89PNG0000", media_type="image/png"),
        "document": DocumentInput(data=b"%PDF-1.4", media_type="application/pdf"),
        "audio": AudioInput(data=b"RIFF", media_type="audio/wav"),
    }[kind]


MEDIA_KINDS = ("image", "document", "audio")


@pytest.mark.asyncio
class TestValidateInputModesMedia:
    @pytest.mark.parametrize("transport", ["jsonrpc", "rest"])
    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize("kind", MEDIA_KINDS)
    async def test_declared_media_passes_http(self, transport: str, streaming: bool, kind: str) -> None:
        server = A2AServer(Agent("srv", config=TestConfig("ok")), validate_input_modes=True)
        factory = (make_test_rest_client_factory if transport == "rest" else make_test_client_factory)(
            server, url="http://test"
        )
        client = Agent(
            "cli",
            config=A2AConfig(
                card_url="http://test", httpx_client_factory=factory, prefer=transport, streaming=streaming
            ),
        )

        reply = await client.ask("look", _media(kind))

        assert reply.response.content == "ok"

    @pytest.mark.parametrize("kind", MEDIA_KINDS)
    async def test_declared_media_passes_grpc(self, kind: str) -> None:
        agent = Agent("srv", config=TestConfig("ok"))
        url = f"127.0.0.1:{pick_free_port('127.0.0.1')}"
        card = build_card(agent, url=url, transports=("grpc",), grpc_url=url)
        grpc_server = A2AServer(agent, validate_input_modes=True).build_grpc(bind=url, grpc_url=url, card=card)
        await grpc_server.start()
        try:
            client = Agent("cli", config=A2AConfig(card_url=url, preset_card=card, prefer="grpc"))

            reply = await client.ask("look", _media(kind))

            assert reply.response.content == "ok"
        finally:
            await grpc_server.stop(grace=0)


@pytest.mark.asyncio
class TestStandardMediaPart:
    """A third-party A2A client follows the card; it sends no `ag2.binary_kind` metadata."""

    @pytest.mark.parametrize(
        ("media_type", "kind"),
        [
            ("image/png", BinaryType.IMAGE),
            ("application/pdf", BinaryType.DOCUMENT),
            ("audio/wav", BinaryType.AUDIO),
            ("video/mp4", BinaryType.VIDEO),
        ],
    )
    async def test_kind_is_inferred_from_the_media_type(self, media_type: str, kind: BinaryType) -> None:
        tracking = TrackingConfig(TestConfig("ok"))

        response = await _send_raw_part(tracking, {"raw": "AAAA", "mediaType": media_type})

        assert "error" not in response
        assert tracking.mock.call_args.args[0] == ModelRequest([
            BinaryInput(b"\x00\x00\x00", media_type=media_type, kind=kind)
        ])

    async def test_explicit_binary_kind_metadata_wins_over_the_media_type(self) -> None:
        tracking = TrackingConfig(TestConfig("ok"))
        part = {"raw": "AAAA", "mediaType": "image/png", "metadata": {"ag2.binary_kind": "document"}}

        await _send_raw_part(tracking, part)

        assert tracking.mock.call_args.args[0] == ModelRequest([
            BinaryInput(b"\x00\x00\x00", media_type="image/png", kind=BinaryType.DOCUMENT)
        ])


async def _send_raw_part(tracking: TrackingConfig, part: Mapping[str, Any]) -> dict[str, Any]:
    app = A2AServer(Agent("srv", config=tracking), validate_input_modes=True).build_jsonrpc(url="http://test")
    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "SendMessage",
        "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [part]}},
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post("/", content=json.dumps(body), headers={"A2A-Version": "1.0"})
    result: dict[str, Any] = response.json()
    return result


@pytest.mark.asyncio
async def test_card_modifier_cannot_widen_what_validation_accepts() -> None:
    async def widen(card: AgentCard) -> AgentCard:
        card.default_input_modes.append("application/x-unknown")
        return card

    app = A2AServer(
        Agent("srv", config=TestConfig("ok")), validate_input_modes=True, card_modifier=widen
    ).build_jsonrpc(url="http://test")
    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": "m1",
                "role": "ROLE_USER",
                "parts": [{"raw": "AAAA", "mediaType": "application/x-unknown"}],
            },
        },
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        await http.get(CARD_PATHS[0])
        response = await http.post("/", content=json.dumps(body), headers={"A2A-Version": "1.0"})

    assert response.json()["error"] == IsPartialDict({"code": -32005})
