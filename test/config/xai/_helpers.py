# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import grpc
from xai_sdk.proto import chat_pb2, chat_pb2_grpc


class FakeChatServer(chat_pb2_grpc.ChatServicer):
    """In-process xAI ``Chat`` service: records requests, replays scripted replies.

    The SDK talks to ``localhost:<port>`` over a local (non-TLS) channel, so the
    real ``XAIConfig`` -> ``AsyncClient`` -> gRPC path runs end to end.
    """

    def __init__(self) -> None:
        self.address = ""
        self.requests: list[chat_pb2.GetCompletionsRequest] = []
        self._responses: list[chat_pb2.GetChatCompletionResponse] = []
        self._chunks: list[chat_pb2.GetChatCompletionChunk] = []

    def reply(self, *responses: chat_pb2.GetChatCompletionResponse) -> None:
        self._responses = list(responses)

    def stream(self, *chunks: chat_pb2.GetChatCompletionChunk) -> None:
        self._chunks = list(chunks)

    async def GetCompletion(  # noqa: N802
        self,
        request: chat_pb2.GetCompletionsRequest,
        context: grpc.aio.ServicerContext,
    ) -> chat_pb2.GetChatCompletionResponse:
        self.requests.append(request)
        return self._responses.pop(0)

    async def GetCompletionChunk(  # noqa: N802
        self,
        request: chat_pb2.GetCompletionsRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[chat_pb2.GetChatCompletionChunk]:
        self.requests.append(request)
        for chunk in self._chunks:
            yield chunk


@asynccontextmanager
async def fake_xai_server() -> AsyncGenerator[FakeChatServer]:
    servicer = FakeChatServer()
    server = grpc.aio.server()
    chat_pb2_grpc.add_ChatServicer_to_server(servicer, server)
    port = server.add_secure_port("localhost:0", grpc.local_server_credentials())
    servicer.address = f"localhost:{port}"
    await server.start()
    try:
        yield servicer
    finally:
        await server.stop(None)
