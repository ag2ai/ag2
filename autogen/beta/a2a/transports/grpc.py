# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any

import grpc
from a2a.server.context import ServerCallContext
from a2a.types import a2a_pb2, a2a_pb2_grpc
from google.protobuf import empty_pb2

if TYPE_CHECKING:
    from a2a.server.request_handlers import RequestHandler

    from ..server import A2AServer


def build_grpc(
    server: "A2AServer",
    *,
    host: str | None = None,
    port: int | None = None,
    server_options: Sequence[tuple[str, Any]] | None = None,
    grpc_server: grpc.aio.Server | None = None,
) -> grpc.aio.Server:
    """Construct (or augment) a ``grpc.aio.Server`` speaking the A2A binding.

    Modes:
      - ``server.build_grpc()`` — fresh ``grpc.aio.Server`` without bind.
      - ``server.build_grpc(host="[::]", port=50051)`` — fresh server with
        insecure bind. Useful for local dev.
      - ``server.build_grpc(grpc_server=my_server)`` — register the A2A
        servicer onto an existing server.
    """
    grpc_server = grpc_server or grpc.aio.server(options=list(server_options) if server_options else None)
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(make_servicer(server), grpc_server)
    if host is not None and port is not None:
        grpc_server.add_insecure_port(f"{host}:{port}")
    return grpc_server


def make_servicer(server: "A2AServer") -> "A2AGrpcServicer":
    """Produce the ``A2AServiceServicer`` for ``server`` without binding ports."""
    return A2AGrpcServicer(server.build_request_handler(), server)


class A2AGrpcServicer(a2a_pb2_grpc.A2AServiceServicer):
    """``A2AServiceServicer`` impl proxying every RPC to the shared handler."""

    def __init__(self, handler: "RequestHandler", server: "A2AServer") -> None:
        self._handler = handler
        self._server = server

    # ---- message ops ----

    async def SendMessage(  # noqa: N802
        self,
        request: a2a_pb2.SendMessageRequest,
        context: grpc.aio.ServicerContext,
    ) -> a2a_pb2.SendMessageResponse:
        result = await self._handler.on_message_send(request, _server_call_context())
        if isinstance(result, a2a_pb2.Task):
            return a2a_pb2.SendMessageResponse(task=result)
        if isinstance(result, a2a_pb2.Message):
            return a2a_pb2.SendMessageResponse(message=result)
        await context.abort(grpc.StatusCode.INTERNAL, f"Unexpected handler result type: {type(result).__name__}")
        raise AssertionError("unreachable")

    async def SendStreamingMessage(  # noqa: N802
        self,
        request: a2a_pb2.SendMessageRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[a2a_pb2.StreamResponse, None]:
        async for event in self._handler.on_message_send_stream(request, _server_call_context()):
            yield _wrap_stream_response(event)

    # ---- task ops ----

    async def GetTask(  # noqa: N802
        self,
        request: a2a_pb2.GetTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> a2a_pb2.Task:
        task = await self._handler.on_get_task(request, _server_call_context())
        if task is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"task {request.id!r} not found")
            raise AssertionError("unreachable")
        return task

    async def CancelTask(  # noqa: N802
        self,
        request: a2a_pb2.CancelTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> a2a_pb2.Task:
        task = await self._handler.on_cancel_task(request, _server_call_context())
        if task is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"task {request.id!r} not found")
            raise AssertionError("unreachable")
        return task

    async def ListTasks(  # noqa: N802
        self,
        request: a2a_pb2.ListTasksRequest,
        context: grpc.aio.ServicerContext,
    ) -> a2a_pb2.ListTasksResponse:
        return await self._handler.on_list_tasks(request, _server_call_context())

    async def SubscribeToTask(  # noqa: N802
        self,
        request: a2a_pb2.SubscribeToTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[a2a_pb2.StreamResponse, None]:
        async for event in self._handler.on_subscribe_to_task(request, _server_call_context()):
            yield _wrap_stream_response(event)

    # ---- card ----

    async def GetExtendedAgentCard(  # noqa: N802
        self,
        request: a2a_pb2.GetExtendedAgentCardRequest,
        context: grpc.aio.ServicerContext,
    ) -> a2a_pb2.AgentCard:
        return await self._handler.on_get_extended_agent_card(request, _server_call_context())

    # ---- push notifications ----

    async def CreateTaskPushNotificationConfig(  # noqa: N802
        self,
        request: a2a_pb2.TaskPushNotificationConfig,
        context: grpc.aio.ServicerContext,
    ) -> a2a_pb2.TaskPushNotificationConfig:
        return await self._handler.on_create_task_push_notification_config(request, _server_call_context())

    async def GetTaskPushNotificationConfig(  # noqa: N802
        self,
        request: a2a_pb2.GetTaskPushNotificationConfigRequest,
        context: grpc.aio.ServicerContext,
    ) -> a2a_pb2.TaskPushNotificationConfig:
        return await self._handler.on_get_task_push_notification_config(request, _server_call_context())

    async def ListTaskPushNotificationConfigs(  # noqa: N802
        self,
        request: a2a_pb2.ListTaskPushNotificationConfigsRequest,
        context: grpc.aio.ServicerContext,
    ) -> a2a_pb2.ListTaskPushNotificationConfigsResponse:
        return await self._handler.on_list_task_push_notification_configs(request, _server_call_context())

    async def DeleteTaskPushNotificationConfig(  # noqa: N802
        self,
        request: a2a_pb2.DeleteTaskPushNotificationConfigRequest,
        context: grpc.aio.ServicerContext,
    ) -> empty_pb2.Empty:
        await self._handler.on_delete_task_push_notification_config(request, _server_call_context())
        return empty_pb2.Empty()


def _wrap_stream_response(
    event: a2a_pb2.Task | a2a_pb2.Message | a2a_pb2.TaskStatusUpdateEvent | a2a_pb2.TaskArtifactUpdateEvent,
) -> a2a_pb2.StreamResponse:
    """Pack a single handler-yielded event into a wire ``StreamResponse``."""
    if isinstance(event, a2a_pb2.Task):
        return a2a_pb2.StreamResponse(task=event)
    if isinstance(event, a2a_pb2.Message):
        return a2a_pb2.StreamResponse(message=event)
    if isinstance(event, a2a_pb2.TaskStatusUpdateEvent):
        return a2a_pb2.StreamResponse(status_update=event)
    if isinstance(event, a2a_pb2.TaskArtifactUpdateEvent):
        return a2a_pb2.StreamResponse(artifact_update=event)
    raise TypeError(f"Unexpected stream event type: {type(event).__name__}")


def _server_call_context() -> ServerCallContext:
    """Construct a minimal ``ServerCallContext`` for unauthenticated gRPC calls.

    Auth-aware deployments should swap this for a real builder that reads
    metadata off the gRPC ``ServicerContext`` and populates the user identity.
    """
    return ServerCallContext()
