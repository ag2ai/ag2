# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared harness for A2UI ⇄ A2A round-trip tests.

A client :class:`~ag2.Agent` talks to an :class:`A2AServer` wired with
an :class:`A2UIAgentExecutor` over an in-process httpx ``ASGITransport`` (the
same path the ``test/a2a`` E2E suite uses). The mock LLM doubles here
mirror the sanctioned pattern in ``test/a2a/_helpers.py`` (``RecordingConfig``):
they capture what the executor feeds the model — the resolved system prompt or
the full synthesized user turn — which ``TrackingConfig`` (last message only)
cannot expose.
"""

from collections.abc import Sequence
from typing import Any

from a2a.client import ClientCallInterceptor
from a2a.client.interceptors import BeforeArgs
from a2a.types import TaskState
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.events import A2ATaskStatusUpdate
from ag2.a2a.testing import make_test_client_factory
from ag2.a2ui._types import A2UIVersion
from ag2.a2ui.a2a import get_a2ui_data, is_a2ui_part
from ag2.a2ui.a2a.executor import A2UIAgentExecutor
from ag2.config import LLMClient, ModelConfig
from ag2.events import BaseEvent, ModelMessage, ModelResponse
from ag2.stream import MemoryStream

# Marker the HITL continuation feeds back so a paused turn can complete instead
# of pausing forever (see CallFunctionThenComplete).
FUNCTION_RESULT_MARK = "FUNCTION-RESULT-DELIVERED"


class CapturingConfig(ModelConfig):
    """A mock LLM config that records the resolved prompt and message list per call.

    ``TrackingConfig`` only records ``messages[-1]``; verifying the synthesized
    *system prompt* (capabilities) or the full synthesized *user turn* (incoming
    envelope rewrites) needs the prompt and the whole message list, so this
    follows the same custom-double pattern as ``a2a/_helpers.RecordingConfig``.
    """

    def __init__(self, response: str = "ok") -> None:
        self._response = response
        self.prompts: list[list[str]] = []
        self.messages: list[list[BaseEvent]] = []

    def copy(self) -> Self:
        return self

    def create(self) -> "CapturingClient":
        return CapturingClient(self._response, self.prompts, self.messages)

    def create_files_client(self) -> None:
        raise NotImplementedError


class CapturingClient(LLMClient):
    def __init__(self, response: str, prompts: list[list[str]], messages: list[list[BaseEvent]]) -> None:
        self._response = response
        self._prompts = prompts
        self._messages = messages

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self._prompts.append(list(context.prompt))
        self._messages.append(list(messages))
        msg = ModelMessage(self._response)
        await context.send(msg)
        return ModelResponse(msg)


class CallFunctionThenComplete(ModelConfig):
    """First turn emits a ``callFunction(wantResponse)``; once the client delivers
    the function result (via the HITL continuation) the next turn completes.

    The A2A executor recreates the LLM client every turn, so the branch keys off
    the conversation history rather than call count.
    """

    def __init__(self, call_block: str) -> None:
        self._call_block = call_block

    def copy(self) -> Self:
        return self

    def create(self) -> "CallFunctionThenCompleteClient":
        return CallFunctionThenCompleteClient(self._call_block)

    def create_files_client(self) -> None:
        raise NotImplementedError


class CallFunctionThenCompleteClient(LLMClient):
    def __init__(self, call_block: str) -> None:
        self._call_block = call_block

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        if FUNCTION_RESULT_MARK in synthesized_text(messages):
            text = "All done."
        else:
            text = f"Opening the link.\n<a2ui-json>\n{self._call_block}\n</a2ui-json>"
        msg = ModelMessage(text)
        await context.send(msg)
        return ModelResponse(msg)


class MetadataInterceptor(ClientCallInterceptor):
    """Splices ``a2uiClientCapabilities`` onto the outgoing message metadata.

    The stock A2A client has no hatch for arbitrary message metadata, so a real
    capability-negotiation round-trip injects it here, mirroring how production
    clients would advertise their renderer's capabilities.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata

    async def before(self, args: BeforeArgs) -> None:
        try:
            message = args.input.message
        except AttributeError:
            return  # not a send-message call (e.g. card fetch)
        message.metadata.update(self._metadata)

    async def after(self, args: Any) -> None:
        return None


def client_for(
    agent: Agent,
    *,
    streaming: bool = False,
    interceptors: Sequence[ClientCallInterceptor] = (),
    hitl_hook: Any = None,
    protocol_version: A2UIVersion = "v0.9",
    validate_responses: bool = True,
    actions: Sequence[Any] = (),
) -> Agent:
    """A client Agent talking to ``agent`` via the A2UI executor over in-process HTTP.

    ``agent`` is a plain ``ag2.Agent``; A2UI config (protocol version,
    validation) lives on the :class:`A2UIAgentExecutor`. Clickable buttons
    (``@a2ui_action`` tools) are declared via ``actions=`` on the executor — the
    same way they are on :class:`A2UIServer` — so the agent stays plain.
    """
    executor = A2UIAgentExecutor(
        agent,
        actions=actions,
        protocol_version=protocol_version,
        validate_responses=validate_responses,
    )
    a2a_server = A2AServer(agent, executor=executor)
    factory = make_test_client_factory(a2a_server, url="http://test")
    kwargs: dict[str, Any] = {}
    if hitl_hook is not None:
        kwargs["hitl_hook"] = hitl_hook
    return Agent(
        "client",
        config=A2AConfig(
            card_url="http://test",
            httpx_client_factory=factory,
            streaming=streaming,
            interceptors=list(interceptors),
        ),
        **kwargs,
    )


def synthesized_text(messages: Sequence[BaseEvent]) -> str:
    """Flatten the text content the executor synthesized into a turn's messages.

    The full ``TextInput.content`` is needed (``repr`` truncates), so this pulls
    every string ``content`` off the message parts and joins them.
    """
    chunks: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []) or []:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                chunks.append(content)
    return "\n".join(chunks)


def subscribe_task_stream(stream: MemoryStream) -> "tuple[list, list[TaskState]]":
    """Collect A2UI DataPart payloads and task states off the client's stream.

    The client flattens A2UI DataParts out of ``reply.response`` (it only decodes
    text and tool-call parts), so the canonical A2UI DataPart is observed on the
    finalization message that rides the ``A2ATaskStatusUpdate`` (streaming mode).
    """
    a2ui_payloads: list = []
    states: list[TaskState] = []

    @stream.subscribe
    async def _collect(event: BaseEvent) -> None:
        if not isinstance(event, A2ATaskStatusUpdate):
            return
        states.append(event.state)
        message = event.update.status.message
        if message is None:
            return
        for part in message.parts:
            if is_a2ui_part(part):
                a2ui_payloads.append(get_a2ui_data(part))

    return a2ui_payloads, states
