# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias
from unittest.mock import MagicMock

from typing_extensions import Self

from ag2 import Context
from ag2.config import LLMClient, ModelConfig, ModelProvider
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
)

if TYPE_CHECKING:
    from ag2.files.protocol import FilesClient

__all__ = (
    "TestConfig",
    "TrackingConfig",
    "Turn",
)

Turn: TypeAlias = "str | ModelResponse | ToolCallEvent | Iterable[ToolCallEvent] | BaseEvent | BaseException"
"""One scripted LLM turn.

* ``str`` — the model replies with that text.
* ``ToolCallEvent`` (or an iterable of them) — the model calls tools.
* ``ModelResponse`` — the response, spelled out in full.
* ``BaseException`` — the call fails with it, the way a provider client would.
* any other ``BaseEvent`` — published to the stream *during* the turn, exactly as
  a provider client streams chunks or reasoning, then the script is read on for
  whatever ends the turn.
"""


class TestClient(LLMClient):
    __test__ = False

    def __init__(
        self,
        *events: "Turn",
        raise_tool_errors: bool = True,
        script: Iterator["Turn"] | None = None,
    ) -> None:
        self.events = iter(events) if script is None else script
        self.raise_tool_errors = raise_tool_errors

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        if self.raise_tool_errors:
            for m in messages:
                if isinstance(m, ToolErrorEvent):
                    raise m.error

        while True:
            try:
                scripted = next(self.events)
            except StopIteration:
                raise RuntimeError("TestConfig script is exhausted") from None

            if isinstance(scripted, BaseException):
                raise scripted

            if isinstance(scripted, str):
                message = ModelMessage(scripted)
                await context.send(message)
                return ModelResponse(message)

            if isinstance(scripted, ModelResponse):
                return scripted

            # A builtin call is not a request for the agent to run a tool: the
            # provider ran it, and the client only publishes it. It falls through
            # to the branch below.
            if isinstance(scripted, ToolCallEvent) and not isinstance(scripted, BuiltinToolCallEvent):
                return ModelResponse(tool_calls=ToolCallsEvent([scripted]))

            if isinstance(scripted, BaseEvent):
                # Anything else a provider publishes mid-turn — a streamed chunk,
                # reasoning, server-side tool activity. It does not end the turn,
                # so keep reading the script for what does.
                await context.send(scripted)
                continue

            return ModelResponse(tool_calls=ToolCallsEvent(list(scripted)))


class TrackingClient(LLMClient):
    def __init__(self, client: LLMClient, mock: MagicMock) -> None:
        self.client = client
        self.mock = mock

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **kwargs: Any,
    ) -> ModelResponse:
        self.mock(messages[-1])
        return await self.client(messages, context=context, **kwargs)


class TrackingConfig(ModelConfig):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.mock = MagicMock()

    @property
    def provider(self) -> ModelProvider:
        return self.config.provider

    @property
    def model(self) -> str | None:
        return self.config.model

    def copy(self) -> Self:
        return self

    def create(self) -> TrackingClient:
        return TrackingClient(self.config.create(), self.mock)

    def create_files_client(self) -> "FilesClient":
        raise NotImplementedError(f"{type(self).__name__} does not support Files API.")


class TestConfig(ModelConfig):
    __test__ = False

    def __init__(
        self,
        *events: "Turn",
        provider: ModelProvider | None = None,
        model: str | None = None,
        raise_tool_errors: bool = True,
        shared_script: bool = False,
    ) -> None:
        """Script the LLM, one :data:`Turn` per positional event.

        Events that only *publish* (chunks, reasoning, builtin tool activity) do
        not consume a turn — they are sent to the stream and the next scripted
        event is read straight away, so a single turn can stream and then fail::

            TestConfig(ModelMessageChunk("Tok"), TimeoutError("dropped"), "Recovered")

        ``raise_tool_errors`` (default ``True``) re-raises any ``ToolErrorEvent``
        it finds in the history, which is the convenient way to assert that a
        tool blew up. Set it to ``False`` to model a *real* provider, which is
        handed a failed tool call as an ordinary result and carries on: a test
        asserting that something **ends the turn** needs that, or it is
        asserting this double's behaviour rather than the agent's.

        Each ``create()`` starts the script over, and an agent creates one
        client per run. A served agent (A2A, the network, …) makes a run per
        request, so a conversation over a transport would hear turn 1 on every
        request. ``shared_script=True`` gives every client this config creates
        one cursor, so the script carries on where the last run left it. It is
        unsound for concurrent consumers — whichever runs first takes the next
        turn — and running past the end raises, so script every turn.

        To hold a model call open — for cancellation, concurrency or barge-in —
        script a transient event of the test's own and park on it in an
        ``async`` observer. The stream awaits subscribers in the sender's task,
        so the call waits for the observer, and cancelling the turn lands in
        its ``await``::

            class Hold(BaseEvent):
                __transient__ = True


            async def hold(event: Hold) -> None:
                entered.set()
                await release.wait()


            Agent(..., config=TestConfig(Hold(), "reply"), observers=[observer(Hold, hold)])

        A sync observer runs in a thread, where cancellation cannot reach it.
        """
        self.events = events
        self._shared = iter(events) if shared_script else None
        self._provider = provider
        self._model = model
        self._raise_tool_errors = raise_tool_errors

    @property
    def provider(self) -> ModelProvider:
        if not self._provider:
            raise NotImplementedError
        return self._provider

    @property
    def model(self) -> str | None:
        return self._model

    def copy(self) -> Self:
        return self

    def create(self) -> TestClient:
        if self._shared is not None:
            return TestClient(raise_tool_errors=self._raise_tool_errors, script=self._shared)
        return TestClient(*self.events, raise_tool_errors=self._raise_tool_errors)

    def create_files_client(self) -> "FilesClient":
        raise NotImplementedError(f"{type(self).__name__} does not support Files API.")
