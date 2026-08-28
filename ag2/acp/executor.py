# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Run one ACP ``session/prompt`` as one AG2 agent turn.

The executor owns the bridge in both directions: ACP content blocks in, AG2
events projected back out as ``session/update`` notifications. It deliberately
runs an ordinary AG2 turn — no simplified execution path — so tool events, human
input, observers and middleware all behave exactly as they do off-protocol.

Modelled on :class:`ag2.a2a.executor.AgentExecutor`, which solves the same
problem for A2A.
"""

import asyncio
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import acp

from ag2.agent import Agent
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    HumanInputRequest,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    TextInput,
)
from ag2.exceptions import HumanInputError
from ag2.history import close_unanswered_tool_calls
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

from .mappers import event_to_session_update, prompt_to_inputs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ag2.context import SubId
    from ag2.hitl import HumanHook
    from ag2.stream import MemoryStream

    from .sessions import AgentSession, SessionStore
    from .types import ContentBlock

logger = logging.getLogger(__name__)

# ACP stop reasons AG2 can report truthfully. The protocol also defines
# ``max_tokens`` / ``max_turn_requests`` / ``refusal``, but an AG2 ``ModelResponse``
# carries no unambiguous signal for those, and guessing one would tell the Client
# something we do not know.
STOP_END_TURN = "end_turn"
STOP_CANCELLED = "cancelled"

# Namespaced context variable carrying the Client's ``_meta`` into the turn.
# AG2 OSS never interprets what is inside; a downstream agent that cares (e.g.
# reading AG2 Space provenance) reads this key, and a generic agent ignores it.
META_VARIABLE = "acp.meta"

# Stands in for a tool result the Client will never get, because the turn was
# cancelled while the tool was still running. See :func:`heal_cancelled_turn`.
CANCELLED_TOOL_RESULT = "The turn was cancelled before this tool finished."

# Put on the ``data`` of the protocol error raised when a turn died because
# nobody could answer a human-input request. Stable across every reason the
# channel can fail, so a Client can branch on it — unlike ``type``, which is
# whichever Python exception happened to come out.
HUMAN_INPUT_ERROR_CATEGORY = "human_input"


class UpdateDeliveryError(RuntimeError):
    """Raised when a ``session/update`` could not be pushed to the Client.

    Fails the turn rather than letting it report success for output nobody
    received.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Could not deliver a session/update to the Client for session {session_id!r}.")
        self.session_id = session_id


class HumanInputUnsupportedError(HumanInputError, RuntimeError):
    """Raised when an agent asks for human input and no ``hitl_hook`` was given.

    ACP elicitation is not wired in this version, so the protocol itself has no
    way to put the question to the Client. Failing loudly beats hanging forever
    on a prompt nobody will be asked to answer. A host that can reach a human by
    its own means passes ``ACPAgent(..., hitl_hook=...)`` and never sees this.

    A :class:`~ag2.exceptions.HumanInputError`, which is what carries it out of
    tool execution to the Client: caught as an ordinary exception there it
    would become a tool result, and the Client would be told the turn
    succeeded. The protocol error it becomes carries
    ``data["category"] == "human_input"``. ``RuntimeError`` is kept in the
    bases for anyone already catching it as one.
    """

    def __init__(self) -> None:
        super().__init__(
            "The agent requested human input, but ACPAgent does not implement ACP "
            "elicitation yet, so there is nobody to ask. Pass ACPAgent(..., hitl_hook=...) "
            "to answer from the hosting application, remove the human-input step, or serve "
            "this agent over a transport that supports it."
        )


class AgentExecutor:
    """Bridge one ACP prompt turn onto :meth:`Agent._execute`.

    ``Agent._execute`` is private API, used here for the same reason
    :class:`ag2.a2a.executor.AgentExecutor` uses it: it is the only entry point
    that takes a pre-built :class:`~ag2.context.Context`, which is how a session's
    accumulated history reaches the turn. The context is handed to
    ``Agent._prepare_turn`` first, so an ACP turn resolves its prompts and
    dependencies through the same path every off-protocol turn uses.
    """

    __slots__ = ("_agent", "_stream_thoughts", "_hitl_hook")

    def __init__(
        self,
        agent: Agent,
        *,
        stream_thoughts: bool = False,
        hitl_hook: "HumanHook | None" = None,
    ) -> None:
        self._agent = agent
        self._stream_thoughts = stream_thoughts
        # ``None`` keeps the transport honest: with nobody to ask, a turn that
        # requests human input fails rather than hanging on an answer that is
        # never coming. A host that *does* have a human supplies the hook.
        self._hitl_hook: HumanHook = _reject_human_input if hitl_hook is None else hitl_hook

    @property
    def agent(self) -> Agent:
        return self._agent

    async def run_turn(
        self,
        *,
        session: "AgentSession",
        store: "SessionStore",
        client: acp.Client,
        blocks: "Sequence[ContentBlock]",
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Run one prompt to completion and return its ACP stop reason.

        Raises :class:`asyncio.CancelledError` if ``session/cancel`` arrives
        mid-turn; the caller turns that into ``stop_reason="cancelled"``. Events
        already streamed stay in the session's history either way.
        """
        stream = store.stream(session)
        delivered, subscription = self._forward_updates(stream, client=client, session_id=session.session_id)

        inputs = prompt_to_inputs(blocks)
        if not inputs:
            # A prompt of blocks we could not map at all. Send an empty text
            # input rather than nothing, so the turn is still well-formed.
            inputs = [TextInput("")]

        try:
            reply = await self._dispatch(ModelRequest(inputs), stream=stream, session=session, meta=meta)
            await self._send_final_text(reply, delivered, client=client, session_id=session.session_id)
        finally:
            # The stream belongs to the session, not to this turn, so the
            # subscriber has to come off explicitly — otherwise every prompt
            # would leave another forwarder behind, and a later turn would send
            # each update once per prompt the session had ever run.
            stream.unsubscribe(subscription)
        return STOP_END_TURN

    def _forward_updates(
        self, stream: "MemoryStream", *, client: acp.Client, session_id: str
    ) -> "tuple[list[str], SubId]":
        """Project this turn's AG2 events onto ``session/update`` notifications.

        A single subscriber handles every event, so notifications reach the
        Client in stream order.

        Returns the text delivered during the turn's *final* model call — what
        the final response is compared against, see :meth:`_send_final_text` —
        along with the subscription id the caller must release when the turn
        ends.
        """
        stream_thoughts = self._stream_thoughts
        current: list[str] = []
        final_call: list[str] = []

        # ``subscribe`` returns the id this binding is released by, so the name
        # holds a ``SubId`` rather than the function.
        @stream.subscribe
        async def subscription(event: BaseEvent) -> None:
            if isinstance(event, ModelResponse):
                # One model call just ended. A turn can contain several — a
                # tool-selection call, then the answering call — and each may
                # stream its own text. Keep only the most recent call's, because
                # ``reply.response`` describes that call and nothing earlier.
                final_call[:] = current
                current.clear()
                return

            update = event_to_session_update(event, stream_thoughts=stream_thoughts)
            if update is None:
                return
            await self._deliver(update, client=client, session_id=session_id)
            # Recorded only *after* the notification is on the wire. Counting a
            # chunk we failed to send would make the de-dup below suppress the
            # final text too, and the Client would end the turn having received
            # nothing at all.
            if isinstance(event, ModelMessageChunk):
                current.append(event.content)

        return final_call, subscription

    async def _send_final_text(
        self,
        reply: Any,
        delivered: list[str],
        *,
        client: acp.Client,
        session_id: str,
    ) -> None:
        """Deliver the final reply text the Client has not already been sent.

        A streaming provider emits the answer as ``ModelMessageChunk``s, which are
        already on the wire — re-sending the assembled text would show the reply
        twice. A non-streaming provider emits no chunks at all, so without this
        the Client would receive an empty turn.

        ``delivered`` is scoped to the turn's *final* model call. Comparing
        against every chunk of the whole turn would break the moment an earlier
        call streamed anything of its own: the totals would differ, and the
        answer would go out a second time.
        """
        message = getattr(reply.response, "message", None)
        final = getattr(message, "content", "") or ""
        if not final or final == "".join(delivered):
            return
        await self._deliver(acp.update_agent_message_text(final), client=client, session_id=session_id)

    async def _deliver(self, update: Any, *, client: acp.Client, session_id: str) -> None:
        """Push one ``session/update``, letting a delivery failure fail the turn.

        Swallowing this would let a turn answer ``stop_reason="end_turn"`` to a
        Client that never received the answer — a success report for work it
        cannot see. The turn's events are already on the session's stream by the
        time this runs, so history survives regardless; what fails is only the
        claim that the Client was told.
        """
        try:
            await client.session_update(session_id=session_id, update=update)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("failed to deliver ACP session/update for session %s", session_id, exc_info=True)
            raise UpdateDeliveryError(session_id) from exc

    async def _dispatch(
        self,
        initial_event: BaseEvent,
        *,
        stream: "MemoryStream",
        session: "AgentSession",
        meta: dict[str, Any] | None,
    ) -> Any:
        agent = self._agent
        if agent.config is None:
            raise RuntimeError(f"Agent {agent.name!r} has no config; cannot serve it over ACP.")

        # The session's own variables, handed over as-is: this is the same dict
        # every turn of this conversation sees, so what a tool writes is still
        # there next prompt. It was seeded by value at ``session/new``, so no
        # other conversation shares it.
        variables = session.variables
        if meta:
            variables[META_VARIABLE] = meta
        else:
            # Metadata belongs to the request that carried it; leaving the
            # previous one in place would misattribute this turn.
            variables.pop(META_VARIABLE, None)

        context = ConversationContext(
            stream,
            dependencies=dict(agent._agent_dependencies),
            variables=variables,
            dependency_provider=agent.dependency_provider,
        )
        # Off-protocol turns get this from ``Agent.ask``, which builds its own
        # context; without it, metrics cannot attribute the turn to its agent.
        context.dependencies[AGENT_CONTEXT_DEPENDENCY_KEY] = agent

        # ``_prepare_turn`` is private, but it is the only entry point that runs the
        # standard preparation an off-protocol turn gets: it fills an *empty*
        # ``context.prompt`` from the static prompt **and** every ``@agent.prompt``
        # hook, and records the resolved config as a context dependency. Seeding the
        # prompt here instead would leave it non-empty and silently skip the hooks —
        # so a served agent would drop the per-request policy those hooks carry,
        # which the same agent applies under ``ask``.
        client = await agent._prepare_turn(initial_event, context, None)

        return await agent._execute(
            initial_event,
            context=context,
            client=client,
            hitl_hook=self._hitl_hook,
        )


def isolate_variables(defaults: dict[Any, Any]) -> dict[Any, Any]:
    """Seed one session's variables from the agent's defaults, by value.

    A shallow copy is not enough. It stops one session *rebinding* a key another
    session reads, but every nested list, dict or object stays a shared
    reference — so one conversation appending to a list is immediately visible in
    all the others. Sessions are supposed to be independent conversations, and a
    budget, an approval list or a workflow accumulator is exactly the kind of
    value that gets stored nested.

    A value that cannot be deep-copied (an open handle, a lock, a client) is kept
    as a shared reference and named in a warning, because silently degrading to
    sharing is how the original bug looked from the outside.
    """
    isolated: dict[Any, Any] = {}
    for key, value in defaults.items():
        try:
            isolated[key] = deepcopy(value)
        except Exception:
            logger.warning(
                "ACP session variable %r could not be copied and is shared across sessions; "
                "mutating it in one conversation will affect the others.",
                key,
            )
            isolated[key] = value
    return isolated


async def heal_cancelled_turn(stream: "MemoryStream") -> int:
    """Close off tool calls a cancelled turn left unanswered. Returns how many.

    The repair itself lives in :func:`ag2.history.close_unanswered_tool_calls`,
    because every surface needs it and not only ACP; what this adds is the
    reason ACP has for asking, which is the text the model reads next turn.

    Still called on the failure path, where it covers the ways a turn can die
    that the turn boundary does not repair itself — a ``session/update`` that
    could not be delivered, a storage error. A human-input failure has already
    been closed off by then, with its own reason, so this finds nothing left to
    do and the transcript is not mislabelled as cancelled. The repair is
    idempotent, which is what makes calling it twice harmless.
    """
    return await close_unanswered_tool_calls(stream.history, result=CANCELLED_TOOL_RESULT)


async def _reject_human_input(event: HumanInputRequest) -> str:
    """Default ``hitl_hook``: fail the turn instead of waiting forever.

    Takes the request and nothing else, because a hook is resolved the way a
    tool is: an un-annotated second parameter is read as another *field* to fill
    from the event, and the turn then dies of a validation error naming this
    function instead of the error that explains the problem.

    Declared as returning ``str`` to satisfy the hook signature; it never
    returns. See :class:`HumanInputUnsupportedError`.
    """
    raise HumanInputUnsupportedError
