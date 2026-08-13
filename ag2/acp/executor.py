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
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.events.tool_events import ToolResult
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

from .mappers import event_to_session_update, prompt_to_inputs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ag2.context import SubId
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


class UpdateDeliveryError(RuntimeError):
    """Raised when a ``session/update`` could not be pushed to the Client.

    Fails the turn rather than letting it report success for output nobody
    received.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Could not deliver a session/update to the Client for session {session_id!r}.")
        self.session_id = session_id


class HumanInputUnsupportedError(RuntimeError):
    """Raised when an agent asks for human input over a transport that has none.

    ACP elicitation is not wired in this version. Failing loudly beats hanging
    forever on a prompt the Client will never be asked to answer.
    """

    def __init__(self) -> None:
        super().__init__(
            "The agent requested human input, but ACPAgent does not implement ACP "
            "elicitation yet, so there is nobody to ask. Remove the human-input step "
            "or serve this agent over a transport that supports it."
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

    __slots__ = ("_agent", "_stream_thoughts")

    def __init__(self, agent: Agent, *, stream_thoughts: bool = False) -> None:
        self._agent = agent
        self._stream_thoughts = stream_thoughts

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
            hitl_hook=_reject_human_input,
        )


def _tool_call_batches(events: "Sequence[BaseEvent]") -> "list[ToolCallsEvent]":
    """Every batch of tool calls in ``events``, from either place they appear.

    A turn persists the model's :class:`ModelResponse` — which already carries
    ``tool_calls`` — *before* the agent emits the matching
    :class:`ToolCallsEvent`. Cancelling in that window leaves history holding
    calls that no ``ToolCallsEvent`` describes, so reading only the latter would
    find nothing to repair and leave the transcript with an unanswered call.

    Batches are deduplicated by call id, because the usual case is both records
    present and describing the same calls.
    """
    batches: list[ToolCallsEvent] = []
    seen: set[str] = set()
    for event in events:
        calls = (
            event.tool_calls.calls
            if isinstance(event, ModelResponse) and event.tool_calls
            else event.calls
            if isinstance(event, ToolCallsEvent)
            else []
        )
        fresh = [call for call in calls if call.id not in seen]
        if not fresh:
            continue
        seen.update(call.id for call in fresh)
        batches.append(ToolCallsEvent(fresh))
    return batches


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

    Cancelling stops the turn wherever it happened to be, which can be *between*
    a tool call and its result. That leaves history holding an assistant
    tool-call with nothing answering it — and providers reject that shape, so the
    session would fail on its next prompt even though cancelling is supposed to
    be a normal, recoverable thing to do.

    Appending a synthetic result per unanswered call keeps the transcript valid
    and tells the model plainly what happened, rather than rewriting history to
    pretend the call was never made.
    """
    events = list(await stream.history.get_events())

    # A batch counts as settled only once a ``ToolResultsEvent`` covers it —
    # that wrapper is what providers serialize. A loose ``ToolResultEvent`` left
    # behind by a tool that *did* finish is not enough on its own: a partially
    # completed batch would otherwise be rebuilt with the finished call missing,
    # and the transcript would carry more tool calls than tool results.
    settled = {r.parent_id for e in events if isinstance(e, ToolResultsEvent) for r in e.results}
    completed = {e.parent_id: e for e in events if isinstance(e, ToolResultEvent) and e.parent_id not in settled}

    repaired: list[ToolResultsEvent] = []
    closed = 0
    for event in _tool_call_batches(events):
        pending = [call for call in event.calls if call.id not in settled]
        if not pending:
            continue
        # Rebuild the *whole* outstanding batch: results that did land, plus a
        # stand-in for each call the cancellation cut short.
        results = []
        for call in pending:
            done = completed.get(call.id)
            results.append(
                done
                if done is not None
                else ToolResultEvent(
                    parent_id=call.id,
                    name=call.name,
                    result=ToolResult(CANCELLED_TOOL_RESULT),
                )
            )
            closed += int(done is None)
        repaired.append(ToolResultsEvent(results))

    if not repaired:
        return 0

    await stream.history.replace([*events, *repaired])
    return closed


async def _reject_human_input(event: BaseEvent, context: Any) -> str:
    """``hitl_hook`` that fails the turn instead of waiting forever.

    Declared as returning ``str`` to satisfy the hook signature; it never
    returns. See :class:`HumanInputUnsupportedError`.
    """
    raise HumanInputUnsupportedError
