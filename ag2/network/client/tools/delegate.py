# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``delegate`` — one-shot consult: open consulting → ask → return reply.

This is the most common multi-agent pattern and gets its own flat tool
(vs the grouped ``channels(action="open") + tasks(...)`` pattern). The
flat surface keeps the LLM's tool list short — ``say`` and
``delegate`` cover the hot path.

``capability`` is recorded as a channel knob and feeds the owner's
``Resume.observed`` on terminal completion via the task mirror.
"""

import asyncio
from typing import TYPE_CHECKING

from ag2.tools import tool

from ...envelope import (
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_EXPIRED,
    EV_CHANNEL_INVITE_REJECT,
    EV_TEXT,
    Envelope,
)
from ..inject import AgentClientInject

if TYPE_CHECKING:
    from ..agent_client import AgentClient

__all__ = ("make_delegate_tool",)


_TERMINAL_CHANNEL_EVENTS = frozenset({
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_EXPIRED,
    EV_CHANNEL_INVITE_REJECT,
})


def make_delegate_tool(agent_client: "AgentClient") -> object:
    """Return a closure-bound ``delegate`` tool."""

    @tool
    async def delegate(
        target: str,
        prompt: str,
        *,
        capability: str | None = None,
        timeout: float = 300.0,
        client: AgentClientInject = None,
    ) -> str:
        """Open a one-shot consulting channel with ``target`` and return its reply.

        target: peer **name** (or agent_id) to consult.
        prompt: the question or request to send.
        capability: optional capability tag. Recorded as a channel
                    knob; the task mirror uses it to update
                    ``Resume.observed`` on terminal completion.
        timeout: max seconds to wait for the reply (default 300s).

        Returns the reply text on success, or an ``Error: ...`` string
        on failure (target unknown, timeout, channel rejected, etc.).
        """
        actual_client = client if client is not None else agent_client

        # Resolve target.
        try:
            target_passport = await actual_client._hub_client.get_agent(target)
        except Exception:
            return f"Error: target {target!r} not found"
        target_id = target_passport.agent_id
        if target_id is None:
            return f"Error: target {target!r} has no agent_id"
        if target_id == actual_client.agent_id:
            # A consulting channel needs a respondent distinct from the
            # initiator. Delegating to self collapses both roles onto one
            # participant, which the consulting adapter rejects with the
            # opaque "consulting requires exactly one respondent". Fail
            # fast here with an actionable message the caller can recover
            # from instead of opening a doomed channel.
            return f"Error: cannot delegate to self (target {target!r} is this agent)"

        # Open consulting channel — handshake awaited inside.
        knobs = {"capability": capability} if capability else None
        try:
            channel = await actual_client.open(
                type="consulting",
                target=target,
                knobs=knobs,
            )
        except Exception as exc:
            return f"Error: failed to open consulting channel: {exc}"

        # Pre-create the inbox BEFORE sending. Otherwise a fast reply
        # (e.g. ``LocalLink`` where dispatch lands on the same loop tick)
        # can hit ``AgentClient.receive`` before ``wait_for_channel_event``
        # creates the queue, and the envelope is silently dropped.
        actual_client.ensure_channel_inbox(channel.channel_id)

        # Suppress the default handler for this channel — we own its
        # lifecycle here; we don't want the handler to ALSO run a turn
        # on the reply envelope when it lands.
        actual_client._suppress_handler(channel.channel_id)
        try:
            # Send the prompt as the initiator's turn. ``depth`` is
            # stamped from the outer handler's depth + 1 so the hub can
            # enforce ``Rule.limits.delegation_depth``.
            try:
                await channel.send(
                    prompt,
                    audience=[target_id],
                    depth=actual_client.current_handling_depth + 1,
                )
            except Exception as exc:
                return f"Error: prompt send failed: {exc}"

            # Wait for the respondent's reply OR a terminal channel
            # event. Terminating events resolve fast so the caller
            # doesn't sit at ``timeout`` (300s default) when the channel
            # was rejected, expired, or closed out-of-band.
            try:
                envelope = await actual_client.wait_for_channel_event(
                    channel_id=channel.channel_id,
                    predicate=_reply_or_terminal_predicate(target_id),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return f"Error: delegate to {target!r} timed out after {timeout}s"
            except Exception as exc:
                return f"Error: delegate to {target!r} failed: {exc}"

            # Terminal channel event → fail-fast with the close reason.
            if envelope.event_type in _TERMINAL_CHANNEL_EVENTS:
                reason = envelope.event_data.get("reason", envelope.event_type)
                return f"Error: delegate to {target!r} channel closed: {reason}"
        finally:
            actual_client._unsuppress_handler(channel.channel_id)
            actual_client.discard_channel_inbox(channel.channel_id)

        body = envelope.event_data.get("text", "")
        return body if isinstance(body, str) else str(body)

    return delegate


def _reply_or_terminal_predicate(target_id: str):
    """Match the respondent's substantive reply OR any terminal channel event."""

    def matches(envelope: Envelope) -> bool:
        if envelope.event_type == EV_TEXT and envelope.sender_id == target_id:
            return True
        return envelope.event_type in _TERMINAL_CHANNEL_EVENTS

    return matches
