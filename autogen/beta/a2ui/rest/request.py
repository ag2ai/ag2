# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Parse the transport-neutral A2UI server request body.

The REST/SSE adapter speaks a minimal, dependency-free JSON contract (no
``ag-ui`` / ``a2a-sdk`` types) so any client that can POST JSON can drive an
:class:`~autogen.beta.a2ui.A2UIAgent`::

    {
      "messages":  [{"role": "user", "content": "show a booking form"}],
      "variables": {"locale": "en"},
      "a2ui":      [{"version": "v0.9", "action": {"name": "confirm", ...}}]
    }

The server is **stateless**: the client sends the full conversation each turn.
Prior messages become history; the trailing run of ``user`` messages plus any
client→server ``a2ui`` envelopes (button clicks / errors) become the current
turn — the latter rewritten to corrective prompts via the shared
``action_to_prompt`` / ``error_to_prompt`` helpers.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from autogen.beta.events import BaseEvent, Input, ModelMessage, ModelRequest, ModelResponse, TextInput

from ..actions import A2UIAction
from ..incoming import action_to_prompt, error_to_prompt, function_response_to_prompt, parse_incoming_message

logger = logging.getLogger(__name__)

# Roles understood in the ``messages`` array. ``tool`` round-trips are not part
# of the native A2UI request contract — client→server interaction flows through
# the ``a2ui`` envelope array instead.
_PROMPT_ROLES = ("system", "developer")


@dataclass(slots=True)
class A2UIServerRequest:
    """A parsed, transport-neutral A2UI turn.

    ``current_inputs`` is the initial event payload for the agent turn: the
    trailing user message(s) plus any prompts synthesized from client→server
    ``a2ui`` action/error envelopes. ``prompt`` carries system/developer
    messages; ``history`` carries the prior conversation turns.
    """

    current_inputs: list[Input] = field(default_factory=list)
    history: list[BaseEvent] = field(default_factory=list)
    prompt: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)


def parse_request(
    body: bytes | str | dict[str, Any],
    *,
    resolve_action: Callable[[str], A2UIAction | None],
) -> A2UIServerRequest:
    """Parse a raw request body into an :class:`A2UIServerRequest`.

    Args:
        body: The raw JSON request body (bytes/str) or an already-decoded dict.
        resolve_action: Looks up a registered :class:`A2UIAction` by name (e.g.
            ``agent.get_action``). Used to rewrite incoming ``action`` envelopes;
            an unregistered action is dropped with a warning rather than leaking
            raw client data into the prompt.

    Returns:
        The parsed turn. When neither ``messages`` nor ``a2ui`` yields any
        current-turn input, ``current_inputs`` is empty and the dispatcher
        falls back to a blank user turn — callers that require a non-empty
        turn should validate before dispatching.

    Raises:
        ValueError: If the body is not valid JSON, not a JSON object, or
            ``messages`` / ``variables`` / ``a2ui`` have the wrong shape.
    """
    data = _decode_body(body)

    raw_messages = data.get("messages", [])
    if not isinstance(raw_messages, list):
        raise ValueError("'messages' must be a list")

    raw_variables = data.get("variables", {})
    if not isinstance(raw_variables, dict):
        raise ValueError("'variables' must be an object")

    raw_a2ui = data.get("a2ui", [])
    if not isinstance(raw_a2ui, list):
        raise ValueError("'a2ui' must be a list")

    prompt, history, current_inputs = _map_messages(raw_messages)
    current_inputs.extend(_map_a2ui_envelopes(raw_a2ui, resolve_action))

    return A2UIServerRequest(
        current_inputs=current_inputs,
        history=history,
        prompt=prompt,
        variables=dict(raw_variables),
    )


def _decode_body(body: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(body, dict):
        return body
    try:
        decoded = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"request body is not valid JSON: {e}") from e
    if not isinstance(decoded, dict):
        raise ValueError("request body must be a JSON object")
    return decoded


def _map_messages(raw_messages: list[Any]) -> tuple[list[str], list[BaseEvent], list[Input]]:
    """Split the ``messages`` array into prompt / history / current-turn inputs.

    Mirrors the AG-UI history mapping: a trailing run of ``user`` messages is
    kept as the current turn (so the LLM sees a meaningful ``messages[-1]``);
    any non-user message flushes the buffered user run into ``history`` as a
    ``ModelRequest``.
    """
    prompt: list[str] = []
    history: list[BaseEvent] = []
    input_buffer: list[Input] = []

    for msg in raw_messages:
        if not isinstance(msg, dict):
            raise ValueError("each message must be an object")
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str):
            raise ValueError("message 'content' must be a string")

        if role == "user":
            input_buffer.append(TextInput(content))
            continue

        if input_buffer:
            history.append(ModelRequest(list(input_buffer)))
            input_buffer = []

        if role in _PROMPT_ROLES:
            if content:
                prompt.append(content)
        elif role == "assistant":
            history.append(ModelResponse(ModelMessage(content) if content else None))
        else:
            raise ValueError(f"unsupported message role: {role!r}")

    return prompt, history, input_buffer


def _map_a2ui_envelopes(
    raw_a2ui: list[Any],
    resolve_action: Callable[[str], A2UIAction | None],
) -> list[Input]:
    """Rewrite client→server ``action`` / ``functionResponse`` / ``error`` envelopes as prompt inputs.

    The server is stateless, so a v1.0 ``functionResponse`` (the client's reply
    to a prior server ``callFunction``) simply arrives as another envelope in
    the next request and is rewritten to a continuation prompt — no pause/resume
    bookkeeping is needed on this transport.
    """
    inputs: list[Input] = []
    for envelope in raw_a2ui:
        result = parse_incoming_message(envelope)
        if result.kind == "action" and result.action is not None:
            action_def = resolve_action(result.action.name)
            prompt = action_to_prompt(result.action, action_def)
            if prompt is None:
                logger.warning(
                    "Dropping A2UI action '%s' — no matching A2UIAction registration.",
                    result.action.name,
                )
                continue
            inputs.append(TextInput(prompt))
        elif result.kind == "functionResponse" and result.function_response is not None:
            if not result.function_response.function_call_id:
                logger.warning(
                    "Dropping A2UI functionResponse — missing functionCallId, cannot correlate to a callFunction.",
                )
                continue
            inputs.append(TextInput(function_response_to_prompt(result.function_response)))
        elif result.kind == "error" and result.error is not None:
            inputs.append(TextInput(error_to_prompt(result.error)))
        else:
            logger.debug("Skipping A2UI envelope of unknown kind: %s", result.parse_error)
    return inputs


__all__ = ("A2UIServerRequest", "parse_request")
