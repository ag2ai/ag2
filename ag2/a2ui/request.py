# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Parse the transport-neutral A2UI server request body — a minimal JSON
contract (``messages``, ``variables``, ``a2ui``, ``a2uiClientCapabilities``).
The server is stateless: the client sends the full conversation each turn.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ag2.events import BaseEvent, Input, ModelMessage, ModelRequest, ModelResponse, TextInput

from ._types import A2UIVersion, JsonObject, JsonValue
from .actions import A2UIEventAction
from .capabilities import A2UIClientCapabilities, parse_client_capabilities
from .incoming import A2UIIncomingParseResult, iter_incoming_prompts, parse_incoming_interactions

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
    ``client_capabilities`` carries the decoded ``a2uiClientCapabilities``, if
    present, so the dispatcher can fold catalog negotiation into the prompt.
    ``client_interactions`` carries the typed client→server envelopes so the
    dispatcher can surface each as an ``A2UIClientEvent`` on the turn's stream.
    """

    current_inputs: list[Input] = field(default_factory=list)
    history: list[BaseEvent] = field(default_factory=list)
    prompt: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    client_capabilities: A2UIClientCapabilities | None = None
    client_interactions: list[A2UIIncomingParseResult] = field(default_factory=list)


def parse_request(
    body: bytes | str | JsonObject,
    *,
    resolve_action: Callable[[str], A2UIEventAction | None],
    version_key: A2UIVersion = "v0.9",
) -> A2UIServerRequest:
    """Parse a raw request body into an :class:`A2UIServerRequest`.

    Args:
        body: The raw JSON request body (bytes/str) or an already-decoded dict.
        resolve_action: Looks up a registered server action by name (e.g.
            ``runtime.get_action``). A matched action's click runs on the server
            (handled in the turn core, kept out of the prompt); an unmatched click
            is rewritten generically so the LLM can react to the button it rendered.
        version_key: The protocol version under which to read
            ``a2uiClientCapabilities`` (the client nests it under its version,
            e.g. ``"v1.0"``). Defaults to ``"v0.9"``; callers serving a non-v0.9
            agent should pass ``agent.schema_manager.version_string`` (the ASGI
            adapter does), else a newer client's capabilities are silently
            dropped.

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
        client_capabilities=parse_client_capabilities(data, version_key=version_key),
        client_interactions=parse_incoming_interactions(raw_a2ui),
    )


def _decode_body(body: bytes | str | JsonObject) -> JsonObject:
    if isinstance(body, dict):
        return body
    try:
        decoded: JsonValue = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"request body is not valid JSON: {e}") from e
    if not isinstance(decoded, dict):
        raise ValueError("request body must be a JSON object")
    return decoded


def _map_messages(raw_messages: list[JsonValue]) -> tuple[list[str], list[BaseEvent], list[Input]]:
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
    raw_a2ui: list[JsonValue],
    resolve_action: Callable[[str], A2UIEventAction | None],
) -> list[Input]:
    """Rewrite client→server ``action`` / ``functionResponse`` / ``error`` envelopes as prompt inputs.

    The server is stateless, so a v1.0 ``functionResponse`` (the client's reply
    to a prior server ``callFunction``) simply arrives as another envelope in
    the next request and is rewritten to a continuation prompt — no pause/resume
    bookkeeping is needed on this transport.
    """
    return [TextInput(prompt) for prompt in iter_incoming_prompts(raw_a2ui, resolve_action)]


__all__ = ("A2UIServerRequest", "parse_request")
