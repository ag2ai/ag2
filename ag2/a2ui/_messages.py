# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What counts as an A2UI message, in each direction."""

from typing import TypeGuard

from ._types import JsonObject, JsonValue, ServerToClientMessage, server_to_client_payload_keys

CLIENT_TO_SERVER_KEYS = ("action", "callAgentFunction", "rendererFunctionResponse", "error")
"""The payload key of each envelope a renderer sends the agent, in the order an envelope is classified."""


def server_message_kind(value: JsonValue) -> str | None:
    """The payload key of a server→client message: an object with exactly one; ``None`` for anything else.

    The payload's contents are left to validation.
    """
    if not isinstance(value, dict):
        return None
    keys = server_to_client_payload_keys().intersection(value)
    return next(iter(keys)) if len(keys) == 1 else None


def is_server_message(value: JsonValue) -> TypeGuard[ServerToClientMessage]:
    """Whether ``value`` is a server→client message envelope (see :func:`server_message_kind`)."""
    return server_message_kind(value) is not None


def mentions_server_message(value: JsonValue) -> bool:
    """Whether ``value``, or an item of it if it is a list, has a server→client payload key holding an object.

    Looser than :func:`is_server_message`: it flags results shaped like messages that are not exactly one.
    """
    candidates = value if isinstance(value, list) else [value]
    return any(
        isinstance(item, dict)
        and any(isinstance(item[key], dict) for key in server_to_client_payload_keys().intersection(item))
        for item in candidates
    )


def is_client_envelope(value: JsonValue) -> TypeGuard[JsonObject]:
    """Whether ``value`` is an envelope a renderer sends the agent: an object with a payload object."""
    return isinstance(value, dict) and any(isinstance(value.get(key), dict) for key in CLIENT_TO_SERVER_KEYS)


__all__ = (
    "CLIENT_TO_SERVER_KEYS",
    "is_client_envelope",
    "is_server_message",
    "mentions_server_message",
    "server_message_kind",
)
