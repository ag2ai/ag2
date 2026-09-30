# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A Gemini tool call's thought signature as an AG-UI encrypted value, and back."""

from base64 import b64decode, b64encode

from ag_ui.core import ReasoningEncryptedValueEvent

from ag2 import events

# `ag2[ag-ui]` does not install google-genai. Without it no Gemini call can
# exist, so there is no signature to read and none to restore.
try:
    from ag2.config.gemini.events import GeminiToolCallEvent
except ImportError:
    GeminiToolCallEvent = None  # type: ignore[assignment,misc]


def encrypted_signature_of(event: events.ToolCallEvent) -> str | None:
    """The call's thought signature as the base64 the wire carries, or `None` if it has none."""
    if GeminiToolCallEvent is None or not isinstance(event, GeminiToolCallEvent) or event.thought_signature is None:
        return None
    return b64encode(event.thought_signature).decode()


def signature_event(call_id: str, signature: str, timestamp: int) -> ReasoningEncryptedValueEvent:
    """The encrypted value that carries a call's signature. Send it after the call starts: a consumer may drop a value whose entity it has not seen."""
    return ReasoningEncryptedValueEvent(
        subtype="tool-call", entity_id=call_id, encrypted_value=signature, timestamp=timestamp
    )


def restore_tool_call(
    provider: str | None,
    *,
    id: str,
    name: str,
    arguments: str,
    encrypted_value: str | None,
) -> events.ToolCallEvent:
    """A replayed tool call, carrying its signature back where the provider needs one."""
    if provider == "gemini" and encrypted_value is not None and GeminiToolCallEvent is not None:
        return GeminiToolCallEvent(id=id, name=name, arguments=arguments, thought_signature=b64decode(encrypted_value))
    return events.ToolCallEvent(id=id, name=name, arguments=arguments)


__all__ = ("encrypted_signature_of", "restore_tool_call", "signature_event")
