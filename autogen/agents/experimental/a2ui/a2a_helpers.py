# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2A extension helpers for A2UI v0.9.

Provides functions for creating A2UI-typed A2A DataParts, checking parts
for A2UI content, and negotiating the A2UI extension during A2A sessions.

Requires the ``a2a`` extra: ``pip install ag2[a2a]``
"""

from typing import Any

from ....a2a.constants import A2UI_MIME_TYPE
from ....import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from a2a.server.agent_execution import RequestContext
    from a2a.types import AgentExtension, DataPart, Part

A2UI_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.9"

MIME_TYPE_KEY = "mimeType"
A2UI_DEFAULT_DELIMITER = "---a2ui_JSON---"
A2UI_DEFAULT_VERSION = "v0.9"
A2UI_DEFAULT_ACTIVITY_TYPE = "a2ui-surface"


@require_optional_import(["a2a"], "a2a")
def create_a2ui_part(a2ui_data: dict[str, Any] | list[dict[str, Any]]) -> Part | list[Part]:
    """Create A2A Part(s) containing A2UI data.

    Each A2UI operation (e.g., ``createSurface``, ``updateComponents``) is
    sent as a separate DataPart with the operation key at the top level of
    the ``data`` field. This matches what genui_a2a's ``A2uiAgentConnector``
    expects when processing A2UI messages from status-update events.

    Args:
        a2ui_data: A2UI message(s) — either a single operation dict or a list
            of operation dicts (e.g., ``[createSurface, updateComponents]``).

    Returns:
        A single ``Part`` if one operation, or a list of ``Part`` objects
        if multiple operations.
    """
    if isinstance(a2ui_data, dict):
        a2ui_data = [a2ui_data]

    parts = [
        Part(
            root=DataPart(
                data=op,
                metadata={MIME_TYPE_KEY: A2UI_MIME_TYPE},
            )
        )
        for op in a2ui_data
    ]
    return parts[0] if len(parts) == 1 else parts


def is_a2ui_part(part: Part) -> bool:
    """Check if an A2A Part contains A2UI data.

    Args:
        part: The A2A Part to check.

    Returns:
        True if the part is a DataPart with A2UI MIME type.
    """
    return (
        isinstance(part.root, DataPart)
        and part.root.metadata is not None
        and part.root.metadata.get(MIME_TYPE_KEY) == A2UI_MIME_TYPE
    )


def get_a2ui_datapart(part: Part) -> DataPart | None:
    """Extract the A2UI DataPart from an A2A Part, if present.

    Args:
        part: The A2A Part to extract from.

    Returns:
        The ``DataPart`` if the part contains A2UI data, None otherwise.
    """
    if is_a2ui_part(part):
        return part.root  # type: ignore[return-value]
    return None


@require_optional_import(["a2a"], "a2a")
def get_a2ui_agent_extension(
    accepts_inline_custom_catalog: bool = False,
) -> AgentExtension:
    """Create the A2UI v0.9 AgentExtension for an A2A Agent Card.

    This extension declaration tells clients that the agent supports
    A2UI output. Include it in the agent's A2A card extensions.

    Args:
        accepts_inline_custom_catalog: Whether the agent accepts inline
            custom catalogs from the client.

    Returns:
        A configured ``AgentExtension`` for A2UI v0.9.
    """
    params: dict[str, Any] = {}
    if accepts_inline_custom_catalog:
        params["acceptsInlineCustomCatalog"] = True

    return AgentExtension(
        uri=A2UI_EXTENSION_URI,
        description="Provides agent-driven UI using the A2UI v0.9 JSON format.",
        params=params if params else None,
    )


@require_optional_import(["a2a"], "a2a")
def try_activate_a2ui_extension(context: RequestContext) -> bool:
    """Activate the A2UI extension if the client requested it.

    Call this in your ``AgentExecutor`` to negotiate A2UI support.
    If the client's request includes the A2UI extension URI, it is
    activated and the function returns True.

    Args:
        context: The A2A request context.

    Returns:
        True if the A2UI extension was activated, False otherwise.
    """
    if A2UI_EXTENSION_URI in context.requested_extensions:
        context.add_activated_extension(A2UI_EXTENSION_URI)
        return True
    return False
