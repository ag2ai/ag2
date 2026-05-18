# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2A AgentExtension builder and activation helper for A2UI v0.9."""

from collections.abc import Sequence

from a2a.server.agent_execution import RequestContext
from a2a.types import AgentExtension

from .._types import JsonValue
from ..constants import A2UI_DEFAULT_CATALOG_ID, A2UI_EXTENSION_URI


def get_a2ui_agent_extension(
    *,
    supported_catalog_ids: Sequence[str] | None = None,
    accepts_inline_catalogs: bool = False,
) -> AgentExtension:
    """Create the A2UI v0.9 ``AgentExtension`` for an A2A Agent Card.

    This extension declaration tells clients that the agent supports
    A2UI output. Include it in the agent's A2A card extensions.

    Args:
        supported_catalog_ids: List of catalog IDs the agent can generate.
            Defaults to ``[A2UI_DEFAULT_CATALOG_ID]`` (the basic catalog).
            Matches ``server_capabilities.json#/v0.9/supportedCatalogIds``.
        accepts_inline_catalogs: Whether the agent accepts inline catalogs
            from the client. Matches
            ``server_capabilities.json#/v0.9/acceptsInlineCatalogs``.

    Returns:
        A configured ``AgentExtension`` for A2UI v0.9.
    """
    catalog_ids = list(supported_catalog_ids) if supported_catalog_ids is not None else [A2UI_DEFAULT_CATALOG_ID]
    params: dict[str, JsonValue] = {"supportedCatalogIds": catalog_ids}
    if accepts_inline_catalogs:
        params["acceptsInlineCatalogs"] = True

    return AgentExtension(
        uri=A2UI_EXTENSION_URI,
        description="Provides agent-driven UI using the A2UI v0.9 JSON format.",
        params=params,
    )


def try_activate_a2ui_extension(context: RequestContext) -> bool:
    """Activate the A2UI extension if the client requested it.

    Call this in an ``AgentExecutor`` to negotiate A2UI support. If the
    client's request includes the A2UI extension URI, it is recorded
    under ``context.metadata['activated_extensions']`` and the function
    returns True.
    """
    requested = getattr(context, "requested_extensions", None) or []
    if A2UI_EXTENSION_URI not in requested:
        return False
    if context.metadata is None:
        context.metadata = {}
    activated = context.metadata.setdefault("activated_extensions", [])
    if A2UI_EXTENSION_URI not in activated:
        activated.append(A2UI_EXTENSION_URI)
    return True
