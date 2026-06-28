# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2A AgentExtension builder and activation helper for A2UI (v0.9 / v0.9.1 / v1.0)."""

from collections.abc import Sequence

from a2a.server.agent_execution import RequestContext
from a2a.types import AgentExtension

from .._types import A2UIVersion, JsonValue
from ..constants import (
    A2UI_DEFAULT_CATALOG_ID_BY_VERSION,
    A2UI_DEFAULT_VERSION,
    A2UI_EXTENSION_URI_BY_VERSION,
)


def get_a2ui_agent_extension(
    *,
    version: A2UIVersion = A2UI_DEFAULT_VERSION,
    supported_catalog_ids: Sequence[str] | None = None,
    accepts_inline_catalogs: bool = False,
) -> AgentExtension:
    """Create the A2UI ``AgentExtension`` for an A2A Agent Card.

    This extension declaration tells clients that the agent supports
    A2UI output. Include it in the agent's A2A card extensions.

    Args:
        version: The A2UI protocol version to advertise: "v0.9" (default),
            "v0.9.1", or "v1.0". Selects the extension ``uri`` and the default
            catalog id.
        supported_catalog_ids: List of catalog IDs the agent can generate.
            Defaults to the basic catalog for ``version``. Matches
            ``server_capabilities.json#/<version>/supportedCatalogIds``.
        accepts_inline_catalogs: Whether the agent accepts inline catalogs
            from the client. Matches
            ``server_capabilities.json#/<version>/acceptsInlineCatalogs``.

    Returns:
        A configured ``AgentExtension`` for the requested A2UI version.
    """
    default_catalog = A2UI_DEFAULT_CATALOG_ID_BY_VERSION[version]
    catalog_source = supported_catalog_ids if supported_catalog_ids is not None else [default_catalog]
    # Spread into a JsonValue list so it drops into the JsonValue-typed params dict
    # (a plain list[str] would trip list invariance).
    catalog_ids: list[JsonValue] = [*catalog_source]
    params: dict[str, JsonValue] = {"supportedCatalogIds": catalog_ids}
    if accepts_inline_catalogs:
        params["acceptsInlineCatalogs"] = True

    return AgentExtension(
        uri=A2UI_EXTENSION_URI_BY_VERSION[version],
        description=f"Provides agent-driven UI using the A2UI {version} JSON format.",
        params=params,
    )


def try_activate_a2ui_extension(
    context: RequestContext,
    *,
    version: A2UIVersion = A2UI_DEFAULT_VERSION,
) -> bool:
    """Activate the A2UI extension if the client requested it.

    Call this in an ``AgentExecutor`` to negotiate A2UI support. If the
    client's request includes the A2UI extension URI for ``version``, it is
    recorded under ``context.metadata['activated_extensions']`` and the
    function returns True.
    """
    extension_uri = A2UI_EXTENSION_URI_BY_VERSION[version]
    requested = getattr(context, "requested_extensions", None) or []
    if extension_uri not in requested:
        return False
    if context.metadata is None:
        context.metadata = {}
    activated = context.metadata.setdefault("activated_extensions", [])
    if extension_uri not in activated:
        activated.append(extension_uri)
    return True
