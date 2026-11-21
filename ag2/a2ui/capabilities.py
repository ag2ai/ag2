# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral A2UI client capability negotiation: decode the client's
advertised catalogs into :class:`A2UIClientCapabilities` and describe them in the
turn's system prompt via :func:`capabilities_to_prompt`.

Negotiation itself happens in the runtime: only the agent's catalogs the client
names are offered for its turn, and with none in common the turn has no A2UI.
Inline catalogs are decoded but never offered: the agent does not accept them
(it does not declare ``acceptsInlineCatalogs``, and the schema says a client
"should only" send them to an agent that does).
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from ._sanitize import sanitize_for_prompt
from ._types import A2UIVersion, JsonObject, JsonValue
from ._versions import (
    A2UI_CLIENT_CAPABILITIES_METADATA_KEY,
    A2UI_RENDERER_CAPABILITIES_METADATA_KEY,
    DEFAULT_PROTOCOL,
    get_protocol,
)

logger = logging.getLogger(__name__)

# Cap how many catalog ids we splice into the prompt,
# bounding the blast radius of a runaway or hostile capabilities payload.
_MAX_CATALOG_IDS = 32


@dataclass(slots=True)
class A2UIClientCapabilities:
    """Decoded ``a2uiClientCapabilities.<version>`` payload."""

    supported_catalog_ids: list[str] = field(default_factory=list)
    inline_catalogs: list[JsonObject] = field(default_factory=list)
    """Decoded for inspection only; the agent does not accept inline catalogs, so they are never offered."""


def parse_client_capabilities(
    metadata: Mapping[str, JsonValue] | None,
    *,
    version_key: A2UIVersion = DEFAULT_PROTOCOL.version,
) -> A2UIClientCapabilities | None:
    """Decode the client's capabilities for protocol version ``version_key``.

    v0.9 and v0.9.1 read ``metadata.a2uiClientCapabilities["v0.9"]``; v1.0 reads
    ``metadata.a2uiRendererCapabilities["1.0"]``. Callers serving a non-default
    agent pass its ``protocol_version``.
    Returns ``None`` if the metadata does not carry the key or does not declare
    the requested version (no capabilities). Capabilities that are present but
    break the schema (``renderer_capabilities.json``: ``supportedCatalogIds`` a
    required array of strings, ``inlineCatalogs`` an array of objects with a
    string ``catalogId``) are logged and read as supporting no catalog, so the
    turn sends no A2UI: a2ui-core rejects them
    (``conformance/agent/renderer_capabilities.yaml``), and guessing at them
    could offer a renderer catalogs it never named.
    """
    if not metadata or (caps := metadata.get((protocol := get_protocol(version_key)).capabilities_key)) is None:
        return None
    if not isinstance(caps, dict):
        return _malformed(protocol.capabilities_key, "is not an object")
    if (v := caps.get(protocol.capabilities_version_key)) is None:
        return None
    if not isinstance(v, dict):
        return _malformed(protocol.capabilities_key, f"entry {protocol.capabilities_version_key!r} is not an object")
    raw_ids = v.get("supportedCatalogIds")
    ids = [x for x in raw_ids if isinstance(x, str)] if isinstance(raw_ids, list) else []
    if not isinstance(raw_ids, list) or len(ids) != len(raw_ids):
        return _malformed(protocol.capabilities_key, "'supportedCatalogIds' is not an array of strings")
    raw_inline = v.get("inlineCatalogs", [])
    inline = (
        [c for c in raw_inline if isinstance(c, dict) and isinstance(c.get("catalogId"), str)]
        if isinstance(raw_inline, list)
        else []
    )
    if not isinstance(raw_inline, list) or len(inline) != len(raw_inline):
        return _malformed(protocol.capabilities_key, "'inlineCatalogs' is not an array of catalogs with a 'catalogId'")
    return A2UIClientCapabilities(supported_catalog_ids=ids, inline_catalogs=inline)


def _malformed(key: str, problem: str) -> A2UIClientCapabilities:
    logger.warning(
        "A2UI renderer capabilities %r are malformed (%s); treating them as supporting no catalog.", key, problem
    )
    return A2UIClientCapabilities()


def capabilities_to_prompt(caps: A2UIClientCapabilities | None, *, catalog_id: str) -> str:
    """Render negotiated client capabilities as a per-turn system-prompt fragment.

    ``catalog_id`` is the catalog the turn's prompt has ``createSurface`` name.
    Returns ``""`` when there is nothing to tell the LLM (no caps, or no
    ``supportedCatalogIds``). When the client lists catalogs that exclude
    ``catalog_id``, a mismatch note is added and a warning is logged; the runtime
    offers only catalogs the client names, so its own prompts never hit this.
    Inline catalogs are not described: the agent does not accept them.
    """
    if caps is None:
        return ""

    supported = caps.supported_catalog_ids[:_MAX_CATALOG_IDS]
    if not supported:
        return ""

    lines = ["## A2UI Client Capabilities", ""]

    if supported:
        lines.append("The connected client renderer supports these component/function catalogs:")
        lines.extend(f"- {sanitize_for_prompt(cid)}" for cid in supported)
        if len(caps.supported_catalog_ids) > _MAX_CATALOG_IDS:
            lines.append(f"- …(+{len(caps.supported_catalog_ids) - _MAX_CATALOG_IDS} more)")
        lines.append("")

        if catalog_id not in caps.supported_catalog_ids:
            logger.warning(
                "A2UI client did not advertise the agent's catalog %r in supportedCatalogIds %r; "
                "the client may be unable to render this agent's UI.",
                catalog_id,
                caps.supported_catalog_ids,
            )
            lines.append(
                f"Note: the client did NOT list this agent's catalog ('{sanitize_for_prompt(catalog_id)}'). "
                "Prefer components and functions the client supports; it may be unable to render others."
            )
            lines.append("")

    return "\n".join(lines).rstrip()


__all__ = (
    "A2UI_CLIENT_CAPABILITIES_METADATA_KEY",
    "A2UI_RENDERER_CAPABILITIES_METADATA_KEY",
    "A2UIClientCapabilities",
    "capabilities_to_prompt",
    "parse_client_capabilities",
)
