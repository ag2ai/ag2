# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral A2UI client capability negotiation: decode the client's
advertised catalogs into :class:`A2UIClientCapabilities` and fold them into the
turn's system prompt via :func:`capabilities_to_prompt`. Advisory only — a
mismatch is logged and surfaced to the LLM but never blocks output.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from ._types import A2UIVersion, JsonObject, JsonValue
from .incoming import sanitize_for_prompt

logger = logging.getLogger(__name__)

A2UI_CLIENT_CAPABILITIES_METADATA_KEY = "a2uiClientCapabilities"

# Default wire-shape version key for the nested capability/data-model objects
# (the client wraps them under their protocol version, e.g. ``{"v0.9": {...}}``).
# Callers serving a v0.9.1 / v1.0 agent pass their version via ``version_key``.
_DEFAULT_VERSION_KEY: A2UIVersion = "v0.9"

# Cap how many catalog ids / inline-component names we splice into the prompt,
# bounding the blast radius of a runaway or hostile capabilities payload.
_MAX_CATALOG_IDS = 32
_MAX_INLINE_NAMES = 40


@dataclass(slots=True)
class A2UIClientCapabilities:
    """Decoded ``a2uiClientCapabilities.<version>`` payload."""

    supported_catalog_ids: list[str] = field(default_factory=list)
    inline_catalogs: list[JsonObject] = field(default_factory=list)


def parse_client_capabilities(
    metadata: Mapping[str, JsonValue] | None,
    *,
    version_key: A2UIVersion = _DEFAULT_VERSION_KEY,
) -> A2UIClientCapabilities | None:
    """Decode ``metadata.a2uiClientCapabilities.<version_key>``.

    ``version_key`` selects the nested version object (defaults to ``"v0.9"``;
    pass the agent's version, e.g. ``"v1.0"``, when serving a newer client).
    Returns ``None`` if the metadata is missing, malformed, or does not declare
    the requested version. ``supportedCatalogIds`` entries are coerced to ``str``
    and non-dict ``inlineCatalogs`` entries are dropped.
    """
    if not metadata:
        return None
    caps = metadata.get(A2UI_CLIENT_CAPABILITIES_METADATA_KEY)
    if not isinstance(caps, dict):
        return None
    v = caps.get(version_key)
    if not isinstance(v, dict):
        return None
    raw_ids = v.get("supportedCatalogIds")
    raw_inline = v.get("inlineCatalogs")
    return A2UIClientCapabilities(
        supported_catalog_ids=[str(x) for x in raw_ids] if isinstance(raw_ids, list) else [],
        inline_catalogs=[c for c in raw_inline if isinstance(c, dict)] if isinstance(raw_inline, list) else [],
    )


def _summarize_inline_catalog(catalog: JsonObject) -> str | None:
    """One-line summary of an inline catalog's components/functions, or ``None`` if it has neither."""
    components = catalog.get("components")
    functions = catalog.get("functions")
    comp_names = sorted(components.keys()) if isinstance(components, dict) else []
    func_names = sorted(functions.keys()) if isinstance(functions, dict) else []
    if not comp_names and not func_names:
        return None

    raw_id = catalog.get("catalogId") or catalog.get("$id")
    label = sanitize_for_prompt(str(raw_id)) if isinstance(raw_id, str) and raw_id else "(no catalogId)"

    parts: list[str] = []
    if comp_names:
        shown = comp_names[:_MAX_INLINE_NAMES]
        names = ", ".join(sanitize_for_prompt(n) for n in shown)
        if len(comp_names) > _MAX_INLINE_NAMES:
            names += f", …(+{len(comp_names) - _MAX_INLINE_NAMES} more)"
        parts.append(f"components: {names}")
    if func_names:
        shown = func_names[:_MAX_INLINE_NAMES]
        names = ", ".join(sanitize_for_prompt(n) for n in shown)
        if len(func_names) > _MAX_INLINE_NAMES:
            names += f", …(+{len(func_names) - _MAX_INLINE_NAMES} more)"
        parts.append(f"functions: {names}")
    return f"catalog '{label}': {'; '.join(parts)}"


def capabilities_to_prompt(caps: A2UIClientCapabilities | None, *, catalog_id: str) -> str:
    """Render negotiated client capabilities as a per-turn system-prompt fragment.

    ``catalog_id`` is the agent's active catalog. Returns ``""`` when there is
    nothing actionable to tell the LLM (no caps, or an empty payload). When the
    client lists catalogs that exclude the agent's ``catalog_id``, a mismatch
    note is added to the prompt and a warning is logged — negotiation is advisory
    and never blocks output. Inline catalogs (if any) are summarized so the LLM
    can target their components/functions.
    """
    if caps is None:
        return ""

    supported = caps.supported_catalog_ids[:_MAX_CATALOG_IDS]
    inline_summaries = [s for s in (_summarize_inline_catalog(c) for c in caps.inline_catalogs) if s]
    if not supported and not inline_summaries:
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

    if inline_summaries:
        lines.append("The client also provided inline catalog definitions you may target:")
        lines.extend(f"- {s}" for s in inline_summaries)
        lines.append("")

    return "\n".join(lines).rstrip()


__all__ = (
    "A2UI_CLIENT_CAPABILITIES_METADATA_KEY",
    "A2UIClientCapabilities",
    "capabilities_to_prompt",
    "parse_client_capabilities",
)
