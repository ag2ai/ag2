# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from mcp.server.extension import validate_extension_identifier

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext

# An SEP-2133 extension map: reverse-DNS identifier -> that extension's settings.
# The settings object is opaque to ag2 — each extension defines its own keys.
ExtensionMap = Mapping[str, Mapping[str, Any]]


def validated_extensions(extensions: ExtensionMap) -> dict[str, dict[str, Any]]:
    """Copy ``extensions`` into the shape the low-level server advertises, validating keys.

    Every identifier is checked against the SEP-2133 grammar (a reverse-DNS
    prefix, a ``/``, then a name) using the SDK's own validator, so a typo fails
    where the server is built rather than on the wire. Values are copied so a
    caller's mapping cannot mutate what is advertised afterwards.
    """
    validated: dict[str, dict[str, Any]] = {}
    for identifier, settings in extensions.items():
        validate_extension_identifier(identifier, owner="MCPServer(extensions=...)")
        validated[identifier] = dict(settings)
    return validated


def client_extension(ctx: "ServerRequestContext[Any, Any] | None", identifier: str) -> dict[str, Any] | None:
    """The settings the connected client advertised for ``identifier``, or ``None``.

    ``None`` means the client said nothing about this extension; ``{}`` means it
    advertised support with no settings. The wire distinguishes the two and so
    does this, because "supported, nothing to configure" is a different answer
    from "not supported". ``None`` also covers the case of no live request at
    all, so a handler calling this outside one gets an answer rather than an
    exception.

    **Reading works in both eras.** A handshake-era client's
    ``capabilities.extensions`` arrives intact and a modern-era request carries
    its own capabilities envelope, so branching on what a client advertised is
    useful today. Advertising in the other direction is not symmetric — see
    :class:`~ag2.mcp.MCPServer`'s ``extensions`` parameter.
    """
    if ctx is None:
        return None
    capabilities = ctx.session.client_capabilities
    extensions = capabilities.extensions if capabilities is not None else None
    if not extensions:
        return None
    settings = extensions.get(identifier)
    return None if settings is None else dict(settings)


__all__ = (
    "ExtensionMap",
    "client_extension",
    "validated_extensions",
)
