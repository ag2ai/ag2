# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from a2a.client import ClientCallContext
from a2a.client.service_parameters import ServiceParametersFactory, with_a2a_extensions
from a2a.types import AgentCard

from .errors import A2AExtensionNotSupportedError

# AG2 client-tools extension: announced via the URI on the AgentCard,
# wire-tagged via the MIME types below. Private to AG2 — never sent to
# non-AG2 servers, never inspected by intermediaries.
EXTENSION_URI = "urn:ag2:client-tools:v1"

MIME_TOOL_SCHEMAS = "application/vnd.ag2.tool-schemas+json"
MIME_TOOL_CALL = "application/vnd.ag2.tool-call+json"
MIME_TOOL_RESULT = "application/vnd.ag2.tool-result+json"
MIME_HISTORY = "application/vnd.ag2.history+json"

# Bidirectional context-variables sync rides on Message.metadata under this key.
CONTEXT_UPDATE_METADATA_KEY = "ag2.context_update"

# Dependency key for splicing extra A2A ``Part``s onto the outgoing message.
EXTRA_PARTS_DEPENDENCY_KEY = "a2a:extra_parts"

# Per-call tenant override in ``context.variables`` — wins over ``A2AConfig.tenant``.
TENANT_VARIABLE_KEY = "a2a:tenant"

# Extension URIs AG2 itself implements — used by the client to satisfy
# ``AgentExtension.required`` checks without explicit user activation.
NATIVE_EXTENSION_URIS = frozenset({EXTENSION_URI})


def validate_extension_activation(card: AgentCard, uris: Sequence[str], *, url: str) -> None:
    """Reconcile activated URIs with the card, in both directions.

    Shared by every path that connects from an ``A2AConfig`` so the
    conversational client and the ``tasks`` / ``push`` helpers agree.
    """
    advertised = {ext.uri for ext in card.capabilities.extensions}
    unknown = [uri for uri in uris if uri not in advertised]
    if unknown:
        raise A2AExtensionNotSupportedError(
            url=url,
            uris=unknown,
            reason="not advertised by the server card",
        )
    unmet = [
        ext.uri
        for ext in card.capabilities.extensions
        if ext.required and ext.uri not in uris and ext.uri not in NATIVE_EXTENSION_URIS
    ]
    if unmet:
        raise A2AExtensionNotSupportedError(
            url=url,
            uris=unmet,
            reason="required by the server card; add these URIs to A2AConfig.extensions",
        )


def extension_call_context(uris: Sequence[str]) -> ClientCallContext | None:
    """Build the call context activating ``uris`` on every RPC it's passed to.

    Carries the ``A2A-Extensions`` service parameter — an HTTP header on
    jsonrpc/rest, gRPC metadata on grpc. The other channel,
    ``Message.extensions``, is the mappers' job. ``None`` when nothing is
    activated, so callers can pass it to ``context=`` unconditionally.
    """
    if not uris:
        return None
    return ClientCallContext(
        service_parameters=ServiceParametersFactory.create([
            with_a2a_extensions(list(uris)),
        ]),
    )
