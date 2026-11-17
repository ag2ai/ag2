# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Everything one request carries from the renderer, decoded once for every transport."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .._types import A2UIVersion, JsonObject, JsonValue
from ..actions import A2UIEventAction
from ..capabilities import A2UIClientCapabilities, parse_client_capabilities
from ..data_model import parse_client_data_model
from ._decode import decode_envelopes
from ._prompts import incoming_prompts
from ._results import A2UIIncomingParseResult


@dataclass(frozen=True, slots=True)
class A2UIRendererInput:
    """What the renderer sent with one request: its interactions, capabilities and data model."""

    interactions: tuple[A2UIIncomingParseResult, ...]
    """Every decoded client→server envelope, in the order the request carried them."""
    prompts: tuple[tuple[str, ...], ...]
    """The agent prompts those interactions rewrite to, one tuple per envelope group passed in."""
    capabilities: A2UIClientCapabilities | None
    """``a2uiClientCapabilities`` (``a2uiRendererCapabilities`` in v1.0); ``None`` when the request has none."""
    data_model: dict[str, JsonObject] | None
    """``a2uiClientDataModel`` (``a2uiRendererDataModel`` in v1.0) surfaces; ``None`` when absent."""


def read_renderer_input(
    protocol_version: A2UIVersion,
    resolve_action: Callable[[str], A2UIEventAction | None],
    *,
    envelope_groups: Iterable[Iterable[JsonValue]],
    capability_sources: Iterable[Mapping[str, JsonValue] | None],
    data_model_source: Mapping[str, JsonValue] | None,
) -> A2UIRendererInput:
    """Decode what the renderer sent with one request.

    Args:
        protocol_version: The served protocol version.
        resolve_action: Looks up a registered server action by name; a click on one yields no prompt.
        envelope_groups: The request's client→server envelopes, grouped as the transport places their prompts (A2A:
            one group per DataPart, so each prompt takes its part's place; elsewhere a single group).
        capability_sources: Where the capabilities may sit, in order; the first that declares them wins.
        data_model_source: Where the data model sits.
    """
    interactions: list[A2UIIncomingParseResult] = []
    prompts: list[tuple[str, ...]] = []
    for envelopes in envelope_groups:
        decoded = decode_envelopes(envelopes, protocol_version=protocol_version)
        interactions.extend(decoded)
        prompts.append(tuple(incoming_prompts(decoded, resolve_action)))
    capabilities = next(
        (
            caps
            for source in capability_sources
            if source and (caps := parse_client_capabilities(source, version_key=protocol_version))
        ),
        None,
    )
    return A2UIRendererInput(
        interactions=tuple(interactions),
        prompts=tuple(prompts),
        capabilities=capabilities,
        data_model=parse_client_data_model(data_model_source, version_key=protocol_version),
    )
