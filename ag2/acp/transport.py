# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Which transport reaches a remote ACP agent, and how a dropped one is reported.

Deliberately dependency-free: :mod:`ag2.acp.client` needs
:class:`ACPTransportError` on every turn, and it must not pull the HTTP/2 and
WebSocket stack that only :mod:`ag2.acp.remote` requires. The transports
themselves are constructed there.
"""

from typing import Literal, get_args
from urllib.parse import urlsplit

from ag2.exceptions import AG2Error

__all__ = ("ACPTransport", "ACPTransportError", "resolve_transport")

ACPTransport = Literal["http", "websocket"]

# The scheme is the fact a caller already has to state, so it picks the
# transport by default; requiring both invites the two to disagree.
_SCHEME_TRANSPORTS: dict[str, ACPTransport] = {
    "http": "http",
    "https": "http",
    "ws": "websocket",
    "wss": "websocket",
}


def resolve_transport(url: str, override: "ACPTransport | None" = None) -> ACPTransport:
    """The transport ``url`` reaches the agent over, unless ``override`` says otherwise.

    Args:
        url: The remote agent's endpoint.
        override: An explicit transport, for a proxy or gateway whose URL scheme
            does not follow the conventions above. Wins over the scheme.

    Raises:
        ValueError: when ``override`` is not a known transport, or when no
            override was given and the URL's scheme is not one AG2 can map.
    """
    if override is not None:
        if override not in get_args(ACPTransport):
            raise ValueError(f"unknown ACP transport {override!r}; expected one of {list(get_args(ACPTransport))}")
        return override
    scheme = urlsplit(url).scheme.lower()
    transport = _SCHEME_TRANSPORTS.get(scheme)
    if transport is None:
        raise ValueError(
            f"cannot infer the ACP transport from url {url!r}: expected an "
            f"{', '.join(sorted(_SCHEME_TRANSPORTS))} scheme, got {scheme or 'none'!r}. "
            "Pass transport='http' or transport='websocket' to say which one to use."
        )
    return transport


class ACPTransportError(AG2Error):
    """The ACP connection to the agent failed while a turn was in flight.

    Raised in place of letting the turn come back empty: a caller who receives a
    blank answer cannot tell a silent network drop from an agent that genuinely
    had nothing to say, and that ambiguity is what makes remote failures
    expensive to diagnose. AG2 does not reconnect or resume — a dropped
    connection fails the turn and the session.
    """

    def __init__(self, transport: str, cause: BaseException) -> None:
        super().__init__(
            f"the ACP {transport} connection to the agent failed during the turn "
            f"({type(cause).__name__}: {cause}); the turn has no answer"
        )
        self.transport = transport
