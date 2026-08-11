# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from typing import Any

import grpc
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers.grpc_handler import GrpcHandler
from a2a.server.tasks import (
    PushNotificationConfigStore,
    PushNotificationSender,
    TaskStore,
)
from a2a.types import AgentCard, a2a_pb2_grpc

from ._common import (
    CardSigner,
    ExtendedCardModifier,
    build_default_handler,
    prepare_public_card,
    sign_card,
    wrap_extended_card_modifier,
)

# ``grpc`` and the generated ``a2a_pb2_grpc`` ship no type information, so
# every ``grpc.aio.*`` annotation below degrades to ``Any`` and the generated
# servicer registration reads as untyped. The ignores are the unfollowed
# import, not a defect here — drop them once stubs are available.


_TLS_PREFIXES = ("grpcs://", "grpc+tls://")
_INSECURE_PREFIXES = ("grpc+insecure://", "grpc://")


def _strip_scheme(url: str, prefixes: Sequence[str]) -> str:
    for prefix in prefixes:
        if url.startswith(prefix):
            return url[len(prefix) :]
    return url


def default_grpc_channel_factory(url: str) -> grpc.aio.Channel:  # type: ignore[no-any-unimported]
    """Build a gRPC channel whose security is selected by the URL scheme.

    ``grpcs://`` and ``grpc+tls://`` use TLS with system CA roots. Existing
    ``grpc://``, ``grpc+insecure://``, and bare ``host:port`` targets remain
    insecure. Use :func:`secure_grpc_channel_factory` for private CAs or mTLS.
    """
    for prefix in _TLS_PREFIXES:
        if url.startswith(prefix):
            return grpc.aio.secure_channel(url[len(prefix) :], grpc.ssl_channel_credentials())
    return grpc.aio.insecure_channel(_strip_scheme(url, _INSECURE_PREFIXES))


def secure_grpc_channel_factory(  # type: ignore[no-any-unimported]
    credentials: grpc.ChannelCredentials | None = None,
    options: Sequence[tuple[str, Any]] = (),
) -> Callable[[str], grpc.aio.Channel]:
    """Create a TLS channel factory with optional custom credentials.

    When ``credentials`` is omitted, system CA roots are used. The returned
    factory accepts every supported gRPC URL spelling but always dials TLS.
    """
    resolved_credentials = credentials if credentials is not None else grpc.ssl_channel_credentials()

    def factory(url: str) -> grpc.aio.Channel:  # type: ignore[no-any-unimported]
        target = _strip_scheme(url, (*_TLS_PREFIXES, *_INSECURE_PREFIXES))
        return grpc.aio.secure_channel(
            target,
            resolved_credentials,
            options=list(options) if options else None,
        )

    return factory


def build_grpc_server(  # type: ignore[no-any-unimported]
    *,
    agent_executor: AgentExecutor,
    agent_card: AgentCard,
    bind: str,
    extended_agent_card: AgentCard | None = None,
    extended_card_modifier: ExtendedCardModifier | None = None,
    card_signer: CardSigner | None = None,
    task_store: TaskStore | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
    push_sender: PushNotificationSender | None = None,
    options: Sequence[tuple[str, Any]] = (),
    server_credentials: grpc.ServerCredentials | None = None,
) -> grpc.aio.Server:
    """``grpc.aio.Server`` exposing A2A service; caller starts/awaits it.

    The server binds insecurely by default for backwards compatibility. Pass
    ``server_credentials`` to bind the port with TLS instead.
    """
    agent_card = prepare_public_card(
        agent_card,
        extended=extended_agent_card is not None,
        push=push_config_store is not None,
        signer=card_signer,
    )
    if extended_agent_card is not None:
        extended_agent_card = sign_card(extended_agent_card, card_signer)
    extended_card_modifier = wrap_extended_card_modifier(extended_card_modifier, card_signer)
    handler = build_default_handler(
        agent_executor=agent_executor,
        agent_card=agent_card,
        extended_agent_card=extended_agent_card,
        extended_card_modifier=extended_card_modifier,
        task_store=task_store,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )
    server = grpc.aio.server(options=list(options) if options else None)
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(GrpcHandler(handler), server)  # type: ignore[no-untyped-call]
    if server_credentials is not None:
        server.add_secure_port(bind, server_credentials)
    else:
        server.add_insecure_port(bind)
    return server
