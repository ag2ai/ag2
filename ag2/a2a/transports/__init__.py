# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal, TypeAlias

from ag2.exceptions import missing_additional_dependency

# Canonical transport identifiers used across server, client, card, and
# config modules. Centralised here (rather than re-typing the Literal
# in every module) so we have one source of truth.
TransportName: TypeAlias = Literal["jsonrpc", "rest", "grpc"]

# The fallback rebinds a name mypy has bound to a class, which it rejects; it
# sees only the real import. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .jsonrpc import build_jsonrpc_asgi
else:
    try:
        from .jsonrpc import build_jsonrpc_asgi
    except ImportError as e:
        build_jsonrpc_asgi = missing_additional_dependency("build_jsonrpc_asgi", "a2a-sdk[http-server]", e)

if TYPE_CHECKING:
    from .rest import build_rest_asgi
else:
    try:
        from .rest import build_rest_asgi
    except ImportError as e:
        build_rest_asgi = missing_additional_dependency("build_rest_asgi", "a2a-sdk[http-server]", e)

if TYPE_CHECKING:
    from .grpc import build_grpc_server, default_grpc_channel_factory
else:
    try:
        from .grpc import build_grpc_server, default_grpc_channel_factory
    except ImportError as e:
        build_grpc_server = missing_additional_dependency("build_grpc_server", "a2a-sdk[grpc]", e)
        default_grpc_channel_factory = missing_additional_dependency("default_grpc_channel_factory", "a2a-sdk[grpc]", e)

__all__ = (
    "TransportName",
    "build_grpc_server",
    "build_jsonrpc_asgi",
    "build_rest_asgi",
    "default_grpc_channel_factory",
)
