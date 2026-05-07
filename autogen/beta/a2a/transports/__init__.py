# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .jsonrpc import build_jsonrpc_asgi
except ImportError as e:
    build_jsonrpc_asgi = missing_optional_dependency("build_jsonrpc_asgi", "a2a-http", e)  # type: ignore[misc]

try:
    from .rest import build_rest_asgi
except ImportError as e:
    build_rest_asgi = missing_optional_dependency("build_rest_asgi", "a2a-http", e)  # type: ignore[misc]

try:
    from .grpc import build_grpc_server, default_grpc_channel_factory
except ImportError as e:
    build_grpc_server = missing_optional_dependency("build_grpc_server", "a2a-grpc", e)  # type: ignore[misc]
    default_grpc_channel_factory = missing_optional_dependency("default_grpc_channel_factory", "a2a-grpc", e)  # type: ignore[misc]

__all__ = (
    "build_grpc_server",
    "build_jsonrpc_asgi",
    "build_rest_asgi",
    "default_grpc_channel_factory",
)
