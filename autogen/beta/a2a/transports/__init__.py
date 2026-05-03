# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .asgi import build_asgi_factory
except ImportError as e:
    build_asgi_factory = missing_optional_dependency("build_asgi_factory", "a2a-http", e)  # type: ignore[misc]

try:
    from .rest import build_rest_factory
except ImportError as e:
    build_rest_factory = missing_optional_dependency("build_rest_factory", "a2a-http", e)  # type: ignore[misc]

try:
    from .grpc import build_grpc_factory, make_servicer
except ImportError as e:
    build_grpc_factory = missing_optional_dependency("build_grpc_factory", "a2a-grpc", e)  # type: ignore[misc]
    make_servicer = missing_optional_dependency("make_servicer", "a2a-grpc", e)  # type: ignore[misc]

__all__ = (
    "build_asgi_factory",
    "build_grpc_factory",
    "build_rest_factory",
    "make_servicer",
)
