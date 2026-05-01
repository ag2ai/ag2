# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .asgi import build_asgi
except ImportError as e:
    build_asgi = missing_optional_dependency("build_asgi", "a2a-http", e)  # type: ignore[misc]

try:
    from .rest import build_rest
except ImportError as e:
    build_rest = missing_optional_dependency("build_rest", "a2a-http", e)  # type: ignore[misc]

try:
    from .grpc import build_grpc, make_servicer
except ImportError as e:
    build_grpc = missing_optional_dependency("build_grpc", "a2a-grpc", e)  # type: ignore[misc]
    make_servicer = missing_optional_dependency("make_servicer", "a2a-grpc", e)  # type: ignore[misc]

__all__ = (
    "build_asgi",
    "build_grpc",
    "build_rest",
    "make_servicer",
)
