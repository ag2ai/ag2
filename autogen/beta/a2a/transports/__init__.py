# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def _missing(name: str, extra: str) -> Any:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            f"`autogen.beta.a2a.transports.{name}` requires the optional `{extra}` extra. "
            f"Install with: pip install ag2[{extra}]"
        )

    _raise.__name__ = f"build_{name}"
    return _raise


try:
    from .asgi import build_asgi
except ImportError:
    build_asgi = _missing("asgi", "a2a")

try:
    from .fastapi import build_fastapi
except ImportError:
    build_fastapi = _missing("fastapi", "a2a")

try:
    from .rest import build_rest
except ImportError:
    build_rest = _missing("rest", "a2a")

try:
    from .grpc import build_grpc, make_servicer
except ImportError:
    build_grpc = _missing("grpc", "a2a-grpc")
    make_servicer = _missing("grpc", "a2a-grpc")

__all__ = (
    "build_asgi",
    "build_fastapi",
    "build_grpc",
    "build_rest",
    "make_servicer",
)
