# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface for the hub.

Phase 3a ships the minimum HTTP endpoints an external client needs to
join the hub, discover actors, create sessions, and read session state.
See ``design/network_v3_redesign.md`` §9.1 for the full table — Phase 3a
covers seven routes (register / discover / describe / session create /
describe / close / WAL read). The remaining admin / activity / knowledge
endpoints ship in Phase 3b.

The module lazy-imports ``starlette`` so the network layer stays
install-optional. Callers that want the HTTP surface install it via
``pip install 'ag2[http]'`` (or the umbrella beta-network extra once
that lands).
"""

from __future__ import annotations

try:
    from .server import HttpServer, build_app
except ImportError:  # pragma: no cover — optional dep fallback
    HttpServer = None  # type: ignore[assignment, misc]
    build_app = None  # type: ignore[assignment, misc]


__all__ = ("HttpServer", "build_app")
