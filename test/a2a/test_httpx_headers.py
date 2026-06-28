# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from ag2.a2a.transports._http import make_httpx_client


def test_factory_client_headers_not_mutated() -> None:
    shared = httpx.AsyncClient(base_url="http://test")
    snapshot = dict(shared.headers)

    def factory() -> httpx.AsyncClient:
        return shared

    with pytest.warns(RuntimeWarning, match="headers"):
        returned = make_httpx_client(
            headers={"X-Tenant": "alpha"},
            timeout=10.0,
            factory=factory,
        )

    assert returned is shared
    assert dict(shared.headers) == snapshot


def test_headers_applied_when_no_factory() -> None:
    client = make_httpx_client(headers={"X-Tenant": "alpha"}, timeout=10.0, factory=None)
    assert client.headers.get("X-Tenant") == "alpha"
