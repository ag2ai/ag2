# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AuthAdapter protocol, NoAuth, and AuthRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from autogen.beta.network.auth import AuthAdapter, AuthRegistry, NoAuth, default_registry
from autogen.beta.network.errors import AuthError
from autogen.beta.network.identity import ActorIdentity, AuthBlock


class _ApiKeyDummy:
    scheme = "api_key"

    def __init__(self, allowed: set[str]) -> None:
        self._allowed = allowed

    async def validate(self, identity: ActorIdentity, claim: dict[str, Any]) -> None:
        token = claim.get("token")
        if token not in self._allowed:
            raise AuthError("bad token")


def test_noauth_accepts_none_scheme() -> None:
    import asyncio

    ident = ActorIdentity(name="x", auth=AuthBlock(scheme="none"))
    asyncio.run(NoAuth().validate(ident, {}))


def test_noauth_rejects_non_none_scheme() -> None:
    import asyncio

    ident = ActorIdentity(name="x", auth=AuthBlock(scheme="jwt"))
    with pytest.raises(AuthError):
        asyncio.run(NoAuth().validate(ident, {}))


def test_registry_dispatches_to_matching_adapter() -> None:
    import asyncio

    registry = AuthRegistry([NoAuth(), _ApiKeyDummy(allowed={"secret"})])
    ok = ActorIdentity(name="a", auth=AuthBlock(scheme="api_key"))
    bad = ActorIdentity(name="a", auth=AuthBlock(scheme="api_key"))
    asyncio.run(registry.validate(ok, {"token": "secret"}))
    with pytest.raises(AuthError):
        asyncio.run(registry.validate(bad, {"token": "wrong"}))


def test_registry_unknown_scheme_raises_auth_error() -> None:
    import asyncio

    registry = AuthRegistry([NoAuth()])
    ident = ActorIdentity(name="x", auth=AuthBlock(scheme="mtls"))
    with pytest.raises(AuthError):
        asyncio.run(registry.validate(ident, {}))


def test_default_registry_ships_noauth() -> None:
    registry = default_registry()
    assert "none" in registry.schemes()


def test_auth_adapter_protocol_runtime_checkable() -> None:
    assert isinstance(NoAuth(), AuthAdapter)
