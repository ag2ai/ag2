# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

from autogen.beta.tools.sandbox import (
    CallableFactory,
    LocalSandbox,
    Sandbox,
    SandboxFactory,
    SingletonFactory,
)


@pytest.mark.asyncio
class TestSingletonFactory:
    async def test_open_yields_wrapped_sandbox(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        factory = SingletonFactory(sandbox)
        async with factory.open() as sb:
            assert sb is sandbox

    async def test_open_returns_same_instance_each_call(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        factory = SingletonFactory(sandbox)
        async with factory.open() as sb1:
            pass
        async with factory.open() as sb2:
            assert sb1 is sb2

    async def test_open_does_not_close_sandbox(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        factory = SingletonFactory(sandbox)
        async with factory.open() as sb:
            await sb.exec([sys.executable, "-c", "pass"])
        result = await sandbox.exec([sys.executable, "-c", "pass"])
        assert result.exit_code == 0


def test_singleton_factory_satisfies_protocol() -> None:
    sandbox = LocalSandbox()
    factory: SandboxFactory = SingletonFactory(sandbox)
    assert isinstance(factory, SandboxFactory)


def test_singleton_factory_exposes_sandbox(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path)
    factory = SingletonFactory(sandbox)
    assert factory.sandbox is sandbox


@pytest.mark.asyncio
class TestCallableFactory:
    async def test_nullary_builder(self, tmp_path: Path) -> None:
        factory = CallableFactory(lambda: LocalSandbox(tmp_path))
        async with factory.open() as sb:
            result = await sb.exec([sys.executable, "-c", "pass"])
        assert result.exit_code == 0

    async def test_context_aware_builder_receives_context(self, tmp_path: Path) -> None:
        seen: list[object] = []

        def build(ctx: object) -> Sandbox:
            seen.append(ctx)
            return LocalSandbox(tmp_path)

        factory = CallableFactory(build)
        async with factory.open(context=None) as sb:
            await sb.exec([sys.executable, "-c", "pass"])
        assert seen == [None]

    async def test_async_builder(self, tmp_path: Path) -> None:
        async def build() -> Sandbox:
            return LocalSandbox(tmp_path)

        factory = CallableFactory(build)
        async with factory.open() as sb:
            result = await sb.exec([sys.executable, "-c", "pass"])
        assert result.exit_code == 0

    async def test_satisfies_protocol(self, tmp_path: Path) -> None:
        factory: SandboxFactory = CallableFactory(lambda: LocalSandbox(tmp_path))
        assert isinstance(factory, SandboxFactory)
