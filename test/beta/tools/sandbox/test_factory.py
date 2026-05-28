# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from autogen.beta.tools.sandbox import LocalSandbox, SandboxFactory, SingletonFactory


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
            await sb.exec(["/bin/echo", "hi"])
        result = await sandbox.exec(["/bin/echo", "still-alive"])
        assert result.exit_code == 0


def test_singleton_factory_satisfies_protocol() -> None:
    sandbox = LocalSandbox()
    factory: SandboxFactory = SingletonFactory(sandbox)
    assert isinstance(factory, SandboxFactory)
