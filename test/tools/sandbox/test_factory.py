# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

from ag2.tools.sandbox import SandboxFactory
from ag2.tools.sandbox.factory import SingletonFactory
from ag2.tools.sandbox.local import LocalSandbox


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
