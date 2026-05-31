# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

from unittest.mock import Mock

import pytest

from autogen.fast_depends import Depends, dependency_provider, inject


@pytest.fixture
def provider():
    yield dependency_provider
    dependency_provider.clear()


def test_not_override(provider):
    mock = Mock()

    def base_dep():  # pragma: no cover
        mock.original()
        return 1

    @inject(dependency_overrides_provider=None)
    def func(d=Depends(base_dep)):
        assert d == 1

    func()

    mock.original.assert_called_once()


def test_sync_override(provider):
    mock = Mock()

    def base_dep():  # pragma: no cover
        mock.original()
        return 1

    def override_dep():
        mock.override()
        return 2

    provider.override(base_dep, override_dep)

    @inject
    def func(d=Depends(base_dep)):
        assert d == 2

    func()

    mock.override.assert_called_once()
    assert not mock.original.called


def test_override_context(provider):
    def base_dep():
        return 1

    def override_dep():
        return 2

    @inject
    def func(d=Depends(base_dep)):
        return d

    with provider.scope(base_dep, override_dep):
        assert func() == 2

    assert func() == 1


def test_sync_by_async_override(provider):
    def base_dep():  # pragma: no cover
        return 1

    async def override_dep():  # pragma: no cover
        return 2

    provider.override(base_dep, override_dep)

    @inject
    def func(d=Depends(base_dep)):
        pass

    with pytest.raises(AssertionError):
        func()


@pytest.mark.anyio
async def test_async_override(provider):
    mock = Mock()

    async def base_dep():  # pragma: no cover
        mock.original()
        return 1

    async def override_dep():
        mock.override()
        return 2

    provider.override(base_dep, override_dep)

    @inject
    async def func(d=Depends(base_dep)):
        assert d == 2

    await func()

    mock.override.assert_called_once()
    assert not mock.original.called


@pytest.mark.anyio
async def test_async_by_sync_override(provider):
    mock = Mock()

    async def base_dep():  # pragma: no cover
        mock.original()
        return 1

    def override_dep():
        mock.override()
        return 2

    provider.override(base_dep, override_dep)

    @inject
    async def func(d=Depends(base_dep)):
        assert d == 2

    await func()

    mock.override.assert_called_once()
    assert not mock.original.called
