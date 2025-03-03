# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import unittest
from unittest.mock import MagicMock, patch

import pytest

from autogen.cache.disk_cache import DiskCache


class TestDiskCache:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.seed = "test_seed"

    @patch("autogen.cache.disk_cache.diskcache.Cache", return_value=MagicMock())
    def test_init(self, mock_cache):
        cache = DiskCache(self.seed)
        assert isinstance(cache.cache, MagicMock)
        mock_cache.assert_called_with(self.seed)

    @patch("autogen.cache.disk_cache.diskcache.Cache", return_value=MagicMock())
    def test_get(self, mock_cache):
        key = "key"
        value = "value"
        cache = DiskCache(self.seed)
        cache.cache.get.return_value = value
        assert cache.get(key) == value
        cache.cache.get.assert_called_with(key, None)

        cache.cache.get.return_value = None
        assert cache.get(key, None) is None

    @patch("autogen.cache.disk_cache.diskcache.Cache", return_value=MagicMock())
    def test_set(self, mock_cache):
        key = "key"
        value = "value"
        cache = DiskCache(self.seed)
        cache.set(key, value)
        cache.cache.set.assert_called_with(key, value)

    @patch("autogen.cache.disk_cache.diskcache.Cache", return_value=MagicMock())
    def test_context_manager(self, mock_cache):
        with DiskCache(self.seed) as cache:
            assert isinstance(cache, DiskCache)
            mock_cache_instance = cache.cache
        mock_cache_instance.close.assert_called()

    @patch("autogen.cache.disk_cache.diskcache.Cache", return_value=MagicMock())
    def test_close(self, mock_cache):
        cache = DiskCache(self.seed)
        cache.close()
        cache.cache.close.assert_called()


if __name__ == "__main__":
    unittest.main()
