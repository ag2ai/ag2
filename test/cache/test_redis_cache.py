# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import pickle
import unittest
from unittest.mock import MagicMock, patch

import pytest

from autogen.cache.redis_cache import RedisCache


@pytest.mark.redis
class TestRedisCache:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.seed = "test_seed"
        self.redis_url = "redis://localhost:6379/0"

    @patch("autogen.cache.redis_cache.redis.Redis.from_url", return_value=MagicMock())
    def test_init(self, mock_redis_from_url):
        cache = RedisCache(self.seed, self.redis_url)
        assert cache.seed == self.seed
        mock_redis_from_url.assert_called_with(self.redis_url)

    @patch("autogen.cache.redis_cache.redis.Redis.from_url", return_value=MagicMock())
    def test_prefixed_key(self, mock_redis_from_url):
        cache = RedisCache(self.seed, self.redis_url)
        key = "test_key"
        expected_prefixed_key = f"autogen:{self.seed}:{key}"
        assert cache._prefixed_key(key) == expected_prefixed_key

    @patch("autogen.cache.redis_cache.redis.Redis.from_url", return_value=MagicMock())
    def test_get(self, mock_redis_from_url):
        key = "key"
        value = "value"
        serialized_value = pickle.dumps(value)
        cache = RedisCache(self.seed, self.redis_url)
        cache.cache.get.return_value = serialized_value
        assert cache.get(key) == value
        cache.cache.get.assert_called_with(f"autogen:{self.seed}:{key}")

        cache.cache.get.return_value = None
        assert cache.get(key) is None

    @patch("autogen.cache.redis_cache.redis.Redis.from_url", return_value=MagicMock())
    def test_set(self, mock_redis_from_url):
        key = "key"
        value = "value"
        serialized_value = pickle.dumps(value)
        cache = RedisCache(self.seed, self.redis_url)
        cache.set(key, value)
        cache.cache.set.assert_called_with(f"autogen:{self.seed}:{key}", serialized_value)

    @patch("autogen.cache.redis_cache.redis.Redis.from_url", return_value=MagicMock())
    def test_context_manager(self, mock_redis_from_url):
        with RedisCache(self.seed, self.redis_url) as cache:
            assert isinstance(cache, RedisCache)
            mock_redis_instance = cache.cache
        mock_redis_instance.close.assert_called()


if __name__ == "__main__":
    unittest.main()
