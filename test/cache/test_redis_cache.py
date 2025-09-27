# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest

# Create a persistent mock for redis module
redis_mock = MagicMock()
redis_mock.Redis.from_url = MagicMock(return_value=MagicMock())

# Mock redis module before importing RedisCache and keep it persistent
sys.modules["redis"] = redis_mock

from autogen.cache.redis_cache import RedisCache


@pytest.mark.redis
class TestRedisCache:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.seed = "test_seed"
        self.redis_url = "redis://localhost:6379/0"

    def test_init(self):
        with patch.object(redis_mock.Redis, "from_url", return_value=MagicMock()) as mock_from_url:
            cache = RedisCache(self.seed, self.redis_url)
            assert cache.seed == self.seed
            mock_from_url.assert_called_with(self.redis_url)

    def test_prefixed_key(self):
        with patch.object(redis_mock.Redis, "from_url", return_value=MagicMock()):
            cache = RedisCache(self.seed, self.redis_url)
            key = "test_key"
            expected_prefixed_key = f"autogen:{self.seed}:{key}"
            assert cache._prefixed_key(key) == expected_prefixed_key

    def test_get(self):
        mock_redis_client = MagicMock()
        with patch.object(redis_mock.Redis, "from_url", return_value=mock_redis_client):
            key = "key"
            value = "value"
            serialized_value = pickle.dumps(value)
            cache = RedisCache(self.seed, self.redis_url)
            cache.cache.get.return_value = serialized_value
            assert cache.get(key) == value
            cache.cache.get.assert_called_with(f"autogen:{self.seed}:{key}")

            cache.cache.get.return_value = None
            assert cache.get(key) is None

    def test_set(self):
        mock_redis_client = MagicMock()
        with patch.object(redis_mock.Redis, "from_url", return_value=mock_redis_client):
            key = "key"
            value = "value"
            serialized_value = pickle.dumps(value)
            cache = RedisCache(self.seed, self.redis_url)
            cache.set(key, value)
            cache.cache.set.assert_called_with(f"autogen:{self.seed}:{key}", serialized_value)

    def test_context_manager(self):
        mock_redis_client = MagicMock()
        with patch.object(redis_mock.Redis, "from_url", return_value=mock_redis_client):
            with RedisCache(self.seed, self.redis_url) as cache:
                assert isinstance(cache, RedisCache)
                mock_redis_instance = cache.cache
            mock_redis_instance.close.assert_called()


if __name__ == "__main__":
    unittest.main()
