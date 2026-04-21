# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Security tests for autogen/cache/redis_cache.py (R3R2-B2).

Verifies that pickle payloads are rejected by default and that JSON
round-trips work correctly.  Redis is mocked throughout.
"""

import os
import pickle
import warnings
from unittest.mock import MagicMock, patch

import pytest

from autogen.cache.redis_cache import _FORMAT_JSON, _FORMAT_PICKLE, RedisCache


def _make_cache(seed: str = "test_seed") -> RedisCache:
    """Build a RedisCache with a fully mocked Redis client."""
    mock_redis = MagicMock()
    with patch("autogen.cache.redis_cache.redis") as mock_redis_module:
        mock_redis_module.Redis.from_url.return_value = mock_redis
        cache = RedisCache(seed, "redis://localhost:6379/0")
        cache.cache = mock_redis
    return cache


# ---------------------------------------------------------------------------
# R3R2-B2: pickle payload rejection
# ---------------------------------------------------------------------------


class TestRedisCachePickleRejection:
    def test_pickle_payload_rejected_by_default(self):
        """pickle.loads on Redis GET bytes MUST raise ValueError without env var."""
        import importlib

        import autogen.cache.redis_cache as mod

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "0"}):
            importlib.reload(mod)
            cache = _make_cache()
            malicious = _FORMAT_PICKLE + pickle.dumps({"secret": "data"})  # pragma: allowlist secret
            cache.cache.get.return_value = malicious
            with pytest.raises(ValueError, match="Refusing to deserialize pickle"):
                cache.get("key")

    def test_adversarial_pickle_rce_payload_blocked(self):
        """Attacker-crafted pickle payload MUST NOT execute code."""

        class _Exploit:
            def __reduce__(self):
                import os

                return (os.system, ("echo PWNED",))

        import importlib

        import autogen.cache.redis_cache as mod

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "0"}):
            importlib.reload(mod)
            cache = _make_cache()
            malicious = _FORMAT_PICKLE + pickle.dumps(_Exploit())
            cache.cache.get.return_value = malicious
            with pytest.raises(ValueError, match="Refusing to deserialize pickle"):
                cache.get("key")

    def test_json_round_trip_works(self):
        """set() must write JSON; get() must read it back correctly."""
        cache = _make_cache()
        original = {"messages": [{"role": "user", "content": "hi"}]}

        # Capture what was stored
        stored: list = []
        cache.cache.set.side_effect = lambda key, val: stored.append(val)

        cache.set("k1", original)
        assert len(stored) == 1
        raw = stored[0]
        assert raw[:1] == _FORMAT_JSON

        # Simulate reading it back
        cache.cache.get.return_value = raw
        result = cache.get("k1")
        assert result == original

    def test_new_writes_use_json_format(self):
        """set() MUST prefix payload with _FORMAT_JSON regardless of env vars."""
        import importlib

        import autogen.cache.redis_cache as mod

        for env_val in ("0", "1"):
            with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": env_val}):
                importlib.reload(mod)
                cache = _make_cache()
                stored: list = []
                cache.cache.set.side_effect = lambda key, val: stored.append(val)
                cache.set("k", "value")
                assert stored[0][:1] == _FORMAT_JSON

    def test_cache_miss_returns_default(self):
        """Missing key MUST return the specified default value."""
        cache = _make_cache()
        cache.cache.get.return_value = None
        assert cache.get("missing", default="sentinel") == "sentinel"

    def test_empty_payload_raises(self):
        """Empty bytes from Redis MUST raise, not return None."""
        cache = _make_cache()
        cache.cache.get.return_value = b""
        with pytest.raises(ValueError, match="Empty cache payload"):
            cache.get("k")

    def test_unknown_format_prefix_raises(self):
        """Unknown format prefix byte MUST raise ValueError."""
        cache = _make_cache()
        cache.cache.get.return_value = b"\x99{}"
        with pytest.raises(ValueError, match="Unknown cache payload format prefix"):
            cache.get("k")

    def test_pickle_read_allowed_with_env_var(self):
        """With AG2_ALLOW_PICKLE_CACHE_READ=1, legacy pickle payloads MAY be read."""
        import importlib

        import autogen.cache.redis_cache as mod

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "1"}):
            importlib.reload(mod)
            cache = _make_cache()
            legacy_data = {"legacy": True}
            raw = _FORMAT_PICKLE + pickle.dumps(legacy_data)
            cache.cache.get.return_value = raw
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = cache.get("k")
            assert result == legacy_data
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
