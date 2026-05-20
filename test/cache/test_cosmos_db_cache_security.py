# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Security tests for autogen/cache/cosmos_db_cache.py (R3R2-B1).

Verifies that pickle payloads are rejected by default and that JSON
round-trips work correctly.  Azure SDK is mocked throughout.
"""

import os
import pickle
from unittest.mock import MagicMock, patch

import pytest

from autogen.cache.cosmos_db_cache import (
    _FORMAT_JSON,
    _FORMAT_PICKLE,
    CosmosDBCache,
)


def _make_cache(seed: str = "test_seed") -> CosmosDBCache:
    """Build a CosmosDBCache skipping __init__ so tests run without azure-cosmos installed."""
    cache = CosmosDBCache.__new__(CosmosDBCache)
    cache.seed = seed
    cache.container = MagicMock()
    return cache


# ---------------------------------------------------------------------------
# R3R2-B1: pickle payload rejection
# ---------------------------------------------------------------------------


@pytest.mark.cosmosdb
class TestCosmosDBCachePickleRejection:
    def test_pickle_payload_rejected_by_default(self):
        """pickle.loads on stored data MUST raise ValueError without env var."""
        cache = _make_cache()
        malicious = _FORMAT_PICKLE + pickle.dumps({"secret": "data"})  # pragma: allowlist secret
        cache.container.read_item.return_value = {"data": malicious}

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "0"}):
            # Reload the module so the constant picks up the patched env
            import importlib

            import autogen.cache.cosmos_db_cache as mod

            importlib.reload(mod)
            cache_reloaded = _make_cache()
            cache_reloaded.container.read_item.return_value = {"data": malicious}
            with pytest.raises(ValueError, match="Refusing to deserialize pickle"):
                cache_reloaded.get("key")

    def test_adversarial_pickle_rce_payload_blocked(self):
        """Attacker-crafted pickle payload MUST NOT execute code."""

        class _Exploit:
            def __reduce__(self):
                import os

                return (os.system, ("echo PWNED",))

        import importlib

        import autogen.cache.cosmos_db_cache as mod

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "0"}):
            importlib.reload(mod)
            cache = _make_cache()
            malicious = _FORMAT_PICKLE + pickle.dumps(_Exploit())
            cache.container.read_item.return_value = {"data": malicious}
            with pytest.raises(ValueError, match="Refusing to deserialize pickle"):
                cache.get("key")

    def test_json_round_trip_works(self):
        """set() must write JSON; get() must read it back correctly."""
        cache = _make_cache()
        original = {"messages": [{"role": "user", "content": "hi"}]}

        # Capture what was upserted
        upserted: list = []
        cache.container.upsert_item.side_effect = lambda item: upserted.append(item)

        cache.set("k1", original)
        assert len(upserted) == 1
        raw = upserted[0]["data"]
        assert raw[:1] == _FORMAT_JSON

        # Now simulate reading it back
        cache.container.read_item.return_value = {"data": raw}
        result = cache.get("k1")
        assert result == original

    def test_new_writes_use_json_format(self):
        """set() MUST prefix payload with _FORMAT_JSON regardless of env vars."""
        import importlib

        import autogen.cache.cosmos_db_cache as mod

        for env_val in ("0", "1"):
            with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": env_val}):
                importlib.reload(mod)
                cache = _make_cache()
                upserted: list = []
                cache.container.upsert_item.side_effect = lambda item: upserted.append(item)
                cache.set("k", "value")
                assert upserted[0]["data"][:1] == _FORMAT_JSON

    def test_empty_payload_raises(self):
        """Empty bytes from Cosmos MUST raise, not silently return None."""
        cache = _make_cache()
        cache.container.read_item.return_value = {"data": b""}
        with pytest.raises(ValueError, match="Empty cache payload"):
            cache.get("k")

    def test_unknown_format_prefix_raises(self):
        """Unknown format prefix MUST raise ValueError."""
        import importlib

        import autogen.cache.cosmos_db_cache as mod

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "0"}):
            importlib.reload(mod)
            cache = _make_cache()
            cache.container.read_item.return_value = {"data": b"\x99{}"}
            with pytest.raises(ValueError, match="Unknown cache payload format prefix"):
                cache.get("k")

    def test_pickle_read_allowed_with_env_var(self):
        """With AG2_ALLOW_PICKLE_CACHE_READ=1, legacy pickle payloads MAY be read."""
        import importlib
        import warnings

        import autogen.cache.cosmos_db_cache as mod

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "1"}):
            importlib.reload(mod)
            cache = _make_cache()
            legacy_data = {"legacy": True}
            raw = _FORMAT_PICKLE + pickle.dumps(legacy_data)
            cache.container.read_item.return_value = {"data": raw}
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = cache.get("k")
            assert result == legacy_data
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_deserialize_unversioned_pickle_with_env_var(self):
        """Unversioned (no prefix) pickle bytes MUST be readable with AG2_ALLOW_PICKLE_CACHE_READ=1."""
        import importlib
        import warnings

        import autogen.cache.cosmos_db_cache as mod

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "1"}):
            importlib.reload(mod)
            cache = _make_cache()
            original = {"key": "value", "num": 42}
            raw = pickle.dumps(original)  # no prefix -- old code format
            cache.container.read_item.return_value = {"data": raw}
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = cache.get("k")
            assert result == original
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "unversioned" in str(dep_warnings[0].message).lower()

    def test_deserialize_unversioned_pickle_without_env_var(self):
        """Unversioned (no prefix) pickle bytes MUST raise ValueError without env var."""
        import importlib

        import autogen.cache.cosmos_db_cache as mod

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_CACHE_READ": "0"}):
            importlib.reload(mod)
            cache = _make_cache()
            raw = pickle.dumps({"key": "value"})  # no prefix -- old code format
            cache.container.read_item.return_value = {"data": raw}
            with pytest.raises(ValueError, match="Unknown cache payload format prefix"):
                cache.get("k")
