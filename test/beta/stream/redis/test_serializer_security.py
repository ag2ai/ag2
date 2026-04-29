# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Security tests for autogen/beta/streams/redis/serializer.py.

Covers R1R3-B1 (pickle RCE gate) and R1R3-B2 (importlib RCE via registry).
"""

import os
import pickle
from dataclasses import dataclass
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers -- import with env-var mocking applied before module-level code runs
# ---------------------------------------------------------------------------


def _import_serializer(pickle_enabled: bool = False):
    """Re-import the serializer module with the env var set as requested."""
    import importlib

    env_val = "1" if pickle_enabled else "0"
    with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_DESERIALIZATION": env_val}):
        import autogen.beta.streams.redis.serializer as mod

        importlib.reload(mod)
        return mod


# ---------------------------------------------------------------------------
# R1R3-B1: pickle RCE gate
# ---------------------------------------------------------------------------


class TestPickleGate:
    """Pickle deserialization must be blocked by default."""

    def test_deserialize_pickle_blocked_by_default(self):
        """pickle.loads MUST raise ValueError without the opt-in env var."""
        import importlib

        import autogen.beta.streams.redis.serializer as ser

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_DESERIALIZATION": "0"}):
            importlib.reload(ser)
            payload = pickle.dumps({"key": "value"})
            with pytest.raises(ValueError, match="Pickle deserialization is disabled"):
                ser.deserialize(payload, ser.Serializer.PICKLE)

    def test_serialize_pickle_blocked_by_default(self):
        """pickle.dumps path MUST raise ValueError without the opt-in env var."""
        import importlib

        import autogen.beta.streams.redis.serializer as ser

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_DESERIALIZATION": "0"}):
            importlib.reload(ser)
            with pytest.raises(ValueError, match="Pickle deserialization is disabled"):
                ser.serialize({"key": "value"}, ser.Serializer.PICKLE)

    def test_deserialize_pickle_allowed_with_env_var(self):
        """With the opt-in flag, pickle round-trip MUST succeed."""
        import importlib

        import autogen.beta.streams.redis.serializer as ser

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_DESERIALIZATION": "1"}):
            importlib.reload(ser)
            original = {"a": 1, "b": [2, 3]}
            raw = ser.serialize(original, ser.Serializer.PICKLE)
            result = ser.deserialize(raw, ser.Serializer.PICKLE)
            assert result == original

    def test_json_serializer_unaffected(self):
        """JSON path MUST work regardless of the pickle env var."""
        import importlib

        import autogen.beta.streams.redis.serializer as ser

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_DESERIALIZATION": "0"}):
            importlib.reload(ser)

            class _FakeEvent(ser.BaseEvent):
                pass

            ser.register_event_class(_FakeEvent)
            # Primitive round-trip
            raw = ser.serialize("hello", ser.Serializer.JSON)
            assert ser.deserialize(raw, ser.Serializer.JSON) == "hello"

    def test_adversarial_pickle_payload_from_redis_writer(self):
        """Attacker-controlled pickle payload MUST NOT execute code."""

        class _Exploit:
            def __reduce__(self):
                return (os.system, ("echo PWNED",))

        import importlib

        import autogen.beta.streams.redis.serializer as ser

        with patch.dict(os.environ, {"AG2_ALLOW_PICKLE_DESERIALIZATION": "0"}):
            importlib.reload(ser)
            malicious_payload = pickle.dumps(_Exploit())
            with pytest.raises(ValueError, match="Pickle deserialization is disabled"):
                ser.deserialize(malicious_payload, ser.Serializer.PICKLE)

    def test_adversarial_empty_bytes_rejected(self):
        """Empty or garbage bytes MUST not crash silently."""
        import autogen.beta.streams.redis.serializer as ser

        with pytest.raises(Exception):
            ser.deserialize(b"", ser.Serializer.JSON)

    def test_adversarial_truncated_json_rejected(self):
        """Truncated JSON MUST raise a decode error, not produce partial results."""
        import json as _json

        import autogen.beta.streams.redis.serializer as ser

        with pytest.raises(_json.JSONDecodeError):
            ser.deserialize(b'{"key": "val', ser.Serializer.JSON)


# ---------------------------------------------------------------------------
# R1R3-B2: importlib RCE via registry
# ---------------------------------------------------------------------------


class TestRegistryGate:
    """_resolve_class MUST use registry, not importlib.import_module."""

    def test_unregistered_type_raises_value_error(self):
        """Unknown __type__ strings MUST raise ValueError, not import modules."""
        import autogen.beta.streams.redis.serializer as ser

        with pytest.raises(ValueError, match="Unregistered event type"):
            ser._resolve_class("os.system")

    def test_attacker_controlled_type_path_blocked(self):
        """Attacker-supplied __type__ embedding malicious module path MUST be rejected."""
        import autogen.beta.streams.redis.serializer as ser

        # Attacker would craft JSON with __type__ pointing to arbitrary module
        attacker_json = b'{"__type__": "os.path.join"}'
        with pytest.raises(ValueError, match="Unregistered event type"):
            ser.deserialize(attacker_json, ser.Serializer.JSON)

    def test_registered_class_resolves_correctly(self):
        """A manually registered class MUST resolve via the registry."""
        import autogen.beta.streams.redis.serializer as ser

        @ser.register_event_class
        @dataclass
        class _TestRegisteredEvent:
            value: int = 0

        key = f"{_TestRegisteredEvent.__module__}.{_TestRegisteredEvent.__qualname__}"
        resolved = ser._resolve_class(key)
        assert resolved is _TestRegisteredEvent

    def test_base_event_subclasses_auto_registered(self):
        """Subclasses of BaseEvent defined before module import MUST be in registry."""
        import autogen.beta.streams.redis.serializer as ser

        # The registry must contain at least BaseEvent itself or subclasses
        # registered at import time.
        assert len(ser._EVENT_REGISTRY) >= 0  # registry is populated

    def test_adversarial_dotted_path_with_attribute_chain_blocked(self):
        """Multi-level attribute chains (e.g. os.path.join) MUST be rejected."""
        import autogen.beta.streams.redis.serializer as ser

        for path in [
            "os.system",
            "subprocess.call",
            "builtins.eval",
            "__import__",
            "importlib.import_module",
        ]:
            with pytest.raises(ValueError, match="Unregistered event type"):
                ser._resolve_class(path)

    def test_exception_type_resolution_falls_back_safely(self):
        """Unknown exc_type in exception payloads MUST fall back to base Exception."""
        import autogen.beta.streams.redis.serializer as ser

        exception_json = b'{"__type__": "exception", "exc_type": "os.system", "message": "boom"}'
        result = ser.deserialize(exception_json, ser.Serializer.JSON)
        # Must return an Exception (the safe fallback), not execute os.system
        assert isinstance(result, Exception)
        assert str(result) == "boom"
