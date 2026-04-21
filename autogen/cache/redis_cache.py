# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import json
import os
import pickle
import warnings
from types import TracebackType
from typing import Any

# Pickle payloads retrieved from Redis allow RCE for any actor with Redis write
# access (the default production cache path).  JSON is the safe default.
# Set AG2_ALLOW_PICKLE_CACHE_READ=1 only to migrate existing caches; new
# writes always use JSON regardless of this flag.
_PICKLE_CACHE_READ_ENABLED: bool = os.environ.get("AG2_ALLOW_PICKLE_CACHE_READ") == "1"

# Version prefix byte to distinguish formats at read time.
_FORMAT_JSON: bytes = b"\x01"
_FORMAT_PICKLE: bytes = b"\x00"

from typing_extensions import Self

from ..import_utils import optional_import_block, require_optional_import
from .abstract_cache_base import AbstractCache

with optional_import_block():
    import redis


@require_optional_import("redis", "redis")
class RedisCache(AbstractCache):
    """Implementation of AbstractCache using the Redis database.

    This class provides a concrete implementation of the AbstractCache
    interface using the Redis database for caching data.

    Attributes:
        seed (Union[str, int]): A seed or namespace used as a prefix for cache keys.
        cache (redis.Redis): The Redis client used for caching.

    Methods:
        __init__(self, seed, redis_url): Initializes the RedisCache with the given seed and Redis URL.
        _prefixed_key(self, key): Internal method to get a namespaced cache key.
        get(self, key, default=None): Retrieves an item from the cache.
        set(self, key, value): Sets an item in the cache.
        close(self): Closes the Redis client.
        __enter__(self): Context management entry.
        __exit__(self, exc_type, exc_value, traceback): Context management exit.
    """

    def __init__(self, seed: str | int, redis_url: str):
        """Initialize the RedisCache instance.

        Args:
            seed (Union[str, int]): A seed or namespace for the cache. This is used as a prefix for all cache keys.
            redis_url (str): The URL for the Redis server.

        """
        self.seed = seed
        self.cache = redis.Redis.from_url(redis_url)

    def _prefixed_key(self, key: str) -> str:
        """Get a namespaced key for the cache.

        Args:
            key (str): The original key.

        Returns:
            str: The namespaced key.
        """
        return f"autogen:{self.seed}:{key}"

    @staticmethod
    def _serialize(value: Any) -> bytes:
        """Serialize value to JSON bytes with a version prefix.

        All new writes use JSON (_FORMAT_JSON prefix).
        """
        payload = json.dumps(value).encode()
        return _FORMAT_JSON + payload

    @staticmethod
    def _deserialize(raw: bytes) -> Any:
        """Deserialize a versioned payload.

        Format:
          b'\\x01' + json-bytes  -- JSON (current)
          b'\\x00' + pickle-bytes -- legacy pickle (read-only, behind env var)
        """
        if not raw:
            raise ValueError("Empty cache payload")
        prefix, body = raw[:1], raw[1:]
        if prefix == _FORMAT_JSON:
            return json.loads(body)
        if prefix == _FORMAT_PICKLE:
            # Legacy pickle path: only allowed with explicit opt-in env var.
            if not _PICKLE_CACHE_READ_ENABLED:
                raise ValueError(
                    "Refusing to deserialize pickle cache payload. "
                    "Set AG2_ALLOW_PICKLE_CACHE_READ=1 to read legacy caches. "
                    "See docs/cache-migration.md"
                )
            warnings.warn(
                "Reading a legacy pickle-serialized cache entry. Re-write the entry (cache.set) to migrate to JSON.",
                DeprecationWarning,
                stacklevel=4,
            )
            return pickle.loads(body)  # noqa: S301
        raise ValueError(f"Unknown cache payload format prefix: {prefix!r}")

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """Retrieve an item from the Redis cache.

        Args:
            key (str): The key identifying the item in the cache.
            default (optional): The default value to return if the key is not found.
                                Defaults to None.

        Returns:
            The deserialized value associated with the key if found, else the default value.
        """
        result = self.cache.get(self._prefixed_key(key))
        if result is None:
            return default
        return self._deserialize(result)

    def set(self, key: str, value: Any) -> None:
        """Set an item in the Redis cache.

        Args:
            key (str): The key under which the item is to be stored.
            value: The value to be stored in the cache.

        Notes:
            Values are serialized using JSON with a version prefix.
        """
        serialized_value = self._serialize(value)
        self.cache.set(self._prefixed_key(key), serialized_value)

    def close(self) -> None:
        """Close the Redis client.

        Perform any necessary cleanup, such as closing network connections.
        """
        self.cache.close()

    def __enter__(self) -> Self:
        """Enter the runtime context related to the object.

        Returns:
            self: The instance itself.
        """
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """Exit the runtime context related to the object.

        Perform cleanup actions such as closing the Redis client.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_val: The exception value if an exception was raised in the context.
            exc_tb: The traceback if an exception was raised in the context.
        """
        self.close()
