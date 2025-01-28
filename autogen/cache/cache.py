# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from types import TracebackType
from typing import Any

from ..doc_utils import export_module
from .abstract_cache_base import AbstractCache
from .cache_factory import CacheFactory


@export_module("autogen")
class Cache(AbstractCache):
    """A wrapper class for managing cache configuration and instances.

    This class provides a unified interface for creating and interacting with
    different types of cache (e.g., Redis, Disk). It abstracts the underlying
    cache implementation details, providing methods for cache operations.

    Attributes:
        config (Dict[str, Any]): A dictionary containing cache configuration.
        cache: The cache instance created based on the provided configuration.
    """

    ALLOWED_CONFIG_KEYS = [
        "cache_seed",
        "redis_url",
        "cache_path_root",
        "cosmos_db_config",
    ]

    @staticmethod
    def redis(cache_seed: str | int = 42, redis_url: str = "redis://localhost:6379/0") -> Cache:
        """Create a Redis cache instance.

        Args:
            cache_seed (Union[str, int], optional): A seed for the cache. Defaults to 42.
            redis_url (str, optional): The URL for the Redis server. Defaults to "redis://localhost:6379/0".

        Returns:
            Cache: A Cache instance configured for Redis.
        """
        return Cache({"cache_seed": cache_seed, "redis_url": redis_url})

    @staticmethod
    def disk(cache_seed: str | int = 42, cache_path_root: str = ".cache") -> Cache:
        """Create a Disk cache instance.

        Args:
            cache_seed (Union[str, int], optional): A seed for the cache. Defaults to 42.
            cache_path_root (str, optional): The root path for the disk cache. Defaults to ".cache".

        Returns:
            Cache: A Cache instance configured for Disk caching.
        """
        return Cache({"cache_seed": cache_seed, "cache_path_root": cache_path_root})

    @staticmethod
    def cosmos_db(
        connection_string: str | None = None,
        container_id: str | None = None,
        cache_seed: str | int = 42,
        client: any | None = None,
    ) -> Cache:
        """Create a Cosmos DB cache instance with 'autogen_cache' as database ID.

        Args:
            connection_string (str, optional): Connection string to the Cosmos DB account.
            container_id (str, optional): The container ID for the Cosmos DB account.
            cache_seed (Union[str, int], optional): A seed for the cache.
            client: Optional[CosmosClient]: Pass an existing Cosmos DB client.

        Returns:
            Cache: A Cache instance configured for Cosmos DB.
        """
        cosmos_db_config = {
            "connection_string": connection_string,
            "database_id": "autogen_cache",
            "container_id": container_id,
            "client": client,
        }
        return Cache({"cache_seed": str(cache_seed), "cosmos_db_config": cosmos_db_config})

    def __init__(self, config: dict[str, Any]):
        """Initialize the Cache with the given configuration.

        Validates the configuration keys and creates the cache instance.

        Args:
            config (Dict[str, Any]): A dictionary containing the cache configuration.

        Raises:
            ValueError: If an invalid configuration key is provided.
        """
        self.config = config
        # Ensure that the seed is always treated as a string before being passed to any cache factory or stored.
        self.config["cache_seed"] = str(self.config.get("cache_seed", 42))

        # validate config
        for key in self.config:
            if key not in self.ALLOWED_CONFIG_KEYS:
                raise ValueError(f"Invalid config key: {key}")
        # create cache instance
        self.cache = CacheFactory.cache_factory(
            seed=self.config["cache_seed"],
            redis_url=self.config.get("redis_url"),
            cache_path_root=self.config.get("cache_path_root"),
            cosmosdb_config=self.config.get("cosmos_db_config"),
        )

    def __enter__(self) -> Cache:
        """Enter the runtime context related to the cache object.

        Returns:
            The cache instance for use within a context block.
        """
        return self.cache.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the runtime context related to the cache object.

        Cleans up the cache instance and handles any exceptions that occurred
        within the context.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_value: The exception value if an exception was raised in the context.
            traceback: The traceback if an exception was raised in the context.
        """
        return self.cache.__exit__(exc_type, exc_value, traceback)

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """Retrieve an item from the cache.

        Args:
            key (str): The key identifying the item in the cache.
            default (optional): The default value to return if the key is not found.
                                Defaults to None.

        Returns:
            The value associated with the key if found, else the default value.
        """
        return self.cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set an item in the cache.

        Args:
            key (str): The key under which the item is to be stored.
            value: The value to be stored in the cache.
        """
        self.cache.set(key, value)

    def close(self) -> None:
        """Close the cache.

        Perform any necessary cleanup, such as closing connections or releasing resources.
        """
        self.cache.close()
