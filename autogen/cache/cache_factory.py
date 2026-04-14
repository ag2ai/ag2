# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import logging
import os
from typing import Any

from ..import_utils import optional_import_block
from .abstract_cache_base import AbstractCache
from .in_memory_cache import InMemoryCache


class CacheFactory:
    @staticmethod
    def cache_factory(
        seed: str | int,
        redis_url: str | None = None,
        cache_path_root: str = ".cache",
        cosmosdb_config: dict[str, Any] | None = None,
    ) -> AbstractCache:
        """Factory function for creating cache instances.

        This function decides whether to create a RedisCache, DiskCache, or CosmosDBCache instance
        based on the provided parameters. If RedisCache is available and a redis_url is provided,
        a RedisCache instance is created. If connection_string, database_id, and container_id
        are provided, a CosmosDBCache is created. Otherwise, a DiskCache instance is used if available,
        or InMemoryCache as a final fallback.

        Args:
            seed (Union[str, int]): Used as a seed or namespace for the cache.
            redis_url (Optional[str]): URL for the Redis server.
            cache_path_root (str): Root path for the disk cache.
            cosmosdb_config (Optional[Dict[str, str]]): Dictionary containing 'connection_string',
                                                       'database_id', and 'container_id' for Cosmos DB cache.

        Returns:
            An instance of RedisCache, DiskCache, CosmosDBCache, or InMemoryCache.

        Examples:
        Creating a Redis cache

        ```python
        redis_cache = cache_factory("myseed", "redis://localhost:6379/0")
        ```
        Creating a Disk cache (requires 'ag2[diskcache]')

        ```python
        disk_cache = cache_factory("myseed", None)
        ```

        Creating a Cosmos DB cache:
        ```python
        cosmos_cache = cache_factory(
            "myseed",
            cosmosdb_config={
                "connection_string": "your_connection_string",
                "database_id": "your_database_id",
                "container_id": "your_container_id",
            },
        )
        ```

        """
        if redis_url:
            with optional_import_block() as result:
                from .redis_cache import RedisCache

            if result.is_successful:
                return RedisCache(seed, redis_url)
            else:
                logging.warning(
                    "RedisCache is not available. Checking other cache options. The last fallback is InMemoryCache."
                )

        if cosmosdb_config:
            with optional_import_block() as result:
                from .cosmos_db_cache import CosmosDBCache

            if result.is_successful:
                return CosmosDBCache.create_cache(seed, cosmosdb_config)
            else:
                logging.warning("CosmosDBCache is not available. Checking other cache options.")

        # Try DiskCache if available
        with optional_import_block() as result:
            from .disk_cache import DiskCache

        if result.is_successful:
            path = os.path.join(cache_path_root, str(seed))
            try:
                return DiskCache(os.path.join(".", path))
            except ImportError:
                logging.warning(
                    "DiskCache requires 'diskcache' package. Install with: pip install 'ag2[diskcache]'. "
                    "Note: diskcache has a critical security vulnerability (CVE-2025-69872). "
                    "Falling back to InMemoryCache."
                )

        # Final fallback to InMemoryCache
        logging.info("Using InMemoryCache as the default cache backend.")
        return InMemoryCache(seed)
