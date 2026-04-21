# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# Install Azure Cosmos DB SDK if not already

import json
import os
import pickle
import warnings
from typing import Any, Optional, TypedDict

# Pickle payloads stored in Cosmos DB can be replaced by a malicious actor with
# database write access, triggering RCE on read.  JSON is the safe default.
# Set AG2_ALLOW_PICKLE_CACHE_READ=1 only to migrate existing pickle-serialized
# caches; all new writes always use JSON regardless of this flag.
_PICKLE_CACHE_READ_ENABLED: bool = os.environ.get("AG2_ALLOW_PICKLE_CACHE_READ") == "1"

# Version prefix stored as the first byte of the serialized payload to
# distinguish JSON (b'\x01') from pickle (b'\x00') at read time.
_FORMAT_JSON: bytes = b"\x01"
_FORMAT_PICKLE: bytes = b"\x00"

from ..import_utils import optional_import_block, require_optional_import
from .abstract_cache_base import AbstractCache

with optional_import_block():
    from azure.cosmos import CosmosClient, PartitionKey
    from azure.cosmos.exceptions import CosmosResourceNotFoundError


@require_optional_import("azure", "cosmosdb")
class CosmosDBConfig(TypedDict, total=False):
    connection_string: str
    database_id: str
    container_id: str
    cache_seed: str | int | None
    client: Optional["CosmosClient"]


@require_optional_import("azure", "cosmosdb")
class CosmosDBCache(AbstractCache):
    """Synchronous implementation of AbstractCache using Azure Cosmos DB NoSQL API.

    This class provides a concrete implementation of the AbstractCache
    interface using Azure Cosmos DB for caching data, with synchronous operations.

    Attributes:
        seed (Union[str, int]): A seed or namespace used as a partition key.
        client (CosmosClient): The Cosmos DB client used for caching.
        container: The container instance used for caching.
    """

    def __init__(self, seed: str | int, cosmosdb_config: CosmosDBConfig):
        """Initialize the CosmosDBCache instance.

        Args:
            seed: A seed or namespace for the cache, used as a partition key.
            cosmosdb_config: The configuration for the Cosmos DB cache.
        """
        self.seed = str(seed)
        self.client = cosmosdb_config.get("client") or CosmosClient.from_connection_string(
            cosmosdb_config["connection_string"]
        )
        database_id = cosmosdb_config.get("database_id", "autogen_cache")
        self.database = self.client.get_database_client(database_id)
        container_id = cosmosdb_config.get("container_id")
        self.container = self.database.create_container_if_not_exists(
            id=container_id, partition_key=PartitionKey(path="/partitionKey")
        )

    @classmethod
    def create_cache(cls, seed: str | int, cosmosdb_config: CosmosDBConfig):
        """Factory method to create a CosmosDBCache instance based on the provided configuration.
        This method decides whether to use an existing CosmosClient or create a new one.
        """
        if "client" in cosmosdb_config and isinstance(cosmosdb_config["client"], CosmosClient):
            return cls.from_existing_client(seed, **cosmosdb_config)
        else:
            return cls.from_config(seed, cosmosdb_config)

    @classmethod
    def from_config(cls, seed: str | int, cosmosdb_config: CosmosDBConfig):
        return cls(str(seed), cosmosdb_config)

    @classmethod
    def from_connection_string(cls, seed: str | int, connection_string: str, database_id: str, container_id: str):
        config = {"connection_string": connection_string, "database_id": database_id, "container_id": container_id}
        return cls(str(seed), config)

    @classmethod
    def from_existing_client(cls, seed: str | int, client: "CosmosClient", database_id: str, container_id: str):
        config = {"client": client, "database_id": database_id, "container_id": container_id}
        return cls(str(seed), config)

    @staticmethod
    def _serialize(value: Any) -> bytes:
        """Serialize value to JSON bytes with a version prefix.

        All new writes use JSON (_FORMAT_JSON prefix).  Pickle is never written
        by this method; read-back pickle compat is handled in _deserialize.
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
        """Retrieve an item from the Cosmos DB cache.

        Args:
            key (str): The key identifying the item in the cache.
            default (optional): The default value to return if the key is not found.

        Returns:
            The deserialized value associated with the key if found, else the default value.
        """
        try:
            response = self.container.read_item(item=key, partition_key=str(self.seed))
            return self._deserialize(response["data"])
        except CosmosResourceNotFoundError:
            return default
        except Exception as e:
            raise e

    def set(self, key: str, value: Any) -> None:
        """Set an item in the Cosmos DB cache.

        Args:
            key (str): The key under which the item is to be stored.
            value: The value to be stored in the cache.

        Notes:
            Values are serialized using JSON with a version prefix.
        """
        try:
            serialized_value = self._serialize(value)
            item = {"id": key, "partitionKey": str(self.seed), "data": serialized_value}
            self.container.upsert_item(item)
        except Exception as e:
            raise e

    def close(self) -> None:
        """Close the Cosmos DB client.

        Perform any necessary cleanup, such as closing network connections.
        """
        # CosmosClient doesn"t require explicit close in the current SDK
        # If you created the client inside this class, you should close it if necessary
        pass

    def __enter__(self):
        """Context management entry.

        Returns:
            self: The instance itself.
        """
        return self

    def __exit__(self, exc_type: type | None, exc_value: Exception | None, traceback: Any | None) -> None:
        """Context management exit.

        Perform cleanup actions such as closing the Cosmos DB client.
        """
        self.close()
