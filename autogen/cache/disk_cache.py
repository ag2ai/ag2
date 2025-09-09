# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from types import TracebackType
from typing import Any

import diskcache
from typing_extensions import Self

from .abstract_cache_base import AbstractCache


class DiskCache(AbstractCache):
    """Implementation of AbstractCache using the DiskCache library.

    This class provides a concrete implementation of the AbstractCache
    interface using the diskcache library for caching data on disk.

    Attributes:
        cache (diskcache.Cache): The DiskCache instance used for caching.

    Methods:
        __init__(self, seed): Initializes the DiskCache with the given seed.
        get(self, key, default=None): Retrieves an item from the cache.
        set(self, key, value): Sets an item in the cache.
        close(self): Closes the cache.
        __enter__(self): Context management entry.
        __exit__(self, exc_type, exc_value, traceback): Context management exit.
    """

    def __init__(self, seed: str | int):
        """Initialize the DiskCache instance.

        Args:
            seed (Union[str, int]): A seed or namespace for the cache. This is used to create
                        a unique storage location for the cache data.

        """
        self.cache = diskcache.Cache(seed)

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

        Perform any necessary cleanup, such as closing file handles or
        releasing resources.
        """
        self.cache.close()

    def __enter__(self) -> Self:
        """Enter the runtime context related to the object.

        Returns:
            self: The instance itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the runtime context related to the object.

        Perform cleanup actions such as closing the cache.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_value: The exception value if an exception was raised in the context.
            traceback: The traceback if an exception was raised in the context.
        """
        self.close()
