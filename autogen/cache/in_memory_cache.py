# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from types import TracebackType
from typing import Any

from typing_extensions import Self

from .abstract_cache_base import AbstractCache


class InMemoryCache(AbstractCache):
    def __init__(self, seed: str | int = ""):
        self._seed = str(seed)
        self._cache: dict[str, Any] = {}

    def _prefixed_key(self, key: str) -> str:
        separator = "_" if self._seed else ""
        return f"{self._seed}{separator}{key}"

    def get(self, key: str, default: Any | None = None) -> Any | None:
        result = self._cache.get(self._prefixed_key(key))
        if result is None:
            return default
        return result

    def set(self, key: str, value: Any) -> None:
        self._cache[self._prefixed_key(key)] = value

    def close(self) -> None:
        pass

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

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_val: The exception value if an exception was raised in the context.
            exc_tb: The traceback if an exception was raised in the context.
        """
        self.close()
