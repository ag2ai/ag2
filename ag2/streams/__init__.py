# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_optional_dependency

# The fallback rebinds a name mypy has bound to a class, which it rejects; it
# sees only the real import. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .redis import RedisStorage, RedisStream, Serializer
else:
    try:
        from .redis import RedisStorage, RedisStream, Serializer
    except ImportError as e:
        RedisStorage = missing_optional_dependency("RedisStorage", "redis", e)
        RedisStream = missing_optional_dependency("RedisStream", "redis", e)
        Serializer = missing_optional_dependency("Serializer", "redis", e)

__all__ = (
    "RedisStorage",
    "RedisStream",
    "Serializer",
)
