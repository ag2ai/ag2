# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_optional_dependency

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
