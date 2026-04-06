# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock


def _missing_optional_dependency(name: str, extra: str, error: ImportError) -> Mock:
    def _raise_helpful_import_error(*args: object, **kwargs: object) -> None:
        raise ImportError(
            f'{name} requires optional dependencies. Install with `pip install "ag2[{extra}]"`'
        ) from error

    return Mock(side_effect=_raise_helpful_import_error)


try:
    from .redis import RedisStorage, RedisStream, Serializer
except ImportError as e:
    RedisStorage = _missing_optional_dependency("RedisStorage", "redis", e)
    RedisStream = _missing_optional_dependency("RedisStream", "redis", e)
    Serializer = _missing_optional_dependency("Serializer", "redis", e)

__all__ = (
    "RedisStorage",
    "RedisStream",
    "Serializer",
)
