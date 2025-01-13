# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any, Literal, Optional

from autogen.logger.base_logger import BaseLogger
from autogen.logger.file_logger import FileLogger
from autogen.logger.sqlite_logger import SqliteLogger

__all__ = ("LoggerFactory",)


class LoggerFactory:
    """Factory class to create logger objects."""

    @staticmethod
    def get_logger(
        logger_type: Literal["sqlite", "file"] = "sqlite", config: Optional[dict[str, Any]] = None
    ) -> BaseLogger:
        """Factory method to create logger objects.

        Args:
            logger_type (Literal["sqlite", "file"], optional): Type of logger. Defaults to "sqlite".
            config (Optional[dict[str, Any]], optional): Configuration for logger. Defaults to None.

        Returns:
            BaseLogger: Logger object
        """
        if config is None:
            config = {}

        if logger_type == "sqlite":
            return SqliteLogger(config)
        elif logger_type == "file":
            return FileLogger(config)
        else:
            raise ValueError(f"[logger_factory] Unknown logger type: {logger_type}")
