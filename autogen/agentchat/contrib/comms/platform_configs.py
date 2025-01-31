# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict


class BaseCommsPlatformConfig(BaseModel, ABC):
    """Base configuration for all communication platform configs."""

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the configuration settings.

        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        pass

    model_config = ConfigDict(extra="allow")


class ReplyMonitorConfig(BaseModel):
    """Configuration for handling platform replies."""

    timeout_minutes: int = Field(default=1, gt=0)
    """How long to wait for replies before timing out."""

    max_reply_messages: int = Field(default=1, gt=0)
    """Maximum number of messages to collect before returning."""

    model_config = ConfigDict(extra="allow")
