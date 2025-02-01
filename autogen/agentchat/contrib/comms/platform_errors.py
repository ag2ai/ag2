# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional


class PlatformError(Exception):
    """Base exception for all communication platform-related errors."""

    def __init__(
        self,
        message: str,
        platform_error: Optional[Any] = None,
        platform_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.platform_error = platform_error
        self.platform_name = platform_name
        self.context = context or {}
        self.message = message

        # Enhance error message with platform details if available
        full_message = f"[{platform_name or 'Unknown Platform'}] {message}"
        if platform_error:
            full_message += f"\nOriginal error: {str(platform_error)}"

        super().__init__(full_message)


class PlatformConnectionError(PlatformError):
    """Raised when there's an error connecting to the platform."""

    pass


class PlatformAuthenticationError(PlatformError):
    """Raised when there's an authentication or authorization error."""

    pass


class PlatformTimeoutError(PlatformError):
    """Raised when a platform operation times out."""

    pass


class PlatformRateLimitError(PlatformError):
    """Raised when platform rate limits are hit."""

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class PlatformMessageError(PlatformError):
    """Raised when there's an error sending/receiving messages."""

    def __init__(self, message: str, message_id: Optional[str] = None, **kwargs):
        self.message_id = message_id
        super().__init__(message, **kwargs)
