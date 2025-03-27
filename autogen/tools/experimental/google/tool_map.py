# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

__all__ = [
    "GoogleToolMapProtocol",
]


@runtime_checkable
class GoogleToolMapProtocol(Protocol):
    @classmethod
    def recommended_scopes(cls) -> list[str]:
        """Defines a required static method without implementation."""
        ...
