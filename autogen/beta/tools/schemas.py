# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(slots=True)
class ToolSchema:
    type: str


__all__ = ("ToolSchema",)
