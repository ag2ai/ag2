# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, Self

from .llms import LLMClient


class ModelConfig(Protocol):
    def copy(self) -> Self: ...

    def create(self) -> LLMClient: ...
