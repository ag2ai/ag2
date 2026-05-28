# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Protocol

from typing_extensions import Self

from .client import LLMClient

if TYPE_CHECKING:
    from autogen.beta.files.protocol import FilesClient


class ModelConfig(Protocol):
    def copy(self) -> Self: ...

    def create(self) -> LLMClient: ...

    def create_files_client(self) -> "FilesClient":
        raise NotImplementedError(f"{type(self).__name__} does not support Files API.")
