# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

__all__ = ["Document", "DocumentType"]


class DocumentType(Enum):
    """Enum for supporting document type."""

    TEXT = auto()
    HTML = auto()
    PDF = auto()
    JSON = auto()


@dataclass
class Document:
    """A wrapper of graph store query results."""

    doctype: DocumentType
    data: Optional[Any] = None
    path_or_url: Optional[str] = field(default_factory=lambda: "")
