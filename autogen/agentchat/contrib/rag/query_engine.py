# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Optional, Protocol, Union, runtime_checkable

from ....doc_utils import export_module

__all__ = ["RAGQueryEngine"]


@export_module("autogen.agentchat.contrib.rag")
@runtime_checkable
class RAGQueryEngine(Protocol):
    """A protocol class that represents a document ingestation and query engine on top of an underlying database.

    This interface defines the basic methods for RAG.
    """

    def add_docs(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths: Optional[list[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Add new documents to the underlying data store."""
        ...

    def query(self, question: str, *args: Any, **kwargs: Any) -> str:
        """Transform a string format question into database query and return the result.

        Args:
            question: a string format question
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments
        """
        ...
