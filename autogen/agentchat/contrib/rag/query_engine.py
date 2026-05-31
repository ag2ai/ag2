# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ....doc_utils import export_module

__all__ = ["RAGQueryEngine"]


@export_module("autogen.agentchat.contrib.rag")
@runtime_checkable
class RAGQueryEngine(Protocol):
    """A protocol class that represents a document ingestation and query engine on top of an underlying database.

    This interface defines the basic methods for RAG.
    """

    def init_db(
        self,
        new_doc_dir: Path | str | None = None,
        new_doc_paths_or_urls: Sequence[Path | str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the database with the input documents or records.

        This method initializes database with the input documents or records.
        Usually, it takes the following steps:\n
        1. connecting to a database.\n
        2. insert records.\n
        3. build indexes etc.\n

        Args:\n
            new_doc_dir (Optional[Union[Path, str]]): A directory containing documents to be ingested.\n
            new_doc_paths_or_urls (Optional[Sequence[Union[Path, str]]]): A list of paths or URLs to documents to be ingested.\n
            *args: Any additional arguments\n
            **kwargs: Any additional keyword arguments\n
        Returns:\n
            bool: True if initialization is successful, False otherwise\n
        """
        ...

    def add_docs(
        self,
        new_doc_dir: Path | str | None = None,
        new_doc_paths_or_urls: Sequence[Path | str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add new documents to the underlying data store."""
        ...

    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
        """Connect to the database.

        Args:
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments
        Returns:
            bool: True if connection is successful, False otherwise
        """
        ...

    def query(self, question: str, *args: Any, **kwargs: Any) -> str:
        """Transform a string format question into database query and return the result.

        Args:
            question: a string format question
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments
        """
        ...
