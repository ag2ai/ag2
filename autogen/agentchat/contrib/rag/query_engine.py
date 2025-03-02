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

    def init_db(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths: Optional[list[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the database with the input documents or records.
        This method initializes database with the input documents or records.
        Usually, it takes the following steps,
        1. connecting to a database.
        2. insert records
        3. build indexes etc.
        Args:
            new_doc_dir: a dir of input documents that are used to create the records in database.
            new_doc_paths:
                a list of input documents that are used to create the records in database.
                a document can be a path to a file or a url.
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        ...

    def add_docs(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths: Optional[list[Union[Path, str]]] = None,
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
