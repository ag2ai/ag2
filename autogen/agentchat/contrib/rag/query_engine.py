# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class VectorDbQueryEngine(Protocol):
    """An abstract base class that represents aquery engine on top of a underlying vector database.

    This interface defines the basic methods for RAG.
    """

    def init_db(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths: Optional[list[Union[Path, str]]] = None,
        /,
        *args,
        **kwargs,
    ) -> bool:
        """This method initializes database with the input documents or records.
        Usually, it takes the following steps,
        1. connecting to a database.
        2. insert records
        3. build indexes etc.

        Args:
            new_doc_dir: a dir of input documents that are used to create the records in database.
            new_doc_paths:
                a list of input documents that are used to create the records in database.
                a document can be a path to a file or a url.

        Returns:
            bool: True if initialization is successful, False otherwise
        """
        ...

    def add_records(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[list[Union[Path, str]]] = None,
        /,
        *args,
        **kwargs,
    ) -> bool:
        """Add new documents to the underlying database and add to the index."""
        ...

    def connect_db(self, *args, **kwargs) -> bool:
        """This method connects to the database."""
        ...

    def query(self, question: str, /, *args, **kwargs) -> str:
        """This method transform a string format question into database query and return the result."""
        ...
