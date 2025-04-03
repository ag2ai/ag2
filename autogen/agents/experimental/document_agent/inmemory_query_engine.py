# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from pydantic import BaseModel

from .... import ConversableAgent
from ....agentchat.contrib.rag import RAGQueryEngine
from ....doc_utils import export_module
from ....llm_config import LLMConfig

__all__ = ["InMemoryQueryEngine"]

# REPLIES
QUERY_NO_INGESTIONS_REPLY = "Sorry, please ingest some documents/URLs before querying."  # Default response for queries without ingested documents
EMPTY_RESPONSE_REPLY = "Sorry, I couldn't find any information on that. If you haven't ingested any documents, please try that."  # Default response for queries without results
ERROR_RESPONSE_REPLY = "Sorry, there was an error processing your query: "  # Default response for queries with errors
COULD_NOT_ANSWER_REPLY = "Sorry, I couldn't answer that question from the ingested documents/URLs"  # Default response for queries that could not be answered


# Documents and Content structure
class DocumentStore(BaseModel):
    ingestation_name: str
    content: str


# Answer question structure
class QueryAnswer(BaseModel):
    could_answer: bool
    answer: str


@export_module("autogen.agents.experimental")
class InMemoryQueryEngine:
    """
    This engine stores ingested documents in memory and then injects them into an internal agent's system message for answering queries.

    This implements the autogen.agentchat.contrib.rag.RAGQueryEngine protocol.
    """

    def __init__(
        self,
        llm_config: Union[LLMConfig, dict[str, Any]],
    ) -> None:
        # Deep copy the llm config to avoid changing the original
        structured_config = copy.deepcopy(llm_config)

        # The query agent will answer with a structured output
        structured_config["response_format"] = QueryAnswer

        # Our agents for querying
        self._query_agent = ConversableAgent(
            name="inmemory_query_agent",
            llm_config=structured_config,
        )

        # In-memory storage for ingested documents
        self._ingested_documents: list[DocumentStore] = []

    def query(self, question: str, *args: Any, **kwargs: Any) -> str:
        """Run a query against the ingested documents and return the answer."""

        # If no documents have been ingested, return an empty response
        if not self._ingested_documents:
            return QUERY_NO_INGESTIONS_REPLY

        # Put the context into the system message
        context_parts = []
        for i, doc in enumerate(self._ingested_documents, 1):
            context_parts.append(f"Ingested File/URL {i} - '{doc.ingestation_name}':\n{doc.content}\n")

        context = "\n".join(context_parts)

        system_message = (
            "You are a query agent tasked with answering questions based on ingested documents.\n\n"
            "AVAILABLE DOCUMENTS:\n"
            + "\n".join([f"- {doc.ingestation_name}" for doc in self._ingested_documents])
            + "\n\n"
            "When answering questions about these documents, use ONLY the information in the following context:\n\n"
            f"{context}\n\n"
            "IMPORTANT: The user will ask about these documents by name. When they do, provide helpful, detailed answers based on the document content above."
        )

        self._query_agent.update_system_message(system_message)

        message = f"Using ONLY the document content in your system message, answer this question: {question}"

        response = self._query_agent.run(
            message=message,
            max_turns=1,
        )

        response.process()

        try:
            # Get the structured output and return the answer
            answer_object = QueryAnswer.model_validate(json.loads(response.summary))  # type: ignore[arg-type]

            if answer_object.could_answer:
                return answer_object.answer
            else:
                if answer_object.answer:
                    return COULD_NOT_ANSWER_REPLY + ": " + answer_object.answer
                else:
                    return COULD_NOT_ANSWER_REPLY

        except Exception as e:
            # Error converting the response to the structured output
            return ERROR_RESPONSE_REPLY + str(e)

    def add_docs(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
    ) -> None:
        """
        Add additional documents to the in-memory store

        Loads new Docling-parsed Markdown files from a specified directory or a list of file paths
        and inserts them into the in-memory store.

        Args:
            new_doc_dir: The directory path from which to load additional documents.
                If provided, all eligible files in this directory are loaded.
            new_doc_paths_or_urls: A list of file paths specifying additional documents to load.
                Each file should be a Docling-parsed Markdown file.
        """
        new_doc_dir = new_doc_dir or ""
        new_doc_paths = new_doc_paths_or_urls or []
        self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths)

    def _load_doc(
        self, input_dir: Optional[Union[Path, str]], input_docs: Optional[Sequence[Union[Path, str]]]
    ) -> None:
        """
        Load documents from a directory and/or a list of file paths into the in-memory store.

        This helper method reads files using native Python file operations and stores them
        in the in-memory document store. It supports reading text-based files, with the primary
        intended use being for documents processed by Docling.

        Args:
            input_dir (Optional[Union[Path, str]]): The directory containing documents to be loaded.
                If provided, all files in the directory will be considered.
            input_docs (Optional[list[Union[Path, str]]]): A list of individual file paths to load.
                Each path must point to an existing file.

        Raises:
            ValueError: If the specified directory does not exist.
            ValueError: If any provided file path does not exist.
            ValueError: If neither input_dir nor input_docs is provided.
        """
        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        # Process directory if provided
        if input_dir:
            # logger.info(f"Loading docs from directory: {input_dir}")
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")

            # Get all files from the directory
            dir_path = Path(input_dir)
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    self._read_and_store_file(file_path)

        # Process individual files if provided
        if input_docs:
            for doc_path in input_docs:
                # logger.info(f"Loading input doc: {doc_path}")
                if not os.path.exists(doc_path):
                    raise ValueError(f"Document file not found: {doc_path}")
                self._read_and_store_file(doc_path)

    def _read_and_store_file(self, file_path: Union[Path, str]) -> None:
        """
        Read a file and store its content in the in-memory document store.

        Args:
            file_path (Union[Path, str]): Path to the file to be read
        """
        file_path = Path(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Store the document in the in-memory store
            document = DocumentStore(ingestation_name=file_path.name, content=content)
            self._ingested_documents.append(document)
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {str(e)}")

    def init_db(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Not required nor implemented for InMemoryQueryEngine"""
        raise NotImplementedError("Method, init_db, not required nor implemented for InMemoryQueryEngine")

    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
        """Not required nor implemented for InMemoryQueryEngine"""
        raise NotImplementedError("Method, connect_db, not required nor implemented for InMemoryQueryEngine")


# mypy will fail if ChromaDBQueryEngine does not implement RAGQueryEngine protocol
if TYPE_CHECKING:
    from ....agentchat.contrib.rag.query_engine import RAGQueryEngine

    def _check_implement_protocol(o: InMemoryQueryEngine) -> RAGQueryEngine:
        return o
