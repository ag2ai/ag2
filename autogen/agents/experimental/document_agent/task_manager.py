# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from autogen import ConversableAgent
from autogen.agentchat.contrib.rag.query_engine import RAGQueryEngine
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.reply_result import ReplyResult
from autogen.agents.experimental.document_agent.chroma_query_engine import VectorChromaQueryEngine
from autogen.agents.experimental.document_agent.parser_utils import docling_parse_docs
from autogen.doc_utils import export_module
from autogen.llm_config import LLMConfig

__all__ = ["TaskManagerAgent"]

logger = logging.getLogger(__name__)

TASK_MANAGER_SYSTEM_MESSAGE = """
You are task manager agent responsible for processing document ingestion and query tasks.

# INSTRUCTIONS:
1) You are provided provided with tools use those tools to process the tasks.
2) ingest_documents: (tool) For processing document ingestion tasks (takes list of paths/URLs)
3) execute_query: (tool) For answering queries using the RAG system (takes list of query strings)

# TASK FLOW (examples):
1)  Query: "Please ingest this PDF file: /path/to/document.pdf"
 TaskManager → ingest_documents -> summary agent
2) Query: "What is machine learning?"
 TaskManager → execute_query -> summary agent
3) Query: "ingest document A.pdf, What is machine learning?"
 TaskManager → ingest_documents -> execute_query -> summary agent

# IMPORTANT:
- Strictly follow the Instruction.
- Use Task flows as reference for tool call Execution.
"""


@export_module("autogen.agents.experimental")
class TaskManagerAgent(ConversableAgent):
    """TaskManagerAgent with integrated tools for document ingestion and query processing."""

    def __init__(
        self,
        name: str = "TaskManagerAgent",
        llm_config: LLMConfig | dict[str, Any] | None = None,
        query_engine: RAGQueryEngine | None = None,
        parsed_docs_path: Path | str | None = None,
        return_agent_success: str = "TaskManagerAgent",
        return_agent_error: str = "SummaryAgent",
        collection_name: str | None = None,
        max_workers: int | None = None,
        custom_system_message: str | None = None,
    ):
        """Initialize the TaskManagerAgent.

        Args:
            name: The name of the agent
            llm_config: The configuration for the LLM
            query_engine: The RAG query engine for document operations
            parsed_docs_path: Path where parsed documents will be stored
            return_agent_success: The agent to return on successful completion of the task
            return_agent_error: The agent to return on error
            collection_name: The collection name for the RAG query engine
            max_workers: Maximum number of threads for concurrent processing (None for default)
            custom_system_message: Custom system message for the TaskManagerAgent
        """
        self.query_engine = query_engine if query_engine else VectorChromaQueryEngine(collection_name=collection_name)
        self.parsed_docs_path = Path(parsed_docs_path) if parsed_docs_path else Path("./parsed_docs")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._temp_citations_store: dict[str, list[dict[str, str]]] = {}

        def _process_single_document(self: "TaskManagerAgent", input_file_path: str) -> tuple[str, bool, str]:
            """Process a single document. Returns (path, success, error_msg)."""
            try:
                output_files = docling_parse_docs(
                    input_file_path=input_file_path,
                    output_dir_path=self.parsed_docs_path,
                    output_formats=["markdown"],
                )

                # Limit to one output markdown file for now.
                if output_files:
                    output_file = output_files[0]
                    if output_file.suffix == ".md":
                        self.query_engine.add_docs(new_doc_paths_or_urls=[output_file])
                        return (input_file_path, True, "")

                return (input_file_path, False, "No valid markdown output generated")
            except Exception as doc_error:
                return (input_file_path, False, str(doc_error))

        def _execute_single_query(self: "TaskManagerAgent", query_text: str) -> tuple[str, str]:
            """Execute a single query. Returns (query, result)."""
            try:
                # Check for citations support
                if (
                    hasattr(self.query_engine, "enable_query_citations")
                    and getattr(self.query_engine, "enable_query_citations", False)
                    and hasattr(self.query_engine, "query_with_citations")
                    and callable(getattr(self.query_engine, "query_with_citations", None))
                ):
                    answer_with_citations = getattr(self.query_engine, "query_with_citations")(query_text)
                    answer = answer_with_citations.answer
                    txt_citations = [
                        {
                            "text_chunk": source.node.get_text(),
                            "file_path": source.metadata.get("file_path", "Unknown"),
                        }
                        for source in answer_with_citations.citations
                    ]
                    logger.info(f"Citations: {txt_citations}")

                    # Store citations in a temporary store that can be accessed by execute_query
                    if not hasattr(self, "_temp_citations_store"):
                        self._temp_citations_store = {}
                    self._temp_citations_store[query_text] = txt_citations

                    return (query_text, f"Query: {query_text}\nAnswer: {answer}")
                else:
                    answer = self.query_engine.query(query_text) if self.query_engine else "Query engine not available"
                    return (query_text, f"Query: {query_text}\nAnswer: {answer}")

            except Exception as query_error:
                logger.warning(f"Failed to execute query '{query_text}': {query_error}")
                return (query_text, f"Query: {query_text}\nAnswer: Error executing query: {query_error}")

        async def ingest_documents(
            documents_to_ingest: list[str], context_variables: ContextVariables
        ) -> ReplyResult | str:
            """Ingest documents from the provided list using concurrent processing.

            Args:
                documents_to_ingest: List of document paths or URLs to ingest
                context_variables: Context variables to store ingestion state

            Returns:
                str: Status message about the ingestion process
            """
            # Safely handle context variable initialization
            if "DocumentsToIngest" not in context_variables:
                context_variables["DocumentsToIngest"] = []
            if "DocumentsIngested" not in context_variables:
                context_variables["DocumentsIngested"] = []
            if "CompletedTaskCount" not in context_variables:
                context_variables["CompletedTaskCount"] = 0
            if "QueriesToRun" not in context_variables:
                context_variables["QueriesToRun"] = []

            # Add current batch to pending ingestions
            context_variables["DocumentsToIngest"].append(documents_to_ingest)

            try:
                # Process documents concurrently using ThreadPoolExecutor
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(self.executor, _process_single_document, self, doc_path)
                    for doc_path in documents_to_ingest
                ]

                # Wait for all documents to be processed
                results = await asyncio.gather(*futures, return_exceptions=True)

                successfully_ingested = []
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"Document processing failed with exception: {result}")
                        continue

                    # Type check to ensure result is the expected tuple
                    if isinstance(result, tuple) and len(result) == 3:
                        doc_path, success, error_msg = result
                        if success:
                            successfully_ingested.append(doc_path)
                        else:
                            logger.warning(f"Failed to ingest document {doc_path}: {error_msg}")
                    else:
                        logger.warning(f"Unexpected result format: {result}")

                # Enhanced logging with agent and tool title
                logger.info("=" * 80)
                logger.info("TOOL: ingest_documents (CONCURRENT)")
                logger.info("AGENT: TaskManagerAgent")
                logger.info(f"DOCUMENTS: {documents_to_ingest}")
                logger.info(f"SUCCESSFULLY INGESTED: {successfully_ingested}")
                logger.info("=" * 80)

                # Update context variables with successful ingestions
                if successfully_ingested:
                    context_variables["DocumentsIngested"].append(successfully_ingested)
                    context_variables["CompletedTaskCount"] += 1

                # Clear processed tasks
                context_variables["DocumentsToIngest"] = []
                context_variables["QueriesToRun"] = []

                return ReplyResult(
                    message=f"Documents ingested successfully: {successfully_ingested}",
                    context_variables=context_variables,
                )

            except Exception as e:
                # Enhanced error logging
                logger.error("=" * 80)
                logger.error("TOOL ERROR: ingest_documents (CONCURRENT)")
                logger.error("AGENT: TaskManagerAgent")
                logger.error(f"ERROR: {e}")
                logger.error(f"DOCUMENTS: {documents_to_ingest}")
                logger.error("=" * 80)

                # Preserve failed documents for retry
                context_variables["DocumentsToIngest"] = [documents_to_ingest]
                return ReplyResult(
                    message=f"Documents ingestion failed: {e}",
                    context_variables=context_variables,
                )

        async def execute_query(queries_to_run: list[str], context_variables: ContextVariables) -> ReplyResult | str:
            """Execute queries from the provided list using concurrent processing.

            Args:
                queries_to_run: List of queries to execute
                context_variables: Context variables to store query state

            Returns:
                str: The answers to the queries or error message
            """
            if not queries_to_run:
                return "No queries to run"

            # Safely handle context variable initialization
            if "QueriesToRun" not in context_variables:
                context_variables["QueriesToRun"] = []
            if "CompletedTaskCount" not in context_variables:
                context_variables["CompletedTaskCount"] = 0
            if "QueryResults" not in context_variables:
                context_variables["QueryResults"] = []
            if "Citations" not in context_variables:
                context_variables["Citations"] = []

            # Add current batch to pending queries
            context_variables["QueriesToRun"].append(queries_to_run)

            try:
                # Clear temporary citations store before processing
                self._temp_citations_store = {}

                # Process queries concurrently using ThreadPoolExecutor
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(self.executor, _execute_single_query, self, query) for query in queries_to_run
                ]

                # Wait for all queries to be processed
                results = await asyncio.gather(*futures, return_exceptions=True)

                answers = []
                all_citations: list[list[dict[str, str]] | None] = []

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Query processing failed with exception: {result}")
                        answers.append(f"Query processing failed: {result}")
                        all_citations.append(None)
                        continue

                    # Type check to ensure result is the expected tuple
                    if isinstance(result, tuple) and len(result) == 2:
                        query_text, answer = result
                        answers.append(answer)

                        # Get citations from temporary store if available
                        citations = self._temp_citations_store.get(query_text, None)
                        all_citations.append(citations)
                    else:
                        logger.warning(f"Unexpected result format: {result}")
                        answers.append(f"Unexpected result format: {result}")
                        all_citations.append(None)

                # Enhanced logging with agent and tool title
                logger.info("=" * 80)
                logger.info("TOOL: execute_query (CONCURRENT)")
                logger.info("AGENT: TaskManagerAgent")
                logger.info(f"QUERIES: {queries_to_run}")
                logger.info("=" * 80)

                # Update context variables
                context_variables["QueriesToRun"].pop(0)  # Remove processed batch
                context_variables["CompletedTaskCount"] += 1

                # Store query results with citations
                query_result = {"query": queries_to_run, "answer": answers, "citations": all_citations}
                context_variables["QueryResults"].append(query_result)
                # Clear temporary citations store after processing
                self._temp_citations_store = {}

                return ReplyResult(
                    message="\n\n".join(answers),
                    context_variables=context_variables,
                )

            except Exception as e:
                error_msg = f"Query failed for queries '{queries_to_run}': {str(e)}"

                # Enhanced error logging
                logger.error("=" * 80)
                logger.error("TOOL ERROR: execute_query (CONCURRENT)")
                logger.error("AGENT: TaskManagerAgent")
                logger.error(f"QUERIES: {queries_to_run}")
                logger.error(f"ERROR: {e}")
                logger.error("=" * 80)

                return ReplyResult(
                    message=error_msg,
                    context_variables=context_variables,
                )

        # Use custom system message if provided, otherwise use default
        system_message = custom_system_message if custom_system_message else TASK_MANAGER_SYSTEM_MESSAGE

        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            functions=[ingest_documents, execute_query],  # Add initiate_tasks
        )

    def __del__(self) -> None:
        """Clean up the ThreadPoolExecutor when the agent is destroyed."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
