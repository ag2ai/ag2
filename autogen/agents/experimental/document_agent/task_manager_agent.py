# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Annotated, Any

from .... import ConversableAgent
from ....agentchat.contrib.rag.query_engine import RAGQueryEngine
from ....agentchat.group.context_variables import ContextVariables
from ....doc_utils import export_module
from ....llm_config import LLMConfig
from ..document_agent.parser_utils import docling_parse_docs

__all__ = ["TaskManagerAgent"]

logger = logging.getLogger(__name__)

TASK_MANAGER_SYSTEM_MESSAGE = """
You are a task manager agent responsible for processing document ingestion and query tasks.

Your workflow:
1. Process the DocumentTask from the triage agent by extracting ingestions and queries
2. Use your tools to handle tasks:
   - Call ingest_documents for each document that needs to be ingested
   - Call execute_query for each query that needs to be answered
3. Continue processing until all tasks are complete

You have two tools available:
- ingest_documents: For processing document ingestion tasks
- execute_query: For answering queries using the RAG system

Always process ingestions before queries to ensure documents are available for querying.
When all tasks are complete, the system will automatically move to summary generation.
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
    ):
        """Initialize the TaskManagerAgent.

        Args:
            name: The name of the agent
            llm_config: The configuration for the LLM
            query_engine: The RAG query engine for document operations
            parsed_docs_path: Path where parsed documents will be stored
        """
        self.query_engine = query_engine
        self.parsed_docs_path = Path(parsed_docs_path) if parsed_docs_path else Path("./parsed_docs")

        def ingest_documents(
            context_variables: Annotated[ContextVariables, "Context variables containing document tasks"],
        ) -> str:
            """Ingest documents from the DocumentsToIngest list.

            Args:
                context_variables: The current context variables

            Returns:
                str: Status message about the ingestion process
            """
            documents_to_ingest = context_variables.get("DocumentsToIngest", [])

            if not documents_to_ingest:
                return "No documents to ingest"

            results = []
            documents_ingested = context_variables.get("DocumentsIngested", [])

            # Process each document
            for ingest_task in list(documents_to_ingest):  # Create copy to avoid modification during iteration
                try:
                    input_file_path = ingest_task.path_or_url

                    # Parse document with Docling
                    output_files = docling_parse_docs(
                        input_file_path=input_file_path,
                        output_dir_path=self.parsed_docs_path,
                        output_formats=["markdown"],
                    )

                    # Process markdown files
                    ingested_files = []
                    for output_file in output_files:
                        if output_file.suffix == ".md":
                            if self.query_engine:
                                self.query_engine.add_docs(new_doc_paths_or_urls=[output_file])
                            ingested_files.append(str(output_file))

                    if ingested_files:
                        # Update context
                        documents_to_ingest.remove(ingest_task)
                        if documents_ingested is not None:
                            documents_ingested.append(input_file_path)
                        current_count = context_variables.get("CompletedTaskCount", 0)
                        context_variables["CompletedTaskCount"] = (current_count or 0) + 1

                        results.append(f"Successfully ingested: {input_file_path}")
                    else:
                        results.append(f"No markdown files generated for: {input_file_path}")

                except Exception as e:
                    results.append(f"Failed to ingest {ingest_task.path_or_url}: {str(e)}")

            # Update context variables
            context_variables["DocumentsToIngest"] = documents_to_ingest
            context_variables["DocumentsIngested"] = documents_ingested

            return "; ".join(results) if results else "No documents processed"

        def execute_query(
            context_variables: Annotated[ContextVariables, "Context variables containing query tasks"],
        ) -> str:
            """Execute queries from the QueriesToRun list.

            Args:
                context_variables: The current context variables

            Returns:
                str: The answer to the query or error message
            """
            queries_to_run = context_variables.get("QueriesToRun", [])

            if not queries_to_run:
                return "No queries to run"

            # Process one query at a time
            query_task = queries_to_run[0]
            query_text = query_task.query

            try:
                # Check for citations support
                if (
                    self.query_engine is not None
                    and hasattr(self.query_engine, "enable_query_citations")
                    and self.query_engine.enable_query_citations
                    and hasattr(self.query_engine, "query_with_citations")
                    and callable(self.query_engine.query_with_citations)
                ):
                    answer_with_citations = self.query_engine.query_with_citations(query_text)
                    answer = answer_with_citations.answer
                    txt_citations = [
                        {
                            "text_chunk": source.node.get_text(),
                            "file_path": source.metadata.get("file_path", "Unknown"),
                        }
                        for source in answer_with_citations.citations
                    ]
                    logger.info(f"Citations: {txt_citations}")
                else:
                    answer = self.query_engine.query(query_text) if self.query_engine else "Query engine not available"
                    txt_citations = []

                # Update context variables
                queries_to_run.pop(0)
                context_variables["QueriesToRun"] = queries_to_run
                current_count = context_variables.get("CompletedTaskCount", 0)
                context_variables["CompletedTaskCount"] = (current_count or 0) + 1

                query_results = context_variables.get("QueryResults", [])
                if query_results is not None:
                    query_results.append({"query": query_text, "answer": answer, "citations": txt_citations})
                context_variables["QueryResults"] = query_results

                return str(answer)

            except Exception as e:
                error_msg = f"Query failed for '{query_text}': {str(e)}"

                # Still remove the failed query and update context
                queries_to_run.pop(0)
                context_variables["QueriesToRun"] = queries_to_run

                query_results = context_variables.get("QueryResults", [])
                if query_results is not None:
                    query_results.append({"query": query_text, "answer": error_msg, "citations": []})
                context_variables["QueryResults"] = query_results

                return error_msg

        super().__init__(
            name=name,
            system_message=TASK_MANAGER_SYSTEM_MESSAGE,
            llm_config=llm_config,
            functions=[ingest_documents, execute_query],
        )

    def process_triage_output(self, document_task: Any, context_variables: ContextVariables) -> None:
        """Process the DocumentTask from triage agent and update context variables.

        Args:
            document_task: The DocumentTask containing ingestions and queries
            context_variables: The current context variables
        """
        # Extract ingestions and queries from DocumentTask
        ingestions = getattr(document_task, "ingestions", [])
        queries = getattr(document_task, "queries", [])

        # Apply deduplication logic
        existing_ingestions = context_variables.get("DocumentsToIngest", [])
        existing_queries = context_variables.get("QueriesToRun", [])
        documents_ingested = context_variables.get("DocumentsIngested", [])

        # Deduplicate ingestions
        unique_ingestions = []
        if existing_ingestions is not None and documents_ingested is not None:
            for ingestion in ingestions:
                ingestion_path = ingestion.path_or_url
                # Check if already pending or already ingested
                already_pending = any(existing.path_or_url == ingestion_path for existing in existing_ingestions)
                already_ingested = ingestion_path in documents_ingested

                if not (already_pending or already_ingested):
                    unique_ingestions.append(ingestion)

        # Deduplicate queries
        unique_queries = []
        if existing_queries is not None:
            for query in queries:
                query_text = query.query
                # Check if already pending
                already_pending = any(existing.query == query_text for existing in existing_queries)

                if not already_pending:
                    unique_queries.append(query)

        # Update context variables
        context_variables["DocumentsToIngest"] = (existing_ingestions or []) + unique_ingestions
        context_variables["QueriesToRun"] = (existing_queries or []) + unique_queries

        # Set task initiated flag
        context_variables["TaskInitiated"] = True
