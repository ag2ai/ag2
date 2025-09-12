# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
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
You are a task manager agent responsible for processing document ingestion and query tasks.

Your workflow:
1. ALWAYS call initiate_tasks first when you receive a message from the DocumentTriageAgent
2. Use your tools to handle tasks:
   - Call ingest_documents with a list of document paths/URLs
   - Call execute_query with a list of query strings
3. Continue processing until all tasks are complete

You have three tools available:
- ingest_documents: For processing document ingestion tasks (takes list of paths/URLs)
- execute_query: For answering queries using the RAG system (takes list of query strings)

Always process ingestions before queries to ensure documents are available for querying.
When all tasks are complete, the system will automatically move to summary generation.

# STRICTLY end if no tasks are left to process.
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
        """
        self.query_engine = query_engine if query_engine else VectorChromaQueryEngine(collection_name=collection_name)
        self.parsed_docs_path = Path(parsed_docs_path) if parsed_docs_path else Path("./parsed_docs")

        def ingest_documents(documents_to_ingest: list[str], context_variables: ContextVariables) -> ReplyResult | str:
            """Ingest documents from the provided list.

            Args:
                documents_to_ingest: List of document paths or URLs to ingest
                context_variables: Context variables to store ingestion state

            Returns:
                str: Status message about the ingestion process
            """
            context_variables["DocumentsToIngest"].append(documents_to_ingest)
            try:
                for input_file_path in documents_to_ingest:
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

                # Enhanced logging with agent and tool title
                logger.info("=" * 80)
                logger.info("🔧 TOOL: ingest_documents")
                logger.info("🤖 AGENT: TaskManagerAgent")
                logger.info(f"📄 DOCUMENTS: {documents_to_ingest}")
                logger.info("=" * 80)

                context_variables["DocumentsIngested"].append(documents_to_ingest)
                context_variables["CompletedTaskCount"] += 1
                context_variables["DocumentsToIngest"] = []
                context_variables["QueriesToRun"] = []
                return ReplyResult(
                    message=f"Documents ingested successfully: {documents_to_ingest}",
                    context_variables=context_variables,
                )

            except Exception as e:
                # Enhanced error logging
                logger.error("=" * 80)
                logger.error("❌ TOOL ERROR: ingest_documents")
                logger.error("🤖 AGENT: TaskManagerAgent")
                logger.error(f"💥 ERROR: {e}")
                logger.error(f"📄 DOCUMENTS: {documents_to_ingest}")
                logger.error("=" * 80)
                context_variables["DocumentsToIngest"] = [documents_to_ingest]
                return ReplyResult(
                    message=f"Documents ingestion failed: {e}",
                    context_variables=context_variables,
                )

        def execute_query(queries_to_run: list[str], context_variables: ContextVariables) -> ReplyResult | str:
            """Execute queries from the provided list.

            Args:
                queries_to_run: List of queries to execute
                context_variables: Context variables to store query state

            Returns:
                str: The answers to the queries or error message
            """
            if not queries_to_run:
                return "No queries to run"
            context_variables["QueriesToRun"].append(queries_to_run)
            try:
                answers = []
                for query_text in queries_to_run:
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
                    else:
                        answer = (
                            self.query_engine.query(query_text) if self.query_engine else "Query engine not available"
                        )
                        txt_citations = []

                    answers.append(f"Query: {query_text}\nAnswer: {answer}")

                # Enhanced logging with agent and tool title
                logger.info("=" * 80)
                logger.info("🔧 TOOL: execute_query")
                logger.info("🤖 AGENT: TaskManagerAgent")
                logger.info(f"❓ QUERIES: {queries_to_run}")
                logger.info("=" * 80)
                context_variables["QueriesToRun"].pop(0)
                context_variables["CompletedTaskCount"] += 1
                context_variables["QueryResults"].append({"query": queries_to_run, "answer": answers})
                return ReplyResult(
                    message="\n\n".join(answers),
                    context_variables=context_variables,
                )

            except Exception as e:
                error_msg = f"Query failed for queries '{queries_to_run}': {str(e)}"

                # Enhanced error logging
                logger.error("=" * 80)
                logger.error("❌ TOOL ERROR: execute_query")
                logger.error("🤖 AGENT: TaskManagerAgent")
                logger.error(f"❓ QUERIES: {queries_to_run}")
                logger.error(f"💥 ERROR: {e}")
                logger.error("=" * 80)

                return ReplyResult(
                    message=error_msg,
                    context_variables=context_variables,
                )

        super().__init__(
            name=name,
            system_message=TASK_MANAGER_SYSTEM_MESSAGE,
            llm_config=llm_config,
            functions=[ingest_documents, execute_query],  # Add initiate_tasks
        )
