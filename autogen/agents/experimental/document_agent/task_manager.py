
import logging
from pathlib import Path
from typing import Annotated, Any

from .... import ConversableAgent, UserProxyAgent
from ....agentchat.contrib.rag.query_engine import RAGQueryEngine
from ....agentchat.group.context_variables import ContextVariables
from ....doc_utils import export_module
from ....llm_config import LLMConfig
from ..document_agent.parser_utils import docling_parse_docs

executer = UserProxyAgent(name="ToolExecutor", human_input_mode="NEVER")

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



TaskManagerAgent = ConversableAgent(
    name="TaskManagerAgent", 
    system_message=TASK_MANAGER_SYSTEM_MESSAGE        
)


TaskManagerAgent.register_for_llm(description="Use this tool to ingest documents")
TaskManagerAgent.register_for_execution(description="Use this tool to ingest documents")
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
            parsed_docs_path = context_variables.get("ParsedDocsPath", "./parsed_docs")
            query_engine = context_variables.get("QueryEngine", None)
            
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
                        output_dir_path=parsed_docs_path,
                        output_formats=["markdown"],
                    )

                    # Process markdown files
                    ingested_files = []
                    for output_file in output_files:
                        if output_file.suffix == ".md":
                            if query_engine:
                                query_engine.add_docs(new_doc_paths_or_urls=[output_file])
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


TaskManagerAgent.register_for_llm(description="Use this tool to execute queries")
TaskManagerAgent.register_for_execution(description="Use this tool to execute queries")



