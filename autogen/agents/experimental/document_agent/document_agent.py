# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from .... import Agent, ConversableAgent, UpdateSystemMessage
from ....agentchat.contrib.rag.query_engine import RAGQueryEngine
from ....agentchat.group.context_variables import ContextVariables
from ....agentchat.group.multi_agent_chat import initiate_group_chat
from ....agentchat.group.patterns.pattern import DefaultPattern
from ....agentchat.group.targets.transition_target import AgentTarget, TerminateTarget
from ....doc_utils import export_module
from ....llm_config import LLMConfig
from ....oai.client import OpenAIWrapper
from .chroma_query_engine import VectorChromaQueryEngine
from .document_utils import Ingest, Query
from .task_manager_agent import TaskManagerAgent

__all__ = ["DocAgent"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = """
    You are a document agent.
    You are given a list of documents to ingest and a list of queries to perform.
    You are responsible for ingesting the documents and answering the queries.
"""


class DocumentTask(BaseModel):
    """The structured output format for task decisions."""

    ingestions: list[Ingest] = Field(description="The list of documents to ingest.")
    queries: list[Query] = Field(description="The list of queries to perform.")


class DocumentTriageAgent(ConversableAgent):
    """The DocumentTriageAgent is responsible for deciding what type of task to perform from user requests."""

    def __init__(self, llm_config: LLMConfig | dict[str, Any] | None = None):
        # Add the structured message to the LLM configuration
        structured_config_list = deepcopy(llm_config)
        structured_config_list["response_format"] = DocumentTask  # type: ignore[index]

        super().__init__(
            name="DocumentTriageAgent",
            system_message=(
                "You are a document triage agent. "
                "You are responsible for deciding what type of task to perform from a user's request and populating a DocumentTask formatted response. "
                "If the user specifies files or URLs, add them as individual 'ingestions' to DocumentTask. "
                "You can access external websites if given a URL, so put them in as ingestions. "
                "Add the user's questions about the files/URLs as individual 'RAG_QUERY' queries to the 'query' list in the DocumentTask. "
                "Don't make up questions, keep it as concise and close to the user's request as possible."
            ),
            human_input_mode="NEVER",
            llm_config=structured_config_list,
        )


@export_module("autogen.agents.experimental")
class DocAgent(ConversableAgent):
    """The DocAgent is responsible for ingest and querying documents.

    Internally, it generates a group chat with a simplified set of agents:
    - TriageAgent: Analyzes user requests and creates DocumentTask
    - TaskManagerAgent: Processes documents and queries using integrated tools
    - SummaryAgent: Provides final summary of all operations
    """

    def __init__(
        self,
        name: str | None = None,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        system_message: str | None = None,
        parsed_docs_path: str | Path | None = None,
        collection_name: str | None = None,
        query_engine: RAGQueryEngine | None = None,
    ):
        """Initialize the DocAgent.

        Args:
            name: The name of the DocAgent.
            llm_config: The configuration for the LLM.
            system_message: The system message for the DocAgent.
            parsed_docs_path: The path where parsed documents will be stored.
            collection_name: The unique name for the data store collection.
            query_engine: The query engine to use for querying documents.
        """
        name = name or "DocAgent"
        llm_config = llm_config or LLMConfig.get_current_llm_config()
        system_message = system_message or DEFAULT_SYSTEM_MESSAGE
        parsed_docs_path = parsed_docs_path or "./parsed_docs"

        # Default Query Engine will be ChromaDB
        if query_engine is None:
            query_engine = VectorChromaQueryEngine(collection_name=collection_name)

        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        self.register_reply([ConversableAgent, None], DocAgent.generate_inner_group_chat_reply, position=0)

        # Initialize agents
        self._triage_agent = DocumentTriageAgent(llm_config=llm_config)

        self._task_manager_agent = TaskManagerAgent(
            llm_config=llm_config,
            query_engine=query_engine,
            parsed_docs_path=parsed_docs_path,
        )

        def update_ingested_documents() -> None:
            """Updates the list of ingested documents for persistence across interactions."""
            if hasattr(self._triage_agent, "context_variables"):
                agent_documents_ingested = self._triage_agent.context_variables.get("DocumentsIngested", [])
                if agent_documents_ingested is not None:
                    for doc in agent_documents_ingested:
                        if doc not in self.documents_ingested:
                            self.documents_ingested.append(doc)

        def create_summary_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
            """Create the summary agent prompt with context information."""
            update_ingested_documents()

            query_results = cast(list[dict[str, Any]], agent.context_variables.get("QueryResults", []))
            documents_ingested = cast(list[str], agent.context_variables.get("DocumentsIngested", []))
            documents_to_ingest = cast(list[Ingest], agent.context_variables.get("DocumentsToIngest", []))
            queries_to_run = cast(list[Query], agent.context_variables.get("QueriesToRun", []))

            system_message = (
                "You are a summary agent and you provide a summary of all completed tasks and the list of queries and their answers. "
                "Output two sections: 'Ingestions:' and 'Queries:' with the results of the tasks. Number the ingestions and queries. "
                "If there are no ingestions output 'No ingestions', if there are no queries output 'No queries' under their respective sections. "
                "Don't add markdown formatting. "
                "For each query, there is one answer and, optionally, a list of citations. "
                "For each citation, it contains two fields: 'text_chunk' and 'file_path'. "
                "Format the Query and Answers and Citations as 'Query:\\nAnswer:\\n\\nCitations:'. Add a number to each query if more than one. Use the context below:\\n"
                "For each query, output the full citation contents and list them one by one, "
                "format each citation as '\\nSource [X] (chunk file_path here):\\n\\nChunk X:\\n(text_chunk here)' and mark a separator between each citation using '\\n#########################\\n\\n'. "
                "If there are no citations at all, DON'T INCLUDE ANY mention of citations. "
                "Include any errors that occurred during processing.\\n"
                f"Documents ingested: {documents_ingested}\\n"
                f"Documents left to ingest: {len(documents_to_ingest)}\\n"
                f"Queries left to run: {len(queries_to_run)}\\n"
                f"Query Results: {query_results}\\n"
            )

            return system_message

        self._summary_agent = ConversableAgent(
            name="SummaryAgent",
            llm_config=llm_config,
            update_agent_state_before_reply=[UpdateSystemMessage(create_summary_agent_prompt)],
        )

        # Set up handoffs - simplified flow
        self._triage_agent.handoffs.set_after_work(target=AgentTarget(agent=self._task_manager_agent))
        self._task_manager_agent.handoffs.set_after_work(target=AgentTarget(agent=self._summary_agent))
        self._summary_agent.handoffs.set_after_work(target=TerminateTarget())

        self.documents_ingested: list[str] = []
        self._group_chat_context_variables: ContextVariables | None = None

    def generate_inner_group_chat_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """Reply function that generates the inner group chat reply for the DocAgent."""
        # Initialize or reuse context variables
        if hasattr(self, "_group_chat_context_variables") and self._group_chat_context_variables is not None:
            context_variables = self._group_chat_context_variables
            # Reset pending tasks for new run
            context_variables["DocumentsToIngest"] = []
        else:
            context_variables = ContextVariables(
                data={
                    "CompletedTaskCount": 0,
                    "DocumentsToIngest": [],
                    "DocumentsIngested": self.documents_ingested,
                    "QueriesToRun": [],
                    "QueryResults": [],
                }
            )
            self._group_chat_context_variables = context_variables

        # Create custom task manager that processes triage output
        class CustomTaskManager(TaskManagerAgent):
            def __init__(self, base_agent: TaskManagerAgent, context_vars: ContextVariables):
                # Copy all attributes from base agent
                self.__dict__.update(base_agent.__dict__)
                self._context_vars = context_vars

            def generate_reply(
                self,
                messages: list[dict[str, Any]] | None = None,
                sender: Agent | None = None,
                **kwargs: Any,
            ) -> str | dict[str, Any] | None:
                # Check if we received a message from triage agent
                if messages and len(messages) > 0:
                    last_message = messages[-1]
                    if (
                        isinstance(last_message, dict)
                        and last_message.get("name") == "DocumentTriageAgent"
                        and "content" in last_message
                    ):
                        # Extract DocumentTask from triage agent's response
                        try:
                            import json

                            document_task_data = json.loads(last_message["content"])

                            # Create DocumentTask object
                            document_task = DocumentTask(
                                ingestions=[Ingest(**ing) for ing in document_task_data.get("ingestions", [])],
                                queries=[Query(**q) for q in document_task_data.get("queries", [])],
                            )

                            # Process the triage output
                            self.process_triage_output(document_task, self._context_vars)
                        except Exception as e:
                            logger.warning(f"Failed to parse DocumentTask: {e}")

                # Continue with normal task processing
                return super().generate_reply(messages, sender)

        # Create custom task manager instance
        custom_task_manager = CustomTaskManager(self._task_manager_agent, context_variables)

        group_chat_agents = [
            self._triage_agent,
            custom_task_manager,
            self._summary_agent,
        ]

        agent_pattern = DefaultPattern(
            initial_agent=self._triage_agent,
            agents=group_chat_agents,
            context_variables=context_variables,
            group_after_work=TerminateTarget(),
        )

        chat_result, context_variables, last_speaker = initiate_group_chat(
            pattern=agent_pattern,
            messages=self._get_document_input_message(messages),
        )

        # Always return the final result since we only have summary termination
        return True, chat_result.summary

    def _get_document_input_message(self, messages: list[dict[str, Any]] | None) -> str:
        """Gets and validates the input message(s) for the document agent."""
        if messages is None:
            raise NotImplementedError("Messages cannot be None.")
        elif (
            isinstance(messages, list)
            and len(messages) > 0
            and "content" in messages[-1]
            and isinstance(messages[-1]["content"], str)
        ):
            return messages[-1]["content"]
        else:
            raise NotImplementedError("Invalid messages format. Must be a list of messages.")
