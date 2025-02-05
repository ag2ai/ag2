# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

from autogen import ConversableAgent, SwarmAgent, UserProxyAgent
from autogen.agentchat.contrib.rag.docling_doc_ingest_agent import DoclingDocIngestAgent
from autogen.agentchat.contrib.swarm_agent import (
    AFTER_WORK,
    ON_CONDITION,
    AfterWorkOption,
    SwarmResult,
    initiate_swarm_chat,
)

DEFAULT_SYSTEM_MESSAGE = """
    You are a document agent.
    You are given a list of documents to ingest and a list of queries to perform.
    You are responsible for ingesting the documents and answering the queries.
"""
DEFAULT_DESCRIPTION = None
TASK_MANAGER_SYSTEM_MESSAGE = """
    You are a task manager agent. You update the context variables based on the task decisions (DocumentTask).
    i.e. output
    {
        "ingestions": [
            {
                "path_or_url": "path_or_url"
            }
        ],
        "queries": [
            {
                "query_type": "RAG_QUERY",
                "query": "query"
            }
        ],
        "query_results": [
            {
                "query": "query",
                "result": "result"
            }
        ]
    }
"""


class QueryType(Enum):
    RAG_QUERY = "RAG_QUERY"
    COMMON_QUESTION = "COMMON_QUESTION"


class Ingest(BaseModel):
    path_or_url: str = Field(description="The path or URL of the documents to ingest.")


class Query(BaseModel):
    query_type: QueryType = Field(description="The type of query to perform for the Document Agent.")
    query: str = Field(description="The query to perform for the Document Agent.")


class DocumentTask(BaseModel):
    """The structured output format for task decisions."""

    ingestions: list[Ingest] = Field(description="The list of documents to ingest.")
    queries: list[Query] = Field(description="The list of queries to perform.")


class DocumentTriageAgent(SwarmAgent):
    def __init__(
        self,
        llm_config: Dict[str, Any],
        name: str = "DocumentTriageAgent",
        *args,
        **kwargs,
    ):
        """
        Initialize the DocumentTriageAgent.

        Args:
        llm_config (Dict[str, Any]): The configuration for the LLM.
        *args: Additional positional arguments to pass to the parent class.
        **kwargs: Additional keyword arguments to pass to the parent class.

        The DocumentTriageAgent is responsible for deciding what type of task to perform from user requests.
        When user uploads new documents or provide links of documents, you should add Ingest task to DocumentTask.
        When user asks common questions, you should add 'COMMON_QUESTION' Query task to DocumentTask.
        When user asks questions about information from existing documents, you add 'RAG_QUERY' Query task to DocumentTask.
        """
        # Add the structured message to the LLM configuration
        structured_config_list = deepcopy(llm_config)
        for config in structured_config_list["config_list"]:
            config["response_format"] = DocumentTask

        super().__init__(
            name=name,
            system_message=(
                "You are a document triage agent."
                "You are responsible for deciding what type of task to perform from user requests."
                "When user uploads new documents or provide links of documents, you should add Ingest task to DocumentTask."
                "When user asks common questions, you should add 'COMMON_QUESTION' Query task to DocumentTask."
                "When user asks questions about information from existing documents, you add 'RAG_QUERY' Query task to DocumentTask."
            ),
            llm_config=structured_config_list,
            *args,
            **kwargs,
        )


class DocumentAgent(ConversableAgent):
    def __init__(
        self,
        name: str = "Document Agent",
        llm_config: Dict[str, Any] = {},
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        description: Optional[str] = DEFAULT_DESCRIPTION,
        **kwargs,
    ):
        """
        Initialize the DocumentAgent.

        Args:
        name (str): The name of the DocumentAgent.
        llm_config (Dict[str, Any]): The configuration for the LLM.
        system_message (str): The system message for the ChatCompletion inference.
        description (str): The description of the DocumentAgent.
        **kwargs: Additional keyword arguments to pass to the parent class.

        The DocumentAgent is responsible for generating a group of agents to solve a task.

        The agents that the DocumentAgent generates are:
        - Triage Agent: responsible for deciding what type of task to perform from user requests.
        - Task Manager Agent: responsible for managing the tasks.
        - Parser Agent: responsible for parsing the documents.
        - Data Ingestion Agent: responsible for ingesting the documents.
        - Query Agent: responsible for answering the user's questions.
        - Summary Agent: responsible for generating a summary of the user's questions.
        """
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER",
            description=description,
            **kwargs,
        )
        self.register_reply([ConversableAgent, None], self.generate_inner_swarm_reply, position=0)

        self._context_variables = {
            "DocumentsToIngest": [],
            "QueriesToRun": [],
            "QueryResults": [],
        }

        self._triage_agent = DocumentTriageAgent(llm_config=llm_config)

        def initiate_tasks(ingestions: list[Ingest], queries: list[Query], context_variables: dict) -> SwarmResult:
            context_variables["DocumentsToIngest"] = ingestions
            context_variables["QueriesToRun"] = queries
            return SwarmResult(
                value="Updated context variables with task decisions", context_variables=context_variables
            )

        self._task_manager_agent = SwarmAgent(
            name="TaskManagerAgent",
            system_message=TASK_MANAGER_SYSTEM_MESSAGE,
            llm_config=llm_config,
            functions=[initiate_tasks],
        )

        self._parser_agent = SwarmAgent(
            name="Parser Agent",
        )

        self._data_ingestion_agent = DoclingDocIngestAgent(
            name="Data Ingestion Agent",
        )

        self._query_agent = SwarmAgent(
            name="Query Agent",
        )

        self._summary_agent = SwarmAgent(
            name="Summary Agent",
        )

        self._triage_agent.register_hand_off(
            hand_to=[
                ON_CONDITION(self._task_manager_agent, "After output task desicisions, transfer to task manager agent"),
            ]
        )

    def generate_inner_swarm_reply(
        self, messages: Union[list[dict[str, Any]], str]
    ) -> tuple[bool, Union[str, dict, None]]:
        user_agent = UserProxyAgent(
            name="UserAgent",
            system_message="A human admin.",
            human_input_mode="ALWAYS",
        )
        chat_result, context_variables, last_speaker = initiate_swarm_chat(
            initial_agent=self.triage_agent,  # Starting agent
            agents=[self._triage_agent, self._task_manager_agent],
            user_agent=user_agent,  # Human user
            messages=messages,
            context_variables=context_variables,  # Context
            after_work=AFTER_WORK(AfterWorkOption.TERMINATE),  # Swarm-level after work hand off
        )
        pass
