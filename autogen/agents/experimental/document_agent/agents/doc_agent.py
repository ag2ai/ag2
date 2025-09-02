# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional

from autogen import ConversableAgent, UpdateSystemMessage
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.multi_agent_chat import initiate_group_chat
from autogen.agentchat.group.patterns.pattern import DefaultPattern
from autogen.agentchat.group.targets.transition_target import AgentTarget, TerminateTarget

from .....doc_utils import export_module
from .....llm_config import LLMConfig
from ..core.base_interfaces import RAGQueryEngine
from ..core.config import DocAgentConfig

__all__ = ["DocAgent2"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = """
You are a document query agent.
You answer questions based on documents that have been previously ingested into the vector database.
You can only answer questions about documents that are in the database.
"""

QUERY_AGENT_SYSTEM_MESSAGE = """
You are a query agent.
You answer the user's questions only using the query function provided to you.
You can only call use the execute_rag_query tool once per turn.
"""

ERROR_AGENT_SYSTEM_MESSAGE = """
You communicate errors to the user. Include the original error messages in full. Use the format:
The following error(s) have occurred:
- Error 1
- Error 2
"""

SUMMARY_AGENT_SYSTEM_MESSAGE = """
You are a summary agent and you provide a summary of all completed tasks and the list of queries and their answers.
Output two sections: 'Ingestions:' and 'Queries:' with the results of the tasks. Number the ingestions and queries.
If there are no ingestions output 'No ingestions', if there are no queries output 'No queries' under their respective sections.
Don't add markdown formatting.
For each query, there is one answer and, optionally, a list of citations.
For each citation, it contains two fields: 'text_chunk' and 'file_path'.
Format the Query and Answers and Citations as 'Query:\nAnswer:\n\nCitations:'. Add a number to each query if more than one.
For each query, output the full citation contents and list them one by one,
format each citation as '\nSource [X] (chunk file_path here):\n\nChunk X:\n(text_chunk here)' and mark a separator between each citation using '\n#########################\n\n'.
If there are no citations at all, DON'T INCLUDE ANY mention of citations.
"""


@export_module("autogen.agents.experimental.document_agent")
class DocAgent2(ConversableAgent):
    """Refactored DocAgent with Query Agent, Error Agent, and Summary Agent.

    This agent uses a multi-agent architecture to handle queries against pre-ingested documents.
    Document ingestion is handled separately by the ingestion service.
    """

    def __init__(
        self,
        name: str | None = "DocAgent",
        llm_config: LLMConfig | dict[str, Any] = {},
        system_message: str | None = None,
        query_engine: RAGQueryEngine | None = None,
        config: DocAgentConfig | None = None,
    ) -> None:
        """Initialize the refactored DocAgent.

        Args:
            name: The name of the DocAgent
            llm_config: The configuration for the LLM
            system_message: The system message for the DocAgent
            query_engine: The RAG query engine to use
            config: Configuration for the DocAgent
        """
        name = name or "DocAgent2"
        llm_config = llm_config
        system_message = system_message or DEFAULT_SYSTEM_MESSAGE
        config = config or DocAgentConfig()

        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        self.config = config
        self.query_engine = query_engine

        # Initialize context variables
        self._context_variables = ContextVariables(
            data={
                "QueriesToRun": [],
                "QueryResults": [],
                "CompletedTaskCount": 0,
            }
        )

        # Create the specialized agents
        self._create_query_agent(llm_config)
        self._create_error_agent(llm_config)
        self._create_summary_agent(llm_config)

        # Register the main reply function
        self.register_reply([ConversableAgent, None], DocAgent2._generate_group_chat_reply)

    def _create_query_agent(self, llm_config: LLMConfig | dict[str, Any]) -> None:
        """Create the Query Agent."""

        def execute_rag_query(context_variables: ContextVariables) -> dict[str, Any]:
            """Execute outstanding RAG queries."""
            if len(context_variables["QueriesToRun"]) == 0:
                return {"content": "No queries to run"}

            query = context_variables["QueriesToRun"][0]
            try:
                if (
                    self.query_engine is not None
                    and hasattr(self.query_engine, "enable_query_citations")
                    and self.query_engine.enable_query_citations
                    and hasattr(self.query_engine, "query_with_citations")
                    and callable(self.query_engine.query_with_citations)
                ):
                    answer_with_citations = self.query_engine.query_with_citations(query)
                    answer = answer_with_citations.answer
                    txt_citations = [
                        {
                            "text_chunk": source.node.get_text(),
                            "file_path": source.metadata["file_path"],
                        }
                        for source in answer_with_citations.citations
                    ]
                    logger.info(f"Citations:\n {txt_citations}")
                else:
                    if self.query_engine is not None:
                        answer = self.query_engine.query(query)
                    else:
                        answer = "No query engine available"
                    txt_citations = []

                context_variables["QueriesToRun"].pop(0)
                context_variables["CompletedTaskCount"] += 1
                context_variables["QueryResults"].append({"query": query, "answer": answer, "citations": txt_citations})

                return {"content": answer}
            except Exception as e:
                return {"content": f"Query failed for '{query}': {e}"}

        self._query_agent = ConversableAgent(
            name="QueryAgent",
            system_message=QUERY_AGENT_SYSTEM_MESSAGE,
            llm_config=llm_config,
            functions=[execute_rag_query],
        )

    def _create_error_agent(self, llm_config: LLMConfig | dict[str, Any]) -> None:
        """Create the Error Agent."""
        self._error_agent = ConversableAgent(
            name="ErrorAgent",
            system_message=ERROR_AGENT_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )

    def _create_summary_agent(self, llm_config: LLMConfig | dict[str, Any]) -> None:
        """Create the Summary Agent."""

        def create_summary_agent_prompt(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str:
            """Create the summary agent prompt with context information."""
            queries_to_run = agent.context_variables.get("QueriesToRun", [])
            query_results = agent.context_variables.get("QueryResults", [])

            system_message = (
                SUMMARY_AGENT_SYSTEM_MESSAGE + "\n"
                f"Queries left to run: {len(queries_to_run) if queries_to_run is not None else 0}\n"
                f"Query Results: {query_results}\n"
            )
            return system_message

        self._summary_agent = ConversableAgent(
            name="SummaryAgent",
            llm_config=llm_config,
            update_agent_state_before_reply=[UpdateSystemMessage(create_summary_agent_prompt)],
        )

    def _generate_group_chat_reply(
        self,
        messages: list[dict[str, Any]],
        sender: Optional["ConversableAgent"],
        config: Any,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """Generate reply using group chat with Query, Error, and Summary agents."""
        if not self.query_engine:
            return True, "No query engine configured. Please set up a RAG backend first."

        # Extract the query from messages
        if messages is None or len(messages) == 0:
            return True, "No messages provided."

        # Get the last message content
        last_message = messages[-1]
        query = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message)

        if not query:
            return True, "No query content found in message."

        # Add query to context
        self._context_variables["QueriesToRun"] = [query]

        # Create group chat agents
        group_chat_agents = [
            self._query_agent,
            self._error_agent,
            self._summary_agent,
        ]

        # Create pattern for group chat
        agent_pattern = DefaultPattern(
            initial_agent=self._query_agent,
            agents=group_chat_agents,
            context_variables=self._context_variables,
            group_after_work=TerminateTarget(),
        )

        # Set up handoffs
        self._query_agent.handoffs.set_after_work(target=AgentTarget(agent=self._summary_agent))
        self._error_agent.handoffs.set_after_work(target=TerminateTarget())
        self._summary_agent.handoffs.set_after_work(target=TerminateTarget())

        try:
            # Initiate group chat
            chat_result, context_variables, last_speaker = initiate_group_chat(
                pattern=agent_pattern,
                messages=query,
            )

            if last_speaker == self._error_agent or last_speaker == self._summary_agent:
                return True, chat_result.summary
            else:
                return True, "Document query completed successfully."

        except Exception as e:
            logger.error(f"Group chat failed: {e}")
            return True, f"Error processing query: {str(e)}"

    def set_query_engine(self, query_engine: RAGQueryEngine) -> None:
        """Set the query engine for this agent."""
        self.query_engine = query_engine

    def run(
        self,
        recipient: Optional["ConversableAgent"] = None,
        clear_history: bool = True,
        silent: bool | None = False,
        cache: Any = None,
        max_turns: int | None = None,
        summary_method: str | Any = None,
        summary_args: dict[str, Any] | None = None,
        message: dict[str, Any] | str | Any = None,
        executor_kwargs: dict[str, Any] | None = None,
        tools: Any = None,
        user_input: bool | None = False,
        msg_to: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the DocAgent with a query message."""
        if recipient is None:
            recipient = self
        return self.initiate_chat(
            recipient=recipient,
            message=message,
            max_turns=max_turns or 1,
        )

    @property
    def name(self) -> str:
        """Get the agent name."""
        return str(self._name)

    @property
    def system_message(self) -> str:
        """Get the system message."""
        return str(self._oai_system_message[0]["content"])
