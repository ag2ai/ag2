# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, List, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    # Avoid circular import
    from ....agentchat.conversable_agent import ConversableAgent
    from .document_agent import Ingest, Query  # Adjust import path as needed for your actual models

__all__ = ["SummaryTaskAvailableCondition"]


class SummaryTaskAvailableCondition(BaseModel):
    """Available condition for determining if a summary task should be performed.

    This condition checks if:
    1. There are no documents left to ingest
    2. There are no queries left to run
    3. The completed task count is truthy

    If all conditions are met, the agent is ready for a summary task.
    """

    documents_var: str = "DocumentsToIngest"
    queries_var: str = "QueriesToRun"
    completed_var: str = "CompletedTaskCount"

    def __init__(
        self,
        documents_var: str = "DocumentsToIngest",
        queries_var: str = "QueriesToRun",
        completed_var: str = "CompletedTaskCount",
        **data: Any,
    ):
        """Initialize with optional custom variable names.

        Args:
            documents_var: Context variable name for documents to ingest list
            queries_var: Context variable name for queries to run list
            completed_var: Context variable name for completed task count
            **data: Additional data for BaseModel initialization
        """
        super().__init__(documents_var=documents_var, queries_var=queries_var, completed_var=completed_var, **data)

    def is_available(self, agent: "ConversableAgent", messages: list[dict[str, Any]]) -> bool:
        """Check if all task conditions are met.

        Args:
            agent: The agent with context variables
            messages: The conversation history (not used)

        Returns:
            True if all conditions are met (ready for summary), False otherwise
        """
        # Get variables from context with appropriate casting
        documents_to_ingest: List[Ingest] = cast(List[Ingest], agent.context_variables.get(self.documents_var, []))

        queries_to_run: List[Query] = cast(List[Query], agent.context_variables.get(self.queries_var, []))

        completed_task_count = bool(agent.context_variables.get(self.completed_var, 0))

        # All conditions must be true for the function to return True
        return len(documents_to_ingest) == 0 and len(queries_to_run) == 0 and completed_task_count
