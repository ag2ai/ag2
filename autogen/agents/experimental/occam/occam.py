# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from .... import ConversableAgent
from ....doc_utils import export_module

__all__ = ["OccamAgent"]


@export_module("autogen.agents")
class OccamAgent(ConversableAgent):
    """Occam.ai Agent"""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the WebSurferAgent.

        Args:
            llm_config (dict[str, Any]): The LLM configuration.
        """

        super().__init__(*args, **kwargs)
