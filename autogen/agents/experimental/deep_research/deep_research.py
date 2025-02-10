# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from .... import ConversableAgent
from ....doc_utils import export_module

# from ..websurfer import WebSurferAgent

__all__ = ["DeepResearchAgent"]


@export_module("autogen.agents.experimental")
class DeepResearchAgent(ConversableAgent):
    DEFAULT_PROMPT = "TODO"

    def __init__(
        self,
        name: str,
        llm_config: dict[str, Any],
        system_message: Optional[Union[str, list[str]]] = DEFAULT_PROMPT,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )
