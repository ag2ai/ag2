# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any, Literal, Optional, Union

from .. import Agent, ConversableAgent, LLMMessageType

system_message = """You are an expert in text analysis.
The user will give you TEXT to analyze.
The user will give you analysis INSTRUCTIONS copied twice, at both the beginning and the end.
You will follow these INSTRUCTIONS in analyzing the TEXT, then give the results of your expert analysis in the format requested."""


class TextAnalyzerAgent(ConversableAgent):
    """(Experimental) Text Analysis agent, a subclass of ConversableAgent designed to analyze text as instructed."""

    def __init__(
        self,
        name="analyzer",
        system_message: Optional[str] = system_message,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        llm_config: Optional[Union[dict[str, Any], bool]] = None,
        **kwargs: Any,
    ):
        """Args:
        name (str): name of the agent.
        system_message (str): system message for the ChatCompletion inference.
        human_input_mode (str): This agent should NEVER prompt the human for input.
        llm_config (dict or False): llm inference configuration.
            Please refer to [OpenAIWrapper.create](/docs/api-reference/autogen/OpenAIWrapper#autogen.OpenAIWrapper.create)
            for available options.
            To disable llm-based auto reply, set to False.
        **kwargs (dict): other kwargs in [ConversableAgent](/docs/api-reference/autogen/ConversableAgent#conversableagent).
        """
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            llm_config=llm_config,
            **kwargs,
        )
        self.register_reply(Agent, TextAnalyzerAgent._analyze_in_reply, position=2)

    def _analyze_in_reply(
        self,
        messages: Optional[list["LLMMessageType"]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
        """Analyzes the given text as instructed, and returns the analysis as a message.
        Assumes exactly two messages containing the text to analyze and the analysis instructions.
        See Teachability.analyze for an example of how to use this method.
        """
        if self.llm_config is False:
            raise ValueError("TextAnalyzerAgent requires self.llm_config to be set in its base class.")
        if messages is None:
            messages = self._oai_messages[sender]  # In case of a direct call.
        assert len(messages) == 2

        # Delegate to the analysis method.
        return True, self.analyze_text(messages[0]["content"], messages[1]["content"])

    def analyze_text(self, text_to_analyze, analysis_instructions):
        """Analyzes the given text as instructed, and returns the analysis."""
        # Assemble the message.
        text_to_analyze = "# TEXT\n" + text_to_analyze + "\n"
        analysis_instructions = "# INSTRUCTIONS\n" + analysis_instructions + "\n"
        msg_text = "\n".join([
            analysis_instructions,
            text_to_analyze,
            analysis_instructions,
        ])  # Repeat the instructions.
        # Generate and return the analysis string.
        return self.generate_oai_reply([{"role": "user", "content": msg_text}], None, None)[1]
