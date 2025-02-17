from typing import Dict, List, Optional, Union
from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.learning.knowledge_models import KnowledgeModel, ResultModel
from autogen.agentchat.contrib.swarm_agent import SwarmAgent


class KnowledgeSharingSwarmAgent(SwarmAgent):
    """
    An extension of the SwarmAgent class that focuses on sharing knowledge between agents
    and updating system prompts with collected knowledge.

    Attributes:
        name (str): Name of the agent.
        llm_config (Optional[dict]): Configuration for the large language model.
        system_message (str): Base system message for the agent.
        knowledge_sources (Optional[List[SwarmAgent]]): Other agents acting as knowledge sources.
        use_own_knowledge (bool): Whether the agent uses its own knowledge.
        knowledge (KnowledgeModel): Model to manage the agent's knowledge.
    """

    def __init__(
        self,
        name: str,
        llm_config: Optional[dict] = None,
        system_message: str = "",
        knowledge_sources: Optional[List[SwarmAgent]] = None,
        use_own_knowledge: bool = True,
        *args,
        **kwargs
    ):
        """
        Initializes the KnowledgeSharingSwarmAgent.

        Args:
            name (str): The name of the agent.
            llm_config (Optional[dict]): Configuration for the LLM.
            system_message (str): Base system message for the agent.
            knowledge_sources (Optional[List[SwarmAgent]]): Additional agents for knowledge sharing.
            use_own_knowledge (bool): Flag to include the agent's own knowledge as a source.
        """
        super().__init__(name=name, llm_config=llm_config, system_message=system_message, *args, **kwargs)

        self._base_system_message = system_message

        # Initialize knowledge sources
        self._knowledge_sources: List[KnowledgeSharingSwarmAgent] = [self] if use_own_knowledge else []
        self._knowledge_sources += knowledge_sources or []

        # Register hooks to update system prompt and function signatures
        self.register_hook("update_agent_state", self._update_system_prompt_with_knowledge_hook)
        self.register_hook("update_agent_state", self._refresh_function_signatures)

        # Initialize knowledge model
        self.knowledge = KnowledgeModel(agent_name=name)

    def register_knowledge_sources(self, knowledge_sources: List[SwarmAgent]) -> None:
        """
        Registers knowledge sources for the agent.

        Args:
            knowledge_sources (List[SwarmAgent]): List of agents acting as knowledge sources.
        """
        self._knowledge_sources = knowledge_sources

    def add_knowledge_source(self, knowledge_source: SwarmAgent) -> None:
        """
        Adds a single knowledge source to the agent.

        Args:
            knowledge_source (SwarmAgent): An agent to add as a knowledge source.
        """
        self._knowledge_sources.append(knowledge_source)

    def add_results(self, result: ResultModel) -> None:
        """
        Adds a result to the agent's knowledge model.

        Args:
            result (ResultModel): A result to add to the knowledge model.
        """
        self.knowledge.results.append(result)

    def collect_knowledge_from_sources_for_prompt(self) -> str:
        """
        Collects knowledge from all sources and formats it into a concise, token-friendly structure.

        Returns:
            str: A formatted string containing the collected knowledge.
        """
        num_sources = len(self._knowledge_sources)
        num_results = sum(len(source.knowledge.results) for source in self._knowledge_sources)
        num_findings = sum(
            len(result.research.findings)
            for source in self._knowledge_sources
            for result in source.knowledge.results
        )

        introduction = (
            "Below are the results of the research conducted by multiple agents. "
            "Use the findings to assist with reasoning, decision-making, or generating insights."
        )
        explanation = (
            f"There are {num_sources} sources, labeled S1 through S{num_sources}. "
            f"In total, they contain {num_results} results, each with its own findings. "
            f"The data below includes {num_findings} total findings.\n"
            "In this list:\n"
            " • Each source is labeled S{i}\n"
            " • Each result under that agent is labeled R{j}\n"
            " • Each finding in a result is labeled F{k}\n\n"
        )

        lines = [introduction, explanation]

        for i, agent in enumerate(self._knowledge_sources, start=1):
            lines.append(f"A{i} ({agent.name})")
            for j, result_model in enumerate(agent.knowledge.results, start=1):
                lines.append(f"  R{j}: {result_model.question}")

                findings = getattr(result_model.research, "findings", [])
                if not findings:
                    lines.append("    (No findings.)")
                    continue

                for k, finding in enumerate(findings, start=1):
                    lines.append(f"    F{k}: {finding.finding}")

            lines.append("")  # Blank line to separate agents

        return "\n".join(lines)

    def _update_system_prompt_with_knowledge_hook(
        self,
        sender: Agent,
        message: Union[dict, str],
        recipient: Agent = None,
        silent: bool = False,
    ) -> Union[dict, str]:
        """
        Hook to update the system prompt with knowledge from sources.

        Args:
            sender (Agent): The agent sending the update.
            message (Union[dict, str]): The incoming message.
            recipient (Agent, optional): The intended recipient.
            silent (bool, optional): Whether to suppress output.

        Returns:
            Union[dict, str]: The modified message.
        """
        knowledge_text = self.collect_knowledge_from_sources_for_prompt()
        new_system_message = f"{self._base_system_message}\n\n{knowledge_text}"
        self.update_system_message(new_system_message)

    def _refresh_function_signatures(
        self,
        sender: Agent,
        message: Union[dict, str],
        recipient: Agent = None,
        silent: bool = False,
    ) -> Union[dict, str]:
        """
        Hook to refresh function signatures, ensuring consistency.

        Args:
            sender (Agent): The agent sending the update.
            message (Union[dict, str]): The incoming message.
            recipient (Agent, optional): The intended recipient.
            silent (bool, optional): Whether to suppress output.

        Returns:
            Union[dict, str]: The modified message.
        """
        for function_name, function in self._function_map.items():
            self.add_single_function(function)
