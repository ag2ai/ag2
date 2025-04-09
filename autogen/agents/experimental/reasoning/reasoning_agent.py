# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import copy
import math
import random
import re
import warnings
from typing import Any, Literal, Optional, Union

from .... import Agent, AssistantAgent, UserProxyAgent
from ....doc_utils import export_module
from ....import_utils import optional_import_block
from ....llm_config import LLMConfig

__all__ = ["ReasoningAgent", "ThinkNode"]

EPSILON = 1e-6

REASONING_AGENT_MESSAGE = """You are a Reasoning AI Assistant. You generate high-quality responses to user questions by taking advantage of a tree-of-thought reasoning process."""

TREEOFTHOUGHT_MESSAGE = """Role: Deep Thinking AI Assistant

End Goal: Generate an efficient thinking trajectory of steps to follow in order to provide a high-quality response to the user. Think deep in complex questions and shallow in simple ones.

Current Task: Given the question and a list of previous thinking steps (the plan trajectory), you have two options:
1) Terminate: if you believe the user's question has been explored, terminate the task.
2) Continue Thinking: generate at least four innovative options for the next step in the thinking process to add in the trajectory. The user will not answer you anything.

### 1) Terminate
#### Instructions:
- Always terminate the task when you believe the user's question has been explored.
- The user wants the response the quicker possible, so when a high quality response can be crafted by the exploration performed, terminate the process.
- Don't terminate from the first step.
- Never provide additional options when terminating the task.

#### Format of Output:
REFLECTION:
*Give a few sentence reflections on the previous steps in the thinking trajectory, explaining why the task has been explored thoroughly.*

** Possible Options:**
Option 1: TERMINATE
<Short description>

### 2) Continue thinking
#### Instructions:
- Continue thinking when you believe that more exploration is needed to provide a high-quality response.
- Review the user's question and the previous steps taken. If only the question is provided and no previous steps, then make your suggestions to initiate the thinking process.
- The options you will provide must be alternatives for the next step in the thinking trajectory. Not steps that consider another option as given. So, make them focused and not too many.
- Identify any mistakes or errors in the previous thinking. If you find any mistakes, include options to correct them in your proposed options.
- If the question is a multi-choice question, you should carefully eliminate obviously wrong choices, look for contextual clues in the question, and use logical reasoning to select the most plausible answer.
- If you need to validate, simulate, or illustrate a reasoning concept (like mathematical expressions, code execution, algorithms, etc.) with Python, place the code in a fenced block like ```python ... ``` and always print the results that you want to see.

#### Options Restrictions:
- Never suggest options that access/consult/cross-check the internet, external sources, literature, datasets, books, or experts.
- Never suggest options in the physical world like conducting experiments or surveys, your approach in practical problems should still be theoretical.
- Never suggest options that require data you do not have, or suggest research to collect them.
- Never use Python when there is no need to.
- Never include a code option without the script to execute (e.g. not: Use python to make the calculations, but: Use this script to make the calculations: ```python... ```).

#### Format of Output:
REFLECTION:
*Give a few sentence reflections on the previous steps in the thinking trajectory, what is wrong and what is good.*

**Possible Options:**
Option 1: <Thinking 1>
<Short description, optional code snippet to execute>

Option 2: <Thinking 2>
<Short description, optional code snippet to execute>

Option 3: <Thinking 3>
<Short description, optional code snippet to execute>

Option 4: <Thinking 4>
<Short description, optional code snippet to execute>

...
"""

EXECUTOR_MESSAGE = "Please provide an answer for the last step in the thinking trajectory, to advance the thinking process. Keep your answers as consise as possible. Never suggest the next step."


@export_module("autogen.agents.experimental")
class ThinkNode:
    def __init__(self, content: str, parent: Optional["ThinkNode"] = None) -> None:
        """A node in a tree structure representing a step in the reasoning process.

        This class implements a tree node that stores content (text describing a reasoning step),
        maintains parent-child relationships, tracks node statistics, and provides utilities
        for traversing/visualizing the reasoning path.

        Args:
            content (str): The text content/description for this reasoning step.
            parent (Optional[ThinkNode]): The parent node in the tree, if any.

        Attributes:
            content (str): The text content/description for this reasoning step.
            value (float): A numeric score/value assigned to this node.
            parent (Optional[ThinkNode]): Reference to the parent node.
            reflection (str): A string containing reflections on the reasoning process.
            rating_details (str): A string providing details about the rating of this node.
            output (Optional[str]): The output generated at this node through the `execute_node` method.
            depth (int): The depth of this node in the tree (root = 0).
            children (list[ThinkNode]): list of child nodes.
            visits (int): Number of times this node has been visited during search.

        The node automatically maintains the tree structure by:
        - Setting its depth based on the parent's depth + 1.
        - Adding itself to the parent's children list if the parent exists.
        - Providing trajectory utilities to get the full path from root to this node.
        """
        self.content: str = content
        self.value: float = 0.0
        self.parent: Optional[ThinkNode] = parent
        self.reflection: str = ""
        self.rating_details: str = ""
        self.output: Optional[str] = None
        self.depth: int = parent.depth + 1 if parent is not None else 0
        self.children: list[ThinkNode] = []
        self.visits: int = 0
        if self.parent:
            self.parent.children.append(self)

    @property
    def _trajectory_arr(self) -> list[str]:
        """Gets the full path from root to this node as a list of strings.

        Returns:
            list[str]: list containing the content of each node from root to current node
        """
        step = f"Content: {self.content}"
        if self.output is not None:
            step += f"\nOutput: {self.output}"
        if self.parent:
            return self.parent._trajectory_arr + [step]
        return ["# Question:\n" + step + "\n---\n"]

    @property
    def trajectory(self) -> str:
        """Get a formatted string representation of the path from root to this node.

        Returns:
            str: A formatted string showing the question and each step in the reasoning process
        """
        traj = self._trajectory_arr
        ans = traj[0]
        ans += "# Trajectory:\n"
        for i, step in enumerate(traj[1:]):
            ans += f"\nStep {i + 1}:\n{step}"
        return ans

    def backpropagate(self, reward: float) -> None:
        """Update the score of this node and its parents using moving average.

        Args:
            reward (float): The reward to backpropagate up the tree.
        """
        node: Optional[ThinkNode] = self
        while node is not None:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def __str__(self) -> str:
        return f"{self.content} -> Depth: {self.depth} Value: {self.value} Visits: {self.visits}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict[str, Any]:
        """Convert ThinkNode to dictionary representation.

        Returns:
            dict[str, Any]: dictionary containing all node attributes and recursive children
        """
        return {
            "content": self.content,
            "value": self.value,
            "depth": self.depth,
            "reflection": self.reflection,
            "rating_details": self.rating_details,
            "output": self.output,
            "visits": self.visits,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], parent: Optional["ThinkNode"] = None) -> "ThinkNode":
        """Create ThinkNode from dictionary representation.

        Args:
            data (dict[str, Any]): dictionary containing node data
            parent (Optional[ThinkNode]): Parent node to attach to

        Returns:
            ThinkNode: Reconstructed node with all children
        """
        node = cls(content=data["content"], parent=parent)
        node.value = data["value"]
        node.depth = data["depth"]
        node.visits = data["visits"]
        node.reflection = data.get("reflection", "")
        node.rating_details = data.get("rating_details", "")
        node.output = data.get("output")

        # Recursively create children
        for child_data in data["children"]:
            cls.from_dict(child_data, parent=node)

        return node

    def visualize_tree(self) -> None:
        """Visualize the tree of thoughts using graphviz."""
        with optional_import_block() as result:
            from graphviz import Digraph

        if not result.is_successful:
            print("Please install graphviz: pip install graphviz")
            return

        dot = Digraph(comment="Tree of Thoughts")
        dot.attr(rankdir="TB")  # Top to Bottom direction

        def add_nodes(node: ThinkNode, node_id: str = "0") -> None:
            # Truncate long content for better visualization
            display_content = (node.content[:50] + "...") if len(node.content) > 50 else node.content

            # Add node with stats
            label = f"{display_content}\n visits: {node.visits}\n value: {node.value}"
            dot.node(node_id, label)

            # Recursively add children
            for i, child in enumerate(node.children):
                child_id = f"{node_id}_{i}"
                add_nodes(child, child_id)
                dot.edge(node_id, child_id)

        add_nodes(self)

        # Render the graph
        try:
            dot.render("tree_of_thoughts", view=False, format="png", cleanup=True)
        except Exception as e:
            print(f"Error rendering graph: {e}")
            print("Make sure graphviz is installed on your system: https://graphviz.org/download/")


def extract_sft_dataset(root: ThinkNode) -> list[dict[str, Any]]:
    """Extract the best trajectory or multiple equally good trajectories for SFT training.

    Args:
        root (ThinkNonde): The root node of the tree.

    Returns:
        list[dict]: list of best trajectories, each one is a pair of instruction and response.
    """
    instruction = root.content
    idx = len("# Question: ") + len(root.content) + 1

    def _find_leaf_nodes(node: ThinkNode) -> list[ThinkNode]:
        """Recursively find all leaf nodes."""
        if not node.children:
            return [node]
        leafs = []
        for child in node.children:
            leafs.extend(_find_leaf_nodes(child))
        return leafs

    # Step 1: Find all leaf nodes
    leaf_nodes = _find_leaf_nodes(root)

    # Step 2: Determine the highest score among leaf nodes
    max_value = max(leaf_nodes, key=lambda x: x.value).value

    # Step 3: Collect all leaf nodes with the highest score
    best_leafs = [leaf for leaf in leaf_nodes if leaf.value == max_value]

    # Step 4: Collect trajectories for all the best leaf nodes
    best_trajectories = [{"instruction": instruction, "response": leaf.trajectory[idx:]} for leaf in best_leafs]

    return best_trajectories


def extract_rlhf_preference_dataset(root: ThinkNode, contrastive_threshold: float = 0.2) -> list[dict[str, Any]]:
    """Extract and generate preference pairs for RLHF training by comparing sibling nodes.

    Args:
        root (ThinkNode): The root node of the tree.
        contrastive_threshold (float): between (0, 1), a distance measure that we are confident to call
            one is positive and another is negative.

    Returns:
        list[dict]: list of preference pairs, where each pair contains two responses and
        indicates which one is preferred.
    """
    preference_pairs = []

    assert contrastive_threshold > 0
    assert contrastive_threshold < 1

    def traverse_tree(node: ThinkNode) -> None:
        """Traverse the tree to compare sibling nodes and collect preferences."""
        if not node.children:
            return  # Leaf node, no comparisons needed

        # Step 1: Compare all sibling nodes
        for i in range(len(node.children)):
            for j in range(len(node.children)):
                if i == j:
                    continue
                child_a, child_b = node.children[i], node.children[j]

                is_a_better = False
                if child_a.visits > 0 and child_b.visits > 0:
                    # for MCTS
                    is_a_better = (
                        child_a.value / child_a.visits - child_b.value / child_b.visits > contrastive_threshold
                    )
                else:
                    # for Beam Search
                    is_a_better = child_a.value - child_b.value > contrastive_threshold
                if is_a_better:
                    preference_pairs.append({
                        "instruction": node.trajectory,
                        "reflection": node.reflection,
                        "preferred_response": f"Step {child_a.depth}: {child_a.content}",
                        "dispreferred_response": f"Step {child_b.depth}: {child_b.content}",
                    })

        # Step 2: Recurse into child nodes
        for child in node.children:
            traverse_tree(child)

    # Start traversal from the root
    traverse_tree(root)

    return preference_pairs


@export_module("autogen.agents.experimental")
class ReasoningAgent(AssistantAgent):
    def __init__(
        self,
        name: str,
        llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = None,
        grader_llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = None,
        max_depth: int = 4,
        beam_size: int = 3,
        answer_approach: Literal["pool", "best"] = "pool",
        reason_config: Optional[dict[str, Any]] = None,
        code_execution_config: Union[dict[str, Any], Literal[False]] = False,
        scope: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a ReasoningAgent that uses tree-of-thought reasoning.

        Args:
            name (str): Name of the agent
            llm_config (Optional[Union[LLMConfig, dict[str, Any]]]): Configuration for the language model
            grader_llm_config (Optional[Union[LLMConfig, dict[str, Any]]]): Optional separate configuration for the grader model. If not provided, uses llm_config
            max_depth (int): Maximum depth of the reasoning tree
            beam_size (int): DEPRECATED. Number of parallel reasoning paths to maintain
            answer_approach (str): DEPRECATED. Either "pool" or "best" - how to generate final answer
            reason_config (Optional[dict[str, Any]]): Configuration for the reasoning method.
                method (str): The search strategy to use. Options:
                    - "beam_search" (default): Uses beam search with parallel paths
                    - "mcts": Uses Monte Carlo Tree Search for exploration
                    - "lats": Uses Language Agent Tree Search with per-step rewards
                    - "dfs": Uses depth-first search (equivalent to beam_search with beam_size=1)

                Common parameters:
                    max_depth (int): Maximum depth of reasoning tree (default: 3)
                    forest_size (int): Number of independent trees to maintain (default: 1)
                    rating_scale (int): Scale for grading responses, e.g. 1-10 (default: 10)
                    interim_execution (bool): Whether to execute the suggested options between the steps.

                Beam Search specific:
                    beam_size (int): Number of parallel paths to maintain (default: 3)
                    answer_approach (str): How to select final answer, "pool" or "best" (default: "pool")
                    batch_grading (bool): Whether to grade all options on each beam at once or one by one (default: False).

                MCTS/LATS specific:
                    nsim (int): Number of simulations to run (default: 3)
                    exploration_constant (float): UCT exploration parameter (default: 1.41)

                Example configs:
                    `{"method": "beam_search", "beam_size": 5, "max_depth": 4}`
                    `{"method": "mcts", "nsim": 10, "exploration_constant": 2.0}`
                    `{"method": "lats", "nsim": 5, "forest_size": 3}`
            code_execution_config (dict or False): config for the code execution.
                To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:
                - work_dir (Optional, str): The working directory for the code execution.
                    If None, a default working directory will be used.
                    The default working directory is the "extensions" directory under
                    "path_to_autogen".
                - use_docker (Optional, list, str or bool): The docker image to use for code execution.
                    Default is True, which means the code will be executed in a docker container. A default list of images will be used.
                    If a list or a str of image name(s) is provided, the code will be executed in a docker container
                    with the first image successfully pulled.
                    If False, the code will be executed in the current environment.
                    We strongly recommend using docker for code execution.
                - timeout (Optional, int): The maximum execution time in seconds.
                - last_n_messages (Experimental, int or str): The number of messages to look back for code execution.
                    If set to 'auto', it will scan backwards through all messages arriving since the agent last spoke, which is typically the last time execution was attempted. (Default: auto)
            scope (Optional[str]): The scope of the agent, includes information on how the agent should operate. It is appended to all system prompts of the internal agents. If None, no scope is added to the prompts.
            **kwargs (Any): Additional keyword arguments passed to parent class
        """
        reason_config = reason_config or {}
        if "verbose" in kwargs:
            warnings.warn(
                "The parameter `verbose` in ReasoningAgent has been deprecated. "
                "Please use the `silent` parameter as other AG2 agents.",
                DeprecationWarning,
            )
            kwargs["silent"] = not kwargs.pop("verbose")

        llm_config = LLMConfig.get_current_llm_config(llm_config)  # type: ignore[arg-type]
        self._scope = scope

        system_msg = kwargs.pop("system_message", REASONING_AGENT_MESSAGE)
        system_msg = self._add_scope(system_msg)

        super().__init__(
            name=name,
            system_message=system_msg,
            llm_config=llm_config,
            code_execution_config=code_execution_config,
            **kwargs,
        )
        self._llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = llm_config
        self._grader_llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = (
            grader_llm_config if grader_llm_config else llm_config
        )

        if max_depth != 4 or beam_size != 3 or answer_approach != "pool":
            warnings.warn(
                "The parameters max_depth, beam_size, and answer_approach have been deprecated. "
                "Please use the reason_config dictionary to configure these settings instead.",
                DeprecationWarning,
            )

        self._reason_config: dict[str, Any] = reason_config or {}
        self._method: Literal["beam_search", "mcts", "lats", "dfs"] = reason_config.get("method", "beam_search")
        if self._method not in ["beam_search", "mcts", "lats", "dfs"]:
            raise ValueError(
                f"Invalid reasoning method specified: '{self._method}'. Should be one of 'beam_search', 'mcts', 'lats', or 'dfs'."
            )

        self._beam_size: int = 1
        if self._method in ["beam_search", "dfs"]:
            if self._method != "dfs":
                self._beam_size = reason_config.get("beam_size", beam_size)
            self._answer_approach: Literal["pool", "best"] = reason_config.get("answer_approach", answer_approach)
            if self._answer_approach not in ["pool", "best"]:
                raise ValueError(
                    f"Invalid answer_approach specified: '{self._answer_approach}'. Should be one of 'pool' or 'best'."
                )
            self._batch_grading: bool = reason_config.get("batch_grading", False)
        elif self._method in ["mcts", "lats"]:
            self._nsim: int = reason_config.get("nsim", 3)
            self._exploration_constant: float = reason_config.get("exploration_constant", 1.41)

        self._max_depth: int = reason_config.get("max_depth", max_depth)
        self._forest_size: int = reason_config.get("forest_size", 1)
        self._rating_scale: int = reason_config.get("rating_scale", 10)
        self._interim_execution: bool = reason_config.get("interim_execution", False)

        self._root: Optional[ThinkNode] = None
        self._lats_context: str = ""
        self.register_reply([Agent, None], ReasoningAgent.generate_forest_response)

        # Initialize llm agent for interim step execution
        self._executor: Optional[AssistantAgent] = None
        # Add scope if provided
        executor_msg = self._add_scope(EXECUTOR_MESSAGE)
        if self._interim_execution:
            self._executor = AssistantAgent(
                name="tot_executor", system_message=executor_msg, llm_config=self._llm_config
            )

        tot_msg = self._add_scope(TREEOFTHOUGHT_MESSAGE)

        # Initialize user proxy agent for code execution
        self._user_proxy: Optional[UserProxyAgent] = None
        if self._code_execution_config:
            # to execute code interim_execution should be True
            if not self._interim_execution:
                raise ValueError(
                    "Code execution is enabled in the system, but interim_execution is set to False. "
                    "Please set interim_execution to True to allow code execution between reasoning steps."
                )

            self._user_proxy = UserProxyAgent(
                name="reasoner_user_proxy",
                human_input_mode="NEVER",
                code_execution_config=self._code_execution_config,
                max_consecutive_auto_reply=1,
            )

            self._code_execution_config = False  # code should only be executed by the user proxy
        else:
            # remove python instructions from the tot message
            tot_msg = "\n".join([
                line for line in tot_msg.split("\n") if not re.compile(r".*(python|```).*").search(line)
            ])

        # Initialize required agents
        self._thinker = AssistantAgent(name="tot_thinker", system_message=tot_msg, llm_config=self._llm_config)
        self._grader = AssistantAgent(name="tot_grader", llm_config=self._grader_llm_config)
        self._prompt_rewriter = AssistantAgent(name="prompt_rewriter", llm_config=self._llm_config)

    def _add_scope(self, system_prompt: str) -> str:
        """Add scope information to the system prompt.

        Args:
            system_prompt (str): The original system prompt.

        Returns:
            str: The modified system prompt with scope information.
        """
        if self._scope:
            return f"Task Scope: {self._scope}\n\n{system_prompt}"
        return system_prompt

    def generate_forest_response(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """Generate a response using tree-of-thought reasoning.

        Args:
            messages (Optional[list[dict[str, Any]]]): Input messages to respond to
            sender (Optional[Agent]): Agent sending the messages
            config (Optional[dict[str, Any]]): Optional configuration

        Returns:
            Tuple[bool, str]: Success flag and generated response
        """
        if sender == self:
            return False, ""  # Defer the LLM call to next reply functions.
        prompt, ground_truth = self._process_prompt(messages, sender)
        if not prompt:
            return True, "TERMINATE"

        forest_answers: list[str] = []
        for _ in range(self._forest_size):
            if self._method in ["beam_search", "dfs"]:
                response = self._beam_reply(prompt, ground_truth)
            elif self._method in ["mcts", "lats"]:
                response = self._mcts_reply(prompt, ground_truth)
            else:
                raise ValueError("Invalid reasoning method specified.")

            forest_answers.append(response)

        if len(forest_answers) == 1:
            return True, forest_answers[0]
        else:
            forest_answers_str = "-" + "\n-".join(forest_answers)
            self.send(
                message=f"""Given a list of different answers provide a complete response to a user's question.
Question:
{prompt}

Answers:
{forest_answers_str}

Final Answer:
""",
                recipient=self,
                request_reply=True,
                silent=self.silent,
            )
            last_msg: Optional[dict[str, Any]] = self.last_message(self)
            if last_msg is None:
                return True, ""
            return True, last_msg["content"].strip()

    def rate_node(self, node: ThinkNode, ground_truth: Optional[str] = None, is_outcome: bool = False) -> float:
        """Rate the quality of a reasoning path or the final answer using the grader agent.

        Args:
            node (ThinkNode): Node containing the reasoning trajectory to evaluate
            ground_truth (Optional[str]): Optional ground truth to provide to the grader
            is_outcome (bool): indicates whether the rating is for an outcome (final answer) or a process (thinking trajectory).

        Returns:
            float: Normalized score between 0 and 1 indicating trajectory quality
        """
        if node.value > 0 and node.rating_details:
            # we already calculated the rating for the node
            return node.value

        # Update Grader's system message
        if is_outcome:
            # Outcome Rating
            message = f"""Please rate the answer on a scale of 1 to {self._rating_scale}, where 1 is the worst and {self._rating_scale} is the best.

A great answer must:
- Directly address the original question
- Be factually accurate and complete
- Show clear logical reasoning

Additionally, a good answer should:
- Be concise and well-structured
- Use appropriate language and tone
- Provide relevant examples or evidence when needed
- Be free of contradictions or inconsistencies

If the answer fails to meet any of the core requirements above, it should be considered a poor response.

Also, rate poory (with 1) trajectories that:
- Require access to internet, experts opinions or external sources.
- Require research, hypotheses or data that are not provided.
- Include solutions in the physical world, like conducting experiments or surveys (code execution is fine).

Please provide your rating along with a brief explanation of your assessment.
"""
        else:
            # Process Rating
            message = f"""Please rate the thinking trajectory on a scale of 1 to {self._rating_scale}, where 1 is the worst and {self._rating_scale} is the best.

A great thinking trajectory must:
- Advance the process of solving the problem.

Additionally, a good trajectory should:
- Be appropriate in conversation.
- Contain no inaccuracies.
- Be free of any odd or irrelevant content.

If the trajectory does not meet one of the above requirements, it is considered a bad response.

Also, rate poory (with 1) trajectories that:
- Require access to internet, experts opinions or external sources.
- Require research, hypotheses or data that are not provided.
- Include solutions in the physical world, like conducting experiments or surveys (code execution is fine).

Please provide your rating along with a brief explanation of your assessment.
"""
        # Add ground truth to the message.
        if ground_truth:
            # override the system message
            message += f"--- Note that the Ground Truth is ---\n{ground_truth}\n---\n"

        # add scope if provided
        message = self._add_scope(message)

        self._grader.update_system_message(message)

        if self._method == "lats":
            prompt = self._lats_context + "\n\n---\n\n" + f"Rate:\n{node.trajectory}"
        else:
            prompt = f"Rate:\n{node.trajectory}"

        self._grader.clear_history()
        self.send(
            message=prompt,
            recipient=self._grader,
            request_reply=True,
            silent=self.silent,
        )
        rating: str = ""
        last_message: Optional[dict[str, Any]] = self._grader.last_message()
        if last_message is not None:
            rating = last_message["content"].strip()
        node.rating_details = rating

        try:
            # Scale rating to [0, 1]
            reward = (float(re.findall(r"[\d.]+", rating)[0]) - 1.0) / (self._rating_scale - 1.0)
        except (IndexError, ValueError):
            reward = 0.0  # Default reward if parsing fails
        return reward

    def rate_batch_nodes(self, nodes: list[ThinkNode], ground_truth: Optional[str] = None) -> list[float]:
        """Rate a batch of nodes using a single call of the grader agent. All the nodes must have the same parent.

        This method evaluates all given nodes while considering the other available options.

        Args:
            nodes (list[ThinkNode]): List of nodes to rate, all nodes must have the same parent
            ground_truth (Optional[str]): Optional ground truth to provide to the grader
        """
        # Assert that all nodes have the same parent and it is not None
        assert all(node.parent == nodes[0].parent for node in nodes), "All nodes must have the same parent."
        assert nodes[0].parent is not None, "Parent node must not be None."

        # Update Grader's system message
        message = f"""You will be provided a thinking trajectory and a list of options for the next step.
Please rate the thinking trajectory created by each option on a scale of 1 to {self._rating_scale}, where 1 is the worst and {self._rating_scale} is the best.

A great thinking trajectory must:
- Advance the process of solving the problem.

Additionally, a good trajectory should:
- Be appropriate in conversation.
- Contain no inaccuracies.
- Be free of any odd or irrelevant content.

If the trajectory does not meet one of the above requirements, it is considered a bad response.

Also, rate poorly (with 1) trajectories that:
- Require access to internet, experts opinions or external sources.
- Require research, hypotheses or data that are not provided.
- Include solutions in the physical world, like conducting experiments or surveys (code execution is fine).

Please provide your rating along with a brief explanation of your assessment.

**Output Format:**
Option 1: <your explanation here for the trajectory>
Rating: <rating>

Option 2: <your explanation here for the trajectory>
Rating: <rating>
...
"""
        # Add ground truth to the message.
        if ground_truth:
            # override the system message
            message += f"--- Note that the Ground Truth is ---\n{ground_truth}\n---\n"

        # add scope if provided
        message = self._add_scope(message)

        self._grader.update_system_message(message)

        # add lats context if necessary
        prompt = f"{self._lats_context}\n\n---\n\n" if self._method == "lats" else ""

        # add current trajectory
        prompt += f"Trajectory:\n{nodes[0].parent.trajectory}\n\n---\n\nOptions:\n"

        # add options
        for i, node in enumerate(nodes):
            prompt += f"\nOption {i + 1}:\n{node.content}"

        self._grader.clear_history()
        self.send(
            message=prompt,
            recipient=self._grader,
            request_reply=True,
            silent=self.silent,
        )
        rating: str = ""
        last_message: Optional[dict[str, Any]] = self._grader.last_message()
        if last_message is not None:
            rating = last_message["content"].strip()

        # Extract ratings and details for each option
        options_with_ratings = re.findall(r"(Option \d+:.*?Rating:\s*[\d.]+)", rating, re.DOTALL)
        ratings = []
        for option in options_with_ratings:
            match = re.search(r"Rating:\s*([\d.]+)", option)
            if match:
                ratings.append(match.group(1))

        # if the response wasn't of the expected format, return default rewards
        if len(ratings) != len(nodes):
            return [0.0] * len(nodes)

        rewards = []
        # Get rewards and assign rating details to corresponding nodes
        for node, rating, details in zip(nodes, ratings, options_with_ratings):
            if node.value > 0 and node.rating_details:
                # we already calculated the rating for the node
                rewards.append(node.value)
                continue
            node.rating_details = details
            rewards.append((float(rating) - 1.0) / (self._rating_scale - 1.0))
        return rewards

    def execute_node(self, node: ThinkNode) -> Optional[str]:
        """Execute the node's content to get the response.

        This method runs the node's content to get the response.
        If the content contains a Python code snippet, it sends the code to the user proxy agent for execution.
        Else, it sends the content to the LLM for generating the response.

        Args:
            node (ThinkNode): The node to run.

        Returns:
            Optional[str]: The response generated by the node, or None if the node is TERMINATE.
        """
        assert isinstance(self._executor, AssistantAgent)

        if node.output is not None:
            return node.output

        if "TERMINATE" in node.content:  # don't use _is_terminal, as the terminal node for beam search can be executed
            return None

        # check for python snippet
        if "```python" in node.content:
            # if code execution is disabled, ask to follow a different approach
            if not self._user_proxy:
                return "Python code execution is disabled. Follow a different approach."
            self._user_proxy.clear_history()
            self.send(
                message=node.content,
                recipient=self._user_proxy,
                request_reply=True,
                silent=self.silent,
            )
            user_proxy_last_msg: Optional[dict[str, Any]] = self._user_proxy.last_message(self)
            user_proxy_last_msg_content: str = user_proxy_last_msg["content"] if user_proxy_last_msg is not None else ""
            return user_proxy_last_msg_content

        # run with the LLM
        prompt = f"{self._lats_context}\n\n---\n\n" if self._method == "lats" else ""
        prompt += f"Trajectory:\n{node.trajectory}\nOutput:"

        self._executor.clear_history()
        self.send(
            message=prompt,
            recipient=self._executor,
            request_reply=True,
            silent=self.silent,
        )

        output = ""
        last_message: Optional[dict[str, Any]] = self._executor.last_message()

        # this agent is not supposed to write Python code, so if there is a need for that ask the thinker to do so
        if last_message is not None:
            if "```python" in last_message["content"]:
                output = (
                    "To execute Python code please provide the exact snippet in a fenced block like ```python ... ```."
                )
            else:
                output = last_message["content"].strip()

        return output

    def _process_prompt(
        self, messages: Optional[list[dict[str, Any]]], sender: Optional[Agent]
    ) -> tuple[Optional[str], Optional[str]]:
        """Process the incoming messages to extract the prompt and ground truth.

        This method checks if the provided messages are None and identifies the prompt.
        If there is only one message, it uses that as the prompt. Otherwise, it asks the question in the messages including also the important information from the previous messages.
        It also looks for a specific keyword "GROUND_TRUTH" in any of the messages to separate the ground truth for evaluation purposes.

        Args:
            messages (Optional[list[dict[str, Any]]]): A list of message dictionaries containing the content to process.
            sender (Optional[Agent]): The agent sending the messages.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing the processed prompt and the ground truth.
            If the prompt is empty, returns (None, None).
        """
        messages = self._oai_messages[sender] if messages is None else messages
        messages_copy = copy.deepcopy(messages)

        # Extract the ground truth for more accurate evaluation.
        # TODO: in the future, allow user to pass a callable (func) to calculate reward.
        ground_truth = None
        for i, message in enumerate(messages_copy):
            if "GROUND_TRUTH" in message["content"]:
                idx = message["content"].find("GROUND_TRUTH")
                messages_copy[i]["content"], ground_truth = message["content"][:idx].rstrip(), message["content"][idx:]
                break

        if len(messages) == 1:
            # First message, no previous context
            prompt = messages_copy[0]["content"]
        else:
            rewriter_message = f"""
Task: Given a list of messages including a previous discussion, write a prompt that summarizes the discussion, including all the useful information, and asks a question.

**Messages:**
{messages_copy}

**Format of Output:**
QUESTION: *Write the initial question asked by the user here.*
SUMMARY: *summarize the existing discussions.*

ACTIVITY LOG:
- *Action 1 performed*
- *Action 2 performed*
- ...

CURRENT_QUESTION: *Write the current/last question to be addressed here. In case the task has been completed, write: "The task has now been completed, write the final response and terminate the task."*
"""
            self._prompt_rewriter.clear_history()
            self.send(
                message=rewriter_message,
                recipient=self._prompt_rewriter,
                request_reply=True,
                silent=self.silent,
            )
            last_msg: Optional[dict[str, Any]] = self._prompt_rewriter.last_message()
            prompt = last_msg["content"].strip() if last_msg is not None else ""

        if not prompt:
            return None, None

        return prompt, ground_truth

    def _beam_reply(self, prompt: str, ground_truth: Optional[str] = None) -> str:
        """Generate a response using tree-of-thought reasoning.

        Implements beam search through a tree of reasoning steps, using the thinker
        agent to generate possible next steps and the grader agent to evaluate paths.

        Args:
            prompt (str): The question or prompt to generate a response for.
            ground_truth (Optional[str]): The ground truth or correct answer for evaluation.

        Returns:
            str: The generated response based on the reasoning process.
        """
        root = ThinkNode(content=prompt, parent=None)
        self._root = root  # save the root node for later visualization
        prev_leafs: list[ThinkNode] = [root]
        final_answers: set[ThinkNode] = set()  # store the final answers

        while prev_leafs and len(final_answers) < self._beam_size:
            new_leafs: list[ThinkNode] = []
            new_leafs_per_beam: list[list[ThinkNode]] = []  # used for batch grading
            for node in prev_leafs:
                if self._is_terminal(node):
                    # Reached max depth; collect possible answers
                    if node.value is None:
                        node.value = self.rate_node(node, ground_truth)
                    final_answers.add(node)
                    continue

                expansion_leafs = self._expand(node)
                new_leafs += expansion_leafs
                new_leafs_per_beam.append(expansion_leafs)

            prev_leafs = new_leafs

            if len(prev_leafs) + len(final_answers) > self._beam_size:
                if len(final_answers) >= self._beam_size:
                    prev_leafs = []  # stop searching, max beam size reached
                    break

                # Rate
                if self._batch_grading:
                    for beam_nodes in new_leafs_per_beam:
                        rewards = self.rate_batch_nodes(beam_nodes, ground_truth)
                        for node, reward in zip(beam_nodes, rewards):
                            node.value = reward
                else:
                    for node in prev_leafs:
                        node.value = self.rate_node(node, ground_truth)
                # Beam search: keep top beam_size leaf nodes
                prev_leafs = sorted(prev_leafs, key=lambda x: x.value if x.value else 0, reverse=True)[
                    : self._beam_size - len(final_answers)
                ]

                # Execute
                if self._interim_execution:
                    for node in prev_leafs:
                        node.output = self.execute_node(node)

        assert final_answers, "No final answers found."
        final_answers_list = list(final_answers)

        if self._answer_approach == "best":
            # Best the final answers
            best_leaf = max(final_answers_list, key=lambda x: x.value)
            message = f"""Given a thinking process, you have to provide a complete response to a user's question.
Question:
{prompt}

Thinking process:
{best_leaf.trajectory}

Final Answer:
"""
        elif self._answer_approach == "pool":
            all_thoughts = "\n\n".join([
                f"--- Possibility {i + 1} ---\n{node.trajectory}\n" for i, node in enumerate(final_answers_list)
            ])
            message = f"""Given a list of thinking processes, you have to provide a complete response to a user's question.
Question:
{prompt}

Thinking processes:
{all_thoughts}

Final Answer:
"""
        self.send(
            message=message,
            recipient=self,
            request_reply=True,
            silent=self.silent,
        )
        last_msg: Optional[dict[str, Any]] = self.last_message(self)
        final_answer: str = last_msg["content"].strip() if last_msg is not None else ""
        return final_answer

    def _mcts_reply(self, prompt: str, ground_truth: Optional[str] = None) -> str:
        """Generate a response using Monte Carlo Tree Search (MCTS) reasoning.

        Args:
            prompt (str): The question or prompt to generate a response for.
            ground_truth (Optional[str]): The ground truth or correct answer for evaluation.

        Returns:
            str: The generated response based on the reasoning process.
        """
        root = ThinkNode(content=prompt, parent=None)
        self._root = root
        answer_nodes: list[ThinkNode] = []

        self._lats_context = "## Here are some previous trajectories and reflections\n\n"  # Store LATS's reflections

        # TODO: future, parallelism with Swarm agent or AsyncOpenAI client.
        for _ in range(self._nsim):
            node = root

            # Selection
            while not self._is_terminal(node) and len(node.children) > 0:
                choices_weights = [
                    (child.value / (child.visits + EPSILON))
                    + self._exploration_constant
                    * math.sqrt(2 * math.log(node.visits + EPSILON) / (child.visits + EPSILON))
                    for child in node.children
                ]
                node = node.children[choices_weights.index(max(choices_weights))]

                # Execution
                if self._interim_execution:
                    node.output = self.execute_node(node)

            # Expansion and Simulation
            while not self._is_terminal(node):
                if len(node.children) == 0:
                    self._expand(node)
                if len(node.children) == 0:
                    node.content += "\nTERMINATE"
                    break
                node = random.choice(node.children)

                # Execution
                if self._interim_execution:
                    node.output = self.execute_node(node)

            # Add answer (leaf) node and evaluate answer
            self.send(
                message=f"""Given a thinking process, you have to provide a complete response to a user's question.
Question:
{prompt}

Thinking process:
{node.trajectory}

Final Answer:
""",
                recipient=self,
                request_reply=True,
                silent=self.silent,
            )
            last_msg: Optional[dict[str, Any]] = self.last_message(self)
            _answer: str = last_msg["content"].strip() if last_msg is not None else ""
            _ans_node = ThinkNode(content=_answer, parent=node)
            reward = self.rate_node(_ans_node, ground_truth, is_outcome=True)
            _ans_node.value = reward
            answer_nodes.append(_ans_node)
            self._lats_context += f"### Previous Tries:\n{node.trajectory}\n\nRating:{_ans_node.rating_details}\n\n"
            node.backpropagate(reward)

        best_ans_node = max(answer_nodes, key=lambda node: node.value)
        return best_ans_node.content

    def _expand(self, node: ThinkNode) -> list[ThinkNode]:
        """Expand the node by generating possible next steps based on the current trajectory.

        This method sends a message to the thinker agent, asking for possible next steps
        that can be taken from the current node's trajectory. It processes the response to
        extract the options provided by the thinker and creates new ThinkNode instances
        for each option.

        Args:
            node (ThinkNode): The node to expand, representing the current state in the reasoning process.

        Returns:
            list[ThinkNode]: A list of new ThinkNode instances created from the options provided by the thinker.
        """
        self._thinker.clear_history()

        if self._method == "lats":
            prompt = (
                self._lats_context
                + "\n\n---\n\n"
                + f"{node.trajectory}\n---\nHow should the thinking process continue?"
            )
        else:
            prompt = f"{node.trajectory}\n---\nHow should the thinking process continue?"

        self.send(
            message=prompt,
            recipient=self._thinker,
            request_reply=True,
            silent=self.silent,
        )
        last_msg: Optional[dict[str, Any]] = self._thinker.last_message()
        reply: str = last_msg["content"].strip() if last_msg is not None else ""
        reflection = re.findall(r"REFLECTION:\s*(.+?)(?=\*\*Possible Options:\*\*|Option \d+:|$)", reply, re.DOTALL)
        if reflection:
            node.reflection += str(reflection[0].strip())
        options = re.findall(r"Option \d+:(.+?)(?=Option \d+:|$)", reply, re.DOTALL)

        option_nodes = [ThinkNode(content=option.strip().rstrip(), parent=node) for option in options]

        return option_nodes

    def _is_terminal(self, node: ThinkNode) -> bool:
        """Check if the node is a terminal state in the reasoning process.

        Args:
            node (ThinkNode): The node to check for terminal state.

        Returns:
            bool: True if the node is terminal, False otherwise.
        """
        return node.depth >= self._max_depth or "TERMINATE" in node.content

    @property
    def method(self) -> str:
        """Get the reasoning method being used.

        Returns:
            str: The name of the reasoning method
        """
        return self._method

    def visualize_tree(self) -> None:
        """Visualize the tree of thoughts using graphviz.

        Raises:
            RuntimeError: If the tree has not been generated yet.
        """
        if self._root:
            self._root.visualize_tree()
        else:
            raise RuntimeError("No tree to visualize. Run the reasoning process first.")

    def extract_sft_dataset(self) -> list[dict[str, Any]]:
        """Extract the best trajectory or multiple equally good trajectories for SFT training.

        Returns:
            list[dict]: list of best trajectories, each one is a pair of instruction and response.

        Raises:
            RuntimeError: If the tree has not been generated yet.
        """
        if self._root:
            return extract_sft_dataset(self._root)
        else:
            raise RuntimeError("No tree to extract dataset from. Run the reasoning process first.")

    def extract_rlhf_preference_dataset(self, contrastive_threshold: float = 0.2) -> list[dict[str, Any]]:
        """Extract and generate preference pairs for RLHF training by comparing sibling nodes.

        Args:
            contrastive_threshold (float): between (0, 1), a distance measure that we are confident to call
                one is positive and another is negative.

        Returns:
            list[dict]: list of preference pairs, where each pair contains two responses and
            indicates which one is preferred.

        Raises:
            RuntimeError: If the tree has not been generated yet.
        """
        if self._root:
            return extract_rlhf_preference_dataset(self._root, contrastive_threshold)
        else:
            raise RuntimeError("No tree to extract dataset from. Run the reasoning process first.")
