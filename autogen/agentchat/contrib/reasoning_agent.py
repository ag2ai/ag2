# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import math
import random
import re
import warnings
from typing import Optional

from ...doc_utils import export_module
from ...import_utils import optional_import_block
from ..agent import Agent
from ..assistant_agent import AssistantAgent

EPSILON = 1e-6

TreeofThought_message = """
Role: Expert Planning AI Assistant

Task: Given a question and a list of previous steps (the plan trajectory), generate at least four innovative options for the next step. The user would not answer you anything.

Instructions:
- Review the user's question and the previous steps taken.
- Identify any mistakes or errors in the previous steps.
- If you find any mistakes, include options to correct them in your proposed options.
- Think creatively to propose at least four possible options that logically build upon or correct the previous steps.
- Reply a single word 'TERMINATE' as an option if you believe the user's question is fully resolved.
- Provide a brief description for each option.
- Present your output in the specified format.
- If the question is a multi-choice question, you should carefully eliminate obviously wrong choices, look for contextual clues in the question, and use logical reasoning to select the most plausible answer.

---

**Format of Output:**

REFLECTION:
*Give a few sentence reflections on the previous steps, what are wrong and what are good.*

**Possible Options:**
Option 1: Correct the error X in the previous steps.
Option 2: Reiterate and understand the user's question.
Option 3: Analyze and validate the results based on the previous steps.
Option 4: Perform Y.
"""


@export_module("autogen")
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
            value (Optional[float]): A numeric score/value assigned to this node.
            parent (Optional[ThinkNode]): Reference to the parent node.
            reflection (str): A string containing reflections on the reasoning process.
            rating_details (str): A string providing details about the rating of this node.
            depth (int): The depth of this node in the tree (root = 0).
            children (List[ThinkNode]): List of child nodes.
            visits (int): Number of times this node has been visited during search.

        The node automatically maintains the tree structure by:
        - Setting its depth based on the parent's depth + 1.
        - Adding itself to the parent's children list if the parent exists.
        - Providing trajectory utilities to get the full path from root to this node.
        """
        self.content = content
        self.value = 0
        self.parent = parent
        self.reflection = ""
        self.rating_details = ""
        self.depth = self.parent.depth + 1 if parent else 0
        self.children = []
        self.visits = 0
        if self.parent:
            self.parent.children.append(self)

    @property
    def _trajectory_arr(self) -> list[str]:
        """Gets the full path from root to this node as a list of strings.

        Returns:
            List[str]: List containing the content of each node from root to current node
        """
        if self.parent:
            return self.parent._trajectory_arr + [self.content]
        return ["# Question:\n" + self.content + "\n---\n"]

    @property
    def trajectory(self) -> str:
        """Get a formatted string representation of the path from root to this node.

        Returns:
            str: A formatted string showing the question and each step in the reasoning process
        """
        traj = self._trajectory_arr
        ans = traj[0]
        for i, option in enumerate(traj[1:]):
            ans += f"\nStep {i + 1}: {option}"
        return ans

    def backpropagate(self, reward: float) -> None:
        """Update the score of this node and its parents using moving average.

        Args:
            reward (float): The reward to backpropagate up the tree.
        """
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def __str__(self) -> str:
        return f"{self.content} -> Depth: {self.depth} Value: {self.value} Visits: {self.visits}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        """Convert ThinkNode to dictionary representation.

        Returns:
            Dict: Dictionary containing all node attributes and recursive children
        """
        return {
            "content": self.content,
            "value": self.value,
            "depth": self.depth,
            "reflection": self.reflection,
            "rating_details": self.rating_details,
            "visits": self.visits,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict, parent: Optional["ThinkNode"] = None) -> "ThinkNode":
        """Create ThinkNode from dictionary representation.

        Args:
            data (Dict): Dictionary containing node data
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

        # Recursively create children
        for child_data in data["children"]:
            cls.from_dict(child_data, parent=node)

        return node


@export_module("autogen")
def visualize_tree(root: ThinkNode) -> None:
    """Visualize the tree of thoughts using graphviz.

    Args:
        root (ThinkNode): The root node of the tree.
    """
    with optional_import_block() as result:
        from graphviz import Digraph

    if not result.is_successful:
        print("Please install graphviz: pip install graphviz")
        return

    dot = Digraph(comment="Tree of Thoughts")
    dot.attr(rankdir="TB")  # Top to Bottom direction

    def add_nodes(node: ThinkNode, node_id: str = "0"):
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

    add_nodes(root)

    # Render the graph
    try:
        dot.render("tree_of_thoughts", view=False, format="png", cleanup=True)
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print("Make sure graphviz is installed on your system: https://graphviz.org/download/")


def extract_sft_dataset(root: ThinkNode) -> list[dict]:
    """Extract the best trajectory or multiple equally good trajectories for SFT training.

    Args:
        root (ThinkNonde): The root node of the tree.

    Returns:
        List[Dict]: List of best trajectories, each one is a pair of instruction and response.
    """
    instruction = root.content
    idx = len("# Question: ") + len(root.content) + 1

    def _find_leaf_nodes(node):
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


def extract_rlhf_preference_dataset(root: ThinkNode, contrastive_threshold: float = 0.2) -> list[dict]:
    """Extract and generate preference pairs for RLHF training by comparing sibling nodes.

    Args:
        root (ThinkNode): The root node of the tree.
        contrastive_threshold (float): between (0, 1), a distance measure that we are confident to call
            one is positive and another is negative.

    Returns:
        List[Dict]: List of preference pairs, where each pair contains two responses and
        indicates which one is preferred.
    """
    preference_pairs = []

    assert contrastive_threshold > 0
    assert contrastive_threshold < 1

    def traverse_tree(node) -> None:
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
                    preference_pairs.append(
                        {
                            "instruction": node.trajectory,
                            "reflection": node.reflection,
                            "preferred_response": f"Step {child_a.depth}: {child_a.content}",
                            "dispreferred_response": f"Step {child_b.depth}: {child_b.content}",
                        }
                    )

        # Step 2: Recurse into child nodes
        for child in node.children:
            traverse_tree(child)

    # Start traversal from the root
    traverse_tree(root)

    return preference_pairs


@export_module("autogen")
class ReasoningAgent(AssistantAgent):
    def __init__(
        self,
        name: str,
        llm_config: dict,
        grader_llm_config: Optional[dict] = None,
        max_depth: int = 4,
        beam_size: int = 3,
        answer_approach: str = "pool",
        verbose: bool = True,
        reason_config: dict = {},
        **kwargs,
    ) -> None:
        """Initialize a ReasoningAgent that uses tree-of-thought reasoning.

        Args:
            name (str): Name of the agent
            llm_config(dict): Configuration for the language model
            grader_llm_config(Optional[dict]): Optional separate configuration for the grader model. If not provided, uses llm_config
            max_depth (int): Maximum depth of the reasoning tree
            beam_size (int): DEPRECATED. Number of parallel reasoning paths to maintain
            answer_approach (str): DEPRECATED. Either "pool" or "best" - how to generate final answer
            verbose (bool): Whether to show intermediate steps

            reason_config (dict): Configuration for the reasoning method. Supported parameters:
                method (str): The search strategy to use. Options:
                    - "beam_search" (default): Uses beam search with parallel paths
                    - "mcts": Uses Monte Carlo Tree Search for exploration
                    - "lats": Uses Language Agent Tree Search with per-step rewards
                    - "dfs": Uses depth-first search (equivalent to beam_search with beam_size=1)

                Common parameters:
                    max_depth (int): Maximum depth of reasoning tree (default: 3)
                    forest_size (int): Number of independent trees to maintain (default: 1)
                    rating_scale (int): Scale for grading responses, e.g. 1-10 (default: 10)

                Beam Search specific:
                    beam_size (int): Number of parallel paths to maintain (default: 3)
                    answer_approach (str): How to select final answer, "pool" or "best" (default: "pool")

                MCTS/LATS specific:
                    nsim (int): Number of simulations to run (default: 3)
                    exploration_constant (float): UCT exploration parameter (default: 1.41)

                Example configs:
                    `{"method": "beam_search", "beam_size": 5, "max_depth": 4}`
                    `{"method": "mcts", "nsim": 10, "exploration_constant": 2.0}`
                    `{"method": "lats", "nsim": 5, "forest_size": 3}`
        """
        super().__init__(name=name, llm_config=llm_config, **kwargs)
        self._verbose = verbose
        self._llm_config = llm_config
        self._grader_llm_config = grader_llm_config if grader_llm_config else llm_config

        if max_depth != 4 or beam_size != 3 or answer_approach != "pool":
            # deprecate warning
            warnings.warn(
                "The parameters max_depth, beam_size, and answer_approach have been deprecated. "
                "Please use the reason_config dictionary to configure these settings instead.",
                DeprecationWarning,
            )

        if reason_config is None:
            reason_config = {}
        self._reason_config = reason_config

        self._method = reason_config.get("method", "beam_search")
        if self._method in ["beam_search", "dfs"]:
            if self._method == "dfs":
                self._beam_size = 1
            else:
                self._beam_size = reason_config.get("beam_size", beam_size)
            self._answer_approach = reason_config.get("answer_approach", answer_approach)
            assert self._answer_approach in ["pool", "best"]
        elif self._method in ["mcts", "lats"]:
            self._nsim = reason_config.get("nsim", 3)
            self._exploration_constant = reason_config.get("exploration_constant", 1.41)

        self._max_depth = reason_config.get("max_depth", max_depth)
        self._forest_size = reason_config.get("forest_size", 1)  # We default use only 1 tree.
        self._rating_scale = reason_config.get("rating_scale", 10)

        self._root = None
        self.register_reply([Agent, None], ReasoningAgent.generate_forest_response)

        self._thinker = AssistantAgent(
            name="tot_thinker", system_message=TreeofThought_message, llm_config=self._llm_config
        )
        self._grader = AssistantAgent(name="tot_grader", llm_config=self._grader_llm_config)

    def generate_forest_response(
        self, messages: list[dict], sender: Agent, config: Optional[dict] = None
    ) -> tuple[bool, str]:
        """Generate a response using tree-of-thought reasoning.

        Args:
            messages (List[Dict[str, Any]]): Input messages to respond to
            sender (Agent): Agent sending the messages
            config (Optional[Dict[str, Any]]): Optional configuration

        Returns:
            Tuple[bool, str]: Success flag and generated response
        """
        if sender == self:
            return False, ""  # Defer the LLM call to next reply functions.
        prompt, ground_truth = self._process_prompt(messages, sender)
        if not prompt:
            return True, "TERMINATE"

        forest_answers = []
        for _ in range(self._forest_size):
            if self._method in ["beam_search", "dfs"]:
                response = self._beam_reply(prompt, ground_truth)
            elif self._method in ["mcts", "lats"]:
                response = self._mtcs_reply(prompt, ground_truth)
            else:
                raise ValueError("Invalid reasoning method specified.")

            forest_answers.append(response)

        if len(forest_answers) == 1:
            return True, forest_answers[0]
        else:
            self.send(
                message=f"Answer the question {prompt}. Here are some students' different answers:\n{{'\n-'.join(forest_answers)}}",
                recipient=self,
                request_reply=True,
                silent=not self._verbose,
            )
            return True, self.last_message(self)["content"].strip()

    def rate_node(self, node: ThinkNode, ground_truth: str = None, is_outcome: bool = False) -> float:
        """Rate the quality of a reasoning path using the grader agent.

        Args:
            node (ThinkNode): Node containing the reasoning trajectory to evaluate
            ground_truth (str): Optional ground truth to provide to the grader
            is_outcome (bool): indicates whether the rating is for an outcome (final answer) or a process (thinking trajectory).

        Returns:
            float: Normalized score between 0 and 1 indicating trajectory quality
        """
        if node.value > 0 and node.rating_details:
            # we already calculated the rating for the node
            return node.value

        # Update Grader's system message
        if is_outcome:
            ## Outcome Rating
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

Please provide your rating along with a brief explanation of your assessment.
"""
        else:
            ## Process Rating
            message = f"""Please rate the thinking trajectory on a scale of 1 to {self._rating_scale}, where 1 is the worst and {self._rating_scale} is the best.

A great thinking trajectory must:
- Advance the process of solving the problem.

Additionally, a good trajectory should:
- Be appropriate in conversation.
- Contain no inaccuracies.
- Be free of any odd or irrelevant content.

If the trajectory does not meet one of the above requirements, it is considered a bad response.

Please provide your rating along with a brief explanation of your assessment.
"""
        ## Add ground truth to the message.
        if ground_truth:
            # override the system message
            message += f"--- Note that the Ground Truth is ---\n{ground_truth}\n---\n"
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
            silent=not self._verbose,
        )
        rating = self._grader.last_message()["content"].strip()
        node.rating_details = rating

        try:
            # Scale rating to [0, 1]
            reward = (float(re.findall(r"[\d.]+", rating)[0]) - 1.0) / (self._rating_scale - 1.0)
        except (IndexError, ValueError):
            reward = 0.0  # Default reward if parsing fails
        return reward

    def _process_prompt(self, messages: list[dict], sender: Agent) -> tuple[Optional[str], Optional[str]]:
        """Process the incoming messages to extract the prompt and ground truth.

        This method checks if the provided messages are None and retrieves the last message's content.
        It also looks for a specific keyword "GROUND_TRUTH" in the prompt to separate the main prompt
        from the ground truth for evaluation purposes.

        Args:
            messages (List[Dict[str, Any]]): A list of message dictionaries containing the content to process.
            sender (Agent): The agent sending the messages.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing the processed prompt and the ground truth.
            If the prompt is empty, returns (None, None).
        """
        messages = self._oai_messages[sender] if messages is None else messages
        prompt = messages[-1]["content"].strip()
        if not prompt:
            return None, None

        # Extract the ground truth for more accurate evaluation.
        # TODO: in the future, allow user to pass a callable (func) to calculate reward.
        if "GROUND_TRUTH" in prompt:
            idx = prompt.find("GROUND_TRUTH")
            prompt, ground_truth = prompt[:idx].rstrip(), prompt[idx:]
        else:
            ground_truth = None
        return prompt, ground_truth

    def _beam_reply(self, prompt: str, ground_truth: str = "") -> str:
        """Generate a response using tree-of-thought reasoning.

        Implements beam search through a tree of reasoning steps, using the thinker
        agent to generate possible next steps and the grader agent to evaluate paths.

        Args:
            prompt (str): The question or prompt to generate a response for.
            ground_truth (str): The ground truth or correct answer for evaluation.

        Returns:
            str: The generated response based on the reasoning process.
        """
        root = ThinkNode(content=prompt, parent=None)
        self._root = root  # save the root node for later visualization
        prev_leafs = [root]

        final_answers = set()  # store the final answers

        while prev_leafs and len(final_answers) < self._beam_size:
            new_leafs = []
            for node in prev_leafs:
                if self._is_terminal(node):
                    # Reached max depth; collect possible answers
                    if node.value is None:
                        node.value = self.rate_node(node, ground_truth)
                    final_answers.add(node)
                    continue

                new_leafs += self._expand(node)

            prev_leafs = new_leafs

            if len(prev_leafs) + len(final_answers) > self._beam_size:
                if len(final_answers) >= self._beam_size:
                    prev_leafs = []  # stop searching, max beam size reached
                    break

                # Rate
                for node in prev_leafs:
                    node.value = self.rate_node(node, ground_truth)
                # Beam search: keep top beam_size leaf nodes
                prev_leafs = sorted(prev_leafs, key=lambda x: x.value if x.value else 0, reverse=True)[
                    : self._beam_size - len(final_answers)
                ]

        assert final_answers, "No final answers found."
        final_answers = list(final_answers)

        if self._answer_approach == "best":
            # Best the final answers
            best_leaf = max(final_answers, key=lambda x: x.value)
            self.send(
                message=f"Answer the question {prompt}. Here is my thinking processes:\n{best_leaf.trajectory}",
                recipient=self,
                request_reply=True,
                silent=not self._verbose,
            )
        elif self._answer_approach == "pool":
            all_thoughts = "\n\n".join(
                [f"--- Possibility {i + 1} ---\n{node.trajectory}\n" for i, node in enumerate(final_answers)]
            )
            self.send(
                message=f"Answer the question {prompt}. You can utilize these students' thinking processes.\n\n{all_thoughts}",
                recipient=self,
                request_reply=True,
                silent=not self._verbose,
            )

        final_answer = self.chat_messages[self][-1]["content"].strip()
        return final_answer

    def _mtcs_reply(self, prompt: str, ground_truth: str = "") -> str:
        """Generate a response using Monte Carlo Tree Search (MCTS) reasoning.

        Args:
            prompt (str): The question or prompt to generate a response for.
            ground_truth (str): The ground truth or correct answer for evaluation.

        Returns:
            str: The generated response based on the reasoning process.
        """
        root = ThinkNode(content=prompt, parent=None)
        self._root = root
        answer_nodes = []

        self._lats_context = "## Here are some previous trajectories and reflections\n\n"  # Store LATS's reflections

        # TODO: future, parallelism with Swarm agent or AsyncOpenAI client.
        for _ in range(self._nsim):
            node = root

            # Selection
            while not self._is_terminal(node) and len(node.children) > 0:
                # TODO: In the original UCT formula, child.value represents the win ratio.
                # Here, we use the average rating rather than the win ratio.
                # The rating might be biased from the LLM, which could affect the bounds of this vanilla UCT equation.
                # More intensive analysis is needed in the future.
                choices_weights = [
                    # exploitation term +
                    (child.value / (child.visits + EPSILON))
                    +
                    # exploration term
                    self._exploration_constant
                    * math.sqrt(2 * math.log(node.visits + EPSILON) / (child.visits + EPSILON))
                    for child in node.children
                ]
                node = node.children[choices_weights.index(max(choices_weights))]

            # Expansion and Simulation
            while not self._is_terminal(node):
                if len(node.children) == 0:
                    self._expand(node)
                if len(node.children) == 0:
                    node.content += "\nTERMINATE"
                    break
                node = random.choice(node.children)

            # Add answer (leaf) node and evaluate answer
            self.send(
                message=f"Answer the question {prompt}. Here is my thinking process:\n{node.trajectory}",
                recipient=self,
                request_reply=True,
                silent=not self._verbose,
            )
            _answer = self.last_message(self)["content"].strip()
            # We add the answer (as a node) to the leaf to help
            # future logging and debugging.
            _ans_node = ThinkNode(content=_answer, parent=node)
            reward = self.rate_node(_ans_node, ground_truth, is_outcome=True)
            _ans_node.value = reward
            answer_nodes.append(_ans_node)
            self._lats_context += f"### Previous Tries:\n{node.trajectory}\n\nRating:{_ans_node.rating_details}\n\n"
            # Backpropagation
            node.backpropagate(reward)

        # Best action
        best_ans_node = max(answer_nodes, key=lambda node: node.value)
        return best_ans_node.content

    def _expand(self, node: ThinkNode) -> list:
        """Expand the node by generating possible next steps based on the current trajectory.

        This method sends a message to the thinker agent, asking for possible next steps
        that can be taken from the current node's trajectory. It processes the response to
        extract the options provided by the thinker and creates new ThinkNode instances
        for each option.

        Args:
            node (ThinkNode): The node to expand, representing the current state in the reasoning process.

        Returns:
            List[ThinkNode]: A list of new ThinkNode instances created from the options provided by the thinker.
        """
        self._thinker.clear_history()

        if self._method == "lats":
            prompt = self._lats_context + "\n\n---\n\n" + f"{node.trajectory}\n---\nWhat are the possible next steps?"
        else:
            prompt = f"{node.trajectory}\n---\nWhat are the possible next steps?"

        self.send(
            message=prompt,
            recipient=self._thinker,
            request_reply=True,
            silent=not self._verbose,
        )
        reply = self._thinker.last_message()["content"].strip()
        reflection = re.findall(r"REFLECTION:\s*(.+?)(?=\*\*Possible Options:\*\*|Option \d+:|$)", reply, re.DOTALL)
        if reflection:
            node.reflection += str(reflection[0].strip())
        # Extract options from reply using regex:
        # - Matches text between "Option N:" and either next "Option N:" or end of string
        # - (?=...) is a lookahead to match option boundary without including it
        # - re.DOTALL allows . to match newlines
        options = re.findall(r"Option \d+:(.+?)(?=Option \d+:|$)", reply, re.DOTALL)

        return [ThinkNode(content=option.strip().rstrip(), parent=node) for option in options]

    def _is_terminal(self, node: ThinkNode) -> bool:
        """Check if the node is a terminal state in the reasoning process.

        Args:
            node (ThinkNode): The node to check for terminal state.

        Returns:
            bool: True if the node is terminal, False otherwise.
        """
        return node.depth >= self._max_depth or "TERMINATE" in node.content

    @property
    def method(self):
        return self._method
