# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import math
import random
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

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

---

**Format of Output:**

**Reflection**
*Give a few sentence reflections on the previous steps, what are wrong and what are good.*

**Possible Options:**
Option 1: Correct the error X in the previous steps.
Option 2: Reiterate and understand the user's question.
Option 3: Analyze and validate the results based on the previous steps.
Option 4: Perform Y.
"""


GRADER_message = "Rate the response on a scale of 1 to 10 (1 being the worst and 10 being the best)."


class ThinkNode:

    def __init__(self, content: str, parent: Optional["ThinkNode"] = None) -> None:
        """A node in a tree structure representing a step in the reasoning process.

        This class implements a tree node that stores content (text describing a reasoning step),
        maintains parent-child relationships, tracks node statistics, and provides utilities
        for traversing/visualizing the reasoning path.

        Args:
            content (str): The text content/description for this reasoning step
            parent (Optional[ThinkNode]): The parent node in the tree, if any

        Attributes:
            content (str): The text content/description for this reasoning step
            value (Optional[float]): A numeric score/value assigned to this node
            parent (Optional[ThinkNode]): Reference to parent node
            depth (int): The depth of this node in the tree (root = 0)
            children (List[ThinkNode]): List of child nodes
            visits (int): Number of times this node has been visited during search

        The node automatically maintains the tree structure by:
        - Setting its depth based on parent's depth + 1
        - Adding itself to parent's children list if parent exists
        - Providing trajectory utilities to get the full path from root to this node
        """
        self.content = content
        self.value = 0
        self.parent = parent
        self.depth = self.parent.depth + 1 if parent else 0
        self.children = []
        self.visits = 0
        self._is_solved = "TERMINATE" in content
        if self._is_solved:
            self._mark_tree_as_solved()
        if self.parent:
            self.parent.children.append(self)

    @property
    def is_solved(self) -> bool:
        """If any solutions exist, we can end the search."""
        return self._is_solved

    def _mark_tree_as_solved(self):
        """Mark all parent nodes as solved when a solution is found."""
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents using moving average."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""

        def get_all_nodes(node):
            all_nodes = [node]
            for child in node.children:
                all_nodes.extend(get_all_nodes(child))
            return all_nodes

        all_nodes = get_all_nodes(self)
        best_node = max(
            all_nodes,
            # Filter out all non-terminal, non-solution trajectories
            key=lambda node: int(len(node.children) == 0 and node.is_solved)
            * (node.value if node.value is not None else 0),
        )
        return best_node

    @property
    def _trajectory_arr(self) -> List[str]:
        """Get the full path from root to this node as a list of strings.

        Returns:
            List[str]: List containing the content of each node from root to current node
        """
        if self.parent:
            return self.parent._trajectory_arr + [self.content]
        return ["# Question: " + self.content]

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

    def __str__(self) -> str:
        return f"{self.content} -> Depth: {self.depth} Value: {self.value} Visits: {self.visits}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict:
        """Convert ThinkNode to dictionary representation.

        Returns:
            Dict: Dictionary containing all node attributes and recursive children
        """
        return {
            "content": self.content,
            "value": self.value,
            "depth": self.depth,
            "visits": self.visits,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict, parent: Optional["ThinkNode"] = None) -> "ThinkNode":
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

        # Recursively create children
        for child_data in data["children"]:
            cls.from_dict(child_data, parent=node)

        return node


def visualize_tree(root: ThinkNode) -> None:
    """
    Visualize the tree of thoughts using graphviz.
    """
    try:
        from graphviz import Digraph
    except ImportError:
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


def extract_sft_dataset(root):
    """
    Extract the best trajectory or multiple equally good trajectories
    for SFT training.

    Args:
        root: The root node of the tree.

    Returns:
        List of best trajectories, where each trajectory is a pair of instruction and response.
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


def extract_rlhf_preference_dataset(root, contrastive_threshold=0.2):
    """
    Extract and generate preference pairs for RLHF training by comparing sibling nodes.

    Args:
        root: The root node of the tree.
        contrastive_threshold (float): between (0, 1), a distance measure that we are confidence to call
            one is positive and another is negative.

    Returns:
        A list of preference pairs, where each pair contains two responses and
        indicates which one is preferred.
    """
    preference_pairs = []

    assert contrastive_threshold > 0
    assert contrastive_threshold < 1

    def traverse_tree(node):
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


class ReasoningAgent(AssistantAgent):
    def __init__(
        self,
        name,
        llm_config,
        max_depth=4,
        beam_size=3,
        answer_approach="pool",
        verbose=True,
        reason_config: dict = None,
        **kwargs,
    ) -> None:
        """Initialize a ReasoningAgent that uses tree-of-thought reasoning.

        Args:
            name: Name of the agent
            llm_config: Configuration for the language model
            max_depth (int): Maximum depth of the reasoning tree
            beam_size (int): DEPRECATED. Number of parallel reasoning paths to maintain
            answer_approach (str): DEPRECATED. Either "pool" or "best" - how to generate final answer
            verbose (bool): Whether to show intermediate steps
            reason_config (dict): Configuration for the reasoning method, e.g.,
                {"method": "mcts"} or
                {"method": "beam_search", "beam_size": 3, "answer_approach": "pool"} or
                {"method": "lats", "max_iterations": 10, "num_candidates": 5}
        """
        super().__init__(name=name, llm_config=llm_config, **kwargs)
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.verbose = verbose
        assert answer_approach in ["pool", "best"]
        self.answer_approach = answer_approach
        self.thinker = AssistantAgent(name="tot_thinker", system_message=TreeofThought_message, llm_config=llm_config)
        self.grader = AssistantAgent(name="tot_grader", system_message=GRADER_message, llm_config=llm_config)

        if reason_config:
            method = reason_config.get("method", "beam_search")
            if method == "beam_search":
                self.register_reply([Agent, None], ReasoningAgent.generate_beam_response)
                if "beam_size" in reason_config:
                    self.beam_size = reason_config["beam_size"]
                if "answer_approach" in reason_config:
                    self.answer_approach = reason_config["answer_approach"]
            elif method == "mcts":
                self.register_reply([Agent, None], ReasoningAgent.generate_mcts_response)
                self.mcts_simulations = reason_config.get("nsim", 10)
                self.exploration_constant = reason_config.get("exploration_constant", 1.41)
            elif method == "lats":
                self.register_reply([Agent, None], ReasoningAgent.generate_lats_response)
                self.lats_max_iterations = reason_config.get("max_iterations", 5)
                self.lats_num_candidates = reason_config.get("num_candidates", 3)

        self._root = None

    def rate_node(self, node: ThinkNode, ground_truth: str = None) -> float:
        """Rate the quality of a reasoning path using the grader agent.

        Args:
            node (ThinkNode): Node containing the reasoning trajectory to evaluate

        Returns:
            float: Normalized score between 0 and 1 indicating trajectory quality
        """
        if ground_truth:
            # override the system message
            self.grader.update_system_message(
                f"Rate the response on a scale of 1 to 10 (1 being the worst and 10 being the best). Use the following as the evaluation criteria: Ground Truth is:\n{ground_truth}"
            )
        else:
            self.grader.update_system_message(GRADER_message)

        self.send(
            message=f"Rate:\n{node.trajectory}",
            recipient=self.grader,
            request_reply=True,
            silent=not self.verbose,
        )
        rating = self.grader.last_message()["content"].strip()
        try:
            # Scale rating to [0, 1]
            reward = (float(re.findall(r"[\d.]+", rating)[0]) - 1) / 4.0
        except (IndexError, ValueError):
            reward = 0.0  # Default reward if parsing fails
        return reward

    def _process_prompt(self, messages, sender):
        """
        Process the incoming messages to extract the prompt and ground truth.

        This method checks if the provided messages are None and retrieves the last message's content.
        It also looks for a specific keyword "GROUND_TRUTH" in the prompt to separate the main prompt
        from the ground truth for evaluation purposes.

        Args:
            messages (List[Dict[str, Any]]): A list of message dictionaries containing the content to process.

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

    def generate_beam_response(self, messages, sender, config=None):
        """Generate a response using tree-of-thought reasoning.

        Implements beam search through a tree of reasoning steps, using the thinker
        agent to generate possible next steps and the grader agent to evaluate paths.

        Args:
            messages: Input messages to respond to
            sender: Agent sending the messages
            config: Optional configuration

        Returns:
            Tuple[bool, str]: Success flag and generated response
        """
        if sender == self:
            return False, ""  # Defer the LLM call to next reply functions.
        prompt, ground_truth = self._process_prompt(messages, sender)
        if not prompt:
            return True, "TERMINATE"

        root = ThinkNode(content=prompt, parent=None)
        self._root = root  # save the root node for later visualization
        prev_leafs = [root]

        final_answers = set()  # store the final answers

        while prev_leafs and len(final_answers) < self.beam_size:
            new_leafs = []
            for node in prev_leafs:
                if self.is_terminal(node):
                    # Reached max depth; collect possible answers
                    if node.value is None:
                        node.value = self.rate_node(node, ground_truth)
                    final_answers.add(node)
                    continue

                new_leafs += self.expand(node)

            prev_leafs = new_leafs

            if len(prev_leafs) + len(final_answers) > self.beam_size:
                if len(final_answers) >= self.beam_size:
                    prev_leafs = []  # stop searching, max beam size reached
                    break

                # Rate
                for node in prev_leafs:
                    node.value = self.rate_node(node, ground_truth)
                # Beam search: keep top beam_size leaf nodes
                prev_leafs = sorted(prev_leafs, key=lambda x: x.value if x.value else 0, reverse=True)[
                    : self.beam_size - len(final_answers)
                ]

        assert final_answers, "No final answers found."
        final_answers = list(final_answers)

        if self.answer_approach == "best":
            # Best the final answers
            best_leaf = max(final_answers, key=lambda x: x.value)
            self.send(
                message=f"Answer the question {prompt}. Here is my thinking processes:\n{best_leaf.trajectory}",
                recipient=self,
                request_reply=True,
                silent=not self.verbose,
            )
        elif self.answer_approach == "pool":
            all_thoughts = "\n\n".join(
                [f"--- Possibility {i+1} ---\n{node.trajectory}\n" for i, node in enumerate(final_answers)]
            )
            self.send(
                message=f"Answer the question {prompt}. You can utilize these students' thinking processes.\n\n{all_thoughts}",
                recipient=self,
                request_reply=True,
                silent=not self.verbose,
            )

        final_answer = self.chat_messages[self][-1]["content"].strip()
        return True, final_answer

    def generate_mcts_response(self, messages, sender, config=None):
        if sender == self:
            return False, ""  # Defer the LLM call to next reply functions.
        prompt, ground_truth = self._process_prompt(messages, sender)
        if not prompt:
            return True, "TERMINATE"

        root = ThinkNode(content=prompt, parent=None)
        self._root = root
        answer_nodes = []

        # TODO: future, parallelism with Swarm agent or AsyncOpenAI client.
        for _ in range(self.mcts_simulations):
            node = root

            # Selection
            while not self.is_terminal(node) and len(node.children) > 0:
                choices_weights = [
                    # exploitation term +
                    (child.value / (child.visits + EPSILON)) +
                    # exploration term
                    self.exploration_constant
                    * math.sqrt((2 * math.log(node.visits + EPSILON) / (child.visits + EPSILON)))
                    for child in node.children
                ]
                node = node.children[choices_weights.index(max(choices_weights))]

            # Expansion and Simulation
            while not self.is_terminal(node):
                if len(node.children) == 0:
                    self.expand(node)
                node = random.choice(node.children)

            # Add answer (leaf) node and evaluate answer
            self.send(
                message=f"Answer the question {prompt}. Here is my thinking process:\n{node.trajectory}",
                recipient=self,
                request_reply=True,
                silent=not self.verbose,
            )
            _answer = self.last_message(self)["content"].strip()
            # We add the answer (as a node) to the leaf to help
            # future logging and debugging.
            _ans_node = ThinkNode(content=_answer, parent=node)
            reward = self.rate_node(_ans_node, ground_truth)
            _ans_node.value = reward
            answer_nodes.append(_ans_node)

            # Backpropagation
            while node is not None:
                node.visits += 1
                if node.value is None:
                    node.value = reward
                else:
                    node.value += reward
                node = node.parent

        # Best action
        best_ans_node = max(answer_nodes, key=lambda node: node.value)
        return True, best_ans_node.content

    def expand(self, node: ThinkNode) -> List:
        """
        Expand the node by generating possible next steps based on the current trajectory.

        This method sends a message to the thinker agent, asking for possible next steps
        that can be taken from the current node's trajectory. It processes the response to
        extract the options provided by the thinker and creates new ThinkNode instances
        for each option.

        Args:
            node (ThinkNode): The node to expand, representing the current state in the reasoning process.

        Returns:
            List[ThinkNode]: A list of new ThinkNode instances created from the options provided by the thinker.
        """
        self.thinker.clear_history()
        self.send(
            message=f"{node.trajectory}\n---\nWhat are the possible next steps?",
            recipient=self.thinker,
            request_reply=True,
            silent=not self.verbose,
        )
        reply = self.thinker.last_message()["content"].strip()

        # Extract options from reply using regex:
        # - Matches text between "Option N:" and either next "Option N:" or end of string
        # - (?=...) is a lookahead to match option boundary without including it
        # - re.DOTALL allows . to match newlines
        options = re.findall(r"Option \d+:(.+?)(?=Option \d+:|$)", reply, re.DOTALL)

        return [ThinkNode(content=option.strip().rstrip(), parent=node) for option in options]

    def is_terminal(self, node):
        return node.depth >= self.max_depth or "TERMINATE" in node.content

    def generate_lats_response(self, messages, sender, config=None):
        """Generate a response using Language Agent Tree Search (LATS)."""
        if sender == self:
            return False, ""

        prompt, ground_truth = self._process_prompt(messages, sender)
        if not prompt:
            return True, "TERMINATE"

        # Initialize root node
        root = ThinkNode(content=prompt, parent=None)
        self._root = root

        # Helper function to determine if we should continue searching
        def should_continue(node, iteration):
            if self._root.is_solved():
                return False
            if iteration >= self.lats_max_iterations:
                return False
            if node.depth >= self.max_depth:
                return False
            return True

        # Main LATS loop
        iteration = 0
        while should_continue(root, iteration):
            # Selection - find best node to expand
            current = root
            while current.children and not self.is_terminal(current):
                # Use UCT formula similar to MCTS
                choices_weights = [
                    (child.value / (child.visits + EPSILON))
                    + 1.41 * math.sqrt(math.log(current.visits + EPSILON) / (child.visits + EPSILON))
                    for child in current.children
                ]
                current = current.children[choices_weights.index(max(choices_weights))]

            # Expansion - generate candidate next steps
            if not self.is_terminal(current):
                self.send(
                    message=f"{current.trajectory}\n---\nWhat are the possible next steps?",
                    recipient=self.thinker,
                    request_reply=True,
                    silent=not self.verbose,
                )
                # TODO: the candidate generation should be done different, refer: https://ag2ai.github.io/ag2/docs/notebooks/lats_search/#candidate-generation,
                # and im not sure how to approach, so for now we will just use the last message.
                candidates = re.findall(
                    r"Option \d+:(.+?)(?=Option \d+:|$)", self.thinker.last_message()["content"].strip(), re.DOTALL
                )

                for candidate in candidates[: self.lats_num_candidates]:
                    child = ThinkNode(content=candidate.strip(), parent=current)
                    # Evaluate candidate and backpropagate
                    reward = self.rate_node(child, ground_truth)
                    child.backpropagate(reward)

            iteration += 1

        # Find best leaf node by traversing tree
        def find_best_leaf(node):
            if not node.children:
                return node
            best_child = max(node.children, key=lambda x: x.value if x.value is not None else 0)
            return find_best_leaf(best_child)

        best_node = find_best_leaf(root)

        # Generate final answer using best trajectory
        self.send(
            message=f"Answer the question {prompt}. Here is my thinking process:\n{best_node.trajectory}",
            recipient=self,
            request_reply=True,
            silent=not self.verbose,
        )

        return True, self.last_message(self)["content"].strip()
