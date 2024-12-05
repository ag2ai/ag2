# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from ..agent import Agent
from ..assistant_agent import AssistantAgent

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
        self.value = None
        self.parent = parent
        self.depth = self.parent.depth + 1 if parent else 0
        self.children = []
        self.visits = 0  # TODO: remove this line if not used.
        if self.parent:
            self.parent.children.append(self)

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


class ReasoningAgent(AssistantAgent):
    def __init__(
        self, name, llm_config, max_depth=4, beam_size=3, answer_approach="pool", verbose=True, **kwargs
    ) -> None:
        """Initialize a ReasoningAgent that uses tree-of-thought reasoning.,

        Args:
            name: Name of the agent
            llm_config: Configuration for the language model
            max_depth (int): Maximum depth of the reasoning tree
            beam_size (int): Number of parallel reasoning paths to maintain
            answer_approach (str): Either "pool" or "best" - how to generate final answer
            verbose (bool): Whether to show intermediate steps
        """
        super().__init__(name=name, llm_config=llm_config, **kwargs)
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.verbose = verbose
        assert answer_approach in ["pool", "best"]
        self.answer_approach = answer_approach
        self.thinker = AssistantAgent(name="tot_thinker", system_message=TreeofThought_message, llm_config=llm_config)

        self.grader = AssistantAgent(
            name="tot_grader",
            system_message="Rate the thinking trajectories for score 1 - 5 (1: worst, 5: best).",
            llm_config=llm_config,
        )
        self.register_reply([Agent, None], ReasoningAgent.generate_response)

        self._root = None

    def rate_node(self, node: ThinkNode) -> float:
        """Rate the quality of a reasoning path using the grader agent.

        Args:
            node (ThinkNode): Node containing the reasoning trajectory to evaluate

        Returns:
            float: Normalized score between 0 and 1 indicating trajectory quality
        """
        self.send(
            message=f"Rate the trajectory:\n{node.trajectory}", recipient=self.grader, request_reply=True, silent=False
        )
        rating = self.grader.last_message()["content"].strip()
        try:
            # Scale rating to [0, 1]
            reward = (float(re.findall(r"[\d.]+", rating)[0]) - 1) / 4.0
        except (IndexError, ValueError):
            reward = 0.0  # Default reward if parsing fails
        return reward

    def generate_response(self, messages, sender, config=None):
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

        messages = self._oai_messages[sender] if messages is None else messages
        prompt = messages[-1]["content"].strip()
        if not prompt:
            return True, "TERMINATE"

        root = ThinkNode(content=prompt, parent=None)
        self._root = root  # save the root node for later visualization
        prev_leafs = [root]

        final_answers = set()  # store the final answers

        while prev_leafs and len(final_answers) < self.beam_size:
            new_leafs = []
            for node in prev_leafs:
                if (self.max_depth and node.depth >= self.max_depth) or "TERMINATE" in node.content:
                    # Reached max depth; collect possible answers
                    if node.value is None:
                        node.value = self.rate_node(node)
                    final_answers.add(node)
                    continue

                self.thinker.clear_history()
                self.send(
                    message=f"{node.trajectory}\n---\nWhat are the possible next steps?",
                    recipient=self.thinker,
                    request_reply=True,
                    silent=False,
                )
                reply = self.thinker.last_message()["content"].strip()

                options = re.findall(
                    r"Option \d+:(.+?)(?=Option \d+:|$)", reply, re.DOTALL
                )  # the options that the thinker provides
                for option in options:
                    new_leafs.append(
                        ThinkNode(content=option.strip().rstrip(), parent=node)
                    )  # each option is a new leaf node

            prev_leafs = new_leafs

            if len(prev_leafs) + len(final_answers) > self.beam_size:
                if len(final_answers) >= self.beam_size:
                    prev_leafs = []  # stop searching, max beam size reached
                    break

                # Rate
                for node in prev_leafs:
                    node.value = self.rate_node(node)
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
