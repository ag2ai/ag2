# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from autogen.function_utils import get_function_schema
from autogen.oai import OpenAIWrapper

from ..agent import Agent
from ..assistant_agent import AssistantAgent
from ..chat import ChatResult
from ..conversable_agent import ConversableAgent
from ..groupchat import GroupChat, GroupChatManager
from ..user_proxy_agent import UserProxyAgent

TreeofThought_message = """
Role: Expert Planning AI Assistant

Task: Given a question and a list of previous steps (the plan trajectory), generate at least four innovative options for the next step. The user would not answer you anything.

Instructions:
- Review the user's question and the previous steps taken.
- Identify any mistakes or errors in the previous steps.
- If you find any mistakes, include optionss to correct them in your proposed options.
- Think creatively to propose at least four possible optionss that logically build upon or correct the previous steps.
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
        self.content = content
        self.value = None
        self.parent = parent
        self.depth = self.parent.depth + 1 if parent else 0
        self.children = []
        self.visits = 0
        if self.parent:
            self.parent.children.append(self)

    @property
    def _trajectory_arr(self) -> List[str]:
        if self.parent:
            return self.parent._trajectory_arr + [self.content]
        return ["# Question: " + self.content]

    @property
    def trajectory(self) -> str:
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
        """Convert ThinkNode to dictionary representation."""
        return {
            "content": self.content,
            "value": self.value,
            "depth": self.depth,
            "visits": self.visits,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict, parent: Optional["ThinkNode"] = None) -> "ThinkNode":
        """Create ThinkNode from dictionary representation."""
        node = cls(content=data["content"], parent=parent)
        node.value = data["value"]
        node.depth = data["depth"]
        node.visits = data["visits"]

        # Recursively create children
        for child_data in data["children"]:
            cls.from_dict(child_data, parent=node)

        return node


class ReasoningAgent(AssistantAgent):

    def __init__(self, name, llm_config, max_depth=4, beam_size=3, answer_approach="pool", verbose=True) -> None:
        super().__init__(name=name, llm_config=llm_config)
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

    def rate_node(self, node: ThinkNode) -> float:
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
        if sender == self:
            return False, ""  # Defer the LLM call to next reply functions.

        messages = self._oai_messages[sender] if messages is None else messages
        prompt = messages[-1]["content"].strip()
        if not prompt:
            return True, "TERMINATE"

        root = ThinkNode(content=prompt, parent=None)
        prev_leafs = [root]

        final_answers = set()  # store the final answers

        while prev_leafs and len(final_answers) < self.beam_size:
            new_leafs = []
            # print("len(final_answers)", len(final_answers))
            # print("len(prev_leafs)", len(prev_leafs))
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
                # print("Options:", options)
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
        # visualize_tree(root) #TODO: inspect if this in necessary
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
