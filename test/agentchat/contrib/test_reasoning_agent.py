# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import os
import random
import sys
from unittest.mock import MagicMock, call, patch

import pytest

from autogen.agentchat.contrib.reasoning_agent import ReasoningAgent, ThinkNode, visualize_tree
from autogen.import_utils import skip_on_missing_imports

from ...conftest import Credentials

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


here = os.path.abspath(os.path.dirname(__file__))

# Test data
TEST_QUESTION = "What is the capital of France?"
TEST_TRAJECTORY = """# Question:
What is the capital of France?
---

Step 1: Let me think about this systematically
Step 2: France is a country in Europe
Step 3: Paris is the capital city of France"""

TEST_CONTENT = "Paris is the capital of France"


@pytest.fixture
def think_node():
    """Create a ThinkNode instance for testing"""
    return ThinkNode(content=TEST_CONTENT)


@pytest.fixture
def reasoning_agent(mock_credentials: Credentials):
    """Create a ReasoningAgent instance for testing"""
    return ReasoningAgent("reasoning_agent", llm_config=mock_credentials.llm_config)


def test_think_node_init(think_node):
    """Test ThinkNode initialization"""
    assert think_node.content == TEST_CONTENT
    assert think_node.value == 0
    assert think_node.parent is None
    assert think_node.depth == 0
    assert think_node.children == []
    assert think_node.visits == 0


def test_think_node_trajectory(think_node):
    """Test ThinkNode trajectory property"""
    first_line = "# Question:\n" + TEST_CONTENT + "\n---\n"
    assert think_node._trajectory_arr == [first_line]
    assert first_line in think_node.trajectory


def test_think_node_str_repr(think_node):
    """Test ThinkNode string representation"""
    expected = f"{TEST_CONTENT} -> Depth: 0 Value: 0 Visits: 0"
    assert str(think_node) == expected
    assert repr(think_node) == expected


def test_think_node_to_dict(think_node):
    """Test ThinkNode to_dict method"""
    node_dict = think_node.to_dict()
    assert node_dict["content"] == TEST_CONTENT
    assert node_dict["value"] == 0
    assert node_dict["depth"] == 0
    assert node_dict["visits"] == 0
    assert node_dict["children"] == []


def test_think_node_from_dict():
    """Test ThinkNode from_dict method"""
    test_dict = {"content": TEST_CONTENT, "value": 0.5, "depth": 1, "visits": 2, "children": []}
    node = ThinkNode.from_dict(test_dict)
    assert node.content == TEST_CONTENT
    assert node.value == 0.5
    assert node.depth == 1
    assert node.visits == 2
    assert node.children == []


@pytest.mark.openai
def test_reasoning_agent_init(reasoning_agent):
    """Test ReasoningAgent initialization"""
    assert reasoning_agent.name == "reasoning_agent"
    assert reasoning_agent._max_depth == 4
    assert reasoning_agent._beam_size == 3
    assert reasoning_agent._answer_approach == "pool"
    assert reasoning_agent._root is None


def test_think_node_with_parent():
    """Test ThinkNode parent-child relationship"""
    parent = ThinkNode(content="Parent node")
    child = ThinkNode(content="Child node", parent=parent)

    assert child.parent == parent
    assert child.depth == 1
    assert child in parent.children
    assert len(parent.children) == 1


def test_think_node_complex_tree():
    """Test ThinkNode in a more complex tree structure"""
    root = ThinkNode(content="Root")
    child1 = ThinkNode(content="Child 1", parent=root)
    child2 = ThinkNode(content="Child 2", parent=root)
    grandchild = ThinkNode(content="Grandchild", parent=child1)

    assert len(root.children) == 2
    assert root.depth == 0
    assert child1.depth == 1
    assert child2.depth == 1
    assert grandchild.depth == 2
    assert "Root" in grandchild.trajectory
    assert "Child 1" in grandchild.trajectory
    assert "Grandchild" in grandchild.trajectory


def test_think_node_serialization_with_children():
    """Test ThinkNode serialization with nested structure"""
    root = ThinkNode(content="Root")
    ThinkNode(content="Child", parent=root)

    # Test to_dict
    root_dict = root.to_dict()
    assert len(root_dict["children"]) == 1
    assert root_dict["children"][0]["content"] == "Child"

    # Test from_dict
    new_root = ThinkNode.from_dict(root_dict)
    assert len(new_root.children) == 1
    assert new_root.children[0].content == "Child"


def test_reasoning_agent_answer(mock_credentials: Credentials):
    for max_depth in range(1, 10):
        for beam_size in range(1, 10):
            for answer_approach in ["pool", "best"]:
                helper_test_reasoning_agent_answer(max_depth, beam_size, answer_approach, mock_credentials)


def helper_test_reasoning_agent_answer(max_depth, beam_size, answer_approach, mock_credentials: Credentials):
    """Test that ReasoningAgent properly terminates when TERMINATE is received"""
    with patch("autogen.agentchat.conversable_agent.ConversableAgent.generate_oai_reply") as mock_oai_reply:
        agent = ReasoningAgent(
            "test_agent",
            llm_config=mock_credentials.llm_config,
            reason_config={"beam_size": beam_size, "answer_approach": answer_approach, "max_depth": max_depth},
        )

        def mock_response(*args, **kwargs):
            # Get the instance that called the mock
            instance = args[0]
            print("INSTANCE:", instance)
            if instance.name == "tot_thinker":
                return True, {
                    "content": """Reflection
Found the answer.

Possible Options:
Option 1: TERMINATE
Option 2: Keep going with an option
Option 3: Another option"""
                }
            elif instance.name == "tot_grader":
                return True, {"content": f"{random.randint(1, 5)}"}
            elif instance.name == "test_agent":
                return True, {"content": "The final answer is here."}
            return True, {"content": "Unknown agent"}

        mock_oai_reply.side_effect = mock_response

        print("OAI REPLY:", agent._thinker.generate_oai_reply)

        response = agent._beam_reply("Test question")
        assert len(response)

    assert "TERMINATE" in agent._thinker.last_message()["content"]

    # Verify we didn't exceed max_depth
    current_node = agent._root
    max_depth_found = 0
    nodes_to_check = [current_node]

    while nodes_to_check:
        node = nodes_to_check.pop(0)
        max_depth_found = max(max_depth_found, node.depth)
        nodes_to_check.extend(node.children)

    assert max_depth_found <= agent._max_depth


@skip_on_missing_imports(["graphviz"], "unknown")
@patch("graphviz.Digraph")
def test_visualize_tree_successful_case(mock_digraph):
    """Test successful tree visualization"""
    # Create a sample tree structure
    root = ThinkNode(content="Root")
    child1 = ThinkNode(content="Child 1", parent=root)
    child2 = ThinkNode(content="Child 2", parent=root)
    grandchild = ThinkNode(content="Grandchild with very long content that should be truncated", parent=child1)

    # Set some values for testing
    root.visits = 1
    root.value = 0.5
    child1.visits = 2
    child1.value = 0.7
    child2.visits = 0
    grandchild.visits = 0

    # Create mock Digraph instance
    mock_graph = MagicMock()
    mock_digraph.return_value = mock_graph

    visualize_tree(root)

    # Verify Digraph initialization
    mock_digraph.assert_called_once()
    mock_graph.attr.assert_called_once_with(rankdir="TB")

    # Verify nodes were added with correct attributes
    expected_calls = [
        call("0", "Root\n visits: 1\n value: 0.5"),
        call("0_0", "Child 1\n visits: 2\n value: 0.7"),
        call("0_1", "Child 2\n visits: 0\n value: 0"),
        call("0_0_0", "Grandchild with very long content that should be t...\n visits: 0\n value: 0"),
    ]
    mock_graph.node.assert_has_calls(expected_calls, any_order=True)

    # Verify edges were created
    expected_edge_calls = [
        call("0", "0_0"),  # Root -> Child1
        call("0", "0_1"),  # Root -> Child2
        call("0_0", "0_0_0"),  # Child1 -> Grandchild
    ]
    mock_graph.edge.assert_has_calls(expected_edge_calls, any_order=True)

    # Verify render was called
    mock_graph.render.assert_called_once_with("tree_of_thoughts", view=False, format="png", cleanup=True)


@skip_on_missing_imports(["graphviz"], "unknown")
@patch("graphviz.Digraph")
def test_visualize_tree_render_failure(mock_digraph):
    """Test visualization when rendering fails"""
    root = ThinkNode(content="Root")

    mock_graph = MagicMock()
    mock_digraph.return_value = mock_graph
    mock_graph.render.side_effect = Exception("Rendering failed")

    with patch("builtins.print") as mock_print:
        visualize_tree(root)
        mock_print.assert_has_calls([
            call("Error rendering graph: Rendering failed"),
            call("Make sure graphviz is installed on your system: https://graphviz.org/download/"),
        ])


def test_reasoning_agent_code_execution(mock_credentials: Credentials):
    """Test that ReasoningAgent properly executes code in responses"""

    # Create agent with code execution enabled
    with patch("autogen.agentchat.conversable_agent.ConversableAgent.generate_oai_reply") as mock_oai_reply:
        agent = ReasoningAgent(
            "test_agent",
            llm_config=mock_credentials.llm_config,
            code_execution_config={"use_docker": False, "work_dir": "mypy_cache"},
        )

        def mock_response(*args, **kwargs):
            instance = args[0]
            if instance.name == "tot_thinker":
                return True, {
                    "content": """Reflection
Let's solve this with Python.

Possible Options:
Option 1: Calculate factorial with Python
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

print(f"Factorial of 5 is {factorial(5)}")
```

Option 2: TERMINATE"""
                }
            elif instance.name == "reasoner_user_proxy":
                # Mock the code execution result
                return True, {"content": "Factorial of 5 is 120"}
            elif instance.name == "test_agent":
                return True, {"content": "The factorial of 5 is 120"}
            return True, {"content": "5"}

        mock_oai_reply.side_effect = mock_response

        # Test code execution
        response = agent._beam_reply("Calculate factorial of 5")

        # Verify code was executed
        assert "Factorial of 5 is 120" in agent._root.children[0].content
        assert "Code Execution Result:" in agent._root.children[0].content
        assert response == "The factorial of 5 is 120"


if __name__ == "__main__":
    pytest.main([__file__])
