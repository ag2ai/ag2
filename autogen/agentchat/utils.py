# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import ast
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from ..doc_utils import export_module
from .agent import Agent


def consolidate_chat_info(
    chat_info: Union[dict[str, Any], list[dict[str, Any]]], uniform_sender: Optional[Agent] = None
) -> None:
    if isinstance(chat_info, dict):
        chat_info = [chat_info]
    for c in chat_info:
        if uniform_sender is None:
            assert "sender" in c, "sender must be provided."
            sender = c["sender"]
        else:
            sender = uniform_sender
        assert "recipient" in c, "recipient must be provided."
        summary_method = c.get("summary_method")
        assert (
            summary_method is None or callable(summary_method) or summary_method in ("last_msg", "reflection_with_llm")
        ), "summary_method must be a string chosen from 'reflection_with_llm' or 'last_msg' or a callable, or None."
        if summary_method == "reflection_with_llm":
            assert sender.client is not None or c["recipient"].client is not None, (
                "llm client must be set in either the recipient or sender when summary_method is reflection_with_llm."
            )


@export_module("autogen")
def gather_usage_summary(agents: list[Agent]) -> dict[str, dict[str, Any]]:
    r"""Gather usage summary from all agents.

    Args:
        agents: (list): List of agents.

    Returns:
        dictionary: A dictionary containing two keys:
            - "usage_including_cached_inference": Cost information on the total usage, including the tokens in cached inference.
            - "usage_excluding_cached_inference": Cost information on the usage of tokens, excluding the tokens in cache. No larger than "usage_including_cached_inference".

    Example:
    ```python
    {
        "usage_including_cached_inference": {
            "total_cost": 0.0006090000000000001,
            "gpt-35-turbo": {
                "cost": 0.0006090000000000001,
                "prompt_tokens": 242,
                "completion_tokens": 123,
                "total_tokens": 365,
            },
        },
        "usage_excluding_cached_inference": {
            "total_cost": 0.0006090000000000001,
            "gpt-35-turbo": {
                "cost": 0.0006090000000000001,
                "prompt_tokens": 242,
                "completion_tokens": 123,
                "total_tokens": 365,
            },
        },
    }
    ```

    Note:
    If none of the agents incurred any cost (not having a client), then the usage_including_cached_inference and usage_excluding_cached_inference will be `{'total_cost': 0}`.
    """

    def aggregate_summary(usage_summary: dict[str, Any], agent_summary: dict[str, Any]) -> None:
        if agent_summary is None:
            return
        usage_summary["total_cost"] += agent_summary.get("total_cost", 0)
        for model, data in agent_summary.items():
            if model != "total_cost":
                if model not in usage_summary:
                    usage_summary[model] = data.copy()
                else:
                    usage_summary[model]["cost"] += data.get("cost", 0)
                    usage_summary[model]["prompt_tokens"] += data.get("prompt_tokens", 0)
                    usage_summary[model]["completion_tokens"] += data.get("completion_tokens", 0)
                    usage_summary[model]["total_tokens"] += data.get("total_tokens", 0)

    usage_including_cached_inference = {"total_cost": 0}
    usage_excluding_cached_inference = {"total_cost": 0}

    for agent in agents:
        if getattr(agent, "client", None):
            aggregate_summary(usage_including_cached_inference, agent.client.total_usage_summary)  # type: ignore[attr-defined]
            aggregate_summary(usage_excluding_cached_inference, agent.client.actual_usage_summary)  # type: ignore[attr-defined]

    return {
        "usage_including_cached_inference": usage_including_cached_inference,
        "usage_excluding_cached_inference": usage_excluding_cached_inference,
    }


def parse_tags_from_content(tag: str, content: Union[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Parses HTML style tags from message contents.

    The parsing is done by looking for patterns in the text that match the format of HTML tags. The tag to be parsed is
    specified as an argument to the function. The function looks for this tag in the text and extracts its content. The
    content of a tag is everything that is inside the tag, between the opening and closing angle brackets. The content
    can be a single string or a set of attribute-value pairs.

    Examples:
        `<img http://example.com/image.png> -> [{"tag": "img", "attr": {"src": "http://example.com/image.png"}, "match": re.Match}]`
        ```<audio text="Hello I'm a robot" prompt="whisper"> ->
                [{"tag": "audio", "attr": {"text": "Hello I'm a robot", "prompt": "whisper"}, "match": re.Match}]```

    Args:
        tag (str): The HTML style tag to be parsed.
        content (Union[str, list[dict[str, Any]]]): The message content to parse. Can be a string or a list of content
            items.

    Returns:
        list[dict[str, str]]: A list of dictionaries, where each dictionary represents a parsed tag. Each dictionary
            contains three key-value pairs: 'type' which is the tag, 'attr' which is a dictionary of the parsed attributes,
            and 'match' which is a regular expression match object.

    Raises:
        ValueError: If the content is not a string or a list.
    """
    results = []
    if isinstance(content, str):
        results.extend(_parse_tags_from_text(tag, content))
    # Handles case for multimodal messages.
    elif isinstance(content, list):
        for item in content:
            if item.get("type") == "text":
                results.extend(_parse_tags_from_text(tag, item["text"]))
    else:
        raise ValueError(f"content must be str or list, but got {type(content)}")

    return results


def _parse_tags_from_text(tag: str, text: str) -> list[dict[str, Any]]:
    pattern = re.compile(f"<{tag} (.*?)>")

    results = []
    for match in re.finditer(pattern, text):
        tag_attr = match.group(1).strip()
        attr = _parse_attributes_from_tags(tag_attr)

        results.append({"tag": tag, "attr": attr, "match": match})
    return results


def _parse_attributes_from_tags(tag_content: str) -> dict[str, str]:
    pattern = r"([^ ]+)"
    attrs = re.findall(pattern, tag_content)
    reconstructed_attrs = _reconstruct_attributes(attrs)

    def _append_src_value(content: dict[str, str], value: Any) -> None:
        if "src" in content:
            content["src"] += f" {value}"
        else:
            content["src"] = value

    content: dict[str, str] = {}
    for attr in reconstructed_attrs:
        if "=" not in attr:
            _append_src_value(content, attr)
            continue

        key, value = attr.split("=", 1)
        if value.startswith("'") or value.startswith('"'):
            content[key] = value[1:-1]  # remove quotes
        else:
            _append_src_value(content, attr)

    return content


def _reconstruct_attributes(attrs: list[str]) -> list[str]:
    """Reconstructs attributes from a list of strings where some attributes may be split across multiple elements."""

    def is_attr(attr: str) -> bool:
        if "=" in attr:
            _, value = attr.split("=", 1)
            if value.startswith("'") or value.startswith('"'):
                return True
        return False

    reconstructed = []
    found_attr = False
    for attr in attrs:
        if is_attr(attr):
            reconstructed.append(attr)
            found_attr = True
        else:
            if found_attr:
                reconstructed[-1] += f" {attr}"
                found_attr = True
            elif reconstructed:
                reconstructed[-1] += f" {attr}"
            else:
                reconstructed.append(attr)
    return reconstructed


@dataclass
@export_module("autogen")
class ContextExpression:
    """A class to evaluate logical expressions using context variables.

    Args:
        expression (str): A string containing a logical expression with context variable references.
            - Variable references use ${var_name} syntax: ${logged_in}, ${attempts}
            - String literals can use normal quotes: 'hello', "world"
            - Supported operators:
                - Logical: not/!, and/&, or/|
                - Comparison: >, <, >=, <=, ==, !=
            - Supported functions:
                - len(${var_name}): Gets the length of a list, string, or other collection
            - Parentheses can be used for grouping
            - Examples:
                - "not ${logged_in} and ${is_admin} or ${guest_checkout}"
                - "!${logged_in} & ${is_admin} | ${guest_checkout}"
                - "len(${orders}) > 0 & ${user_active}"
                - "len(${cart_items}) == 0 | ${checkout_started}"

    Raises:
        SyntaxError: If the expression cannot be parsed
        ValueError: If the expression contains disallowed operations
    """

    expression: str

    def __post_init__(self) -> None:
        # Validate the expression immediately upon creation
        try:
            # Extract variable references and replace with placeholders
            self._variable_names = self._extract_variable_names(self.expression)

            # Convert symbolic operators to Python keywords
            python_expr = self._convert_to_python_syntax(self.expression)

            # Sanitize for AST parsing
            sanitized_expr = self._prepare_for_ast(python_expr)

            # Use ast to parse and validate the expression
            self._ast = ast.parse(sanitized_expr, mode="eval")

            # Verify it only contains allowed operations
            self._validate_operations(self._ast.body)

            # Store the Python-syntax version for evaluation
            self._python_expr = python_expr

        except SyntaxError as e:
            raise SyntaxError(f"Invalid expression syntax in '{self.expression}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Error validating expression '{self.expression}': {str(e)}")

    def _extract_variable_names(self, expr: str) -> list[str]:
        """Extract all variable references ${var_name} from the expression."""
        # Find all patterns like ${var_name}
        matches = re.findall(r"\${([^}]*)}", expr)
        return matches

    def _convert_to_python_syntax(self, expr: str) -> str:
        """Convert symbolic operators to Python keywords."""
        # We need to be careful about operators inside string literals
        # First, temporarily replace string literals with placeholders
        string_literals = []

        def replace_string_literal(match: re.Match[str]) -> str:
            string_literals.append(match.group(0))
            return f"__STRING_LITERAL_{len(string_literals) - 1}__"

        # Replace both single and double quoted strings
        expr_without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", replace_string_literal, expr)

        # Handle the NOT operator (!) - no parentheses handling needed
        # Replace standalone ! before variables or expressions
        expr_without_strings = re.sub(r"!\s*(\${|\()", "not \\1", expr_without_strings)

        # Handle AND and OR operators - simpler approach without parentheses handling
        expr_without_strings = re.sub(r"\s+&\s+", " and ", expr_without_strings)
        expr_without_strings = re.sub(r"\s+\|\s+", " or ", expr_without_strings)

        # Now put string literals back
        for i, literal in enumerate(string_literals):
            expr_without_strings = expr_without_strings.replace(f"__STRING_LITERAL_{i}__", literal)

        return expr_without_strings

    def _prepare_for_ast(self, expr: str) -> str:
        """Convert the expression to valid Python for AST parsing by replacing variables with placeholders."""
        # Replace ${var_name} with var_name for AST parsing
        processed_expr = expr
        for var_name in self._variable_names:
            processed_expr = processed_expr.replace(f"${{{var_name}}}", var_name)

        return processed_expr

    def _validate_operations(self, node: ast.AST) -> None:
        """Recursively validate that only allowed operations exist in the AST."""
        allowed_node_types = (
            # Boolean operations
            ast.BoolOp,
            ast.UnaryOp,
            ast.And,
            ast.Or,
            ast.Not,
            # Comparison operations
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            # Basic nodes
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Expression,
            # Support for basic numeric operations in comparisons
            ast.Num,
            ast.NameConstant,
            # Support for negative numbers
            ast.USub,
            ast.UnaryOp,
            # Support for string literals
            ast.Str,
            ast.Constant,
            # Support for function calls (specifically len())
            ast.Call,
        )

        if not isinstance(node, allowed_node_types):
            raise ValueError(f"Operation type {type(node).__name__} is not allowed in logical expressions")

        # Special validation for function calls - only allow len()
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id == "len"):
                raise ValueError(f"Only the len() function is allowed, got: {getattr(node.func, 'id', 'unknown')}")
            if len(node.args) != 1:
                raise ValueError(f"len() function must have exactly one argument, got {len(node.args)}")

        # Special validation for Compare nodes
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    raise ValueError(f"Comparison operator {type(op).__name__} is not allowed")

        # Recursively check child nodes
        for child in ast.iter_child_nodes(node):
            self._validate_operations(child)

    def evaluate(self, context_variables: dict[str, Any]) -> bool:
        """Evaluate the expression using the provided context variables.

        Args:
            context_variables: Dictionary of context variables to use for evaluation

        Returns:
            bool: The result of evaluating the expression
        """
        # Create a modified expression that we can safely evaluate
        eval_expr = self._python_expr  # Use the Python-syntax version

        # First, handle len() functions with variable references inside
        len_pattern = r"len\(\${([^}]*)}\)"
        len_matches = list(re.finditer(len_pattern, eval_expr))

        # Process all len() operations first
        for match in len_matches:
            var_name = match.group(1)
            var_value = context_variables.get(var_name, [])

            # Calculate the length - works for lists, strings, dictionaries, etc.
            try:
                length_value = len(var_value)
            except TypeError:
                # If the value doesn't support len(), treat as 0
                length_value = 0

            # Replace the len() expression with the actual length
            full_match = match.group(0)
            eval_expr = eval_expr.replace(full_match, str(length_value))

        # Then replace remaining variable references with their values
        for var_name in self._variable_names:
            # Skip variables that were already processed in len() expressions
            if any(m.group(1) == var_name for m in len_matches):
                continue

            # Get the value from context, defaulting to False if not found
            var_value = context_variables.get(var_name, False)

            # Format the value appropriately based on its type
            if isinstance(var_value, (bool, int, float)):
                formatted_value = str(var_value)
            elif isinstance(var_value, str):
                formatted_value = f"'{var_value}'"  # Quote strings
            elif isinstance(var_value, (list, dict, tuple)):
                # For collections, convert to their boolean evaluation
                formatted_value = str(bool(var_value))
            else:
                formatted_value = str(var_value)

            # Replace the variable reference with the formatted value
            eval_expr = eval_expr.replace(f"${{{var_name}}}", formatted_value)

        try:
            return eval(eval_expr)  # type: ignore[no-any-return]
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{self.expression}': {str(e)}")

    def __str__(self) -> str:
        return f"ContextExpression('{self.expression}')"
