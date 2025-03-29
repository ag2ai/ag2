# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.agentchat.group.context_str import ContextStr
from autogen.agentchat.group.context_variables import ContextVariables


class TestContextStr:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.simple_template = "Hello, {name}!"
        self.complex_template = "User {user_id} has {num_items} items in cart: {items}"

        self.simple_context = ContextVariables(data={"name": "World"})
        self.complex_context = ContextVariables(
            data={"user_id": "12345", "num_items": 3, "items": ["apple", "banana", "orange"]}
        )

        self.simple_context_str = ContextStr(self.simple_template)
        self.complex_context_str = ContextStr(self.complex_template)

    def test_init(self) -> None:
        # Test initialisation with a template
        context_str = ContextStr("Test {variable}")
        assert context_str.template == "Test {variable}"

    def test_str(self) -> None:
        # Test string representation
        context_str = ContextStr("Test {variable}")
        str_representation = str(context_str)
        assert str_representation == "ContextStr, unformatted: Test {variable}"

        # Ensure the string representation does not attempt substitution
        assert "{variable}" in str_representation

    def test_format_simple(self) -> None:
        # Call format method
        result = self.simple_context_str.format(self.simple_context)

        # Verify the result
        assert result == "Hello, World!"

    def test_format_complex(self) -> None:
        # Call format method
        result = self.complex_context_str.format(self.complex_context)

        # Check the exact expected output with standard Python formatting
        assert result == "User 12345 has 3 items in cart: ['apple', 'banana', 'orange']"

    def test_format_with_error(self) -> None:
        """Test handling that would cause errors in string.format()."""
        # Create a template with nested attributes that standard format() can't handle
        nested_template = "Welcome {user}!"

        # Create a context with a complex object that can't be directly formatted
        nested_context = ContextVariables(data={"user": {"name": "Alice", "account": {"id": "ACC123", "balance": 500}}})

        # Create a new ContextStr
        nested_context_str = ContextStr(nested_template)

        # Call format method
        result = nested_context_str.format(nested_context)

        # The dict should be converted to string representation
        assert result is not None
        assert result == "Welcome {'name': 'Alice', 'account': {'id': 'ACC123', 'balance': 500}}!"

    def test_format_missing_variable(self) -> None:
        """Test what happens when we reference a variable not in context."""
        # Reference a variable not in the context
        missing_var_template = "Hello, {missing}!"
        missing_var_context_str = ContextStr(missing_var_template)

        # Raise a KeyError
        with pytest.raises(KeyError):
            missing_var_context_str.format(self.simple_context)
