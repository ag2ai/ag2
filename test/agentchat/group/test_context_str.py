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

        self.simple_context_str = ContextStr(template=self.simple_template)
        self.complex_context_str = ContextStr(template=self.complex_template)

    def test_init(self) -> None:
        # Test initialisation with a template
        context_str = ContextStr(template="Test {variable}")
        assert context_str.template == "Test {variable}"

    def test_str(self) -> None:
        # Test string representation
        context_str = ContextStr(template="Test {variable}")
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
        nested_context_str = ContextStr(template=nested_template)

        # Call format method
        result = nested_context_str.format(nested_context)

        # The dict should be converted to string representation
        assert result is not None
        assert result == "Welcome {'name': 'Alice', 'account': {'id': 'ACC123', 'balance': 500}}!"

    def test_format_missing_variable(self) -> None:
        """Test what happens when we reference a variable not in context."""
        # Reference a variable not in the context
        missing_var_template = "Hello, {missing}!"
        missing_var_context_str = ContextStr(template=missing_var_template)

        # Raise a KeyError
        with pytest.raises(KeyError):
            missing_var_context_str.format(self.simple_context)

    def test_format_empty_context(self) -> None:
        """Test formatting with empty context variables."""
        # Create a template
        template = "Hello, {name}!"
        context_str = ContextStr(template=template)

        # Create empty context variables
        empty_context = ContextVariables()

        # Call format method
        result = context_str.format(empty_context)

        # Should return the template as is when context is empty
        assert result == template

    def test_format_no_placeholders(self) -> None:
        """Test formatting a string with no placeholders."""
        # Create a template with no placeholders
        template = "Hello, World!"
        context_str = ContextStr(template=template)

        # Format with any context
        result = context_str.format(self.simple_context)

        # Should return the template as is
        assert result == template

    def test_format_repeated_placeholders(self) -> None:
        """Test formatting with repeated placeholders."""
        # Create a template with repeated placeholders
        template = "Hello, {name}! Your name is {name}."
        context_str = ContextStr(template=template)

        # Format with context
        result = context_str.format(self.simple_context)

        # Both instances should be replaced
        assert result == "Hello, World! Your name is World."

    def test_format_various_data_types(self) -> None:
        """Test formatting with various data types in context variables."""
        # Create a template with various data types
        template = (
            "String: {string}, Integer: {integer}, Float: {float}, Boolean: {boolean}, List: {list}, Dict: {dict}"
        )
        context_str = ContextStr(template=template)

        # Create context with various data types
        context = ContextVariables(
            data={
                "string": "text",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
                "list": [1, 2, 3],
                "dict": {"key": "value"},
            }
        )

        # Format with context
        result = context_str.format(context)

        # All types should be formatted correctly
        assert (
            result == "String: text, Integer: 42, Float: 3.14, Boolean: True, List: [1, 2, 3], Dict: {'key': 'value'}"
        )
