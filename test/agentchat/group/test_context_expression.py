# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.agentchat.group.context_expression import ContextExpression
from autogen.agentchat.group.context_variables import ContextVariables


class TestContextExpressionNewSyntax:
    """Tests for the ContextExpression class with the new ${var_name} syntax."""

    def test_basic_boolean_operations(self) -> None:
        """Test basic boolean operations."""
        context = ContextVariables(
            data={
                "var_true": True,
                "var_false": False,
            }
        )

        # Test simple variable lookup
        assert ContextExpression("${var_true}").evaluate(context) is True
        assert ContextExpression("${var_false}").evaluate(context) is False

        # Test NOT operation - without parentheses
        assert ContextExpression("not ${var_true}").evaluate(context) is False
        assert ContextExpression("not ${var_false}").evaluate(context) is True

        # Test AND operation
        assert ContextExpression("${var_true} and ${var_true}").evaluate(context) is True
        assert ContextExpression("${var_true} and ${var_false}").evaluate(context) is False
        assert ContextExpression("${var_false} and ${var_true}").evaluate(context) is False
        assert ContextExpression("${var_false} and ${var_false}").evaluate(context) is False

        # Test OR operation
        assert ContextExpression("${var_true} or ${var_true}").evaluate(context) is True
        assert ContextExpression("${var_true} or ${var_false}").evaluate(context) is True
        assert ContextExpression("${var_false} or ${var_true}").evaluate(context) is True
        assert ContextExpression("${var_false} or ${var_false}").evaluate(context) is False

    def test_symbolic_operators(self) -> None:
        """Test the symbolic operators (!, &, |)."""
        context = ContextVariables(
            data={
                "var_true": True,
                "var_false": False,
            }
        )

        # Test NOT operator with ! - without parentheses
        assert ContextExpression("!${var_true}").evaluate(context) is False
        assert ContextExpression("!${var_false}").evaluate(context) is True

        # Test AND operator with &
        assert ContextExpression("${var_true} & ${var_true}").evaluate(context) is True
        assert ContextExpression("${var_true} & ${var_false}").evaluate(context) is False
        assert ContextExpression("${var_false} & ${var_true}").evaluate(context) is False
        assert ContextExpression("${var_false} & ${var_false}").evaluate(context) is False

        # Test OR operator with |
        assert ContextExpression("${var_true} | ${var_true}").evaluate(context) is True
        assert ContextExpression("${var_true} | ${var_false}").evaluate(context) is True
        assert ContextExpression("${var_false} | ${var_true}").evaluate(context) is True
        assert ContextExpression("${var_false} | ${var_false}").evaluate(context) is False

        # Test mixed operators
        assert ContextExpression("${var_true} & !${var_false}").evaluate(context) is True
        assert ContextExpression("!${var_true} | ${var_true}").evaluate(context) is True

    def test_mixed_syntax(self) -> None:
        """Test mixing symbolic and keyword operators."""
        context = ContextVariables(
            data={
                "var_true": True,
                "var_false": False,
            }
        )

        # Mixing different styles
        assert ContextExpression("${var_true} and !${var_false}").evaluate(context) is True
        assert ContextExpression("not ${var_true} | ${var_true}").evaluate(context) is True
        assert ContextExpression("${var_true} & (${var_false} or ${var_true})").evaluate(context) is True

    def test_numeric_comparisons(self) -> None:
        """Test numeric comparisons."""
        context = ContextVariables(
            data={
                "num_zero": 0,
                "num_one": 1,
                "num_ten": 10,
                "num_negative": -5,
            }
        )

        # Test equal comparison
        assert ContextExpression("${num_zero} == 0").evaluate(context) is True
        assert ContextExpression("${num_one} == 1").evaluate(context) is True
        assert ContextExpression("${num_one} == 2").evaluate(context) is False

        # Test not equal comparison
        assert ContextExpression("${num_zero} != 1").evaluate(context) is True
        assert ContextExpression("${num_one} != 1").evaluate(context) is False

        # Test greater than comparison
        assert ContextExpression("${num_ten} > 5").evaluate(context) is True
        assert ContextExpression("${num_one} > 5").evaluate(context) is False

        # Test less than comparison
        assert ContextExpression("${num_one} < 5").evaluate(context) is True
        assert ContextExpression("${num_ten} < 5").evaluate(context) is False

        # Test greater than or equal comparison
        assert ContextExpression("${num_ten} >= 10").evaluate(context) is True
        assert ContextExpression("${num_ten} >= 11").evaluate(context) is False

        # Test less than or equal comparison
        assert ContextExpression("${num_one} <= 1").evaluate(context) is True
        assert ContextExpression("${num_one} <= 0").evaluate(context) is False

        # Test with negative numbers
        assert ContextExpression("${num_negative} < 0").evaluate(context) is True
        assert ContextExpression("${num_negative} == -5").evaluate(context) is True
        assert ContextExpression("${num_negative} > -10").evaluate(context) is True

    def test_comparisons_with_symbolic_operators(self) -> None:
        """Test combining comparisons with symbolic operators."""
        context = ContextVariables(data={"num_one": 1, "num_ten": 10, "logged_in": True, "is_admin": False})

        # Test numeric comparisons with symbolic operators
        assert ContextExpression("${num_ten} > 5 & ${num_one} < 5").evaluate(context) is True
        assert ContextExpression("${num_ten} == 10 | ${num_one} == 0").evaluate(context) is True
        assert ContextExpression("!(${num_one} > 5) & ${num_ten} >= 10").evaluate(context) is True

        # Test complex mixed expressions
        assert ContextExpression("${logged_in} & (${num_ten} > ${num_one})").evaluate(context) is True
        assert ContextExpression("${logged_in} & !${is_admin} & ${num_one} == 1").evaluate(context) is True

    def test_string_comparisons(self) -> None:
        """Test string comparisons."""
        context = ContextVariables(
            data={
                "str_empty": "",
                "str_hello": "hello",
                "str_world": "world",
            }
        )

        # Test equal comparison with string literals
        assert ContextExpression("${str_hello} == 'hello'").evaluate(context) is True
        assert ContextExpression("${str_hello} == 'world'").evaluate(context) is False
        assert ContextExpression('${str_hello} == "hello"').evaluate(context) is True

        # Test not equal comparison
        assert ContextExpression("${str_hello} != 'world'").evaluate(context) is True
        assert ContextExpression("${str_hello} != 'hello'").evaluate(context) is False

        # Test empty string comparison
        assert ContextExpression("${str_empty} == ''").evaluate(context) is True
        assert ContextExpression("${str_empty} != 'hello'").evaluate(context) is True

        # Test string comparison between variables
        assert ContextExpression("${str_hello} != ${str_world}").evaluate(context) is True
        assert ContextExpression("${str_hello} == ${str_hello}").evaluate(context) is True

    def test_string_comparisons_with_symbolic_operators(self) -> None:
        """Test string comparisons with symbolic operators."""
        context = ContextVariables(data={"str_hello": "hello", "str_world": "world", "is_premium": True})

        # Test string comparisons with symbolic operators
        assert ContextExpression("${str_hello} == 'hello' & ${is_premium}").evaluate(context) is True
        assert ContextExpression("${str_hello} != 'world' | !${is_premium}").evaluate(context) is True
        assert ContextExpression("!(${str_hello} == ${str_world}) & ${is_premium}").evaluate(context) is True

    def test_complex_expressions(self) -> None:
        """Test complex expressions with nested operations."""
        context = ContextVariables(
            data={
                "user_logged_in": True,
                "is_admin": False,
                "has_permission": True,
                "user_age": 25,
                "min_age": 18,
                "max_attempts": 3,
                "current_attempts": 2,
                "username": "john_doe",
            }
        )

        # Test nested boolean operations
        assert ContextExpression("${user_logged_in} and (${is_admin} or ${has_permission})").evaluate(context) is True

        # Test mixed boolean and comparison operations
        assert ContextExpression("${user_logged_in} and ${user_age} >= ${min_age}").evaluate(context) is True

        # Test with string literals
        assert ContextExpression("${username} == 'john_doe' and ${has_permission}").evaluate(context) is True

        # Test complex nested expressions with strings and numbers
        assert (
            ContextExpression(
                "${user_logged_in} and (${username} == 'john_doe') and (${user_age} > ${min_age})"
            ).evaluate(context)
            is True
        )

        # Test with multiple literals
        assert (
            ContextExpression(
                "${user_age} > 18 and ${username} == 'john_doe' and ${current_attempts} < ${max_attempts}"
            ).evaluate(context)
            is True
        )

    def test_complex_expressions_with_symbolic_operators(self) -> None:
        """Test complex expressions with symbolic operators."""
        context = ContextVariables(
            data={
                "user_logged_in": True,
                "is_admin": False,
                "has_permission": True,
                "user_age": 25,
                "min_age": 18,
                "max_attempts": 3,
                "current_attempts": 2,
                "username": "john_doe",
            }
        )

        # Test complex expressions with symbolic operators
        assert (
            ContextExpression(
                "${user_logged_in} & (${is_admin} | ${has_permission}) & ${user_age} > ${min_age}"
            ).evaluate(context)
            is True
        )

        # Test deeply nested expressions with mixed operators
        assert (
            ContextExpression(
                "!${is_admin} & (${username} == 'john_doe' | ${user_age} > 30) & ${current_attempts} < ${max_attempts}"
            ).evaluate(context)
            is True
        )

        # Test with complex parenthetical expressions
        assert (
            ContextExpression("(${user_logged_in} & ${has_permission}) | (${is_admin} & ${user_age} >= 21)").evaluate(
                context
            )
            is True
        )

    def test_missing_variables(self) -> None:
        """Test behavior with missing variables."""
        context = ContextVariables(
            data={
                "var_true": True,
            }
        )

        # Missing variables should default to False
        with pytest.raises(KeyError):
            ContextExpression("${non_existent_var}").evaluate(context) is False

        with pytest.raises(KeyError):
            assert ContextExpression("${var_true} and ${non_existent_var}").evaluate(context) is False

        with pytest.raises(KeyError):
            assert ContextExpression("${var_true} or ${non_existent_var}").evaluate(context) is True

        with pytest.raises(KeyError):
            assert ContextExpression("not ${non_existent_var}").evaluate(context) is True

        # Test with symbolic operators
        with pytest.raises(KeyError):
            assert ContextExpression("${var_true} & ${non_existent_var}").evaluate(context) is False

        with pytest.raises(KeyError):
            assert ContextExpression("${var_true} | ${non_existent_var}").evaluate(context) is True

        with pytest.raises(KeyError):
            assert ContextExpression("!${non_existent_var}").evaluate(context) is True

    def test_real_world_examples(self) -> None:
        """Test real-world examples with the new syntax."""
        context = ContextVariables(
            data={
                "logged_in": True,
                "is_admin": False,
                "has_order_id": True,
                "order_delivered": True,
                "return_started": False,
                "attempts": 2,
                "customer_angry": True,
                "manager_already_involved": False,
                "customer_name": "Alice Smith",
                "is_premium_customer": True,
                "account_type": "premium",
            }
        )

        # Authentication example - removed parentheses
        assert ContextExpression("${logged_in} and not ${is_admin}").evaluate(context) is True

        # Order processing with string
        assert (
            ContextExpression("${has_order_id} and ${order_delivered} and ${customer_name} == 'Alice Smith'").evaluate(
                context
            )
            is True
        )

        # Account type check
        assert ContextExpression("${is_premium_customer} and ${account_type} == 'premium'").evaluate(context) is True

        # Complex business rule
        assert (
            ContextExpression(
                "${logged_in} and ${customer_angry} and not ${manager_already_involved} and ${account_type} == 'premium'"
            ).evaluate(context)
            is True
        )

    def test_real_world_examples_with_symbolic_operators(self) -> None:
        """Test real-world examples with symbolic operators."""
        context = ContextVariables(
            data={
                "logged_in": True,
                "is_admin": False,
                "has_order_id": True,
                "order_delivered": True,
                "return_started": False,
                "attempts": 2,
                "max_attempts": 5,
                "customer_angry": True,
                "manager_already_involved": False,
                "customer_name": "Alice Smith",
                "is_premium_customer": True,
                "account_type": "premium",
            }
        )

        # Authentication example with symbolic operators
        assert ContextExpression("${logged_in} & !${is_admin}").evaluate(context) is True

        # Order processing with symbolic operators
        assert (
            ContextExpression("${has_order_id} & ${order_delivered} & ${customer_name} == 'Alice Smith'").evaluate(
                context
            )
            is True
        )

        # Complex business rule with symbolic operators
        assert (
            ContextExpression(
                "${logged_in} & ${customer_angry} & !${manager_already_involved} & ${account_type} == 'premium'"
            ).evaluate(context)
            is True
        )

        # Mixed operator business logic
        assert (
            ContextExpression(
                "${logged_in} & (${is_premium_customer} | ${attempts} < ${max_attempts}) & !${return_started}"
            ).evaluate(context)
            is True
        )

    def test_precedence_with_symbolic_operators(self) -> None:
        """Test operator precedence with symbolic operators."""
        context = ContextVariables(
            data={
                "a": True,
                "b": False,
                "c": True,
            }
        )

        # Test precedence: NOT > AND > OR
        # a & !b | c  should be interpreted as  (a & (!b)) | c
        assert ContextExpression("${a} & !${b} | ${c}").evaluate(context) is True

        # Test with explicit parentheses
        assert ContextExpression("${a} & (!${b} | ${c})").evaluate(context) is True
        assert ContextExpression("(${a} & !${b}) | ${c}").evaluate(context) is True
        assert ContextExpression("${a} & (${b} | ${c})").evaluate(context) is True

        # Test complex precedence
        assert ContextExpression("!${a} | ${b} & ${c}").evaluate(context) is False  # (!a) | (b & c)
        assert ContextExpression("!(${a} | ${b}) & ${c}").evaluate(context) is False  # (!(a | b)) & c

    def test_length_operations(self) -> None:
        """Test length operations with lists and other collections."""
        context = ContextVariables(
            data={
                "empty_list": [],
                "non_empty_list": [1, 2, 3],
                "single_item": ["test"],
                "empty_string": "",
                "non_empty_string": "hello",
                "empty_dict": {},
                "non_empty_dict": {"key": "value"},
            }
        )

        # Test length with empty list
        assert ContextExpression("len(${empty_list}) == 0").evaluate(context) is True
        assert ContextExpression("len(${empty_list}) != 0").evaluate(context) is False
        assert ContextExpression("len(${empty_list}) > 0").evaluate(context) is False
        assert ContextExpression("len(${empty_list}) < 1").evaluate(context) is True

        # Test length with non-empty list
        assert ContextExpression("len(${non_empty_list}) == 3").evaluate(context) is True
        assert ContextExpression("len(${non_empty_list}) != 0").evaluate(context) is True
        assert ContextExpression("len(${non_empty_list}) > 0").evaluate(context) is True
        assert ContextExpression("len(${non_empty_list}) >= 3").evaluate(context) is True
        assert ContextExpression("len(${non_empty_list}) < 5").evaluate(context) is True

        # Test length with single item list
        assert ContextExpression("len(${single_item}) == 1").evaluate(context) is True

        # Test length with strings
        assert ContextExpression("len(${empty_string}) == 0").evaluate(context) is True
        assert ContextExpression("len(${non_empty_string}) == 5").evaluate(context) is True
        assert ContextExpression("len(${non_empty_string}) > len(${empty_string})").evaluate(context) is True

        # Test length with dictionaries
        assert ContextExpression("len(${empty_dict}) == 0").evaluate(context) is True
        assert ContextExpression("len(${non_empty_dict}) == 1").evaluate(context) is True

        # Test length with non-existent variables
        with pytest.raises(KeyError):
            assert ContextExpression("len(${non_existent}) == 0").evaluate(context) is True

        # Test length with symbolic operators
        assert ContextExpression("len(${non_empty_list}) > 0 & len(${empty_list}) == 0").evaluate(context) is True
        assert ContextExpression("len(${empty_list}) == 0 | len(${non_empty_list}) == 0").evaluate(context) is True
        assert ContextExpression("!(len(${empty_list}) > 0)").evaluate(context) is True

        # Test complex expressions with length
        assert (
            ContextExpression(
                "len(${non_empty_list}) > 0 & (len(${empty_list}) == 0 | len(${single_item}) == 1)"
            ).evaluate(context)
            is True
        )
        assert (
            ContextExpression(
                "len(${empty_dict}) == 0 & len(${non_empty_dict}) == 1 & len(${non_empty_string}) == 5"
            ).evaluate(context)
            is True
        )
