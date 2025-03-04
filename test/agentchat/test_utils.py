# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.agentchat.utils import ContextExpression


class TestContextExpressionNewSyntax:
    """Tests for the ContextExpression class with the new ${var_name} syntax."""

    def test_basic_boolean_operations(self) -> None:
        """Test basic boolean operations."""
        context = {
            "var_true": True,
            "var_false": False,
        }

        # Test simple variable lookup
        assert ContextExpression("${var_true}").evaluate(context) is True
        assert ContextExpression("${var_false}").evaluate(context) is False

        # Test NOT operation
        assert ContextExpression("not(${var_true})").evaluate(context) is False
        assert ContextExpression("not(${var_false})").evaluate(context) is True

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

    def test_numeric_comparisons(self) -> None:
        """Test numeric comparisons."""
        context = {
            "num_zero": 0,
            "num_one": 1,
            "num_ten": 10,
            "num_negative": -5,
        }

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

    def test_string_comparisons(self) -> None:
        """Test string comparisons."""
        context = {
            "str_empty": "",
            "str_hello": "hello",
            "str_world": "world",
        }

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

    def test_complex_expressions(self) -> None:
        """Test complex expressions with nested operations."""
        context = {
            "user_logged_in": True,
            "is_admin": False,
            "has_permission": True,
            "user_age": 25,
            "min_age": 18,
            "max_attempts": 3,
            "current_attempts": 2,
            "username": "john_doe",
        }

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

    def test_missing_variables(self) -> None:
        """Test behavior with missing variables."""
        context = {
            "var_true": True,
        }

        # Missing variables should default to False
        assert ContextExpression("${non_existent_var}").evaluate(context) is False
        assert ContextExpression("${var_true} and ${non_existent_var}").evaluate(context) is False
        assert ContextExpression("${var_true} or ${non_existent_var}").evaluate(context) is True
        assert ContextExpression("not(${non_existent_var})").evaluate(context) is True

    def test_real_world_examples(self) -> None:
        """Test real-world examples with the new syntax."""
        context = {
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

        # Authentication example
        assert ContextExpression("${logged_in} and not(${is_admin})").evaluate(context) is True

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
                "${logged_in} and ${customer_angry} and not(${manager_already_involved}) and ${account_type} == 'premium'"
            ).evaluate(context)
            is True
        )
