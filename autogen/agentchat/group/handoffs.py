# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, overload

from pydantic import BaseModel, Field

from .on_condition import OnCondition
from .on_context_condition import OnContextCondition
from .targets.transition_target import TransitionTarget

__all__ = ["Handoffs"]


class Handoffs(BaseModel):
    """
    Container for all handoff transition conditions of a ConversableAgent.

    Three types of conditions can be added, each with a different order and time of use:
    1. OnContextConditions (evaluated without an LLM)
    2. OnConditions (evaluated with an LLM)
    3. After work TransitionTarget (if no other transition is triggered)

    Supports method chaining:
    agent.handoffs.add_context_conditions([condition1]) \
                   .add_llm_condition(condition2) \
                   .set_after_work(after_work)
    """

    context_conditions: list[OnContextCondition] = Field(default_factory=list)
    llm_conditions: list[OnCondition] = Field(default_factory=list)
    after_work: Optional[TransitionTarget] = None

    def add_context_condition(self, condition: OnContextCondition) -> "Handoffs":
        """
        Add a single context condition.

        Args:
            condition: The OnContextCondition to add

        Returns:
            Self for method chaining
        """
        # Validate that it is an OnContextCondition
        if not isinstance(condition, OnContextCondition):
            raise TypeError(f"Expected an OnContextCondition instance, got {type(condition).__name__}")

        self.context_conditions.append(condition)
        return self

    def add_context_conditions(self, conditions: list[OnContextCondition]) -> "Handoffs":
        """
        Add multiple context conditions.

        Args:
            conditions: List of OnContextConditions to add

        Returns:
            Self for method chaining
        """
        # Validate that it is a list of OnContextConditions
        if not all(isinstance(condition, OnContextCondition) for condition in conditions):
            raise TypeError("All conditions must be of type OnContextCondition")

        self.context_conditions.extend(conditions)
        return self

    def add_llm_condition(self, condition: OnCondition) -> "Handoffs":
        """
        Add a single LLM condition.

        Args:
            condition: The OnCondition to add

        Returns:
            Self for method chaining
        """
        # Validate that it is an OnCondition
        if not isinstance(condition, OnCondition):
            raise TypeError(f"Expected an OnCondition instance, got {type(condition).__name__}")

        self.llm_conditions.append(condition)
        return self

    def add_llm_conditions(self, conditions: list[OnCondition]) -> "Handoffs":
        """
        Add multiple LLM conditions.

        Args:
            conditions: List of OnConditions to add

        Returns:
            Self for method chaining
        """
        # Validate that it is a list of OnConditions
        if not all(isinstance(condition, OnCondition) for condition in conditions):
            raise TypeError("All conditions must be of type OnCondition")

        self.llm_conditions.extend(conditions)
        return self

    def set_after_work(self, target: TransitionTarget) -> "Handoffs":
        """
        Set the after work target (only one allowed).

        Args:
            target: The after work TransitionTarget to set

        Returns:
            Self for method chaining
        """
        if not isinstance(target, TransitionTarget):
            raise TypeError(f"Expected a TransitionTarget instance, got {type(target).__name__}")

        self.after_work = target
        return self

    @overload
    def add(self, condition: OnContextCondition) -> "Handoffs": ...

    @overload
    def add(self, condition: OnCondition) -> "Handoffs": ...

    def add(self, condition: Union[OnContextCondition, OnCondition]) -> "Handoffs":
        """
        Add a single condition (OnContextCondition or OnCondition).

        Args:
            condition: The condition to add (OnContextCondition or OnCondition)

        Raises:
            TypeError: If the condition type is not supported

        Returns:
            Self for method chaining
        """
        # This add method is a helper method designed to make it easier for
        # adding handoffs without worrying about the specific type.
        if isinstance(condition, OnContextCondition):
            return self.add_context_condition(condition)
        elif isinstance(condition, OnCondition):
            return self.add_llm_condition(condition)
        else:
            raise TypeError(f"Unsupported condition type: {type(condition).__name__}")

    def add_many(self, conditions: list[Union[OnContextCondition, OnCondition]]) -> "Handoffs":
        """
        Add multiple conditions of any supported types (OnContextCondition and OnCondition).

        Args:
            conditions: List of conditions to add

        Raises:
            TypeError: If an unsupported condition type is provided

        Returns:
            Self for method chaining
        """
        # This add_many method is a helper method designed to make it easier for
        # adding handoffs without worrying about the specific type.
        context_conditions = []
        llm_conditions = []

        for condition in conditions:
            if isinstance(condition, OnContextCondition):
                context_conditions.append(condition)
            elif isinstance(condition, OnCondition):
                llm_conditions.append(condition)
            else:
                raise TypeError(f"Unsupported condition type: {type(condition).__name__}")

        if context_conditions:
            self.add_context_conditions(context_conditions)
        if llm_conditions:
            self.add_llm_conditions(llm_conditions)

        return self

    def clear(self) -> "Handoffs":
        """
        Clear all handoff conditions.

        Returns:
            Self for method chaining
        """
        self.context_conditions.clear()
        self.llm_conditions.clear()
        self.after_work = None
        return self

    def get_llm_conditions_by_target_type(self, target_type: type) -> list[OnCondition]:
        """
        Get OnConditions for a specific target type.

        Args:
            target_type: The type of condition to retrieve

        Returns:
            List of conditions of the specified type, or None if none exist
        """
        return [on_condition for on_condition in self.llm_conditions if on_condition.has_target_type(target_type)]

    def get_context_conditions_by_target_type(self, target_type: type) -> list[OnContextCondition]:
        """
        Get OnContextConditions for a specific target type.

        Args:
            target_type: The type of condition to retrieve

        Returns:
            List of conditions of the specified type, or None if none exist
        """
        return [
            on_context_condition
            for on_context_condition in self.context_conditions
            if on_context_condition.has_target_type(target_type)
        ]

    def get_llm_conditions_requiring_wrapping(self) -> list[OnCondition]:
        """
        Get LLM conditions that have targets that require wrapping.

        Returns:
            List of LLM conditions that require wrapping
        """
        return [condition for condition in self.llm_conditions if condition.target_requires_wrapping()]

    def get_context_conditions_requiring_wrapping(self) -> list[OnContextCondition]:
        """
        Get context conditions that have targets that require wrapping.

        Returns:
            List of context conditions that require wrapping
        """
        return [condition for condition in self.context_conditions if condition.target_requires_wrapping()]

    def set_llm_function_names(self) -> None:
        """
        Set the LLM function names for all LLM conditions, creating unique names for each function.
        """
        for i, condition in enumerate(self.llm_conditions):
            # Function names are made unique and allow multiple OnCondition's to the same agent
            condition.llm_function_name = f"transfer_to_{condition.target.normalized_name()}_{i + 1}"
