# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Every condition form the user guide teaches, as the checker sees it.

Checked by ``test/typing/test_condition_dsl_plugin.py``; not imported by the
suite. ``reveal_type`` lines are the assertions — mypy reports them as notes.
"""

from ag2.events import BaseEvent, Condition, Field, ToolCallEvent


class Order(BaseEvent):
    total: int = Field(default=0)
    label: str = Field(default="")


def takes_condition(condition: Condition) -> bool:
    return condition(object())


# Class-level field access is the descriptor, not the field's value type.
reveal_type(Order.total)  # N: Revealed type is "ag2.events.base.FieldInfo"

# Instance access still yields the declared value type.
reveal_type(Order(total=1).total)  # N: Revealed type is "int"

# Every comparison operator builds a Condition.
reveal_type(Order.total == 1)  # N: Revealed type is "ag2.events.conditions.Condition"
reveal_type(Order.total != 1)  # N: Revealed type is "ag2.events.conditions.Condition"
reveal_type(Order.total < 1)  # N: Revealed type is "ag2.events.conditions.Condition"
reveal_type(Order.total <= 1)  # N: Revealed type is "ag2.events.conditions.Condition"
reveal_type(Order.total > 1)  # N: Revealed type is "ag2.events.conditions.Condition"
reveal_type(Order.total >= 1)  # N: Revealed type is "ag2.events.conditions.Condition"
reveal_type(Order.label.is_(None))  # N: Revealed type is "ag2.events.conditions.Condition"

# The form the user guide teaches, against a shipped event.
takes_condition(ToolCallEvent.name == "search")

# Composition: operators and their method spellings.
takes_condition((Order.total > 0) & (Order.total < 10))
takes_condition((Order.total < 0) | (Order.total > 10))
takes_condition(~(Order.label == "x"))
takes_condition((Order.total > 0).and_(Order.total < 10))
takes_condition((Order.total < 0).or_(Order.total > 10))
takes_condition((Order.label == "x").not_())

# An event class is a condition on its own type.
takes_condition(~Order)
takes_condition(Order.not_())
takes_condition(Order.or_(ToolCallEvent))

# A field that does not exist is an error, not a silent Any.
takes_condition(Order.missing == 1)  # E: "type[Order]" has no attribute "missing"  [attr-defined]
