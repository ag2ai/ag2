import operator
import types
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias


class Condition(ABC):
    @abstractmethod
    def __call__(self, event: Any) -> bool:
        raise NotImplementedError

    def __and__(self, other: Any) -> "AndCondition":
        if not isinstance(other, Condition):
            other = TypeCondition(other)
        return AndCondition(self, other)

    def and_(self, other: Any) -> "AndCondition":
        return self & other

    def __or__(self, other: Any) -> "OrCondition":
        if not isinstance(other, Condition):
            other = TypeCondition(other)
        return OrCondition(self, other)

    def or_(self, other: Any) -> "OrCondition":
        return self | other

    def __invert__(self) -> "NotCondition":
        return NotCondition(self)

    def not_(self) -> "NotCondition":
        return ~self


ClassInfo: TypeAlias = type | types.UnionType | tuple["ClassInfo", ...]


class TypeCondition(Condition):
    def __init__(self, t: ClassInfo) -> None:
        self._expected_type = t

    def __call__(self, event: Any) -> bool:
        return isinstance(event, self._expected_type)

    def __repr__(self) -> str:
        if isinstance(self._expected_type, type):
            return f"IsType({self._expected_type.__name__})"
        elif isinstance(self._expected_type, tuple):
            names = [t.__name__ if isinstance(t, type) else str(t) for t in self._expected_type]
            return f"IsType({' | '.join(names)})"
        else:
            return f"IsType({self._expected_type})"


class AndCondition(Condition):
    def __init__(self, *conditions: Condition) -> None:
        self.conditions = conditions

    def __call__(self, event: Any) -> bool:
        return all(condition(event) for condition in self.conditions)

    def __repr__(self) -> str:
        flattened: list[Condition] = []
        for c in self.conditions:
            if isinstance(c, AndCondition):
                flattened.extend(c.conditions)
            else:
                flattened.append(c)
        parts = " & ".join(f"{c!r}" for c in flattened)
        return f"And({parts})"


class OrCondition(Condition):
    def __init__(self, *conditions: Condition) -> None:
        self.conditions = conditions

    def __call__(self, event: Any) -> bool:
        return any(condition(event) for condition in self.conditions)

    def __repr__(self) -> str:
        flattened: list[Condition] = []
        for c in self.conditions:
            if isinstance(c, OrCondition):
                flattened.extend(c.conditions)
            else:
                flattened.append(c)
        parts = " | ".join(f"{c!r}" for c in flattened)
        return f"Or({parts})"


class NotCondition(Condition):
    def __init__(self, condition: Condition) -> None:
        self.condition = condition

    def __call__(self, event: Any) -> bool:
        return not self.condition(event)

    def __repr__(self) -> str:
        return f"~{self.condition!r}"


class OpCondition(Condition):
    def __init__(
        self,
        op_func: Callable[[Any, Any], bool],
        field_name: str,
        value: Any,
        event_class: type,
    ) -> None:
        self.op_func = op_func
        self.field_name = field_name
        self.value = value
        self.event_class = event_class

    def __call__(self, event: Any) -> bool:
        if not isinstance(event, self.event_class):
            return False

        field_value = getattr(event, self.field_name, None)
        return self.op_func(field_value, self.value)

    def __repr__(self) -> str:
        op_name = op_names.get(self.op_func, "?")
        return f"Is({self.event_class.__name__}.{self.field_name} {op_name} {self.value!r})"


def check_eq(x: Any, y: Any) -> Any:
    if not isinstance(x, type(y)):
        return False
    return x == y


op_names = {
    check_eq: "==",
    operator.ne: "!=",
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
    operator.is_: "is",
}
