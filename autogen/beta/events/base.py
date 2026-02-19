import operator
from typing import Any

from .conditions import Condition, OpCondition, OrCondition, TypeCondition, check_eq


class Field:
    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: Any | None, owner: type) -> Any:
        self.event_class = owner
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self.name] = value

    def __eq__(self, other: Any) -> Condition:  # type: ignore[override]
        return OpCondition(check_eq, self.name, other, self.event_class)

    def __ne__(self, other: Any) -> Condition:  # type: ignore[override]
        return OpCondition(operator.ne, self.name, other, self.event_class)

    def __lt__(self, other: Any) -> Condition:
        return OpCondition(operator.lt, self.name, other, self.event_class)

    def __le__(self, other: Any) -> Condition:
        return OpCondition(operator.le, self.name, other, self.event_class)

    def __gt__(self, other: Any) -> Condition:
        return OpCondition(operator.gt, self.name, other, self.event_class)

    def __ge__(self, other: Any) -> Condition:
        return OpCondition(operator.ge, self.name, other, self.event_class)

    def is_(self, other: Any) -> Condition:
        return OpCondition(operator.is_, self.name, other, self.event_class)


class EventMeta(type):
    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> type:
        annotations = namespace.get("__annotations__", {})

        for field_name in annotations:
            namespace[field_name] = Field(field_name)

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        namespace["__init__"] = __init__

        def __repr__(self) -> str:
            fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
            return f"{self.__class__.__name__}({fields})"

        namespace["__repr__"] = __repr__

        return super().__new__(mcs, name, bases, namespace)

    def __or__(self, other: Any) -> Any:
        return TypeCondition(self) | other

    def or_(self, other: Any) -> OrCondition:
        return TypeCondition(self) | other


class BaseEvent(metaclass=EventMeta):
    pass
