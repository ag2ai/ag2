# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import operator
from collections.abc import Callable
from types import EllipsisType
from typing import Any

from .conditions import Condition, NotCondition, OpCondition, OrCondition, TypeCondition, check_eq


class Field:
    def __init__(
        self,
        default: Any = Ellipsis,
        *,
        default_factory: Callable[[], Any] | EllipsisType = Ellipsis,
    ) -> None:
        self.name = ""

        self.__default = default
        self.__default_factory = default_factory

    def get_default(self) -> Any:
        if self.__default_factory is not Ellipsis:
            return self.__default_factory()
        return self.__default

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
        # fields is populated after class creation (supports Python 3.14+ PEP 649 lazy annotations)
        fields: dict[str, Field] = {}

        def __init__(self: object, **kwargs: Any) -> None:  # noqa: N807
            kwargs = {
                name: default for name, f in fields.items() if (default := f.get_default()) is not Ellipsis
            } | kwargs

            for key, value in kwargs.items():
                setattr(self, key, value)

        namespace["__init__"] = __init__

        def __repr__(self: object) -> str:  # noqa: N807
            fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
            return f"{self.__class__.__name__}({fields})"

        if "__repr__" not in namespace and "__str__" not in namespace:
            namespace["__repr__"] = __repr__

        cls = super().__new__(mcs, name, bases, namespace)

        # Get annotations in a Python 3.14+ compatible way (PEP 649: lazy annotation evaluation
        # means __annotations__ is no longer eagerly populated in the class namespace dict).
        try:
            # Python 3.14+
            import annotationlib  # pyright: ignore[reportMissingImports]

            annotations = annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)
        except ImportError:
            annotations = vars(cls).get("__annotations__", {})

        for field_name in annotations:
            raw = namespace.get(field_name)
            if not raw:
                field = Field()
            elif isinstance(raw, Field):
                field = raw
            else:
                field = Field(raw)

            if not field.name:
                field.name = field_name

            fields[field_name] = field
            setattr(cls, field_name, field)

        return cls

    def __or__(cls, other: Any) -> Any:
        return TypeCondition(cls).or_(other)

    def or_(cls, other: Any) -> OrCondition:
        return TypeCondition(cls).or_(other)

    def not_(cls) -> NotCondition:
        return TypeCondition(cls).not_()


class BaseEvent(metaclass=EventMeta):
    pass
