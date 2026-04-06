# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import operator
from collections.abc import Callable
from types import EllipsisType
from typing import Any

from typing_extensions import dataclass_transform

from .conditions import Condition, NotCondition, OpCondition, OrCondition, TypeCondition, check_eq


class Field:
    def __init__(
        self,
        default: Any = Ellipsis,
        *,
        default_factory: Callable[[], Any] | EllipsisType = Ellipsis,
        init: bool = True,
        repr: bool = True,
        compare: bool = True,
        hash: bool | None = None,
        kw_only: bool = True,
    ) -> None:
        self.name = ""

        self.init = init
        self.repr = repr
        self.compare = compare
        self.hash = hash
        self.kw_only = kw_only

        self._default = default
        self._default_factory = default_factory

    def get_default(self) -> Any:
        if self._default_factory is not Ellipsis:
            return self._default_factory()
        return self._default

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


class _ConditionMeta(type):
    """Metaclass providing class-level condition operators (|, or_, not_)."""

    def __or__(cls, other: Any) -> Any:
        return TypeCondition(cls).or_(other)

    def or_(cls, other: Any) -> OrCondition:
        return TypeCondition(cls).or_(other)

    def not_(cls) -> NotCondition:
        return TypeCondition(cls).not_()


def _process_fields(cls: type) -> None:
    """Process annotations and set up Field descriptors for a class."""
    fields: dict[str, Field] = {}

    # Get annotations in a Python 3.14+ compatible way (PEP 649: lazy annotation evaluation
    # means __annotations__ is no longer eagerly populated in the class namespace dict).
    try:
        # Python 3.14+
        import annotationlib  # pyright: ignore[reportMissingImports]

        annotations = annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)
    except ImportError:
        annotations = vars(cls).get("__annotations__", {})

    own_namespace = vars(cls)
    for field_name in annotations:
        raw = own_namespace.get(field_name)
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

    cls._event_fields_ = fields  # type: ignore[attr-defined]


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Field,),
)
class BaseEvent(metaclass=_ConditionMeta):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _process_fields(cls)

    def __init__(self, **kwargs: Any) -> None:
        # Apply defaults first, then user-provided kwargs so that
        # property setters (e.g. content -> _content) aren't overwritten
        # by a field default applied afterwards.
        defaults: dict[str, Any] = {}
        for klass in reversed(type(self).__mro__):
            for name, f in getattr(klass, "_event_fields_", {}).items():
                if name not in kwargs:
                    default = f.get_default()
                    if default is not Ellipsis:
                        defaults[name] = default

        for key, value in defaults.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented

        for klass in type(self).__mro__:
            for name, f in getattr(klass, "_event_fields_", {}).items():
                if f.compare and getattr(self, name) != getattr(other, name):
                    return False

        return True

    def __repr__(self) -> str:
        hidden = set()
        for klass in type(self).__mro__:
            for name, f in getattr(klass, "_event_fields_", {}).items():
                if not f.repr:
                    hidden.add(name)

        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_") and k not in hidden)
        return f"{self.__class__.__name__}({fields})"
