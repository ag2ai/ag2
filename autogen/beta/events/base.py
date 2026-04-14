# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import operator
import time
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
    # Subclasses may set ``__transient__ = True`` to mark themselves as
    # ephemeral streaming / lifecycle artifacts that should NOT be persisted
    # to durable storage by default.  Examples: ModelMessageChunk (superseded
    # by ModelResponse), TaskProgress (superseded by TaskCompleted), observer
    # lifecycle bookkeeping.
    # NOTE: no type annotation — must NOT be processed as an event Field.
    __transient__ = False

    # Auto-populated Unix timestamp (seconds since epoch) for every event.
    # compare=False: timestamps don't affect equality checks.
    # repr=False: keeps repr() output clean.
    created_at: float = Field(default_factory=time.time, compare=False, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _process_fields(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # MRO walk: map positional args and collect defaults.
        positional_names: list[str] = []
        defaults: dict[str, Any] = {}
        seen: set[str] = set()

        for klass in reversed(type(self).__mro__):
            for name, f in getattr(klass, "_event_fields_", {}).items():
                if name not in seen:
                    if not f.kw_only:
                        positional_names.append(name)
                    if name not in kwargs:
                        default = f.get_default()
                        if default is not Ellipsis:
                            defaults[name] = default
                seen.add(name)

        if args:
            if len(args) > len(positional_names):
                raise TypeError(
                    f"{type(self).__name__}() takes {len(positional_names)} "
                    f"positional argument(s) but {len(args)} were given"
                )
            for name, value in zip(positional_names, args):
                if name in kwargs:
                    raise TypeError(f"{type(self).__name__}() got multiple values for argument '{name}'")
                kwargs[name] = value
                defaults.pop(name, None)

        # Apply defaults first, then user-provided kwargs so that
        # property setters (e.g. content -> _content) aren't overwritten
        # by a field default applied afterwards.
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize this event to a JSON-compatible dictionary."""
        from ._serialization import event_to_dict

        return event_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseEvent":
        """Reconstruct an event from a serialized dictionary.

        Filters input to only fields known by this class (via MRO Field
        descriptors), then constructs via ``cls(**filtered)``.
        """
        from ._serialization import deserialize_payload

        # Collect known field names across the MRO
        known_fields: set[str] = set()
        for klass in cls.__mro__:
            for name, f in getattr(klass, "_event_fields_", {}).items():
                known_fields.add(name)

        # Deserialize nested events/special types, then filter to known fields
        deserialized = deserialize_payload(data)
        filtered = {k: v for k, v in deserialized.items() if k in known_fields}
        return cls(**filtered)


# BaseEvent's own fields (e.g. created_at) are not processed by
# __init_subclass__ since that only fires for *sub*classes.
_process_fields(BaseEvent)
