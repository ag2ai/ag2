# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A mypy plugin that teaches the checker the event condition DSL.

An event field is declared with the type of its *value*, so without help a
checker reads ``ToolCallEvent.name`` as ``str`` and ``ToolCallEvent.name ==
"search"`` as ``bool`` — a ``bool`` handed to an API that wants a ``Condition``.
This plugin makes class-level access to an event field resolve to the field
descriptor instead, whose comparison operators already return ``Condition``.

Enable it the way the pydantic plugin is enabled, in ``pyproject.toml``::

    [tool.mypy]
    plugins = ["ag2.mypy"]
"""

from collections.abc import Callable

from mypy.nodes import SymbolTableNode, TypeInfo, Var
from mypy.options import Options
from mypy.plugin import AttributeContext, Plugin
from mypy.types import Instance, Type

BASE_EVENT_FULLNAME = "ag2.events.base.BaseEvent"
FIELD_INFO_FULLNAME = "ag2.events.base.FieldInfo"


class AG2Plugin(Plugin):
    """Resolves class-level event field access as the field descriptor."""

    def __init__(self, options: Options) -> None:
        super().__init__(options)
        self._field_info: TypeInfo | None = None

    def get_class_attribute_hook(self, fullname: str) -> Callable[[AttributeContext], Type] | None:
        class_fullname, _, attr_name = fullname.rpartition(".")
        if not class_fullname or attr_name.startswith("_"):
            return None

        symbol = self.lookup_fully_qualified(class_fullname)
        if symbol is None or not isinstance(symbol.node, TypeInfo):
            return None

        info = symbol.node
        if not any(base.fullname == BASE_EVENT_FULLNAME for base in info.mro):
            return None

        # Searched along the MRO, not just this class: an event that adds no
        # fields of its own still filters on the ones it inherits, and
        # ``BuiltinToolCallEvent.name`` — the form every builtin tool uses — is
        # declared two classes up.
        #
        # Only annotated data attributes become fields; methods, properties and
        # class-level markers such as ``__transient__`` are left alone.
        if not any(_is_field(base.names.get(attr_name)) for base in info.mro):
            return None

        return self._as_field_descriptor

    def _as_field_descriptor(self, ctx: AttributeContext) -> Type:
        if self._field_info is None:
            symbol = self.lookup_fully_qualified(FIELD_INFO_FULLNAME)
            if symbol is None or not isinstance(symbol.node, TypeInfo):
                # The events package is not in this build; leave the type alone
                # rather than guess at one.
                return ctx.default_attr_type
            self._field_info = symbol.node
        return Instance(self._field_info, [])


def _is_field(attr: "SymbolTableNode | None") -> bool:
    return attr is not None and isinstance(attr.node, Var) and not attr.node.is_classvar


def plugin(version: str) -> type[Plugin]:
    return AG2Plugin
