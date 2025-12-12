# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

from .base_event import BaseEvent, resolve_print_callable, wrap_event


@wrap_event
class PrintEvent(BaseEvent):
    """Print message"""

    objects: list[str]
    """List of objects to print"""
    sep: str
    """Separator between objects"""
    end: str
    """End of the print"""

    def __init__(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False, uuid: UUID | None = None):
        objects_as_string = [self._to_json(x) for x in objects]

        super().__init__(uuid=uuid, objects=objects_as_string, sep=sep, end=end)

    def _to_json(self, obj: Any) -> str:
        if isinstance(obj, str):
            return obj

        if hasattr(obj, "model_dump_json"):
            return obj.model_dump_json()  # type: ignore [no-any-return]
        try:
            return json.dumps(obj)
        except Exception:
            return str(obj)
            # return repr(obj)

    def print(self, f: Callable[..., Any] | None = None) -> None:
        f = resolve_print_callable(f)
        f(*self.objects, sep=self.sep, end=self.end, flush=True)
