# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import EllipsisType
from typing import Annotated, Any

from fast_depends.library import CustomField

from .context import Context as ContextType
from .utils import CONTEXT_OPTION_NAME


class Inject(CustomField):
    param_name: str

    def __init__(
        self,
        real_name: str = "",
        *,
        default: Any = Ellipsis,
        default_factory: Callable[[], Any] | EllipsisType = Ellipsis,
        cast: bool = False,
    ) -> None:
        self.name = real_name
        self.default = default
        self.default_factory = default_factory
        super().__init__(
            cast=cast,
            required=(default is Ellipsis),
        )

    def use(self, /, **kwargs: Any) -> dict[str, Any]:
        if ctx := kwargs.get(CONTEXT_OPTION_NAME):
            assert self.param_name

            name = self.name or self.param_name
            if opt := ctx.dependencies.get(name):
                kwargs[self.param_name] = opt
            elif self.default is not Ellipsis:
                kwargs[self.param_name] = ctx.dependencies[name] = self.default
            elif self.default_factory is not Ellipsis:
                kwargs[self.param_name] = ctx.dependencies[name] = self.default_factory()
        return kwargs


class Variable(CustomField):
    param_name: str

    def __init__(
        self,
        real_name: str = "",
        *,
        default: Any = Ellipsis,
        default_factory: Callable[[], Any] | EllipsisType = Ellipsis,
        cast: bool = False,
    ) -> None:
        self.name = real_name
        self.default = default
        self.default_factory = default_factory
        super().__init__(
            cast=cast,
            required=(default is Ellipsis),
        )

    def use(self, /, **kwargs: Any) -> dict[str, Any]:
        if ctx := kwargs.get(CONTEXT_OPTION_NAME):
            assert self.param_name

            name = self.name or self.param_name
            if opt := ctx.variables.get(name):
                kwargs[self.param_name] = opt
            elif self.default is not Ellipsis:
                kwargs[self.param_name] = ctx.variables[name] = self.default
            elif self.default_factory is not Ellipsis:
                kwargs[self.param_name] = ctx.variables[name] = self.default_factory()
        return kwargs


class ContextField(CustomField):
    def use(self, /, **kwargs: Any) -> dict[str, Any]:
        if ctx := kwargs.get(CONTEXT_OPTION_NAME):
            assert self.param_name

            kwargs[self.param_name] = ctx
        return kwargs


# Wrap context to Custom field to make it option name agnostic
# `ctx: Context`
# `context: Context`
# `anything: Context`
# are equal now
Context = Annotated[
    ContextType,
    ContextField(cast=False),
]
