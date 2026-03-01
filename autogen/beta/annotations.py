from typing import Annotated, Any

from fast_depends.library import CustomField

from .stream import Context as ContextType
from .tools.tool import CONTEXT_OPTION_NAME


class Inject(CustomField):
    param_name: str

    def __init__(
        self,
        real_name: str = "",
        *,
        default: Any = Ellipsis,
        cast: bool = False,
    ) -> None:
        self.name = real_name
        self.default = default
        super().__init__(
            cast=cast,
            required=(default is Ellipsis),
        )

    def use(self, /, **kwargs: Any) -> dict[str, Any]:
        if ctx := kwargs.get(CONTEXT_OPTION_NAME):
            if opt := ctx.container.get(self.name or self.param_name):
                kwargs[self.param_name] = opt
            elif not self.required:
                kwargs[self.param_name] = self.default
        return kwargs


class ContextField(CustomField):
    def use(self, /, **kwargs: Any) -> dict[str, Any]:
        if ctx := kwargs.get(CONTEXT_OPTION_NAME):
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
