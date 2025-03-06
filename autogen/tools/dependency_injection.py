# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import sys
from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, get_type_hints

from fast_depends import Depends as FastDepends
from fast_depends import inject
from fast_depends.dependencies import model

from ..agentchat import Agent
from ..doc_utils import export_module
from ..tools.field import Field
from ..tools.function_utils import fix_staticmethod, remove_params

if TYPE_CHECKING:
    from ..agentchat.conversable_agent import ConversableAgent

__all__ = [
    "BaseContext",
    "ChatContext",
    "Depends",
    "get_context_params",
    "inject_params",
    "on",
]


@export_module("autogen.tools")
class BaseContext(ABC):
    """Base class for context classes.

    This is the base class for defining various context types that may be used
    throughout the application. It serves as a parent for specific context classes.
    """

    pass


@export_module("autogen.tools")
class ChatContext(BaseContext):
    """ChatContext class that extends BaseContext.

    This class is used to represent a chat context that holds a list of messages.
    It inherits from `BaseContext` and adds the `messages` attribute.
    """

    def __init__(self, agent: "ConversableAgent") -> None:
        """Initializes the ChatContext with an agent.

        Args:
            agent: The agent to use for retrieving chat messages.
        """
        self._agent = agent

    @property
    def chat_messages(self) -> dict[Agent, list[dict[Any, Any]]]:
        """The messages in the chat.

        Returns:
            A dictionary of agents and their messages.
        """
        return self._agent.chat_messages

    @property
    def last_message(self) -> Optional[dict[str, Any]]:
        """The last message in the chat.

        Returns:
            The last message in the chat.
        """
        return self._agent.last_message()


T = TypeVar("T")


def on(x: T) -> Callable[[], T]:
    def inner(_x: T = x) -> T:
        return _x

    return inner


@export_module("autogen.tools")
def Depends(x: Any) -> Any:  # noqa: N802
    """Creates a dependency for injection based on the provided context or type.

    Args:
        x: The context or dependency to be injected.

    Returns:
        A FastDepends object that will resolve the dependency for injection.
    """
    if isinstance(x, BaseContext):
        return FastDepends(lambda: x)

    return FastDepends(x)


def get_context_params(func: Callable[..., Any], subclass: Union[type[BaseContext], type[ChatContext]]) -> list[str]:
    """Gets the names of the context parameters in a function signature.

    Args:
        func: The function to inspect for context parameters.
        subclass: The subclass to search for.

    Returns:
        A list of parameter names that are instances of the specified subclass.
    """
    sig = inspect.signature(func)
    return [p.name for p in sig.parameters.values() if _is_context_param(p, subclass=subclass)]


def _is_context_param(
    param: inspect.Parameter, subclass: Union[type[BaseContext], type[ChatContext]] = BaseContext
) -> bool:
    # param.annotation.__args__[0] is used to handle Annotated[MyContext, Depends(MyContext(b=2))]
    param_annotation = param.annotation.__args__[0] if hasattr(param.annotation, "__args__") else param.annotation
    return isinstance(param_annotation, type) and issubclass(param_annotation, subclass)


def _is_depends_param(param: inspect.Parameter) -> bool:
    return isinstance(param.default, model.Depends) or (
        hasattr(param.annotation, "__metadata__")
        and type(param.annotation.__metadata__) == tuple
        and isinstance(param.annotation.__metadata__[0], model.Depends)
    )


def _remove_injected_params_from_signature(func: Callable[..., Any]) -> Callable[..., Any]:
    # This is a workaround for Python 3.9+ where staticmethod.__func__ is accessible
    if sys.version_info >= (3, 9) and isinstance(func, staticmethod) and hasattr(func, "__func__"):
        func = fix_staticmethod(func)

    sig = inspect.signature(func)
    params_to_remove = [p.name for p in sig.parameters.values() if _is_context_param(p) or _is_depends_param(p)]
    remove_params(func, sig, params_to_remove)
    return func


def _string_metadata_to_description_field(func: Callable[..., Any]) -> Callable[..., Any]:
    type_hints = get_type_hints(func, include_extras=True)

    for _, annotation in type_hints.items():
        # Check if the annotation itself has metadata (using __metadata__)
        if hasattr(annotation, "__metadata__"):
            metadata = annotation.__metadata__
            if metadata and isinstance(metadata[0], str):
                # Replace string metadata with Field
                annotation.__metadata__ = (Field(description=metadata[0]),)
        # For Python < 3.11, annotations like `Optional` are stored as `Union`, so metadata
        # would be in the first element of __args__ (e.g., `__args__[0]` for `int` in `Optional[int]`)
        elif hasattr(annotation, "__args__") and hasattr(annotation.__args__[0], "__metadata__"):
            metadata = annotation.__args__[0].__metadata__
            if metadata and isinstance(metadata[0], str):
                # Replace string metadata with Field
                annotation.__args__[0].__metadata__ = (Field(description=metadata[0]),)
    return func


def _set_return_annotation_to_any(f: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(f):

        @functools.wraps(f)
        async def _a_wrapped_func(*args: Any, **kwargs: Any) -> Any:
            return await f(*args, **kwargs)

        wrapped_func = _a_wrapped_func

    else:

        @functools.wraps(f)
        def _wrapped_func(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        wrapped_func = _wrapped_func

    sig = inspect.signature(f)

    # Change the return annotation directly on the signature of the wrapper
    wrapped_func.__signature__ = sig.replace(return_annotation=Any)  # type: ignore[attr-defined]

    return wrapped_func


def inject_params(f: Callable[..., Any]) -> Callable[..., Any]:
    """Injects parameters into a function, removing injected dependencies from its signature.

    This function is used to modify a function by injecting dependencies and removing
    injected parameters from the function's signature.

    Args:
        f: The function to modify with dependency injection.

    Returns:
        The modified function with injected dependencies and updated signature.
    """
    # This is a workaround for Python 3.9+ where staticmethod.__func__ is accessible
    if sys.version_info >= (3, 9) and isinstance(f, staticmethod) and hasattr(f, "__func__"):
        f = fix_staticmethod(f)

    f = _string_metadata_to_description_field(f)
    f = _set_return_annotation_to_any(f)
    f = inject(f)
    f = _remove_injected_params_from_signature(f)

    return f
