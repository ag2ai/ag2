# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from contextlib import contextmanager, suppress
from functools import wraps
from logging import getLogger
from typing import Any, Callable, Generic, Optional, Type, TypeVar, Union

__all__ = ["optional_import_block", "patch_object", "require_optional_import", "skip_on_missing_imports"]

logger = getLogger(__name__)


class Result:
    def __init__(self) -> None:
        self._failed: Optional[bool] = None

    @property
    def is_successful(self) -> bool:
        if self._failed is None:
            raise ValueError("Result not set")
        return not self._failed


@contextmanager
def optional_import_block() -> Generator[Result, None, None]:
    """Guard a block of code to suppress ImportErrors

    A context manager to temporarily suppress ImportErrors.
    Use this to attempt imports without failing immediately on missing modules.

    Example:
    ```python
    with optional_import_block():
        import some_module
        import some_other_module
    ```
    """
    result = Result()
    try:
        yield result
        result._failed = False
    except ImportError as e:
        # Ignore ImportErrors during this context
        logger.info(f"Ignoring ImportError: {e}")
        result._failed = True


def get_missing_imports(modules: Union[str, Iterable[str]]) -> list[str]:
    """Get missing modules from a list of module names

    Args:
        modules: Module name or list of module names

    Returns:
        List of missing module names
    """
    if isinstance(modules, str):
        modules = [modules]

    return [m for m in modules if m not in sys.modules]


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class PatchObject(ABC, Generic[T]):
    def __init__(self, o: T, missing_modules: Iterable[str], dep_target: str):
        if not self.accept(o):
            raise ValueError(f"Cannot patch object of type {type(o)}")

        self.o = o
        self.missing_modules = list(missing_modules)
        self.dep_target = dep_target

    @classmethod
    @abstractmethod
    def accept(cls, o: Any) -> bool: ...

    @abstractmethod
    def patch(self) -> T: ...

    def get_object_with_metadata(self) -> Any:
        return self.o

    @property
    def msg(self) -> str:
        o = self.get_object_with_metadata()
        plural = len(self.missing_modules) > 1
        fqn = f"{o.__module__}.{o.__name__}" if hasattr(o, "__module__") else o.__name__
        modules_str = ", ".join([f"'{m}'" for m in self.missing_modules])
        return f"Module{'s' if plural else ''} {modules_str} needed for {fqn} {'are' if plural else 'is'} missing, please install it using 'pip install ag2[{self.dep_target}]'"

    def copy_metadata(self, retval: T) -> None:
        """Copy metadata from original object to patched object

        Args:
            retval: Patched object

        """
        o = self.o
        if hasattr(o, "__doc__"):
            retval.__doc__ = o.__doc__
        if hasattr(o, "__name__"):
            retval.__name__ = o.__name__  # type: ignore[attr-defined]
        if hasattr(o, "__module__"):
            retval.__module__ = o.__module__

    _registry: list[Type["PatchObject[Any]"]] = []

    @classmethod
    def register(cls) -> Callable[[Type["PatchObject[Any]"]], Type["PatchObject[Any]"]]:
        def decorator(subclass: Type["PatchObject[Any]"]) -> Type["PatchObject[Any]"]:
            cls._registry.append(subclass)
            return subclass

        return decorator

    @classmethod
    def create(cls, o: T, *, missing_modules: Iterable[str], dep_target: str) -> Optional["PatchObject[T]"]:
        for subclass in cls._registry:
            if subclass.accept(o):
                return subclass(o, missing_modules, dep_target)
        return None


@PatchObject.register()
class PatchCallable(PatchObject[F]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isfunction(o) or inspect.ismethod(o)

    def patch(self) -> F:
        f: Callable[..., Any] = self.o

        @wraps(f.__call__)  # type: ignore[operator]
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)  # type: ignore[arg-type]

        return _call  # type: ignore[return-value]


@PatchObject.register()
class PatchStatic(PatchObject[F]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        # return inspect.ismethoddescriptor(o)
        return isinstance(o, staticmethod)

    def patch(self) -> F:
        f: Callable[..., Any] = self.o.__func__  # type: ignore[attr-defined]

        @wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)  # type: ignore[arg-type]

        return staticmethod(_call)  # type: ignore[return-value]

    def get_object_with_metadata(self) -> Any:
        return self.o.__func__  # type: ignore[attr-defined]


@PatchObject.register()
class PatchInit(PatchObject[F]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.ismethoddescriptor(o) and o.__name__ == "__init__"

    def patch(self) -> F:
        f: Callable[..., Any] = self.o

        @wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)  # type: ignore[arg-type]

        return staticmethod(_call)  # type: ignore[return-value]

    def get_object_with_metadata(self) -> Any:
        return self.o


@PatchObject.register()
class PatchProperty(PatchObject[Any]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isdatadescriptor(o) and hasattr(o, "fget")

    def patch(self) -> property:
        if not hasattr(self.o, "fget"):
            raise ValueError(f"Cannot patch property without getter: {self.o}")
        f: Callable[..., Any] = self.o.fget

        @wraps(f)
        def _call(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(self.msg)

        self.copy_metadata(_call)

        return property(_call)

    def get_object_with_metadata(self) -> Any:
        return self.o.fget


@PatchObject.register()
class PatchClass(PatchObject[Type[Any]]):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isclass(o)

    def patch(self) -> Type[Any]:
        # Patch __init__ method if possible

        for name, member in inspect.getmembers(self.o):
            if name.startswith("__") and name != "__init__":
                continue
            patched = patch_object(
                member, missing_modules=self.missing_modules, dep_target=self.dep_target, fail_if_not_patchable=False
            )
            with suppress(AttributeError):
                setattr(self.o, name, patched)

        return self.o


def patch_object(o: T, *, missing_modules: Iterable[str], dep_target: str, fail_if_not_patchable: bool = True) -> T:
    patcher = PatchObject.create(o, missing_modules=missing_modules, dep_target=dep_target)
    if fail_if_not_patchable and patcher is None:
        raise ValueError(f"Cannot patch object of type {type(o)}")

    return patcher.patch() if patcher else o


def require_optional_import(modules: Union[str, Iterable[str]], dep_target: str) -> Callable[[T], T]:
    """Decorator to handle optional module dependencies

    Args:
        modules: Module name or list of module names required
        dep_target: Target name for pip installation (e.g. 'test' in pip install ag2[test])
    """
    missing_modules = get_missing_imports(modules)

    if not missing_modules:

        def decorator(o: T) -> T:
            return o
    else:

        def decorator(o: T) -> T:
            return patch_object(o, missing_modules=missing_modules, dep_target=dep_target)

    return decorator


def skip_on_missing_imports(modules: Union[str, Iterable[str]], dep_target: Optional[str] = None) -> Callable[[T], T]:
    """Decorator to skip a test if an optional module is missing

    Args:
        module: Module name
        dep_target: Target name for pip installation (e.g. 'test' in pip install ag2[test])
    """
    missing_modules = get_missing_imports(modules)
    # Add pytest.mark.dep_target decorator
    # For example, if dep_target is "jupyter-executor" add pytest.mark.jupyter_executor
    mark_name = dep_target.replace("-", "_") if dep_target else "openai"

    if not missing_modules:

        def decorator(o: T) -> T:
            import pytest

            pytest_mark_o = getattr(pytest.mark, mark_name)(o)
            return pytest_mark_o  # type: ignore[no-any-return]
    else:

        def decorator(o: T) -> T:
            import pytest

            install_target = "" if dep_target is None else f"[{dep_target}]"
            pytest_mark_o = getattr(pytest.mark, mark_name)(o)
            return pytest.mark.skip(  # type: ignore[return-value,no-any-return]
                f"Missing module{'s' if len(missing_modules) > 1 else ''}: {', '.join(missing_modules)}. Install using 'pip install ag2{install_target}'"
            )(pytest_mark_o)

    return decorator
