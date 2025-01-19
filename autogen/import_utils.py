# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps
from logging import getLogger
from typing import Any, Callable, Generator, Iterable, Optional, Type, TypeVar, Union

__all__ = ["optional_import_block", "require_optional_import"]

logger = getLogger(__name__)


@contextmanager
def optional_import_block() -> Generator[None, None, None]:
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
    try:
        yield
    except ImportError as e:
        # Ignore ImportErrors during this context
        logger.info(f"Ignoring ImportError: {e}")
        pass


def get_missing_imports(modules: Union[str, Iterable[str]]):
    """Get missing modules from a list of module names

    Args:
        modules: Module name or list of module names

    Returns:
        List of missing module names
    """
    if isinstance(modules, str):
        modules = [modules]

    return [m for m in modules if m not in globals()]


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class PatchObject(ABC):
    def __init__(self, o: T, missing_modules: Iterable[str], dep_target: str):
        if not self.accept(o):
            raise ValueError(f"Cannot patch object of type {type(o)}")

        self.o = o
        self.missing_modules = missing_modules
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
            retval.__name__ = o.__name__
        if hasattr(o, "__module__"):
            retval.__module__ = o.__module__

    _registry: list[Type["PatchObject"]] = []

    @classmethod
    def register(cls) -> Callable[[Type["PatchObject"]], Type["PatchObject"]]:
        def decorator(subclass: Type["PatchObject"]):
            cls._registry.append(subclass)
            return subclass

        return decorator

    @classmethod
    def create(cls, o: T, *, missing_modules: Iterable[str], dep_target: str) -> Optional["PatchObject"]:
        print(f"{cls._registry=}")
        for subclass in cls._registry:
            if subclass.accept(o):
                return subclass(o, missing_modules, dep_target)
        return None


@PatchObject.register()
class PatchCallable(PatchObject):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isfunction(o) or inspect.ismethod(o)

    def patch(self) -> F:
        f: Callable[..., Any] = self.o

        @wraps(f.__call__)
        def _call(*args, **kwargs):
            raise ImportError(self.msg)

        self.copy_metadata(_call)

        return _call


@PatchObject.register()
class PatchStatic(PatchObject):
    @classmethod
    def accept(cls, o: Any) -> bool:
        # return inspect.ismethoddescriptor(o)
        return isinstance(o, staticmethod)

    def patch(self) -> F:
        f: Callable[..., Any] = self.o.__func__

        @wraps(f)
        def _call(*args, **kwargs):
            raise ImportError(self.msg)

        self.copy_metadata(_call)

        return staticmethod(_call)

    def get_object_with_metadata(self) -> Any:
        return self.o.__func__


@PatchObject.register()
class PatchInit(PatchObject):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.ismethoddescriptor(o) and o.__name__ == "__init__"

    def patch(self) -> F:
        f: Callable[..., Any] = self.o

        @wraps(f)
        def _call(*args, **kwargs):
            raise ImportError(self.msg)

        self.copy_metadata(_call)

        return staticmethod(_call)

    def get_object_with_metadata(self) -> Any:
        return self.o


@PatchObject.register()
class PatchProperty(PatchObject):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isdatadescriptor(o) and hasattr(o, "fget")

    def patch(self) -> property:
        if not hasattr(self.o, "fget"):
            raise ValueError(f"Cannot patch property without getter: {self.o}")
        f: Callable[..., Any] = self.o.fget

        @wraps(f)
        def _call(*args, **kwargs):
            raise ImportError(self.msg)

        self.copy_metadata(_call)

        return property(_call)

    def get_object_with_metadata(self) -> Any:
        return self.o.fget


_count = 0


@PatchObject.register()
class PatchClass(PatchObject):
    @classmethod
    def accept(cls, o: Any) -> bool:
        return inspect.isclass(o)

    def patch(self) -> F:
        # Patch __init__ method if possible

        for name, member in inspect.getmembers(self.o):
            if name.startswith("__") and name != "__init__":
                continue
            patched = patch_object(
                member, missing_modules=self.missing_modules, dep_target=self.dep_target, fail_if_not_patchable=False
            )
            setattr(self.o, name, patched)

        return self.o


def patch_object(o: T, *, missing_modules: Iterable[str], dep_target: str, fail_if_not_patchable: bool = True) -> T:
    print(f"Patching object {o=}")
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
