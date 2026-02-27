from typing import Protocol, Self

from .llms import LLMClient


class ModelConfig(Protocol):
    def copy(self) -> Self: ...

    def create(self) -> LLMClient: ...
