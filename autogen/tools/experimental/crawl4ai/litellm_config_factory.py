# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Callable

from ....oai import get_first_llm_config

__all__ = ["LiteLLmConfigFactory"]


class LiteLLmConfigFactory(ABC):
    _factories: set["LiteLLmConfigFactory"] = set()

    @classmethod
    def create_lite_llm_config(cls, llm_config: dict[str, Any]) -> dict[str, Any]:
        first_llm_config = get_first_llm_config(llm_config)
        for factory in LiteLLmConfigFactory._factories:
            if factory.accepts(first_llm_config):
                return factory.create(first_llm_config)

        raise ValueError("Could not find a factory for the given config.")

    @classmethod
    def register_factory(cls) -> Callable[[type["LiteLLmConfigFactory"]], type["LiteLLmConfigFactory"]]:
        def decorator(factory: type["LiteLLmConfigFactory"]) -> type["LiteLLmConfigFactory"]:
            cls._factories.add(factory())
            return factory

        return decorator

    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> dict[str, Any]:
        model = first_llm_config.pop("model")
        api_type = first_llm_config.pop("api_type", "openai")

        first_llm_config["provider"] = f"{api_type}/{model}"
        return first_llm_config

    @classmethod
    @abstractmethod
    def get_api_type(cls) -> str: ...

    @classmethod
    def accepts(cls, first_llm_config: dict[str, Any]) -> bool:
        return first_llm_config.get("api_type", "openai") == cls.get_api_type()  # type: ignore [no-any-return]


@LiteLLmConfigFactory.register_factory()
class DefaultLiteLLmConfigFactory(LiteLLmConfigFactory):
    @classmethod
    def get_api_type(cls) -> str:
        raise NotImplementedError("DefaultLiteLLmConfigFactory does not have an API type.")

    @classmethod
    def accepts(cls, first_llm_config: dict[str, Any]) -> bool:
        non_base_api_types = ["google", "ollama"]
        return first_llm_config.get("api_type", "openai") not in non_base_api_types


@LiteLLmConfigFactory.register_factory()
class GoogleLiteLLmConfigFactory(LiteLLmConfigFactory):
    @classmethod
    def get_api_type(cls) -> str:
        return "google"

    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> dict[str, Any]:
        # api type must be changed before calling super().create
        # litellm uses gemini as the api type for google
        first_llm_config["api_type"] = "gemini"
        first_llm_config = super().create(first_llm_config)

        return first_llm_config

    @classmethod
    def accepts(cls, first_llm_config: dict[str, Any]) -> bool:
        api_type: str = first_llm_config.get("api_type", "")
        return api_type == cls.get_api_type() or api_type == "gemini"


@LiteLLmConfigFactory.register_factory()
class OllamaLiteLLmConfigFactory(LiteLLmConfigFactory):
    @classmethod
    def get_api_type(cls) -> str:
        return "ollama"

    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> dict[str, Any]:
        first_llm_config = super().create(first_llm_config)
        if "client_host" in first_llm_config:
            first_llm_config["api_base"] = first_llm_config.pop("client_host")

        return first_llm_config
