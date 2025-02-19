# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable

from ....import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from langchain_anthropic import ChatAnthropic
    from langchain_core.language_models import BaseChatModel
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_openai import AzureChatOpenAI, ChatOpenAI


__all__ = ["LangchainFactory"]


@require_optional_import(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
    "browser-use",
)
class LangchainFactory(ABC):
    _factories: set["LangchainFactory"] = set()

    @classmethod
    def create_base_chat_model(cls, llm_config: dict[str, Any]) -> "BaseChatModel":  # type: ignore [no-any-unimported]
        first_llm_config = cls.get_first_llm_config(llm_config)
        for factory in LangchainFactory._factories:
            if factory.accepts(first_llm_config):
                return factory.create(first_llm_config)

        raise ValueError("Could not find a factory for the given config.")

    @classmethod
    def register_factory(cls) -> Callable[[type["LangchainFactory"]], type["LangchainFactory"]]:
        def decorator(factory: type["LangchainFactory"]) -> type["LangchainFactory"]:
            cls._factories.add(factory())
            return factory

        return decorator

    @classmethod
    def get_first_llm_config(cls, llm_config: dict[str, Any]) -> dict[str, Any]:
        llm_config = deepcopy(llm_config)
        if "config_list" not in llm_config:
            if "model" in llm_config:
                return llm_config
            raise ValueError("llm_config must be a valid config dictionary.")

        if len(llm_config["config_list"]) == 0:
            raise ValueError("Config list must contain at least one config.")

        return llm_config["config_list"][0]  # type: ignore [no-any-return]

    @classmethod
    def prepare_config(cls, first_llm_config: dict[str, Any]) -> dict[str, Any]:
        first_llm_config.pop("api_type", "openai")
        return first_llm_config

    @classmethod
    @abstractmethod
    def create(cls, first_llm_config: dict[str, Any]) -> "BaseChatModel":  # type: ignore [no-any-unimported]
        ...

    @classmethod
    @abstractmethod
    def get_api_type(cls) -> str: ...

    @classmethod
    def accepts(cls, first_llm_config: dict[str, Any]) -> bool:
        return first_llm_config.get("api_type", "openai") == cls.get_api_type()  # type: ignore [no-any-return]


@LangchainFactory.register_factory()
class ChatOpenAIFactory(LangchainFactory):
    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> "ChatOpenAI":  # type: ignore [no-any-unimported]
        first_llm_config = cls.prepare_config(first_llm_config)

        return ChatOpenAI(**first_llm_config)

    @classmethod
    def get_api_type(cls) -> str:
        return "openai"


@LangchainFactory.register_factory()
class DeepSeekFactory(ChatOpenAIFactory):
    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> "ChatOpenAI":  # type: ignore [no-any-unimported]
        if "base_url" not in first_llm_config:
            raise ValueError("base_url is required for deepseek api type.")
        return super().create(first_llm_config)

    @classmethod
    def get_api_type(cls) -> str:
        return "deepseek"


@LangchainFactory.register_factory()
class ChatAnthropicFactory(LangchainFactory):
    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> "ChatAnthropic":  # type: ignore [no-any-unimported]
        first_llm_config = cls.prepare_config(first_llm_config)

        return ChatAnthropic(**first_llm_config)

    @classmethod
    def get_api_type(cls) -> str:
        return "anthropic"


@LangchainFactory.register_factory()
class ChatGoogleGenerativeAIFactory(LangchainFactory):
    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> "ChatGoogleGenerativeAI":  # type: ignore [no-any-unimported]
        first_llm_config = cls.prepare_config(first_llm_config)

        return ChatGoogleGenerativeAI(**first_llm_config)

    @classmethod
    def get_api_type(cls) -> str:
        return "google"


@LangchainFactory.register_factory()
class AzureChatOpenAIFactory(LangchainFactory):
    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> "AzureChatOpenAI":  # type: ignore [no-any-unimported]
        first_llm_config = cls.prepare_config(first_llm_config)
        for param in ["base_url", "api_version"]:
            if param not in first_llm_config:
                raise ValueError(f"{param} is required for azure api type.")
        first_llm_config["azure_endpoint"] = first_llm_config.pop("base_url")

        return AzureChatOpenAI(**first_llm_config)

    @classmethod
    def get_api_type(cls) -> str:
        return "azure"


@LangchainFactory.register_factory()
class ChatOllamaFactory(LangchainFactory):
    @classmethod
    def create(cls, first_llm_config: dict[str, Any]) -> "ChatOllama":  # type: ignore [no-any-unimported]
        first_llm_config = cls.prepare_config(first_llm_config)
        first_llm_config["base_url"] = first_llm_config.pop("client_host", None)
        if "num_ctx" not in first_llm_config:
            # In all Browser Use examples, num_ctx is set to 32000
            first_llm_config["num_ctx"] = 32000

        return ChatOllama(**first_llm_config)

    @classmethod
    def get_api_type(cls) -> str:
        return "ollama"
