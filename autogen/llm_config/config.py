# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import re
from collections.abc import Iterable
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from autogen.oai.anthropic import AnthropicEntryDict, AnthropicLLMConfigEntry
from autogen.oai.bedrock import BedrockEntryDict, BedrockLLMConfigEntry
from autogen.oai.cerebras import CerebrasEntryDict, CerebrasLLMConfigEntry
from autogen.oai.client import (
    AzureOpenAIEntryDict,
    AzureOpenAILLMConfigEntry,
    DeepSeekEntyDict,
    DeepSeekLLMConfigEntry,
    OpenAIEntryDict,
    OpenAILLMConfigEntry,
)
from autogen.oai.cohere import CohereEntryDict, CohereLLMConfigEntry
from autogen.oai.gemini import GeminiEntryDict, GeminiLLMConfigEntry
from autogen.oai.groq import GroqEntryDict, GroqLLMConfigEntry
from autogen.oai.mistral import MistralEntryDict, MistralLLMConfigEntry
from autogen.oai.ollama import OllamaEntryDict, OllamaLLMConfigEntry
from autogen.oai.together import TogetherEntryDict, TogetherLLMConfigEntry

from ..doc_utils import export_module
from .entry import ApplicationConfig, LLMConfigEntry


# Meta class to allow LLMConfig.current and LLMConfig.default to be used as class properties
class MetaLLMConfig(type):
    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def current(cls) -> "LLMConfig":
        current_llm_config = LLMConfig.get_current_llm_config(llm_config=None)
        if current_llm_config is None:
            raise ValueError("No current LLMConfig set. Are you inside a context block?")
        return current_llm_config  # type: ignore[return-value]

    @property
    def default(cls) -> "LLMConfig":
        return cls.current


ConfigItem: TypeAlias = (
    LLMConfigEntry
    | AnthropicEntryDict
    | BedrockEntryDict
    | CerebrasEntryDict
    | CohereEntryDict
    | AzureOpenAIEntryDict
    | OpenAIEntryDict
    | DeepSeekEntyDict
    | MistralEntryDict
    | GroqEntryDict
    | OllamaEntryDict
    | GeminiEntryDict
    | TogetherEntryDict
    | dict[str, Any]
)


@export_module("autogen")
class LLMConfig(metaclass=MetaLLMConfig):
    _current_llm_config: ContextVar["LLMConfig"] = ContextVar("current_llm_config")

    def __init__(
        self,
        *,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        config_list: Iterable[ConfigItem] | dict[str, Any] = (),
        check_every_ms: int | None = None,
        allow_format_str_template: bool | None = None,
        response_format: str | dict[str, Any] | BaseModel | type[BaseModel] | None = None,
        timeout: int | None = None,
        seed: int | None = None,
        cache_seed: int | None = None,
        parallel_tool_calls: bool | None = None,
        tools: Iterable[Any] = (),
        functions: Iterable[Any] = (),
        routing_method: Literal["fixed_order", "round_robin"] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the LLMConfig object.

        Args:
            config_list: A list of LLM configuration entries or dictionaries.
            temperature: The sampling temperature for LLM generation.
            check_every_ms: The interval (in milliseconds) to check for updates
            allow_format_str_template: Whether to allow format string templates.
            response_format: The format of the response (e.g., JSON, text).
            timeout: The timeout for LLM requests in seconds.
            seed: The random seed for reproducible results.
            cache_seed: The seed for caching LLM responses.
            parallel_tool_calls: Whether to enable parallel tool calls.
            tools: A list of tools available for the LLM.
            functions: A list of functions available for the LLM.
            max_tokens: The maximum number of tokens to generate.
            top_p: The nucleus sampling probability.
            routing_method: The method used to route requests (e.g., fixed_order, round_robin).
            **kwargs: Additional keyword arguments for future extensions.

        Examples:
            ```python
            # Example 1: create config from `kwargs` options
            config = LLMConfig(
                model="gpt-4o-mini",
                api_key=os.environ["OPENAI_API_KEY"],
            )

            # Example 2: create config from `config_list` dictionary
            config = LLMConfig(
                config_list={
                    "model": "gpt-4o-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                }
            )

            # Example 3: create config from `config_list` list
            config = LLMConfig(
                config_list=[
                    {
                        "model": "gpt-4o-mini",
                        "api_key": os.environ["OPENAI_API_KEY"],
                    },
                    {
                        "model": "gpt-4",
                        "api_key": os.environ["OPENAI_API_KEY"],
                    },
                ]
            )
            ```
        """
        app_config = ApplicationConfig(
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
        )

        application_level_options = app_config.model_dump(exclude_none=True)

        final_config_list: list[LLMConfigEntry | dict[str, Any]] = []

        if isinstance(config_list, dict):
            config_list = [config_list]

        for c in filter(bool, (*config_list, kwargs)):
            if isinstance(c, LLMConfigEntry):
                final_config_list.append(c.apply_application_config(app_config))
                continue

            else:
                final_config_list.append({
                    "api_type": "openai",  # default api_type
                    **application_level_options,
                    **c,
                })

        self._model = _LLMConfig(
            **application_level_options,
            config_list=final_config_list,
            check_every_ms=check_every_ms,
            seed=seed,
            allow_format_str_template=allow_format_str_template,
            response_format=response_format,
            timeout=timeout,
            cache_seed=cache_seed,
            tools=tools or [],
            functions=functions or [],
            parallel_tool_calls=parallel_tool_calls,
            routing_method=routing_method,
        )

    # used by BaseModel to create instance variables
    def __enter__(self) -> "LLMConfig":
        # Store previous context and set self as current
        self._token = LLMConfig._current_llm_config.set(self)
        return self

    def __exit__(self, exc_type: type[Exception], exc_val: Exception, exc_tb: Any) -> None:
        LLMConfig._current_llm_config.reset(self._token)

    @classmethod
    def get_current_llm_config(cls, llm_config: "LLMConfig | None" = None) -> "LLMConfig | None":
        if llm_config is not None:
            return llm_config
        try:
            return (LLMConfig._current_llm_config.get()).copy()
        except LookupError:
            return None

    def _satisfies_criteria(self, value: Any, criteria_values: Any) -> bool:
        if value is None:
            return False

        if isinstance(value, list):
            return bool(set(value) & set(criteria_values))  # Non-empty intersection
        else:
            return value in criteria_values

    @classmethod
    def from_json(
        cls,
        *,
        env: str | None = None,
        path: str | Path | None = None,
        file_location: str | None = None,
        **kwargs: Any,
    ) -> "LLMConfig":
        from autogen.oai.openai_utils import config_list_from_json

        if env is None and path is None:
            raise ValueError("Either 'env' or 'path' must be provided")
        if env is not None and path is not None:
            raise ValueError("Only one of 'env' or 'path' can be provided")

        config_list = config_list_from_json(
            env_or_file=env if env is not None else str(path), file_location=file_location
        )
        return LLMConfig(config_list=config_list, **kwargs)

    def where(self, *, exclude: bool = False, **kwargs: Any) -> "LLMConfig":
        from autogen.oai.openai_utils import filter_config

        filtered_config_list = filter_config(config_list=self.config_list, filter_dict=kwargs, exclude=exclude)
        if len(filtered_config_list) == 0:
            raise ValueError(f"No config found that satisfies the filter criteria: {kwargs}")

        kwargs = self.model_dump()
        kwargs["config_list"] = filtered_config_list

        return LLMConfig(**kwargs)

    # @functools.wraps(BaseModel.model_dump)
    def model_dump(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        d = self._model.model_dump(*args, exclude_none=exclude_none, **kwargs)
        return {k: v for k, v in d.items() if not (isinstance(v, list) and len(v) == 0)}

    # @functools.wraps(BaseModel.model_dump_json)
    def model_dump_json(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> str:
        # return self._model.model_dump_json(*args, exclude_none=exclude_none, **kwargs)
        d = self.model_dump(*args, exclude_none=exclude_none, **kwargs)
        return json.dumps(d)

    # @functools.wraps(BaseModel.model_validate)
    def model_validate(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.model_validate(*args, **kwargs)

    @functools.wraps(BaseModel.model_validate_json)
    def model_validate_json(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.model_validate_json(*args, **kwargs)

    @functools.wraps(BaseModel.model_validate_strings)
    def model_validate_strings(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.model_validate_strings(*args, **kwargs)

    def __eq__(self, value: Any) -> bool:
        if not isinstance(value, LLMConfig):
            return NotImplemented
        return self._model == value._model

    def _getattr(self, o: object, name: str) -> Any:
        val = getattr(o, name)
        return val

    def get(self, key: str, default: Any | None = None) -> Any:
        val = getattr(self._model, key, default)
        return val

    def __getitem__(self, key: str) -> Any:
        try:
            return self._getattr(self._model, key)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        try:
            setattr(self._model, key, value)
        except ValueError:
            raise ValueError(f"'{self.__class__.__name__}' object has no field '{key}'")

    def __getattr__(self, name: Any) -> Any:
        try:
            return self._getattr(self._model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_model":
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self._model, key)

    def __repr__(self) -> str:
        d = self.model_dump()
        r = [f"{k}={repr(v)}" for k, v in d.items()]

        s = f"LLMConfig({', '.join(r)})"
        # Replace any keys ending with 'key' or 'token' values with stars for security
        s = re.sub(
            r"(['\"])(\w*(key|token))\1:\s*(['\"])([^'\"]*)(?:\4)", r"\1\2\1: \4**********\4", s, flags=re.IGNORECASE
        )
        return s

    def __copy__(self) -> "LLMConfig":
        return LLMConfig(**self.model_dump())

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "LLMConfig":
        return self.__copy__()

    def copy(self) -> "LLMConfig":
        return self.__copy__()

    def deepcopy(self, memo: dict[int, Any] | None = None) -> "LLMConfig":
        return self.__deepcopy__(memo)

    def __str__(self) -> str:
        return repr(self)

    def items(self) -> Iterable[tuple[str, Any]]:
        d = self.model_dump()
        return d.items()

    def keys(self) -> Iterable[str]:
        d = self.model_dump()
        return d.keys()

    def values(self) -> Iterable[Any]:
        d = self.model_dump()
        return d.values()

    _base_model_classes: dict[tuple[type["LLMConfigEntry"], ...], type[BaseModel]] = {}


class _LLMConfig(ApplicationConfig):
    check_every_ms: int | None
    seed: int | None
    allow_format_str_template: bool | None
    response_format: str | dict[str, Any] | BaseModel | type[BaseModel] | None
    timeout: int | None
    cache_seed: int | None
    parallel_tool_calls: bool | None

    tools: list[Any]
    functions: list[Any]

    config_list: list[  # type: ignore[valid-type]
        Annotated[
            AnthropicLLMConfigEntry
            | CerebrasLLMConfigEntry
            | BedrockLLMConfigEntry
            | AzureOpenAILLMConfigEntry
            | DeepSeekLLMConfigEntry
            | OpenAILLMConfigEntry
            | CohereLLMConfigEntry
            | GeminiLLMConfigEntry
            | GroqLLMConfigEntry
            | MistralLLMConfigEntry
            | OllamaLLMConfigEntry
            | TogetherLLMConfigEntry,
            Field(discriminator="api_type"),
        ],
    ] = Field(..., min_length=1)

    routing_method: Literal["fixed_order", "round_robin"] | None

    # Following field is configuration for pydantic to disallow extra fields
    model_config = ConfigDict(extra="forbid")
