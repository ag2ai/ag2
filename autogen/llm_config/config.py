# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import re
import warnings
from collections.abc import Iterable
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self, deprecated

from autogen.doc_utils import export_module

from .entry import ApplicationConfig, LLMConfigEntry
from .types import ConfigEntries
from .utils import config_list_from_json, filter_config


# Meta class to allow LLMConfig.current and LLMConfig.default to be used as class properties
class MetaLLMConfig(type):
    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    @deprecated(
        "`LLMConfig.current / .default` properties are deprecated. "
        "Pass config object to usage explicitly instead. "
        "Scheduled for removal in 0.11.0 version."
    )
    def current(cls) -> "LLMConfig":
        current_llm_config = LLMConfig.get_current_llm_config(llm_config=None)
        if current_llm_config is None:
            raise ValueError("No current LLMConfig set. Are you inside a context block?")
        return current_llm_config  # type: ignore[return-value]

    @property
    @deprecated(
        "`LLMConfig.current / .default` properties are deprecated. "
        "Pass config object to usage explicitly instead. "
        "Scheduled for removal in 0.11.0 version."
    )
    def default(cls) -> "LLMConfig":
        return cls.current


ConfigItem: TypeAlias = LLMConfigEntry | ConfigEntries | dict[str, Any]


@export_module("autogen")
class LLMConfig(metaclass=MetaLLMConfig):
    _current_llm_config: ContextVar["LLMConfig"] = ContextVar("current_llm_config")
    config_list: list[ConfigEntries]

    def __init__(
        self,
        *configs: ConfigItem,
        top_p: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
        config_list: Annotated[
            Iterable[ConfigItem] | dict[str, Any],
            deprecated(
                "`LLMConfig(config_list=[{'model': ..., 'api_key': ...}, ...])` syntax is deprecated. "
                "Use `LLMConfig({'api_key': ..., 'model': ...}, ...)` instead. "
                "Scheduled for removal in 0.11.0 version."
            ),
        ] = (),
        **kwargs: Annotated[
            Any,
            deprecated(
                "`LLMConfig(api_key=..., model=...)` syntax is deprecated. "
                "Use `LLMConfig({'api_key': ..., 'model': ...})` instead. "
                "Scheduled for removal in 0.11.0 version."
            ),
        ],
    ) -> None:
        r"""Initializes the LLMConfig object.

        Args:
            *configs: A list of LLM configuration entries or dictionaries.
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
            # Example 1: create config from one model dictionary
            config = LLMConfig({
                "model": "gpt-5-mini",
                "api_key": os.environ["OPENAI_API_KEY"],
            })

            # Example 2: create config from list of dictionaries
            config = LLMConfig(
                {
                    "model": "gpt-5-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
                {
                    "model": "gpt-4",
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
            )

            # Example 3 (deprecated): create config from `kwargs` options
            config = LLMConfig(
                model="gpt-5-mini",
                api_key=os.environ["OPENAI_API_KEY"],
            )

            # Example 4 (deprecated): create config from `config_list` dictionary
            config = LLMConfig(
                config_list={
                    "model": "gpt-5-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                }
            )

            # Example 5 (deprecated): create config from `config_list` list
            config = LLMConfig(
                config_list=[
                    {
                        "model": "gpt-5-mini",
                        "api_key": os.environ["OPENAI_API_KEY"],
                    },
                    {
                        "model": "gpt-5",
                        "api_key": os.environ["OPENAI_API_KEY"],
                    },
                ]
            )
            ```
        """
        if isinstance(config_list, dict):
            config_list = [config_list]

        if kwargs:
            warnings.warn(
                (
                    "`LLMConfig(api_key=..., model=...)` syntax is deprecated. "
                    "Use `LLMConfig({'api_key': ..., 'model': ...})` instead. "
                    "Scheduled for removal in 0.11.0 version."
                ),
                DeprecationWarning,
            )

        if config_list:
            warnings.warn(
                (
                    "`LLMConfig(config_list=[{'model': ..., 'api_key': ...}, ...])` syntax is deprecated. "
                    "Use `LLMConfig({'api_key': ..., 'model': ...}, ...)` instead. "
                    "Scheduled for removal in 0.11.0 version."
                ),
                DeprecationWarning,
            )

        app_config = ApplicationConfig(
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
        )

        application_level_options = app_config.model_dump(exclude_none=True)

        final_config_list: list[LLMConfigEntry | dict[str, Any]] = []
        for c in filter(bool, (*configs, *config_list, kwargs)):
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

    @classmethod
    def ensure_config(cls, config: "LLMConfig | ConfigItem | Iterable[ConfigItem]", /) -> "LLMConfig":
        """Transforms passed objects to LLMConfig object.

        Method to use for `Agent(llm_config={...})` cases.

        >>> LLMConfig.ensure_config(LLMConfig(...))
        LLMConfig(...)

        >>> LLMConfig.ensure_config(LLMConfigEntry(...))
        LLMConfig(LLMConfigEntry(...))

        >>> LLMConfig.ensure_config({"model": "gpt-o3"})
        LLMConfig(OpenAILLMConfigEntry(model="o3"))

        >>> LLMConfig.ensure_config([{"model": "gpt-o3"}, ...])
        LLMConfig(OpenAILLMConfigEntry(model="o3"), ...)

        >>> (deprecated) LLMConfig.ensure_config({"config_list": [{ "model": "gpt-o3" }, ...]})
        LLMConfig(OpenAILLMConfigEntry(model="o3"), ...)
        """
        if isinstance(config, LLMConfig):
            return config.copy()

        if isinstance(config, LLMConfigEntry):
            return LLMConfig(config)

        if isinstance(config, dict):
            if "config_list" in config:  # backport compatibility
                return LLMConfig(**config)
            return LLMConfig(config)

        return LLMConfig(*config)

    @deprecated(
        "`with llm_config: ...` context manager is deprecated. "
        "Pass config object to usage explicitly instead. "
        "Scheduled for removal in 0.11.0 version."
    )
    def __enter__(self) -> "LLMConfig":
        warnings.warn(
            (
                "`with llm_config: ...` context manager is deprecated. "
                "Pass config object to usage explicitly instead. "
                "Scheduled for removal in 0.11.0 version."
            ),
            DeprecationWarning,
        )

        self._token = LLMConfig._current_llm_config.set(self)
        return self

    def __exit__(self, exc_type: type[Exception], exc_val: Exception, exc_tb: Any) -> None:
        LLMConfig._current_llm_config.reset(self._token)

    @classmethod
    @deprecated(
        "`LLMConfig.current / .default` properties are deprecated. "
        "Pass config object to usage explicitly instead. "
        "Scheduled for removal in 0.11.0 version."
    )
    def get_current_llm_config(cls, llm_config: "LLMConfig | None" = None) -> "LLMConfig | None":
        warnings.warn(
            (
                "`LLMConfig.current / .default` properties are deprecated. "
                "Pass config object to usage explicitly instead. "
                "Scheduled for removal in 0.11.0 version."
            ),
            DeprecationWarning,
        )

        if llm_config is not None:
            return llm_config
        try:
            return (LLMConfig._current_llm_config.get()).copy()
        except LookupError:
            return None

    @classmethod
    def from_json(
        cls,
        *,
        env: str | None = None,
        path: str | Path | None = None,
        file_location: str | None = None,
        filter_dict: dict[str, list[str | None] | set[str | None]] | None = None,
        **kwargs: Any,
    ) -> Self:
        if env is None and path is None:
            raise ValueError("Either 'env' or 'path' must be provided")

        if env is not None and path is not None:
            raise ValueError("Only one of 'env' or 'path' can be provided")

        config_list = config_list_from_json(
            env_or_file=env if env is not None else str(path),
            file_location=file_location,
            filter_dict=filter_dict,
        )

        return cls(*config_list, **kwargs)

    def where(self, *, exclude: bool = False, **kwargs: Any) -> "LLMConfig":
        filtered_config_list = filter_config(
            config_list=[c.model_dump() for c in self.config_list],
            filter_dict=kwargs,
            exclude=exclude,
        )

        if len(filtered_config_list) == 0:
            raise ValueError(f"No config found that satisfies the filter criteria: {kwargs}")

        kwargs = self.model_dump()
        del kwargs["config_list"]
        return LLMConfig(*filtered_config_list, **kwargs)

    def model_dump(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        d = self._model.model_dump(*args, exclude_none=exclude_none, **kwargs)
        return {k: v for k, v in d.items() if not (isinstance(v, list) and len(v) == 0)}

    def model_dump_json(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> str:
        # return self._model.model_dump_json(*args, exclude_none=exclude_none, **kwargs)
        d = self.model_dump(*args, exclude_none=exclude_none, **kwargs)
        return json.dumps(d)

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
        options = self._model.model_dump(exclude={"config_list"})
        return LLMConfig(*self._model.config_list, **options)

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

    config_list: list[
        Annotated[
            ConfigEntries,
            Field(discriminator="api_type"),
        ],
    ] = Field(..., min_length=1)

    routing_method: Literal["fixed_order", "round_robin"] | None

    # Following field is configuration for pydantic to disallow extra fields
    model_config = ConfigDict(extra="forbid")
