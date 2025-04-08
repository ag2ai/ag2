# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import builtins
import importlib
import inspect
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from logging import getLogger
from pathlib import Path
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
)

import fastapi
import requests
from datamodel_code_generator import DataModelType
from fastapi_code_generator.__main__ import generate_code
from mcp.server.fastmcp import FastMCP
from pydantic_core import PydanticUndefined

from .fastapi_code_generator_helpers import patch_get_parameter_type
from .security import BaseSecurity, BaseSecurityParameters

if TYPE_CHECKING:
    from autogen.agentchat import ConversableAgent

__all__ = ["MCPProxy"]

logger = getLogger(__name__)


@contextmanager
def optional_temp_path(path: Optional[str] = None) -> Iterator[Path]:
    if path is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    else:
        yield Path(path)


@contextmanager
def add_to_builtins(new_globals: dict[str, Any]) -> Iterator[None]:
    old_globals = {key: getattr(builtins, key, None) for key in new_globals}

    try:
        for key, value in new_globals.items():
            setattr(builtins, key, value)  # Inject new global
        yield
    finally:
        for key, value in old_globals.items():
            if value is None:
                delattr(builtins, key)  # Remove added globals
            else:
                setattr(builtins, key, value)  # Restore original value


class MCPProxy:
    def __init__(self, servers: list[dict[str, Any]], title: Optional[str] = None, **kwargs: Any) -> None:
        """Proxy class to generate client from OpenAPI schema."""
        self._servers = servers
        self._title = title or "MCP Proxy"
        self._kwargs = kwargs
        self._registered_funcs: list[Callable[..., Any]] = []
        self._globals: dict[str, Any] = {}

        self._security: dict[str, list[BaseSecurity]] = {}
        self._security_params: dict[Optional[str], BaseSecurityParameters] = {}

        self._mcp = FastMCP(title=self._title)

    @staticmethod
    def _convert_camel_case_within_braces_to_snake(text: str) -> str:
        # Function to convert camel case to snake case
        def camel_to_snake(match: re.Match[str]) -> str:
            return re.sub(r"(?<!^)(?=[A-Z])", "_", match.group(1)).lower()

        # Find all occurrences inside curly braces and apply camel_to_snake
        result = re.sub(r"\{([a-zA-Z0-9]+)\}", lambda m: "{" + camel_to_snake(m) + "}", text)

        return result

    @staticmethod
    def _get_params(path: str, func: Callable[..., Any]) -> tuple[set[str], set[str], Optional[str], bool]:
        sig = inspect.signature(func)

        params_names = set(sig.parameters.keys())

        path_params = set(re.findall(r"\{(.+?)\}", path))
        if not path_params.issubset(params_names):
            raise ValueError(f"Path params {path_params} not in {params_names}")

        body = "body" if "body" in params_names else None

        security = "security" in params_names

        q_params = set(params_names) - path_params - {body} - {"security"}

        return q_params, path_params, body, security

    def _process_params(
        self, path: str, func: Callable[[Any], Any], **kwargs: Any
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        path = MCPProxy._convert_camel_case_within_braces_to_snake(path)
        q_params, path_params, body, security = MCPProxy._get_params(path, func)

        expanded_path = path.format(**{p: kwargs[p] for p in path_params})

        url = self._servers[0]["url"] + expanded_path

        body_dict = {}
        if body and body in kwargs:
            body_value = kwargs[body]
            if isinstance(body_value, dict):
                body_dict = {"json": body_value}
            elif hasattr(body_value, "model_dump"):
                body_dict = {"json": body_value.model_dump()}
            else:
                body_dict = {"json": body_value.dict()}

        body_dict["headers"] = {"Content-Type": "application/json"}
        if security:
            q_params, body_dict = kwargs["security"].add_security(q_params, body_dict)
            # body_dict["headers"][security] = kwargs["security"]

        params = {k: v for k, v in kwargs.items() if k in q_params}

        return url, params, body_dict

    def set_security_params(self, security_params: BaseSecurityParameters, name: Optional[str] = None) -> None:
        if name is not None:
            security = self._security.get(name)
            if security is None:
                raise ValueError(f"Security is not set for '{name}'")

            for match_security in security:
                if match_security.accept(security_params):
                    break
            else:
                raise ValueError(f"Security parameters {security_params} do not match security {security}")

        self._security_params[name] = security_params

    def _get_matching_security(
        self, security: list[BaseSecurity], security_params: BaseSecurityParameters
    ) -> BaseSecurity:
        # check if security matches security parameters
        for match_security in security:
            if match_security.accept(security_params):
                return match_security
        raise ValueError(f"Security parameters {security_params} does not match any given security {security}")

    def _get_security_params(self, name: str) -> tuple[Optional[BaseSecurityParameters], Optional[BaseSecurity]]:
        # check if security is set for the method
        security = self._security.get(name)
        if not security:
            return None, None

        security_params = self._security_params.get(name)
        if security_params is None:
            # check if default security parameters are set
            security_params = self._security_params.get(None)
            if security_params is None:
                raise ValueError(
                    f"Security parameters are not set for {name} and there are no default security parameters"
                )

        match_security = self._get_matching_security(security, security_params)

        return security_params, match_security

    def _request(
        self,
        method: Literal["put", "get", "post", "head", "delete", "patch"],
        path: str,
        description: Optional[str] = None,
        security: Optional[list[BaseSecurity]] = None,
        **kwargs: Any,
    ) -> Callable[..., dict[str, Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
            name = func.__name__

            if security is not None:
                self._security[name] = security

            @self._mcp.tool()
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
                url, params, body_dict = self._process_params(path, func, **kwargs)

                security = self._security.get(name)
                if security is not None:
                    security_params, matched_security = self._get_security_params(name)
                    if security_params is None:
                        raise ValueError(f"Security parameters are not set for '{name}'")
                    else:
                        security_params.apply(params, body_dict, matched_security)  # type: ignore [arg-type]

                response = getattr(requests, method)(url, params=params, **body_dict)
                return response.json()  # type: ignore [no-any-return]

            wrapper._description = (  # type: ignore [attr-defined]
                description or func.__doc__.strip() if func.__doc__ is not None else None
            )

            self._registered_funcs.append(wrapper)

            return wrapper

        return decorator  # type: ignore [return-value]

    def put(self, path: str, **kwargs: Any) -> Callable[..., dict[str, Any]]:
        return self._request("put", path, **kwargs)

    def get(self, path: str, **kwargs: Any) -> Callable[..., dict[str, Any]]:
        return self._request("get", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Callable[..., dict[str, Any]]:
        return self._request("post", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Callable[..., dict[str, Any]]:
        return self._request("delete", path, **kwargs)

    def head(self, path: str, **kwargs: Any) -> Callable[..., dict[str, Any]]:
        return self._request("head", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> Callable[..., dict[str, Any]]:
        return self._request("patch", path, **kwargs)

    @classmethod
    def _get_template_dir(cls) -> Path:
        path = Path(__file__).parents[3] / "templates"
        if not path.exists():
            raise RuntimeError(f"Template directory {path.resolve()} not found.")
        return path

    @classmethod
    def generate_code(
        cls,
        input_text: str,
        output_dir: Path,
        disable_timestamp: bool = False,
        custom_visitors: Optional[list[Path]] = None,
    ) -> str:
        if custom_visitors is None:
            custom_visitors = []
        custom_visitors.append(Path(__file__).parent / "security_schema_visitor.py")

        with patch_get_parameter_type():
            generate_code(
                input_name="openapi.json",
                input_text=input_text,
                encoding="utf-8",
                output_dir=output_dir,
                template_dir=cls._get_template_dir(),
                disable_timestamp=disable_timestamp,
                custom_visitors=custom_visitors,
                output_model_type=DataModelType.PydanticV2BaseModel,
            )
            # Use unique file name for main.py
            main_name = "main"
            main_path = output_dir / f"{main_name}.py"
            shutil.move(output_dir / "main.py", main_path)

            # Change "from models import" to "from models_unique_name import"
            with main_path.open("r") as f:
                main_py_code = f.read()
            main_py_code = main_py_code.replace("from .models import", "from models import")
            # Removing "from __future__ import annotations" to avoid ForwardRef issues, should be fixed in fastapi_code_generator
            main_py_code = main_py_code.replace("from __future__ import annotations", "")

            with main_path.open("w") as f:
                f.write(main_py_code)

            # Use unique file name for models.py
            models_name = "models"
            models_path = output_dir / f"{models_name}.py"
            shutil.move(output_dir / "models.py", models_path)

            return main_name

    def set_globals(self, main: ModuleType, suffix: str) -> None:
        xs = {k: v for k, v in main.__dict__.items() if not k.startswith("__")}
        self._globals = {
            k: v for k, v in xs.items() if hasattr(v, "__module__") and v.__module__ in [f"models_{suffix}", "typing"]
        }

    @classmethod
    def create(
        cls,
        *,
        openapi_json: Optional[str] = None,
        openapi_url: Optional[str] = None,
        client_source_path: Optional[str] = None,
        servers: Optional[list[dict[str, Any]]] = None,
    ) -> "MCPProxy":
        if (openapi_json is None) == (openapi_url is None):
            raise ValueError("Either openapi_json or openapi_url should be provided")

        if openapi_json is None and openapi_url is not None:
            with requests.get(openapi_url, timeout=10) as response:
                response.raise_for_status()
                openapi_json = response.text

        if servers:
            openapi_parsed = json.loads(openapi_json)  # type: ignore [arg-type]
            openapi_parsed["servers"] = servers
            openapi_json = json.dumps(openapi_parsed)

        with optional_temp_path(client_source_path) as td:
            suffix = td.name  # noqa F841

            main_name = cls.generate_code(  # noqa F841
                input_text=openapi_json,  # type: ignore [arg-type]
                output_dir=td,
            )
            # add td to sys.path
            try:
                sys.path.append(str(td))
                main = importlib.import_module(main_name, package=td.name)  # nosemgrep
            finally:
                sys.path.remove(str(td))

            client: MCPProxy = main.app  # type: ignore [attr-defined]
            client.set_globals(main, suffix=suffix)

            return client

    def _get_functions_to_register(
        self,
        functions: Optional[Iterable[Union[str, Mapping[str, Mapping[str, str]]]]] = None,
    ) -> dict[Callable[..., Any], dict[str, Union[str, None]]]:
        if functions is None:
            return {
                f: {
                    "name": None,
                    "description": f._description if hasattr(f, "_description") else None,
                }
                for f in self._registered_funcs
            }

        functions_with_name_desc: dict[str, dict[str, Union[str, None]]] = {}

        for f in functions:
            if isinstance(f, str):
                functions_with_name_desc[f] = {"name": None, "description": None}
            elif isinstance(f, dict):
                functions_with_name_desc.update({
                    k: {
                        "name": v.get("name", None),
                        "description": v.get("description", None),
                    }
                    for k, v in f.items()
                })
            else:
                raise ValueError(f"Invalid type {type(f)} for function {f}")

        funcs_to_register: dict[Callable[..., Any], dict[str, Union[str, None]]] = {
            f: functions_with_name_desc[f.__name__]
            for f in self._registered_funcs
            if f.__name__ in functions_with_name_desc
        }
        missing_functions = set(functions_with_name_desc.keys()) - {f.__name__ for f in funcs_to_register}
        if missing_functions:
            raise ValueError(f"Following functions {missing_functions} are not valid functions")

        return funcs_to_register

    @staticmethod
    def _remove_pydantic_undefined_from_tools(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            if "parameters" not in function or "properties" not in function["parameters"]:
                continue

            required = function["parameters"].get("required", [])
            for param_name, param_value in function["parameters"]["properties"].items():
                if "default" not in param_value:
                    continue

                default = param_value.get("default")
                if (
                    isinstance(default, (fastapi.params.Path, fastapi.params.Query))
                    and param_value["default"].default is PydanticUndefined
                ):
                    param_value.pop("default")
                    # We removed the default value, so we need to add the parameter to the required list
                    if param_name not in required:
                        required.append(param_name)

        return tools

    def _register_for_llm(
        self,
        agent: "ConversableAgent",
        functions: Optional[Iterable[Union[str, Mapping[str, Mapping[str, str]]]]] = None,
    ) -> None:
        funcs_to_register = self._get_functions_to_register(functions)

        with add_to_builtins(
            new_globals=self._globals,
        ):
            for f, v in funcs_to_register.items():
                agent.register_for_llm(name=v["name"], description=v["description"])(f)

            agent.llm_config["tools"] = MCPProxy._remove_pydantic_undefined_from_tools(agent.llm_config["tools"])

    def _register_for_execution(
        self,
        agent: "ConversableAgent",
        functions: Optional[Iterable[Union[str, Mapping[str, Mapping[str, str]]]]] = None,
    ) -> None:
        funcs_to_register = self._get_functions_to_register(functions)

        for f, v in funcs_to_register.items():
            agent.register_for_execution(name=v["name"])(f)

    def get_functions(self) -> list[str]:
        raise DeprecationWarning("Use function_names property instead of get_functions method")

    @property
    def function_names(self) -> list[str]:
        return [f.__name__ for f in self._registered_funcs]

    def get_function(self, name: str) -> Callable[..., dict[str, Any]]:
        for f in self._registered_funcs:
            if f.__name__ == name:
                return f
        raise ValueError(f"Function {name} not found")

    def set_function(self, name: str, func: Callable[..., dict[str, Any]]) -> None:
        for i, f in enumerate(self._registered_funcs):
            if f.__name__ == name:
                self._registered_funcs[i] = func
                return

        raise ValueError(f"Function {name} not found")

    def inject_parameters(self, name: str, **kwargs: Any) -> None:
        raise NotImplementedError("Injecting parameters is not implemented yet")
        # for f in self._registered_funcs:
        #     if f.__name__ == name:
        #         return

        # raise ValueError(f"Function {name} not found")
