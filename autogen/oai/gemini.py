# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""Create a OpenAI-compatible client for Gemini features.


Example:
    llm_config={
        "config_list": [{
            "api_type": "google",
            "model": "gemini-pro",
            "api_key": os.environ.get("GOOGLE_GEMINI_API_KEY"),
            "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                    ],
            "top_p":0.5,
            "max_tokens": 2048,
            "temperature": 1.0,
            "top_k": 5
            }
    ]}

    agent = autogen.AssistantAgent("my_agent", llm_config=llm_config)

Resources:
- https://ai.google.dev/docs
- https://cloud.google.com/vertex-ai/docs/generative-ai/migrate-from-azure
- https://blog.google/technology/ai/google-gemini-pro-imagen-duet-ai-update/
- https://ai.google.dev/api/python/google/generativeai/ChatSession
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
import random
import re
import time
import warnings
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import google.generativeai as genai
import PIL
import requests
import vertexai
from google.ai.generativelanguage import Content, FunctionCall, FunctionDeclaration, FunctionResponse, Part, Tool
from google.ai.generativelanguage_v1beta.types import Schema
from google.auth.credentials import Credentials
from jsonschema import ValidationError
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from openai.types.completion_usage import CompletionUsage
from PIL import Image
from pydantic import BaseModel
from vertexai.generative_models import (
    Content as VertexAIContent,
)
from vertexai.generative_models import FunctionDeclaration as vaiFunctionDeclaration
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmBlockThreshold as VertexAIHarmBlockThreshold
from vertexai.generative_models import HarmCategory as VertexAIHarmCategory
from vertexai.generative_models import Image as VertexAIImage
from vertexai.generative_models import Part as VertexAIPart
from vertexai.generative_models import SafetySetting as VertexAISafetySetting
from vertexai.generative_models import (
    Tool as vaiTool,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Google's Gemini API.

    Please visit this [page](https://github.com/microsoft/autogen/issues/2387) for the roadmap of Gemini integration
    of AutoGen.
    """

    # Mapping, where Key is a term used by Autogen, and Value is a term used by Gemini
    PARAMS_MAPPING = {
        "max_tokens": "max_output_tokens",
        # "n": "candidate_count", # Gemini supports only `n=1`
        "stop_sequences": "stop_sequences",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_output_tokens": "max_output_tokens",
    }

    def _initialize_vertexai(self, **params):
        if "google_application_credentials" in params:
            # Path to JSON Keyfile
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = params["google_application_credentials"]
        vertexai_init_args = {}
        if "project_id" in params:
            vertexai_init_args["project"] = params["project_id"]
        if "location" in params:
            vertexai_init_args["location"] = params["location"]
        if "credentials" in params:
            assert isinstance(
                params["credentials"], Credentials
            ), "Object type google.auth.credentials.Credentials is expected!"
            vertexai_init_args["credentials"] = params["credentials"]
        if vertexai_init_args:
            vertexai.init(**vertexai_init_args)

    def __init__(self, **kwargs):
        """Uses either either api_key for authentication from the LLM config
        (specifying the GOOGLE_GEMINI_API_KEY environment variable also works),
        or follows the Google authentication mechanism for VertexAI in Google Cloud if no api_key is specified,
        where project_id and location can also be passed as parameters. Previously created credentials object can be provided,
        or a Service account key file can also be used. If neither a service account key file, nor the api_key are passed,
        then the default credentials will be used, which could be a personal account if the user is already authenticated in,
        like in Google Cloud Shell.

        Args:
            api_key (str): The API key for using Gemini.
                credentials (google.auth.credentials.Credentials): credentials to be used for authentication with vertexai.
            google_application_credentials (str): Path to the JSON service account key file of the service account.
                Alternatively, the GOOGLE_APPLICATION_CREDENTIALS environment variable
                can also be set instead of using this argument.
            project_id (str): Google Cloud project id, which is only valid in case no API key is specified.
            location (str): Compute region to be used, like 'us-west1'.
                This parameter is only valid in case no API key is specified.
        """
        self.api_key = kwargs.get("api_key", None)
        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
            if self.api_key is None:
                self.use_vertexai = True
                self._initialize_vertexai(**kwargs)
            else:
                self.use_vertexai = False
        else:
            self.use_vertexai = False
        if not self.use_vertexai:
            assert ("project_id" not in kwargs) and (
                "location" not in kwargs
            ), "Google Cloud project and compute location cannot be set when using an API Key!"

        if "response_format" in kwargs and kwargs["response_format"] is not None:
            warnings.warn("response_format is not supported for Gemini. It will be ignored.", UserWarning)

    def message_retrieval(self, response) -> List:
        """
        Retrieve and return a list of strings or a list of Choice.Message from the response.

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        return [choice.message for choice in response.choices]

    def cost(self, response) -> float:
        return response.cost

    @staticmethod
    def get_usage(response) -> Dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        # ...  # pragma: no cover
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": response.cost,
            "model": response.model,
        }

    def create(self, params: Dict) -> ChatCompletion:

        if self.use_vertexai:
            self._initialize_vertexai(**params)
        else:
            assert ("project_id" not in params) and (
                "location" not in params
            ), "Google Cloud project and compute location cannot be set when using an API Key!"
        model_name = params.get("model", "gemini-pro")

        if model_name == "gemini-pro-vision":
            raise ValueError(
                "Gemini 1.0 Pro vision ('gemini-pro-vision') has been deprecated, please consider switching to a different model, for example 'gemini-1.5-flash'."
            )
        elif not model_name:
            raise ValueError(
                "Please provide a model name for the Gemini Client. "
                "You can configure it in the OAI Config List file. "
                "See this [LLM configuration tutorial](https://ag2ai.github.io/ag2/docs/topics/llm_configuration/) for more details."
            )

        params.get("api_type", "google")  # not used
        messages = params.get("messages", [])
        stream = params.get("stream", False)
        n_response = params.get("n", 1)
        system_instruction = params.get("system_instruction", None)
        response_validation = params.get("response_validation", True)
        if "tools" in params:
            tools = self._tools_to_gemini_tools(params["tools"])
        else:
            tools = None

        generation_config = {
            gemini_term: params[autogen_term]
            for autogen_term, gemini_term in self.PARAMS_MAPPING.items()
            if autogen_term in params
        }
        if self.use_vertexai:
            safety_settings = GeminiClient._to_vertexai_safety_settings(params.get("safety_settings", {}))
        else:
            safety_settings = params.get("safety_settings", {})

        if stream:
            warnings.warn(
                "Streaming is not supported for Gemini yet, and it will have no effect. Please set stream=False.",
                UserWarning,
            )
            stream = False

        if n_response > 1:
            warnings.warn("Gemini only supports `n=1` for now. We only generate one response.", UserWarning)

        autogen_tool_calls = []

        # Maps the function call ids to function names so we can inject it into FunctionResponse messages
        self.tool_call_function_map: Dict[str, str] = {}

        # A. create and call the chat model.
        gemini_messages = self._oai_messages_to_gemini_messages(messages)
        if self.use_vertexai:
            model = GenerativeModel(
                model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_instruction,
                tools=tools,
            )

            chat = model.start_chat(history=gemini_messages[:-1], response_validation=response_validation)
        else:
            model = genai.GenerativeModel(
                model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_instruction,
                tools=tools,
            )

            genai.configure(api_key=self.api_key)
            chat = model.start_chat(history=gemini_messages[:-1])

        response = chat.send_message(gemini_messages[-1].parts, stream=stream, safety_settings=safety_settings)

        # Extract text and tools from response
        ans = ""
        random_id = random.randint(0, 10000)
        prev_function_calls = []
        for part in response.parts:

            # Function calls
            if fn_call := part.function_call:

                # If we have a repeated function call, ignore it
                if fn_call not in prev_function_calls:
                    autogen_tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=random_id,
                            function={
                                "name": fn_call.name,
                                "arguments": (
                                    json.dumps({key: val for key, val in fn_call.args.items()})
                                    if fn_call.args is not None
                                    else ""
                                ),
                            },
                            type="function",
                        )
                    )

                    prev_function_calls.append(fn_call)
                    random_id += 1

            # Plain text content
            elif text := part.text:
                ans += text

        # If we have function calls, ignore the text
        # as it can be Gemini guessing the function response
        if len(autogen_tool_calls) != 0:
            ans = ""
        else:
            autogen_tool_calls = None

        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count

        # 3. convert output
        message = ChatCompletionMessage(
            role="assistant", content=ans, function_call=None, tool_calls=autogen_tool_calls
        )
        choices = [
            Choice(finish_reason="tool_calls" if autogen_tool_calls is not None else "stop", index=0, message=message)
        ]

        response_oai = ChatCompletion(
            id=str(random.randint(0, 1000)),
            model=model_name,
            created=int(time.time()),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cost=calculate_gemini_cost(self.use_vertexai, prompt_tokens, completion_tokens, model_name),
        )

        return response_oai

    def _oai_content_to_gemini_content(self, message: Dict[str, Any]) -> Tuple[List, str]:
        """Convert AutoGen content to Gemini parts, catering for text and tool calls"""
        rst = []

        if message["role"] == "tool":
            # Tool call recommendation

            function_name = self.tool_call_function_map[message["tool_call_id"]]

            if self.use_vertexai:
                rst.append(
                    VertexAIPart.from_function_response(
                        name=function_name, response={"result": self._to_json_or_str(message["content"])}
                    )
                )
            else:
                rst.append(
                    Part(
                        function_response=FunctionResponse(
                            name=function_name, response={"result": self._to_json_or_str(message["content"])}
                        )
                    )
                )

            return rst, "tool"
        elif "tool_calls" in message and len(message["tool_calls"]) != 0:
            for tool_call in message["tool_calls"]:

                function_id = tool_call["id"]
                function_name = tool_call["function"]["name"]
                self.tool_call_function_map[function_id] = function_name

                if self.use_vertexai:
                    rst.append(
                        VertexAIPart.from_dict(
                            {
                                "functionCall": {
                                    "name": function_name,
                                    "args": json.loads(tool_call["function"]["arguments"]),
                                }
                            }
                        )
                    )
                else:
                    rst.append(
                        Part(
                            function_call=FunctionCall(
                                name=function_name,
                                args=json.loads(tool_call["function"]["arguments"]),
                            )
                        )
                    )

            return rst, "tool_call"

        elif isinstance(message["content"], str):
            content = message["content"]
            if content == "":
                content = "empty"  # Empty content is not allowed.
            if self.use_vertexai:
                rst.append(VertexAIPart.from_text(content))
            else:
                rst.append(Part(text=content))

            return rst, "text"

        # For images the message contains a list of text items
        if isinstance(message["content"], list):
            has_image = False
            for msg in message["content"]:
                if isinstance(msg, dict):
                    assert "type" in msg, f"Missing 'type' field in message: {msg}"
                    if msg["type"] == "text":
                        if self.use_vertexai:
                            rst.append(VertexAIPart.from_text(text=msg["text"]))
                        else:
                            rst.append(Part(text=msg["text"]))
                    elif msg["type"] == "image_url":
                        if self.use_vertexai:
                            img_url = msg["image_url"]["url"]
                            img_part = VertexAIPart.from_uri(img_url, mime_type="image/png")
                            rst.append(img_part)
                        else:
                            b64_img = get_image_data(msg["image_url"]["url"])
                            rst.append(Part(inline_data={"mime_type": "image/png", "data": b64_img}))

                        has_image = True
                    else:
                        raise ValueError(f"Unsupported message type: {msg['type']}")
                else:
                    raise ValueError(f"Unsupported message type: {type(msg)}")
            return rst, "image" if has_image else "text"
        else:
            raise Exception("Unable to convert content to Gemini format.")

    def _concat_parts(self, parts: List[Part]) -> List:
        """Concatenate parts with the same type.
        If two adjacent parts both have the "text" attribute, then it will be joined into one part.
        """
        if not parts:
            return []

        concatenated_parts = []
        previous_part = parts[0]

        for current_part in parts[1:]:
            if previous_part.text != "":
                if self.use_vertexai:
                    previous_part = VertexAIPart.from_text(previous_part.text + current_part.text)
                else:
                    previous_part.text += current_part.text
            else:
                concatenated_parts.append(previous_part)
                previous_part = current_part

        if previous_part.text == "":
            if self.use_vertexai:
                previous_part = VertexAIPart.from_text("empty")
            else:
                previous_part.text = "empty"  # Empty content is not allowed.
        concatenated_parts.append(previous_part)

        return concatenated_parts

    def _oai_messages_to_gemini_messages(self, messages: list[Dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages from OAI format to Gemini format.
        Make sure the "user" role and "model" role are interleaved.
        Also, make sure the last item is from the "user" role.
        """
        rst = []
        for message in messages:
            parts, part_type = self._oai_content_to_gemini_content(message)
            role = "user" if message["role"] in ["user", "system"] else "model"

            if part_type == "text":
                rst.append(
                    VertexAIContent(parts=parts, role=role)
                    if self.use_vertexai
                    else rst.append(Content(parts=parts, role=role))
                )
            elif part_type == "tool":
                rst.append(
                    VertexAIContent(parts=parts, role="function")
                    if self.use_vertexai
                    else rst.append(Content(parts=parts, role="function"))
                )
            elif part_type == "tool_call":
                rst.append(
                    VertexAIContent(parts=parts, role="function")
                    if self.use_vertexai
                    else rst.append(Content(parts=parts, role="function"))
                )
            elif part_type == "image":
                # Image has multiple parts, some can be text and some can be image based
                text_parts = []
                image_parts = []
                for part in parts:
                    if isinstance(part, Part):
                        # Text or non-Vertex AI image part
                        text_parts.append(part)
                    elif isinstance(part, VertexAIPart):
                        # Image
                        image_parts.append(part)
                    else:
                        raise Exception("Unable to process image part")

                if len(text_parts) > 0:
                    rst.append(
                        VertexAIContent(parts=text_parts, role=role)
                        if self.use_vertexai
                        else rst.append(Content(parts=text_parts, role=role))
                    )

                if len(image_parts) > 0:
                    rst.append(
                        VertexAIContent(parts=image_parts, role=role)
                        if self.use_vertexai
                        else rst.append(Content(parts=image_parts, role=role))
                    )

            if len(rst) != 0 and rst[-1] is None:
                rst.pop()

        # The Gemini is restrict on order of roles, such that
        # 1. The messages should be interleaved between user and model.
        # 2. The last message must be from the user role.
        # We add a dummy message "continue" if the last role is not the user.
        if rst[-1].role not in ["user", "function"]:
            text_part, type = self._oai_content_to_gemini_content({"content": "continue"})
            rst.append(
                VertexAIContent(parts=text_part, role="user")
                if self.use_vertexai
                else Content(parts=text_part, role="user")
            )

        return rst

    def _tools_to_gemini_tools(self, tools: List[Dict[str, Any]]) -> List[Tool]:
        """Create Gemini tools (as typically requires Callables)"""

        functions = []
        for tool in tools:
            if self.use_vertexai:
                function = vaiFunctionDeclaration(
                    name=tool["function"]["name"],
                    description=tool["function"]["description"],
                    parameters=tool["function"]["parameters"],
                )
            else:
                function = GeminiClient._create_gemini_function_declaration(tool)
            functions.append(function)

        if self.use_vertexai:
            return [vaiTool(function_declarations=functions)]
        else:
            return [Tool(function_declarations=functions)]

    @staticmethod
    def _create_gemini_function_declaration(tool: Dict) -> FunctionDeclaration:
        function_declaration = FunctionDeclaration()
        function_declaration.name = tool["function"]["name"]
        function_declaration.description = tool["function"]["description"]
        if len(tool["function"]["parameters"]["properties"]) != 0:
            function_declaration.parameters = GeminiClient._create_gemini_function_parameters(
                copy.deepcopy(tool["function"]["parameters"])
            )

        return function_declaration

    @staticmethod
    def _create_gemini_function_declaration_schema(json_data) -> Schema:
        """Recursively creates Schema objects for FunctionDeclaration."""
        param_schema = Schema()
        param_type = json_data["type"]

        """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        INTEGER = 2
        NUMBER = 3
        OBJECT = 4
        ARRAY = 5
        BOOLEAN = 6
        """

        if param_type == "integer":
            param_schema.type_ = 2
        elif param_type == "number":
            param_schema.type_ = 3
        elif param_type == "string":
            param_schema.type_ = 1
        elif param_type == "boolean":
            param_schema.type_ = 6
        elif param_type == "array":
            param_schema.type_ = 5
            if "items" in json_data:
                param_schema.items = GeminiClient._create_gemini_function_declaration_schema(json_data["items"])
            else:
                print("Warning: Array schema missing 'items' definition.")
        elif param_type == "object":
            param_schema.type_ = 4
            param_schema.properties = {}
            if "properties" in json_data:
                for prop_name, prop_data in json_data["properties"].items():
                    param_schema.properties[prop_name] = GeminiClient._create_gemini_function_declaration_schema(
                        prop_data
                    )
                else:
                    print("Warning: Object schema missing 'properties' definition.")

        elif param_type in ("null", "any"):
            param_schema.type_ = 1  # Treating these as strings for simplicity
        else:
            print(f"Warning: Unsupported parameter type '{param_type}'.")

        if "description" in json_data:
            param_schema.description = json_data["description"]

        return param_schema

    def _create_gemini_function_parameters(function_parameter: dict[str, any]) -> dict[str, any]:
        """Convert function parameters to Gemini format, recursive"""

        function_parameter["type_"] = function_parameter["type"].upper()

        # Parameter properties and items
        if "properties" in function_parameter:
            for key in function_parameter["properties"]:
                function_parameter["properties"][key] = GeminiClient._create_gemini_function_parameters(
                    function_parameter["properties"][key]
                )

        if "items" in function_parameter:
            function_parameter["items"] = GeminiClient._create_gemini_function_parameters(function_parameter["items"])

        # Remove any attributes not needed
        for attr in ["type", "default"]:
            if attr in function_parameter:
                del function_parameter[attr]

        return function_parameter

    @staticmethod
    def _to_vertexai_safety_settings(safety_settings):
        """Convert safety settings to VertexAI format if needed,
        like when specifying them in the OAI_CONFIG_LIST
        """
        if isinstance(safety_settings, list) and all(
            [
                isinstance(safety_setting, dict) and not isinstance(safety_setting, VertexAISafetySetting)
                for safety_setting in safety_settings
            ]
        ):
            vertexai_safety_settings = []
            for safety_setting in safety_settings:
                if safety_setting["category"] not in VertexAIHarmCategory.__members__:
                    invalid_category = safety_setting["category"]
                    logger.error(f"Safety setting category {invalid_category} is invalid")
                elif safety_setting["threshold"] not in VertexAIHarmBlockThreshold.__members__:
                    invalid_threshold = safety_setting["threshold"]
                    logger.error(f"Safety threshold {invalid_threshold} is invalid")
                else:
                    vertexai_safety_setting = VertexAISafetySetting(
                        category=safety_setting["category"],
                        threshold=safety_setting["threshold"],
                    )
                    vertexai_safety_settings.append(vertexai_safety_setting)
            return vertexai_safety_settings
        else:
            return safety_settings

    @staticmethod
    def _to_json_or_str(data: str) -> Union[Dict, str]:
        try:
            json_data = json.loads(data)
            return json_data
        except (json.JSONDecodeError, ValidationError):
            return data


def get_image_data(image_file: str, use_b64=True) -> bytes:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        content = response.content
    elif re.match(r"data:image/(?:png|jpeg);base64,", image_file):
        return re.sub(r"data:image/(?:png|jpeg);base64,", "", image_file)
    else:
        image = Image.open(image_file).convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        content = buffered.getvalue()

    if use_b64:
        return base64.b64encode(content).decode("utf-8")
    else:
        return content


def calculate_gemini_cost(use_vertexai: bool, input_tokens: int, output_tokens: int, model_name: str) -> float:

    def total_cost_mil(cost_per_mil_input: float, cost_per_mil_output: float):
        # Cost per million
        return cost_per_mil_input * input_tokens / 1e6 + cost_per_mil_output * output_tokens / 1e6

    def total_cost_k(cost_per_k_input: float, cost_per_k_output: float):
        # Cost per thousand
        return cost_per_k_input * input_tokens / 1e3 + cost_per_k_output * output_tokens / 1e3

    model_name = model_name.lower()
    up_to_128k = input_tokens <= 128000

    if use_vertexai:
        # Vertex AI pricing - based on Text input
        # https://cloud.google.com/vertex-ai/generative-ai/pricing#vertex-ai-pricing

        if "gemini-1.5-flash" in model_name:
            if up_to_128k:
                return total_cost_k(0.00001875, 0.000075)
            else:
                return total_cost_k(0.0000375, 0.00015)

        elif "gemini-1.5-pro" in model_name:
            if up_to_128k:
                return total_cost_k(0.0003125, 0.00125)
            else:
                return total_cost_k(0.000625, 0.0025)

        elif "gemini-1.0-pro" in model_name:
            return total_cost_k(0.000125, 0.00001875)

        else:
            warnings.warn(
                f"Cost calculation is not implemented for model {model_name}. Cost will be calculated zero.",
                UserWarning,
            )
            return 0

    else:
        # Non-Vertex AI pricing

        if "gemini-1.5-flash-8b" in model_name:
            # https://ai.google.dev/pricing#1_5flash-8B
            if up_to_128k:
                return total_cost_mil(0.0375, 0.15)
            else:
                return total_cost_mil(0.075, 0.3)

        elif "gemini-1.5-flash" in model_name:
            # https://ai.google.dev/pricing#1_5flash
            if up_to_128k:
                return total_cost_mil(0.075, 0.3)
            else:
                return total_cost_mil(0.15, 0.6)

        elif "gemini-1.5-pro" in model_name:
            # https://ai.google.dev/pricing#1_5pro
            if up_to_128k:
                return total_cost_mil(1.25, 5.0)
            else:
                return total_cost_mil(2.50, 10.0)

        elif "gemini-1.0-pro" in model_name:
            # https://ai.google.dev/pricing#1_5pro
            return total_cost_mil(0.50, 1.5)

        else:
            warnings.warn(
                f"Cost calculation is not implemented for model {model_name}. Cost will be calculated zero.",
                UserWarning,
            )
            return 0
