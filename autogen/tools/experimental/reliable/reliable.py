# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import copy
import functools
import inspect
import json
import logging
import re
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ....agentchat import initiate_group_chat
from ....agentchat.agent import Agent
from ....agentchat.conversable_agent import ConversableAgent
from ....agentchat.group import AgentTarget, ReplyResult, TerminateTarget
from ....agentchat.group.context_variables import ContextVariables
from ....agentchat.group.patterns import DefaultPattern
from ....doc_utils import export_module
from ....llm_config import LLMConfig
from ....tools.tool import Tool

__all__ = ("ReliableTool",)

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Represents the outcome of a single validation step."""

    model_config = ConfigDict(extra="forbid")
    validation_result: bool
    justification: str

    def __str__(self) -> str:
        status = "Passed" if self.validation_result else "Failed"
        return f"Validation Result: {status}\nJustification: {self.justification}"

    def format(self) -> str:
        """Returns the JSON representation for AutoGen compatibility."""
        return self.model_dump_json()


class ExecutionAttempt(BaseModel):
    """Stores the state of a single attempt to execute and validate the function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    timestamp: float = Field(default_factory=time.time)
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    hypothesis: Optional[str] = None
    error: Optional[str] = None
    result_data: Optional[Any] = None
    result_str: Optional[str] = None
    validation: Optional[ValidationResult] = None

    @property
    def did_execute_successfully(self) -> bool:
        """Check if the attempt executed without raising an error."""
        return self.error is None

    @property
    def did_validate_successfully(self) -> bool:
        """Check if the attempt passed validation."""
        return self.validation is not None and self.validation.validation_result


class ReliableToolContext(BaseModel):
    """Main context object holding the overall state and history of attempts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    task: str
    reliable_tool_name: str
    start_time: float = Field(default_factory=time.time)
    dynamic_validation_input: Optional[str] = None
    attempts: List[ExecutionAttempt] = Field(default_factory=list)
    initial_messages: Optional[List[Dict]] = Field(
        default=None, description="Initial messages provided to the tool run."
    )
    initial_ground_truth: Optional[List[str]] = Field(
        default=None, description="Initial ground truth strings provided."
    )

    @property
    def attempt_count(self) -> int:
        """Return the number of attempts made."""
        return len(self.attempts)

    @property
    def latest_attempt(self) -> Optional[ExecutionAttempt]:
        """Return the most recent attempt, if any."""
        return self.attempts[-1] if self.attempts else None

    @property
    def is_complete_and_successful(self) -> bool:
        """Check if the process finished with a validated successful attempt."""
        latest = self.latest_attempt
        return latest is not None and latest.did_execute_successfully and latest.did_validate_successfully

    def get_final_result_data(self) -> Any:
        """Return the result_data from the first successful and validated attempt."""
        for attempt in reversed(self.attempts):
            if attempt.did_execute_successfully and attempt.did_validate_successfully:
                return attempt.result_data
        return None

    def get_failure_summary(self) -> str:
        """Provide a summary of why the overall execution failed."""
        latest = self.latest_attempt
        if latest is None:
            return "No execution was made."  # Modified log
        if not latest.did_execute_successfully:
            return f"Execution failed: {latest.error}"
        if not latest.did_validate_successfully:
            justification = latest.validation.justification if latest.validation else "validation result missing"
            return f"Execution succeeded but failed validation (Justification: {justification})"
        return "Execution completed but overall status indicates failure (Internal inconsistency)."


# --- Helper Functions ---
# TODO: This is super AI generated overkill, just rip out and replace with how it should actually be done
def _configure_llm_for_structured_output(
    llm_config: Optional[Union[LLMConfig, dict, bool]], structured_output_type: Type[BaseModel]
) -> Optional[Union[LLMConfig, dict, bool]]:
    """Configure LLM config for structured output using a Pydantic model."""
    if llm_config is None or llm_config is False:
        return llm_config
    if not issubclass(structured_output_type, BaseModel):
        raise TypeError(f"{structured_output_type} must be a Pydantic BaseModel subclass.")

    llm_config_obj = ConversableAgent._validate_llm_config(llm_config)

    if llm_config_obj is False:
        return False

    # Try setting response_format attribute directly (Pydantic v2 style in some LLMConfig models)
    response_format_set = False
    if hasattr(llm_config_obj, "response_format"):
        try:
            setattr(llm_config_obj, "response_format", structured_output_type)
            response_format_set = True
        except Exception as e:
            logger.warning("Could not set 'response_format' attribute directly: %s. Trying dict access.", e)

    # Fallback or alternative: Modify as a dictionary if possible
    if not response_format_set:
        if isinstance(llm_config_obj, dict):
            llm_config_obj["response_format"] = structured_output_type
            response_format_set = True
        # If it's an object but doesn't have the attribute, check if it behaves like a dict
        elif hasattr(llm_config_obj, "__setitem__") and hasattr(llm_config_obj, "__getitem__"):
            try:
                llm_config_obj["response_format"] = structured_output_type
                response_format_set = True
            except Exception as e:
                logger.warning("Could not set 'response_format' via item access: %s", e)

    if not response_format_set:
        logger.error(
            "LLMConfig object type (%s) does not support setting 'response_format'. Structured output may fail.",
            type(llm_config_obj).__name__,
        )

    def _remove_conflicts(config_obj):
        removed = False
        conflicting_keys = ["tools", "tool_choice", "functions"]
        if isinstance(config_obj, dict):
            for key in conflicting_keys:
                if key in config_obj:
                    del config_obj[key]
                    removed = True
        elif hasattr(config_obj, "__dict__"):  # Basic check for object attributes
            for key in conflicting_keys:
                if hasattr(config_obj, key) and getattr(config_obj, key, None):
                    try:
                        # Try setting to None or empty list/dict as appropriate
                        default_empty = [] if key in ["tools", "functions"] else None
                        setattr(config_obj, key, default_empty)
                        removed = True
                    except AttributeError:
                        logger.warning("Could not remove conflicting key '%s' from LLM config object.", key)
        return removed

    if _remove_conflicts(llm_config_obj):
        logger.debug(
            "Removed conflicting 'tools'/'tool_choice'/'functions' from validator LLM config for structured output (response_format=%s)",
            structured_output_type.__name__,
        )

    # Handle config_list if present
    config_list = None
    if isinstance(llm_config_obj, dict):
        config_list = llm_config_obj.get("config_list")
    elif hasattr(llm_config_obj, "config_list"):
        config_list = getattr(llm_config_obj, "config_list", None)

    if isinstance(config_list, list):
        new_config_list = []
        for item in config_list:
            item_copy = copy.deepcopy(item)  # Deep copy to avoid modifying original list items
            if isinstance(item_copy, dict):
                item_copy["response_format"] = structured_output_type
                _remove_conflicts(item_copy)
            elif hasattr(item_copy, "__dict__"):  # Basic check for object
                # Try setting attribute
                if hasattr(item_copy, "response_format"):
                    try:
                        setattr(item_copy, "response_format", structured_output_type)
                    except Exception:
                        pass  # Ignore if fails
                # Try item access
                elif hasattr(item_copy, "__setitem__"):
                    try:
                        item_copy["response_format"] = structured_output_type
                    except Exception:
                        pass  # Ignore if fails
                _remove_conflicts(item_copy)
            else:  # Non-dict, non-object item? Keep as is.
                pass
            new_config_list.append(item_copy)

        # Update the config_list in the main object
        if isinstance(llm_config_obj, dict):
            llm_config_obj["config_list"] = new_config_list
        elif hasattr(llm_config_obj, "config_list"):
            setattr(llm_config_obj, "config_list", new_config_list)

    logger.debug("Prepared LLM config for validator (response_format=%s)", structured_output_type.__name__)
    return llm_config_obj


def _get_last_non_empty_message_content(messages: Optional[List[Dict]]) -> Optional[str]:
    """Get content of the last message with non-empty content."""
    if not messages:
        return None
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        # TODO: No idea how multimodal would come into play here, AI generated
        if isinstance(content, list) and content:  # Handle multimodal content
            combined_text = []
            try:
                # Prioritize text parts
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                        combined_text.append(item["text"].strip())
                if combined_text:
                    return "\n".join(combined_text)

                # If no text parts, serialize the first non-text part as JSON or string
                if content:
                    first_item = content[0]
                    if isinstance(first_item, dict):
                        return json.dumps(first_item)  # Serialize first complex item
                    else:
                        return str(first_item).strip()  # Convert simple item to string
            except Exception as e:
                logger.warning("Error extracting content from multimodal message item: %s. Content: %s", e, content)
    return None  # No non-empty content found


def _get_reliable_tool_context(context_variables: ContextVariables, context_key: str) -> ReliableToolContext:
    """Retrieve and validate the ReliableToolContext from ContextVariables."""
    context_data = context_variables.get(context_key)
    if context_data is None:
        raise KeyError(f"ReliableToolContext key '{context_key}' not found.")
    try:
        if isinstance(context_data, ReliableToolContext):
            return context_data
        if isinstance(context_data, str):
            return ReliableToolContext.model_validate_json(context_data)
        if isinstance(context_data, dict):
            return ReliableToolContext.model_validate(context_data)
        raise TypeError(
            f"Unexpected type {type(context_data)} for context key '{context_key}'. Expected ReliableToolContext, str, or dict."
        )
    except (ValidationError, json.JSONDecodeError, TypeError) as e:
        preview = f" Preview: '{str(context_data)[:100]}...'" if isinstance(context_data, (str, dict)) else ""
        logger.error(
            "Failed loading ReliableToolContext '%s'. Error: %s. Type: %s.%s",
            context_key,
            e,
            type(context_data).__name__,
            preview,
            exc_info=True,
        )
        raise ValueError(f"Failed loading ReliableToolContext key '{context_key}': {e}") from e


def _set_reliable_tool_context(
    context_variables: ContextVariables, context_key: str, context: ReliableToolContext
) -> None:
    """Serialize and store the ReliableToolContext in ContextVariables."""
    if not isinstance(context, ReliableToolContext):
        raise TypeError(f"Object to set must be a ReliableToolContext, got {type(context)}.")
    try:
        context_variables[context_key] = context.model_dump_json(warnings="warn")
    except Exception as e:
        context_dict_str = "N/A"
        try:
            context_dict_str = str(context.model_dump(warnings="warn", exclude={"attempts"}))[:500]
        except Exception:
            pass
        logger.critical(
            "Failed serializing ReliableToolContext key '%s': %s. Context (partial): %s",
            context_key,
            e,
            context_dict_str,
            exc_info=True,
        )
        raise ValueError(f"Critical error serializing ReliableToolContext: {e}") from e


def get_runner_prompt(task: str, agent_system_message: str, internal_tool_name: str) -> str:
    """Generate the system prompt for the internal runner agent."""
    # Keep this prompt focused on the runner's job: making the tool call
    return f"""
You are an AI assistant responsible for invoking a specific function based on the user's task and conversation history.
Function to call: '{internal_tool_name}'
Analyze the previous attempt's outcome (if any, visible in history) and adjust the function arguments accordingly for this retry. If this is the first attempt, determine the best initial arguments based on the task and initial context.

You MUST invoke the function '{internal_tool_name}' exactly one time per response using a tool call format that the system can execute.
Do NOT just output text explaining what you would do, or asking for confirmation. Directly make the tool call.
Analyze the task description and *full conversation history* carefully to determine the correct arguments for the function call.
You MUST provide a 'hypothesis' argument summarizing the expected outcome or result format of the function call based on the inputs.

Base Instructions:
{agent_system_message}

Current Task:
{task}
"""


def get_validator_prompt(
    task: str, base_validator_system_message: str, dynamic_validation_addition: Optional[str] = None
) -> str:
    """Generate the system prompt for the internal validator agent."""
    # This prompt is now simpler as the full context will be provided in the message list
    dynamic_section = (
        f"\n\nAdditional Dynamic Requirements for This Specific Run:\n{dynamic_validation_addition.strip()}"
        if dynamic_validation_addition and dynamic_validation_addition.strip()
        else ""
    )
    return f"""
You are an AI validation assistant. You will receive a curated message list containing:
1. Initial context messages (original request, potentially prior conversation).
2. Provided ground truth information (if any).
3. The final result (or error message) of a function call intended to accomplish the task.

Your goal is to validate if the *final function call result/error* meets ALL requirements based on the *entire context provided in the message list*. Consider the base task description, base validation rules, initial context/ground truth, and any dynamic requirements below.

Base Task Description (for reference):
{task}

Base Validation Rules/Context:
{base_validator_system_message}{dynamic_section}

Evaluate the *final function call result/error* (presented at the end of the message list) based on *all* information provided.
Your output MUST be a JSON object strictly conforming to the ValidationResult schema:
{{
  "validation_result": boolean, // True if the result is valid, False otherwise
  "justification": "string" // Explain your reasoning clearly, especially for failures.
}}
Respond ONLY with this JSON object and nothing else. Do not add explanations or conversational text before or after the JSON.
"""


# --- Reliable Function Wrapper ---
def reliable_function_wrapper(tool_function: Callable, validator: ConversableAgent, context_variables_key: str):
    """Wraps the target function, returning a sync or async wrapper.

    Adds 'hypothesis' and 'context_variables' keyword-only arguments.
    Returns a ReplyResult targeting the validator.
    """
    is_original_func_async = inspect.iscoroutinefunction(tool_function)
    tool_sig = inspect.signature(tool_function)

    def _handle_error(
        attempt: ExecutionAttempt, context_vars: ContextVariables, context: ReliableToolContext, e: Exception
    ) -> ReplyResult:
        """Shared logic to handle execution error."""
        err_msg = f"{type(e).__name__}: {e}"
        logger.error(
            "Function execution error for '%s': %s",
            getattr(tool_function, "__name__", "unknown_func"),
            err_msg,
            exc_info=True,
        )
        attempt.error = err_msg
        err_msg_validator = f"Function execution failed with error: {err_msg}."
        if attempt not in context.attempts:
            context.attempts.append(attempt)
        try:
            _set_reliable_tool_context(context_vars, context_variables_key, context)
        except Exception as e_ctx:
            logger.error("Failed saving context after error: %s", e_ctx, exc_info=True)
        # Target the validator even on error, so it knows the attempt failed execution
        return ReplyResult(context_variables=context_vars, target=AgentTarget(validator), message=err_msg_validator)

    if not is_original_func_async:

        @functools.wraps(tool_function)
        def sync_wrapper(*args, hypothesis: str, context_variables: ContextVariables, **kwargs) -> ReplyResult:
            try:
                context = _get_reliable_tool_context(context_variables, context_variables_key)
            except (KeyError, ValueError) as e_get_ctx:
                logger.error("Wrapper cannot get context before sync execution: %s", e_get_ctx, exc_info=True)
                return ReplyResult(message=f"Critical Error: Could not retrieve ReliableToolContext ({e_get_ctx})")
            attempt = ExecutionAttempt(args=list(args), kwargs=kwargs, hypothesis=hypothesis)
            try:
                result = tool_function(*args, **kwargs)
                res_data, res_str, final_reply_result = None, "", None
                # Process result (common Autogen patterns)
                if isinstance(result, ReplyResult):
                    res_data = result.message
                    res_str = (
                        str(result.message)
                        if result.message is not None
                        else (str(res_data) if res_data is not None else "")
                    )
                    final_reply_result = result
                elif isinstance(result, tuple) and len(result) == 2:
                    res_data = result[0]
                    second_element = result[1]
                    if isinstance(second_element, ReplyResult):
                        res_str = (
                            str(second_element.message)
                            if second_element.message is not None
                            else (str(res_data) if res_data is not None else "")
                        )
                        final_reply_result = second_element
                    else:
                        try:
                            res_str = str(second_element)
                        except Exception as str_e:
                            logger.warning(
                                "Could not convert second element of tuple result to string: %s. Using repr(). Result data: %s",
                                str_e,
                                res_data,
                            )
                            res_str = repr(second_element)
                else:
                    res_data = result
                    try:
                        res_str = str(res_data)
                    except Exception as str_e:
                        logger.warning("Could not convert result to string: %s. Using repr().", str_e)
                        res_str = repr(res_data)

                attempt.result_data = res_data
                attempt.result_str = res_str
                context.attempts.append(attempt)
                try:
                    _set_reliable_tool_context(context_variables, context_variables_key, context)
                except Exception as e_ctx:
                    logger.error(
                        "Failed saving context after successful sync execution: %s", e_ctx, exc_info=True
                    )  # Non-fatal

                if final_reply_result is None:
                    final_reply_result = ReplyResult(
                        context_variables=context_variables, target=AgentTarget(validator), message=res_str or ""
                    )
                else:
                    final_reply_result.target = AgentTarget(validator)
                    final_reply_result.context_variables = context_variables
                    final_reply_result.message = (
                        final_reply_result.message if final_reply_result.message is not None else (res_str or "")
                    )

                return final_reply_result

            except Exception as e:
                return _handle_error(attempt, context_variables, context, e)

        wrapper_func = sync_wrapper
    else:

        @functools.wraps(tool_function)
        async def async_wrapper(*args, hypothesis: str, context_variables: ContextVariables, **kwargs) -> ReplyResult:
            try:
                context = _get_reliable_tool_context(context_variables, context_variables_key)
            except (KeyError, ValueError) as e_get_ctx:
                logger.error("Wrapper cannot get context before async execution: %s", e_get_ctx, exc_info=True)
                return ReplyResult(message=f"Critical Error: Could not retrieve ReliableToolContext ({e_get_ctx})")
            attempt = ExecutionAttempt(args=list(args), kwargs=kwargs, hypothesis=hypothesis)
            try:
                result = await tool_function(*args, **kwargs)
                res_data, res_str, final_reply_result = None, "", None
                if isinstance(result, ReplyResult):
                    res_data = result.message
                    res_str = (
                        str(result.message)
                        if result.message is not None
                        else (str(res_data) if res_data is not None else "")
                    )
                    final_reply_result = result
                elif isinstance(result, tuple) and len(result) == 2:
                    res_data = result[0]
                    second_element = result[1]
                    if isinstance(second_element, ReplyResult):
                        res_str = (
                            str(second_element.message)
                            if second_element.message is not None
                            else (str(res_data) if res_data is not None else "")
                        )
                        final_reply_result = second_element
                    else:
                        try:
                            res_str = str(second_element)
                        except Exception as str_e:
                            logger.warning(
                                "Could not convert second element of tuple result to string (async): %s. Using repr(). Result data: %s",
                                str_e,
                                res_data,
                            )
                            res_str = repr(second_element)
                else:
                    res_data = result
                    try:
                        res_str = str(res_data)
                    except Exception as str_e:
                        logger.warning("Could not convert result to string (async): %s. Using repr().", str_e)
                        res_str = repr(res_data)

                attempt.result_data = res_data
                attempt.result_str = res_str
                context.attempts.append(attempt)
                try:
                    _set_reliable_tool_context(context_variables, context_variables_key, context)
                except Exception as e_ctx:
                    logger.error(
                        "Failed saving context after successful async execution: %s", e_ctx, exc_info=True
                    )  # Non-fatal

                if final_reply_result is None:
                    final_reply_result = ReplyResult(
                        context_variables=context_variables, target=AgentTarget(validator), message=res_str or ""
                    )
                else:
                    final_reply_result.target = AgentTarget(validator)
                    final_reply_result.context_variables = context_variables
                    final_reply_result.message = (
                        final_reply_result.message if final_reply_result.message is not None else (res_str or "")
                    )

                return final_reply_result
            except Exception as e:
                return _handle_error(attempt, context_variables, context, e)

        wrapper_func = async_wrapper

    # Adapt Signature
    # TODO: Clean this up, more AI generated overcomplicated stuff
    params = list(tool_sig.parameters.values())
    pos_or_kw_params, kw_only_params, var_pos_param, var_kw_param = [], [], None, None
    for p in params:
        if p.kind == inspect.Parameter.POSITIONAL_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            pos_or_kw_params.append(p)
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            var_pos_param = p
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            kw_only_params.append(p)
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            var_kw_param = p

    new_kw_only_params = [
        inspect.Parameter(
            "hypothesis", inspect.Parameter.KEYWORD_ONLY, annotation=str, default=inspect.Parameter.empty
        ),
        inspect.Parameter(
            "context_variables",
            inspect.Parameter.KEYWORD_ONLY,
            annotation=ContextVariables,
            default=inspect.Parameter.empty,
        ),
    ]

    wrapper_params = (
        pos_or_kw_params
        + ([var_pos_param] if var_pos_param else [])
        + kw_only_params
        + new_kw_only_params
        + ([var_kw_param] if var_kw_param else [])
    )
    wrapper_func.__signature__ = inspect.Signature(parameters=wrapper_params, return_annotation=ReplyResult)
    return wrapper_func


@export_module("autogen.tools.experimental")
class ReliableToolError(Exception):
    """Custom exception for errors during ReliableTool execution."""

    def __init__(self, message: str, final_context: Optional[ReliableToolContext] = None):
        super().__init__(message)
        self.final_context = final_context


@export_module("autogen.tools.experimental")
class ReliableTool(Tool):
    """Wraps a function/tool for reliable execution via internal validation.

    Injects 'messages' and 'ground_truth' into the validator's context if provided
    either during initialization or via the run methods (run methods take precedence).
    """

    INTERNAL_TOOL_NAME_PREFIX = "execute_"

    def __init__(
        self,
        name: str,
        func_or_tool: Union[Callable, Tool],
        runner_llm_config: Union[LLMConfig, dict, bool],
        validator_llm_config: Union[LLMConfig, dict, bool],
        description: Optional[str] = None,
        runner_system_message: str = "You are a helpful AI assistant.",
        validator_system_message: str = "Validate the function result based on the task and context.",
        max_retries: int = 3,
        agent_kwargs: Optional[dict] = None,
        enable_dynamic_validation: bool = False,
        messages: Optional[List[Dict]] = None,
        ground_truth: Optional[List[str]] = None,
    ) -> None:
        self._original_func, original_name, original_description = self._extract_func_details(func_or_tool)
        self._is_original_func_async = inspect.iscoroutinefunction(self._original_func)

        self._runner_llm_config = ConversableAgent._validate_llm_config(runner_llm_config)
        self._validator_llm_config = ConversableAgent._validate_llm_config(validator_llm_config)

        if self._validator_llm_config is False:
            raise ValueError("Validator agent requires a valid LLM configuration.")

        # Basic check if validator LLM config seems usable
        try:
            if isinstance(self._validator_llm_config, LLMConfig):
                validator_model_specified = getattr(self._validator_llm_config, "model", None) or any(
                    getattr(cfg, "model", None)
                    for cfg in getattr(self._validator_llm_config, "config_list", [])
                    if hasattr(cfg, "model")
                )
                if not validator_model_specified:
                    warnings.warn(
                        f"Validator LLM config for tool '{name}' does not explicitly specify a 'model'. Structured output might fail.",
                        UserWarning,
                    )
            elif isinstance(self._validator_llm_config, dict):
                if not self._validator_llm_config.get("model") and not any(
                    cfg.get("model")
                    for cfg in self._validator_llm_config.get("config_list", [])
                    if isinstance(cfg, dict)
                ):
                    warnings.warn(
                        f"Validator LLM config dict for tool '{name}' does not explicitly specify a 'model'. Structured output might fail.",
                        UserWarning,
                    )
        except Exception as e:
            logger.warning(f"Could not fully check validator LLM config for tool '{name}' due to error: {e}")

        self._runner_system_message = runner_system_message
        self._validator_system_message = validator_system_message
        self.max_retries = max_retries
        self._context_variables_key = f"{name}_ReliableToolContext_{id(self)}"
        self._agent_kwargs = copy.deepcopy(agent_kwargs) if agent_kwargs is not None else {}
        self._original_func_name = original_name
        self.enable_dynamic_validation = enable_dynamic_validation

        # <<< Store init-time messages and ground truth safely
        self._init_messages = copy.deepcopy(messages) if messages else None
        self._init_ground_truth = copy.deepcopy(ground_truth) if ground_truth else None

        self._tool_description = description if description is not None else original_description

        public_entry_point_func = self._define_public_entry_point(
            self._is_original_func_async, self.enable_dynamic_validation
        )

        super().__init__(
            name=name,
            description=self._tool_description,
            func_or_tool=public_entry_point_func,
        )

        self._validator_name = f"{self.name}_Validator"
        self._runner_name = f"{self.name}_Runner"
        # Pass self to setup methods to access instance vars if needed by hooks later
        self._validator = self._setup_validator_agent()
        self._reliable_func_wrapper = reliable_function_wrapper(
            self._original_func, self._validator, self._context_variables_key
        )
        self._runner = self._setup_runner_agent()
        self._register_internal_hooks()  # Register hooks after agents are created

    def _define_public_entry_point(self, is_async: bool, enable_dynamic: bool) -> Callable:
        """
        Helper to define the correct public entry point function for the agent.
        This function's signature should only contain parameters expected from the AGENT.
        """
        if not is_async:
            if enable_dynamic:
                # Agent provides task and optional validation prompt
                def sync_entry_point_with_validation(
                    task: str, validation_prompt_addition: Optional[str] = None
                ) -> Any:
                    return self.run(task=task, validation_prompt_addition=validation_prompt_addition)

                return sync_entry_point_with_validation
            else:
                # Agent only provides task
                def sync_entry_point_without_validation(task: str) -> Any:
                    return self.run(task=task, validation_prompt_addition=None)

                return sync_entry_point_without_validation
        else:  # is_async
            if enable_dynamic:
                # Agent provides task and optional validation prompt
                async def async_entry_point_with_validation(
                    task: str, validation_prompt_addition: Optional[str] = None
                ) -> Any:
                    return await self.a_run(task=task, validation_prompt_addition=validation_prompt_addition)

                return async_entry_point_with_validation
            else:
                # Agent only provides task
                async def async_entry_point_without_validation(task: str) -> Any:
                    return await self.a_run(task=task, validation_prompt_addition=None)

                return async_entry_point_without_validation

    def _extract_func_details(self, func_or_tool: Union[Callable, Tool]) -> Tuple[Callable, str, str]:
        """Extracts function, name, and description (Unchanged)."""
        default_desc_template = "Executes the '{name}' function."
        if isinstance(func_or_tool, Tool):
            func = getattr(func_or_tool, "func", None)
            if not callable(func):
                raise TypeError(
                    f"Tool '{func_or_tool.name}' provided but its 'func' attribute is not callable or missing."
                )
            name = func_or_tool.name
            desc = func_or_tool.description
            if not desc or desc == f"Tool '{name}'." or desc == "No description provided.":
                func_doc = inspect.getdoc(func)
                if func_doc:
                    desc = func_doc.strip()
                else:
                    try:
                        sig = inspect.signature(func)
                    except ValueError:
                        sig_desc = "(signature unavailable)"
                    else:
                        sig_desc = f"Parameters: ({', '.join(sig.parameters.keys())})"
                    desc = f"{default_desc_template.format(name=name)} {sig_desc}"
            return func, name, desc
        elif callable(func_or_tool):
            name = getattr(func_or_tool, "__name__", "callable_function")
            doc = inspect.getdoc(func_or_tool)
            if doc:
                desc = doc.strip()
            else:
                try:
                    sig = inspect.signature(func_or_tool)
                except ValueError:
                    sig_desc = "(signature unavailable)"
                else:
                    sig_desc = f"Parameters: ({', '.join(sig.parameters.keys())})"
                desc = f"{default_desc_template.format(name=name)} {sig_desc}"
            return func_or_tool, name, desc
        raise TypeError(
            "Input 'func_or_tool' must be a callable or an autogen.Tool instance with a callable 'func' attribute."
        )

    def _setup_validator_agent(self) -> ConversableAgent:
        """Create and configure the internal validator agent (Unchanged)."""
        try:
            structured_llm_config = _configure_llm_for_structured_output(
                copy.deepcopy(self._validator_llm_config), ValidationResult
            )
            if structured_llm_config is False:
                raise ValueError("Validator LLM config resolved to False.")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to configure validator LLM for structured output: {e}") from e

        merged_kwargs = {"llm_config": structured_llm_config, "human_input_mode": "NEVER", **self._agent_kwargs}
        return ConversableAgent(
            name=self._validator_name, system_message="[Validator Prompt Updated Per Run]", **merged_kwargs
        )

    def _setup_runner_agent(self) -> ConversableAgent:
        """Create and configure the internal runner agent (Unchanged)."""
        runner_llm_config_copy = copy.deepcopy(self._runner_llm_config)
        merged_kwargs = {"llm_config": runner_llm_config_copy, "human_input_mode": "NEVER", **self._agent_kwargs}
        runner = ConversableAgent(
            name=self._runner_name, system_message="[Runner Prompt Updated Per Run]", **merged_kwargs
        )

        internal_tool_name = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"
        internal_tool = Tool(
            name=internal_tool_name, description=self._tool_description, func_or_tool=self._reliable_func_wrapper
        )

        try:
            internal_tool.register_tool(runner)
            logger.info("Successfully registered internal tool '%s' with runner '%s'", internal_tool_name, runner.name)
        except Exception as e:
            wrapper_sig = getattr(self._reliable_func_wrapper, "__signature__", "N/A")
            logger.error(
                f"Failed to register internal tool '{internal_tool_name}' for runner '{runner.name}'. Wrapper Signature: {wrapper_sig}. Error: {e}",
                exc_info=True,
            )
            raise TypeError(
                f"Critical error registering internal tool '{internal_tool_name}'. Original Error: {e}"
            ) from e

        # Post-registration cleanup (Unchanged)
        # TODO: Why is this here?
        if runner.llm_config and runner.llm_config is not False:
            runner_config_obj = runner.llm_config

            def _remove_response_format(config_obj):
                removed = False
                key = "response_format"
                if isinstance(config_obj, dict):
                    if key in config_obj:
                        del config_obj[key]
                        removed = True
                elif hasattr(config_obj, key):
                    try:
                        setattr(config_obj, key, None)
                        removed = True
                    except AttributeError:
                        logger.warning("Could not remove 'response_format' from runner LLM config object attribute.")
                return removed

            if _remove_response_format(runner_config_obj):
                logger.debug("Removed 'response_format' from runner LLM config.")
            config_list = None
            if isinstance(runner_config_obj, dict):
                config_list = runner_config_obj.get("config_list")
            elif hasattr(runner_config_obj, "config_list"):
                config_list = getattr(runner_config_obj, "config_list", None)
            if isinstance(config_list, list):
                for item in config_list:
                    if _remove_response_format(item):
                        logger.debug("Removed 'response_format' from runner config_list item.")
        return runner

    def _register_internal_hooks(self) -> None:
        """Register hooks for internal group chat interaction."""
        self._validator.register_hook(
            hookable_method="process_message_before_send", hook=self._validator_structured_output_hook
        )
        self._validator.register_hook(
            hookable_method="process_all_messages_before_reply", hook=self._validator_construct_context_hook
        )
        self._runner.register_hook(hookable_method="process_message_before_send", hook=self._ensure_function_call_hook)

    def _validator_structured_output_hook(
        self, sender: Agent, message: Union[dict, str], recipient: Agent, silent: bool
    ) -> Union[dict, str]:
        """Processes validator LLM output, expecting ValidationResult JSON, updates context, returns formatted JSON string."""
        if sender.name != self._validator_name:
            return message

        message_text = None
        if isinstance(message, str):
            message_text = message.strip()
        elif isinstance(message, dict) and isinstance(message.get("content"), str):
            message_text = message["content"].strip()
        elif isinstance(message, dict) and isinstance(message.get("content"), dict):
            try:
                validation_result_obj = ValidationResult.model_validate(message["content"])
                message_text = validation_result_obj.model_dump_json()
                logger.debug("Validator hook received pre-parsed dict content, validated and re-serialized.")
            except (ValidationError, TypeError) as e:
                logger.warning(
                    "Validator hook received dict content, but failed ValidationResult validation: %s. Content: %s",
                    e,
                    message.get("content"),
                )
                try:
                    message_text = json.dumps(message.get("content"))
                except TypeError:
                    message_text = str(message.get("content"))

        if not message_text:
            logger.warning("Validator hook: Received empty content from validator: %s", message)
            validation_result_obj = ValidationResult(
                validation_result=False, justification="Validator produced empty or invalid output format"
            )
            self._try_update_context_validation(sender, validation_result_obj)
            return validation_result_obj.format()

        validation_result_obj: Optional[ValidationResult] = None
        # TODO: Json regex seems overcomplicated
        try:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", message_text, re.DOTALL)
            parsed_json = None
            if json_match:
                json_str = json_match.group(1)
                logger.debug("Validator hook extracted JSON from markdown block.")
                parsed_json = json.loads(json_str)
            else:
                parsed_json = json.loads(message_text)

            validation_result_obj = ValidationResult.model_validate(parsed_json)
            status = "PASSED" if validation_result_obj.validation_result else "FAILED"
            log_level = logging.INFO if status == "PASSED" else logging.WARNING
            logger.log(
                log_level,
                f"Validator Hook: Parsed Validation - {status}. Justification: {validation_result_obj.justification}",
            )

            validator_agent = sender
            if not hasattr(validator_agent, "handoffs"):
                logger.error("Validator agent instance missing 'handoffs'.")
            else:
                if not validation_result_obj.validation_result:
                    logger.info("Validation failed, setting handoff to runner: %s", self._runner_name)
                    validator_agent.handoffs.set_after_work(target=AgentTarget(self._runner))
                else:
                    logger.info("Validation passed, setting handoff to TerminateTarget.")
                    validator_agent.handoffs.set_after_work(target=TerminateTarget())

        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(
                f"Validator Hook: Failed to parse/validate output as ValidationResult JSON ({type(e).__name__}): {e}. Raw: ```\n{message_text}\n```"
            )
            validation_result_obj = ValidationResult(
                validation_result=False,
                justification=f"Validation output parsing/schema failed ({type(e).__name__}: {e})",
            )
            if hasattr(sender, "handoffs"):
                sender.handoffs.set_after_work(target=AgentTarget(self._runner))
        except Exception as e:
            logger.error(
                f"Validator Hook: Unexpected error ({type(e).__name__}): {e}. Raw: ```\n{message_text}\n```",
                exc_info=True,
            )
            validation_result_obj = ValidationResult(
                validation_result=False, justification=f"Unexpected error in validation hook ({type(e).__name__})"
            )
            if hasattr(sender, "handoffs"):
                sender.handoffs.set_after_work(target=AgentTarget(self._runner))

        if validation_result_obj:
            self._try_update_context_validation(sender, validation_result_obj)
        else:
            fallback_validation = ValidationResult(
                validation_result=False, justification="Internal error: Validation object creation failed."
            )
            self._try_update_context_validation(sender, fallback_validation)

        return (validation_result_obj or fallback_validation).format()

    def _try_update_context_validation(self, sender: Agent, validation_result: ValidationResult) -> None:
        """Helper to attempt updating the validation state in the ReliableToolContext."""
        try:
            context_vars = getattr(sender, "context_variables", None)
            # Try fallback reference if direct attribute fails
            if not isinstance(context_vars, ContextVariables) and hasattr(sender, "_pattern_context_vars_ref"):
                context_vars = sender._pattern_context_vars_ref

            if not isinstance(context_vars, ContextVariables):
                logger.error(
                    "Validator hook: Cannot access valid ContextVariables on sender '%s'. Cannot update validation state.",
                    sender.name,
                )
                return

            tool_context = _get_reliable_tool_context(context_vars, self._context_variables_key)
            latest_attempt = tool_context.latest_attempt
            if latest_attempt and latest_attempt.validation is None:
                latest_attempt.validation = validation_result
                _set_reliable_tool_context(context_vars, self._context_variables_key, tool_context)
                logger.info(
                    "Validator hook: Updated validation status: %s",
                    "Passed" if validation_result.validation_result else "Failed",
                )
            elif latest_attempt:
                logger.warning("Validator hook: Validation was already set. Not overwriting.")
            else:
                logger.warning(
                    "Validator hook: No execution found in context '%s' to update.", self._context_variables_key
                )
        except (KeyError, ValueError, TypeError) as e_ctx:
            logger.error(
                "Validator hook: Failed to get/set ReliableToolContext for validation update: %s", e_ctx, exc_info=True
            )
        except Exception as e_unexp:
            logger.error(
                "Validator hook: Unexpected error during context update for validation: %s", e_unexp, exc_info=True
            )

    def _validator_construct_context_hook(self, messages: List[Dict], **kwargs: Any) -> List[Dict]:
        """
        Constructs the EXACT message list for the validator's LLM call.
        Ignores the incoming `messages` history except for the last message (result).
        Builds: [SystemPrompt] + [InitialMessages] + [GroundTruth] + [ResultMessage]
        """
        sender = self._validator
        logger.debug("Validator Construct Context Hook running for agent %s.", sender.name)
        if not messages:
            logger.error("Validator Construct Context Hook: Received empty message list.")
            return [
                {"role": "system", "content": sender.system_message},
                {"role": "user", "content": "[Error: No messages received]"},
            ]

        # 1. Get Validator's System Prompt
        # Attempt to get the current system message directly from the sender agent instance
        system_prompt = getattr(sender, "system_message", None)
        if not system_prompt or not isinstance(system_prompt, str):
            # Fallback: Try to get it from the first message if it looks like a system prompt
            if messages[0].get("role") == "system":
                system_prompt = messages[0].get("content")
            else:
                logger.warning("Validator Construct Context Hook: Could not determine validator system prompt.")
                system_prompt = "[System Prompt Unavailable]"  # Placeholder

        system_message_dict = {"role": "system", "content": system_prompt}

        # 2. Get ReliableToolContext to retrieve initial messages/ground truth
        tool_context: Optional[ReliableToolContext] = None
        try:
            context_vars = getattr(sender, "context_variables", None)
            # Try fallback reference
            if not isinstance(context_vars, ContextVariables) and hasattr(sender, "_pattern_context_vars_ref"):
                context_vars = sender._pattern_context_vars_ref

            if isinstance(context_vars, ContextVariables):
                tool_context = _get_reliable_tool_context(context_vars, self._context_variables_key)
            else:
                logger.error("Validator Construct Context Hook: Could not access ContextVariables on sender.")

        except (KeyError, ValueError, TypeError) as e_ctx:
            logger.error("Validator Construct Context Hook: Failed to get ReliableToolContext: %s", e_ctx)

        # 3. Get Initial Messages & Ground Truth from Context
        initial_messages_to_inject = []
        if tool_context and tool_context.initial_messages:
            logger.debug("Injecting %d initial messages from context.", len(tool_context.initial_messages))
            initial_messages_to_inject.extend(copy.deepcopy(tool_context.initial_messages))  # Use deepcopy

        ground_truth_messages_to_inject = []
        if tool_context and tool_context.initial_ground_truth:
            logger.debug("Injecting %d ground truth strings from context.", len(tool_context.initial_ground_truth))
            for i, gt in enumerate(tool_context.initial_ground_truth):
                ground_truth_messages_to_inject.append({
                    "role": "user",  # Use user role for visibility
                    "content": f"[[Provided Ground Truth {i + 1}]]:\n{gt}",
                })

        # 4. Get the Last Function Result/Error
        last_message = messages[-1]
        last_content = _get_last_non_empty_message_content([last_message])  # Use helper
        if last_content is None:
            last_content = f"[Warning: Could not extract text content from last message: {last_message}]"
            logger.warning("Validator Construct Context Hook: %s", last_content)

        # Format the result message clearly for the validator
        result_message_dict = {
            "role": "user",  # Present as user input to the LLM
            "content": f"--- Function Result/Error to Validate ---\n```\n{last_content}\n```\n--- End of Result/Error ---",
        }

        # 5. Construct the Final Message List
        final_messages = (
            [system_message_dict] + initial_messages_to_inject + ground_truth_messages_to_inject + [result_message_dict]
        )

        # Optional: Log the constructed history for debugging
        try:
            logger.debug(
                "Validator Construct Context Hook - Final constructed messages for LLM:\n%s",
                json.dumps(final_messages, indent=2),
            )
        except Exception:
            logger.debug("Validator Construct Context Hook - Could not serialize final messages for logging.")

        return final_messages

    def _ensure_function_call_hook(
        self, sender: Agent, message: Union[dict, str], recipient: Agent, silent: bool
    ) -> Union[dict, str]:
        """Encourages the runner LLM to make the required tool call."""
        if sender.name != self._runner_name:
            return message

        is_tool_call = False
        content_str = ""
        tool_calls_list = []
        if isinstance(message, dict):
            tool_calls_list = message.get("tool_calls")
            is_tool_call = isinstance(tool_calls_list, list) and len(tool_calls_list) > 0
            content_str = message.get("content", "") or ""
        elif isinstance(message, str):
            content_str = message

        tool_name_expected = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"
        correct_tool_called = False
        if is_tool_call and isinstance(tool_calls_list, list):
            for call in tool_calls_list:
                if isinstance(call, dict) and call.get("type") == "function":
                    func_info = call.get("function")
                    if isinstance(func_info, dict) and func_info.get("name") == tool_name_expected:
                        correct_tool_called = True
                        break

        needs_reminder = (not correct_tool_called) and not (
            (
                "error" in content_str.lower()
                and (
                    "tool" in content_str.lower()
                    or "function" in content_str.lower()
                    or "argument" in content_str.lower()
                )
            )
            or "hypothesis" in content_str.lower()
        )

        if needs_reminder:
            logger.warning(
                "Runner '%s' did not generate required tool call for '%s'. Appending reminder.",
                self._runner_name,
                tool_name_expected,
            )
            reminder = f"\n\n[[System Reminder: You MUST invoke the function '{tool_name_expected}' using a tool call. Provide all required arguments including 'hypothesis'.]]"
            if isinstance(message, dict):
                message["content"] = (message.get("content") or "") + reminder
                if "tool_calls" in message:
                    message["tool_calls"] = []
            elif isinstance(message, str):
                message = message + reminder
            else:
                logger.error("Unexpected message type in _ensure_function_call_hook: %s", type(message))
                message = {
                    "role": "assistant",
                    "content": f"[[Internal Error: Unexpected message type {type(message)}]]",
                }
        return message

    def _execute_internal_group_chat(
        self,
        task: str,
        initial_context: ContextVariables,
        dynamic_validation_str: Optional[str] = None,
    ) -> Tuple[Optional[Union[Dict, ReplyResult]], ContextVariables, Optional[Agent]]:
        """Sets up and runs the internal synchronous group chat for ONE execution cycle."""
        try:
            current_tool_context = _get_reliable_tool_context(initial_context, self._context_variables_key)
        except (KeyError, ValueError) as e:
            raise ReliableToolError(f"Failed loading initial context for internal chat: {e}", final_context=None) from e

        internal_tool_name = f"{self.INTERNAL_TOOL_NAME_PREFIX}{self._original_func_name}"

        try:
            runner_prompt = get_runner_prompt(task, self._runner_system_message, internal_tool_name)
            self._runner.update_system_message(runner_prompt)

            validator_prompt = get_validator_prompt(task, self._validator_system_message, dynamic_validation_str)
            self._validator.update_system_message(validator_prompt)

            # Store context ref on agents for hooks
            self._validator.context_variables = initial_context
            self._runner.context_variables = initial_context
            self._validator._pattern_context_vars_ref = initial_context

        except Exception as e:
            logger.error("Failed updating agent system messages or context references: %s", e, exc_info=True)
            raise ReliableToolError(
                f"Failed setting up agent prompts/context for execution: {e}", final_context=current_tool_context
            ) from e

        # Construct initial message list for the *runner* agent.
        # Retrieve the context again to ensure we have the most up-to-date initial messages/GT
        # (e.g., if _process_run updated them just before calling this).
        messages_for_runner_history = []
        try:
            existing_context = _get_reliable_tool_context(initial_context, self._context_variables_key)
            messages_to_inject = existing_context.initial_messages
            ground_truth_to_inject = existing_context.initial_ground_truth
            if messages_to_inject:
                messages_for_runner_history.extend(copy.deepcopy(messages_to_inject))
            if ground_truth_to_inject:
                for i, gt in enumerate(ground_truth_to_inject):
                    messages_for_runner_history.append({
                        "role": "user",
                        "content": f"[[Provided Ground Truth {i + 1}]]:\n{gt}",
                    })
        except (KeyError, ValueError) as e:
            logger.error(
                "Failed getting context for runner history setup: %s. Proceeding without initial messages/GT.", e
            )
            # Fallback to the context we fetched at the start of the method if needed, though it might be stale
            existing_context = current_tool_context

        task_message = {
            "role": "user",
            "content": f"[[Task Kickoff]]: Please execute the required function call for the task: {task}",
        }
        final_initial_messages_for_runner = messages_for_runner_history + [task_message]

        agent_pattern = DefaultPattern(
            agents=[self._runner, self._validator],
            initial_agent=self._runner,
            context_variables=initial_context,  # Pass context vars to the pattern
        )

        # TODO: Figure out the right multiplier to guarantee self.max_retries number of runner llm invocations
        max_internal_rounds = self.max_retries * 4
        logger.debug(f"Setting max internal chat rounds to {max_internal_rounds}")

        logger.info(f"--- Starting ReliableTool '{self.name}' Internal Chat ---")
        last_reply, final_context, last_agent = None, initial_context, None
        try:
            last_reply, final_context, last_agent = initiate_group_chat(
                pattern=agent_pattern,
                messages=final_initial_messages_for_runner,  # Start runner with the task context
                max_rounds=max_internal_rounds,
            )
            logger.info(
                f"--- ReliableTool '{self.name}' Internal Chat Finished (Last Agent: {getattr(last_agent, 'name', 'N/A')}) ---"
            )
            final_context = final_context if isinstance(final_context, ContextVariables) else initial_context
            return last_reply, final_context, last_agent
        except Exception as e:
            logger.critical(f"CRITICAL ERROR during internal group chat ({type(e).__name__} - {e}", exc_info=True)
            try:
                crashed_context_vars = final_context or initial_context
            except Exception:
                crashed_context_vars = initial_context
            try:
                crashed_tool_context = _get_reliable_tool_context(crashed_context_vars, self._context_variables_key)
            except Exception:
                crashed_tool_context = current_tool_context  # Use context from start if crashed one is bad
            raise ReliableToolError(
                f"Internal group chat failed critically: {e}", final_context=crashed_tool_context
            ) from e

    def _process_run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None,
        messages: Optional[List[Dict]] = None,  # Run-time messages
        ground_truth: Optional[List[str]] = None,  # Run-time ground truth
    ) -> Any:
        """Internal synchronous logic for reliable execution."""
        current_context_variables = context_variables if context_variables is not None else ContextVariables()
        if not isinstance(current_context_variables, ContextVariables):
            raise TypeError(f"Expected context_variables as ContextVariables or None, got {type(context_variables)}")

        dynamic_criteria = validation_prompt_addition

        # Determine which messages/GT to use (run-time take precedence over init-time)
        effective_messages = copy.deepcopy(messages) if messages is not None else self._init_messages
        effective_ground_truth = copy.deepcopy(ground_truth) if ground_truth is not None else self._init_ground_truth
        # Log which source is being used if relevant data exists
        msg_source_logged = False
        gt_source_logged = False
        if effective_messages:
            source = "run-time" if messages is not None else "init-time"
            logger.debug(f"[{self.name}] Using {source} messages for this run.")
            msg_source_logged = True
        if effective_ground_truth:
            source = "run-time" if ground_truth is not None else "init-time"
            logger.debug(f"[{self.name}] Using {source} ground truth for this run.")
            gt_source_logged = True
        if not msg_source_logged and not gt_source_logged:
            logger.debug(f"[{self.name}] No initial messages or ground truth provided for this run.")

        # Initialize or retrieve tool context state
        tool_context_needs_update = False
        if self._context_variables_key not in current_context_variables:
            initial_tool_context = ReliableToolContext(
                task=task,
                reliable_tool_name=self.name,
                dynamic_validation_input=dynamic_criteria,
                # Use effective messages/GT determined above
                initial_messages=effective_messages,  # Already deepcopied or None
                initial_ground_truth=effective_ground_truth,  # Already deepcopied or None
            )
            try:
                _set_reliable_tool_context(current_context_variables, self._context_variables_key, initial_tool_context)
                log_msg_init = f"Initialized ReliableToolContext for '{self.name}'"
                if effective_messages:
                    log_msg_init += " with initial messages"
                if effective_ground_truth:
                    log_msg_init += " with ground truth"
                if not effective_messages and not effective_ground_truth:
                    log_msg_init += " (no initial messages/GT)"
                log_msg_init += "."
                logger.info(log_msg_init)
            except ValueError as e_set:
                raise ReliableToolError(f"Failed to initialize ReliableToolContext: {e_set}") from e_set
        else:
            existing_context = _get_reliable_tool_context(current_context_variables, self._context_variables_key)
            # Update context if run-time args differ from stored values or init-time defaults differ
            # Check if the effective data for *this run* differs from what's *currently stored*
            if (
                existing_context.initial_messages != effective_messages
                or existing_context.initial_ground_truth != effective_ground_truth
                or existing_context.task != task  # Also update task if changed
                or existing_context.dynamic_validation_input != dynamic_criteria
            ):  # Also update dynamic input if changed
                # Log specifically what changed
                changes = []
                if existing_context.initial_messages != effective_messages:
                    changes.append("initial messages")
                if existing_context.initial_ground_truth != effective_ground_truth:
                    changes.append("ground truth")
                if existing_context.task != task:
                    changes.append("task")
                if existing_context.dynamic_validation_input != dynamic_criteria:
                    changes.append("dynamic validation input")

                logger.warning(
                    f"Existing ReliableToolContext found for '{self.name}', but current run inputs differ ({', '.join(changes)}). Updating context."
                )
                existing_context.initial_messages = effective_messages
                existing_context.initial_ground_truth = effective_ground_truth
                existing_context.task = task
                existing_context.dynamic_validation_input = dynamic_criteria
                tool_context_needs_update = True

            if tool_context_needs_update:
                try:
                    _set_reliable_tool_context(current_context_variables, self._context_variables_key, existing_context)
                except ValueError as e_set:
                    # Log error but don't necessarily fail the whole run, proceed with potentially stale context
                    logger.error("Failed to update existing ReliableToolContext with new run data: %s", e_set)
            else:
                logger.info(
                    "Reusing existing ReliableToolContext for '%s'. Run data matches stored context.", self.name
                )

        final_tool_context: Optional[ReliableToolContext] = None
        try:
            _, chat_context_variables, _ = self._execute_internal_group_chat(
                task=task,
                initial_context=current_context_variables,
                dynamic_validation_str=dynamic_criteria,
            )

            if isinstance(chat_context_variables, ContextVariables):
                current_context_variables = chat_context_variables
            else:
                logger.warning("Internal chat did not return updated ContextVariables.")

            final_tool_context = _get_reliable_tool_context(current_context_variables, self._context_variables_key)
            latest_attempt_obj = final_tool_context.latest_attempt
            if not latest_attempt_obj:
                raise ReliableToolError(
                    "Critical internal error: No execution recorded after chat cycle.", final_context=final_tool_context
                )

            if latest_attempt_obj.validation is None:
                logger.warning("[%s]: Validation state missing after execution. Assuming validation failed.", self.name)
                latest_attempt_obj.validation = ValidationResult(
                    validation_result=False, justification="Validation result missing after execution."
                )
                try:
                    _set_reliable_tool_context(
                        current_context_variables, self._context_variables_key, final_tool_context
                    )
                except Exception as e_ctx_save:
                    logger.error("Failed saving context after assuming validation failure: %s", e_ctx_save)

            if final_tool_context.is_complete_and_successful:
                logger.info("ReliableTool '%s' succeeded.", self.name)
                return final_tool_context.get_final_result_data()
            else:
                failure_reason = final_tool_context.get_failure_summary()
                logger.warning("ReliableTool '%s' failed. Reason: %s", self.name, failure_reason)
                raise ReliableToolError(
                    f"ReliableTool '{self.name}' failed. Last failure: {failure_reason}",
                    final_context=final_tool_context,
                )
        except (KeyError, ValueError) as e_get_ctx:
            logger.critical(
                "[%s] Execution failed critically retrieving context: %s", self.name, e_get_ctx, exc_info=True
            )
            try:
                final_tool_context = _get_reliable_tool_context(current_context_variables, self._context_variables_key)
            except Exception:
                final_tool_context = None
            raise ReliableToolError(
                f"Critical error retrieving context during execution: {e_get_ctx}", final_context=final_tool_context
            ) from e_get_ctx
        except Exception as e_unexp:
            logger.critical("[%s] Execution failed unexpectedly: %s", self.name, e_unexp, exc_info=True)
            try:
                final_tool_context = _get_reliable_tool_context(current_context_variables, self._context_variables_key)
            except Exception:
                final_tool_context = None
            # Check if it's already a ReliableToolError, if so, re-raise it directly
            if isinstance(e_unexp, ReliableToolError):
                raise e_unexp
            else:
                raise ReliableToolError(f"Unexpected error: {e_unexp}", final_context=final_tool_context) from e_unexp

    def run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None,
        messages: Optional[List[Dict]] = None,  # <<< Keep for direct user calls
        ground_truth: Optional[List[str]] = None,  # <<< Keep for direct user calls
    ) -> Any:
        """Synchronous execution entry point for user code or agent calls."""
        if self._is_original_func_async:
            raise TypeError(f"Sync 'run()' called for async tool '{self.name}'. Use 'a_run()'.")

        # When called via the agent wrapper, messages/ground_truth will be None here.
        # When called directly by user code, they might have values.
        # _process_run handles the logic of using these or init-time values.
        return self._process_run(
            task=task,
            context_variables=context_variables,
            validation_prompt_addition=validation_prompt_addition,
            messages=messages,  # Pass along whatever was provided (None or user data)
            ground_truth=ground_truth,  # Pass along whatever was provided (None or user data)
            # Do not pass **kwargs here unless _process_run expects them
        )

    async def a_run(
        self,
        task: str,
        context_variables: Optional[ContextVariables] = None,
        validation_prompt_addition: Optional[str] = None,
        messages: Optional[List[Dict]] = None,  # <<< Keep for direct user calls
        ground_truth: Optional[List[str]] = None,  # <<< Keep for direct user calls
    ) -> Any:
        """Asynchronous execution entry point for user code or agent calls."""
        if not self._is_original_func_async:
            warnings.warn(
                f"Running sync function '{self._original_func_name}' wrapped by ReliableTool '{self.name}' asynchronously using run_in_executor.",
                UserWarning,
            )

        # Always run _process_run in executor for now, simplifies async handling within it.
        # TODO: Make _process_run truly async if original func is async.
        loop = asyncio.get_running_loop()
        try:
            # Use functools.partial to properly capture args for the executor
            func_call = functools.partial(
                self._process_run,
                task=task,
                context_variables=context_variables,
                validation_prompt_addition=validation_prompt_addition,
                messages=messages,  # Pass along whatever was provided (None or user data)
                ground_truth=ground_truth,  # Pass along whatever was provided (None or user data)
            )
            result = await loop.run_in_executor(None, func_call)
            return result
        except Exception as e:
            logger.debug("[%s] a_run caught exception from executor: %s", self.name, e)
            # Reraise the original exception
            raise e
