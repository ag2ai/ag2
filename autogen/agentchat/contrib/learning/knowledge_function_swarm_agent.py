# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
from inspect import Parameter, Signature, signature
from typing import Callable, List, Optional, Union

from autogen.agentchat.agent import Agent
from autogen.agentchat.contrib.learning.function_models import (
    CRITERIA_RESULTS_DEF,
    HYPOTHESIS_DEF,
    PARAMETERS_DEF,
    QUESTION_RESPONSE_RESULT_DEF,
    RESULT_DEF,
    VALIDATION_CRITERIA_DEF,
    VALIDATION_RESULT_RESULT_DEF,
    AutoGenResult,
    FunctionResponse,
    ValidationResult,
)
from autogen.agentchat.contrib.learning.knowledge_sharing_swarm_agent import KnowledgeSharingSwarmAgent
from autogen.agentchat.contrib.swarm_agent import (
    AFTER_WORK,
    AfterWorkOption,
    SwarmAgent,
    SwarmResult,
    initiate_swarm_chat,
)


class KnowledgeFunctionSwarmAgent(KnowledgeSharingSwarmAgent):
    def __init__(
        self,
        name: str,
        researcher_llm_config: dict,
        result_validator_llm_config: dict,
        functions: List[Callable],
        user_notes: List[str] = [],
        agent_system_message: str = "",
        max_rounds: int = 10,
        use_own_knowledge: bool = True,
        verbose: bool = True,
        knowledge_sources: Optional[List[SwarmAgent]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            llm_config=None,  # None for now, it shouldn't do LLM things
            system_message="",
            knowledge_sources=knowledge_sources,
            use_own_knowledge=use_own_knowledge,
            **kwargs,
        )

        self._verbose = verbose
        self.user_notes = user_notes
        self._agent_system_message = agent_system_message
        self.max_rounds = max_rounds

        self._result_validator = SwarmAgent(
            name="ResultValidator-" + name,
            llm_config=_configure_structured_output(result_validator_llm_config, ValidationResult),
        )

        self._result_validator.register_hook(
            hookable_method="process_message_before_send", hook=self._result_validator_structured_output_hook
        )
        self._result_validator.register_hook(
            hookable_method="process_all_messages_before_reply", hook=self._only_previous_function_results_hook
        )
        self._original_function = functions[0]
        self._functions: list[SubSwarmFunctionWrapper] = [
            SubSwarmFunctionWrapper(function, self._function_response_callback, self._result_validator)
            for function in functions
        ]
        # A hack because it wants the name to be in there and I just want to keep it with this name for now
        self._memory = SwarmAgent(name="memory")
        self._researcher = SwarmAgent(
            name="Researcher-" + name,
            llm_config=researcher_llm_config,
            functions=self._functions,
        )
        self._researcher.register_hand_off(AFTER_WORK(self._result_validator))

    # System too
    def _only_previous_function_results_hook(self, messages: list[dict]) -> list[dict]:
        system_message = messages[0]
        function_results = _get_last_non_empty_message(messages)

        messages = [system_message, function_results]

    def _result_validator_structured_output_hook(
        self, sender: Agent, message: Union[dict, str], recipient: SwarmAgent, silent: bool
    ) -> Union[dict, str]:
        message_text = message if isinstance(message, str) else message.get("content", message)

        validation_result: ValidationResult = ValidationResult.parse_raw(message_text)
        self._pending_validation_result = validation_result
        if validation_result.result:
            self._result_validator.register_hand_off(AFTER_WORK(AfterWorkOption.TERMINATE))
        else:
            self._result_validator.register_hand_off(AFTER_WORK(self._researcher))
        return str(validation_result)

    # TODO Function results should be in a dictionary keyed by function for multi-function
    def _function_response_callback(self, function_response: FunctionResponse):
        self._pending_function_response = function_response

    def _get_messages(self, question):
        return self.get_function_memories_as_messages() + [
            {
                "role": "user",
                # "name": "user",
                "content": question,
            }
        ]

    # Maybe figure out a way to do currying or dependency injection to provide stuff as default more easily.
    def auto_gen_func(self, question: str, context_variables: dict = {}) -> AutoGenResult:
        self._pending_function_response: FunctionResponse = None
        self._pending_validation_result = None

        # Must use findings and also must be run before every invocation
        self._update_prompts(question)

        chat_result, final_context_variables, last_active_agent = initiate_swarm_chat(
            initial_agent=self._researcher,
            agents=[self._researcher, self._result_validator, self._memory],
            # Trying out passing the memories in as messages
            messages=self._get_messages(f"Invoke the function to answer the question.\nQuestion: {question}"),
            max_rounds=self.max_rounds,
            # I think this works as a reference everywhere basically
            context_variables=context_variables,
        )

        auto_gen_result: AutoGenResult = AutoGenResult(
            result=self._pending_function_response.result_data,
            question=question,
            function_response=self._pending_function_response,
            validation_result=self._pending_validation_result,
        )

        return auto_gen_result

    def set_agent_system_message(self, new_message):
        self._agent_system_message = new_message

    def _update_prompts(self, question):
        # We want the code from the original function
        self._researcher.update_system_message(
            get_researcher_prompt(question, self._agent_system_message, self._original_function)
        )
        self._result_validator.update_system_message(get_result_validator_prompt(question))


class SubSwarmFunctionWrapper:
    def __init__(self, tool_function, function_response_callback, results_validator):
        self.tool_function = tool_function
        self.function_response_callback = function_response_callback
        self.results_validator = results_validator

    def __call__(self, context_variables, hypothesis, validation_criteria, *args, **kwargs) -> SwarmResult:
        try:
            # Call the tool function
            (result_data, result_str) = self.tool_function(context_variables=context_variables, *args, **kwargs)
            function_response: FunctionResponse = FunctionResponse(
                function=self.tool_function,
                hypothesis=hypothesis,
                args=args,
                kwargs=kwargs,
                result_data=result_data,
                result_str=result_str,
                validation_criteria=validation_criteria,
            )
            self.function_response_callback(function_response)
            # When the function returns a SwarmResult, then expect that this function call terminates the research and the LearningAgent passes to the result.
            return SwarmResult(
                context_variables=context_variables,
                agent=self.results_validator,
                values=function_response.to_validation_submission(),
            )
        except Exception as e:
            # Capture the exception and debug it, not sure if I need to do this actually?
            return str(e)

    @property
    def __signature__(self):
        """
        Generate a custom function signature for the wrapper.

        Includes context_variables and hypothesis as parameters.
        """
        tool_sig = signature(self.tool_function)
        params = list(tool_sig.parameters.values())

        context_variables_param = Parameter(
            "context_variables",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=dict,
        )

        # TODO Related to prompting, make sure this is actually helpful to do
        # Add hypothesis parameter to function no matter what
        hypothesis_param = Parameter(
            "hypothesis",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=str,
        )

        validation_criteria_param = Parameter(
            "validation_criteria",
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=list[str],
        )

        # Insert the new parameters before args/kwargs
        params = [context_variables_param, hypothesis_param, validation_criteria_param] + [
            p for p in params if p.name not in ("context_variables", "hypothesis", "validation_criteria")
        ]

        # Return the updated signature
        return Signature(parameters=params)

    @property
    def __doc__(self):
        """
        Dynamically generate a docstring that includes additional parameters.

        Combines the original tool function's docstring with the added parameters.
        """
        # Retrieve the original docstring
        original_doc = self.tool_function.__doc__ or ""

        # Append documentation for
        # added parameters, yes its at the end, and it does assumt you do docstrings exactly like this
        added_docs = """:param context_variables: dict, shared state for the execution context.
:param validation_criteria: list[str], Criteria to evaluate in order to validate that the response is correct.
:param hypothesis: str, A hypothesis to be tested by executing the function. Should take the form of "if X then Y"
:return: str, the result of the function execution."""
        return f"{original_doc.strip()}\n\n{added_docs}"

    @property
    def __name__(self):
        # Forward the name of the tool function
        return self.tool_function.__name__


# non-destructive
# TODO Not sure if it is acceptable to mess with a llm_config
def _configure_structured_output(llm_config, structured_output_type):
    llm_config_copy = copy.deepcopy(llm_config)
    for config in llm_config_copy["config_list"]:
        config["response_format"] = structured_output_type
    return llm_config_copy


# TODO Check if there is an actual util for this
def _get_last_non_empty_message(messages):
    for message in reversed(messages):
        if message["content"]:
            return message["content"]
    return None


def get_researcher_prompt(question, agent_system_message, function) -> str:
    return f"""
Task: You are responsible for invoking the function provided to you to produce a result that answers the question below.

Do not ask anything of the user, continue researching until completion.
Do not ask for permission for anything, continue researching until completion.

{agent_system_message}

Instructions:
- Invoke the function to attempt to answer the question using a single invocation
- To facilitate the thinking process, always use the parameters:
    - hypothesis: {HYPOTHESIS_DEF}
    - validation_criteria: {VALIDATION_CRITERIA_DEF}
- Work through any errors that come up when invoking the function.
- You may be provided examples of successful function invocations and their results as messages.
    Leverage this information to guide future function invocations.
- You may be provided background information as messaegs.
    Leverage this information to guide future function invocations.

Question:
{question}

To help guide your input, the code for the function you are invoking is:
{inspect.getsource(function)}
"""


# TODO maybe need to fix and provide more detail here?
def get_result_validator_prompt(question) -> str:
    return f"""
Task: You are responsible for verifying that the result you are provided adequately answers the question and meets the validation criteria.

Question:
{question}

Input Structure:
Hypothesis: {HYPOTHESIS_DEF}
Validation Criteria: {VALIDATION_CRITERIA_DEF}
Parameters: {PARAMETERS_DEF}
Result: {RESULT_DEF}.

Instructions
- Fill out the provided data model with your validation results.
- result: {VALIDATION_RESULT_RESULT_DEF}
- question_response_result: {QUESTION_RESPONSE_RESULT_DEF}
- criteria_results: {CRITERIA_RESULTS_DEF}
"""
