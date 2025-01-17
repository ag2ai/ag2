# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field


def format_args_kwargs(*args, **kwargs):
    """
    Converts all args and kwargs into a multi-line string, one line per argument.
    """
    result = []

    # Handle positional arguments
    for i, arg in enumerate(args):
        result.append(f"arg[{i}]: {arg}")

    # Handle keyword arguments
    for key, value in kwargs.items():
        result.append(f"{key}: {value}")

    # Join all lines into a single string
    return "\n".join(result)


HYPOTHESIS_DEF = "Hypothesis for how the result will answer the question, in the form 'if X then Y'."
VALIDATION_CRITERIA_DEF = "List of criteria to evaluate in order to validate the correctness of the response."
FUNCTION_RESPONSE_DEF = "The entire response from the function call."
PARAMETERS_DEF = "The parameters used in the function invocation."
RESULT_DEF = " The result of the function invocation."


class FunctionResponse(BaseModel):
    function: Annotated[Callable, Field(description="The function that was called.")]
    hypothesis: Annotated[str, Field(description=HYPOTHESIS_DEF)]
    args: Annotated[list[Any], Field(description="The args for the function invocation")]
    kwargs: Annotated[dict, Field(description="The kwargs for the function invocation")]
    result_data: Annotated[Any, Field(description=RESULT_DEF + " As Any type.")]
    result_str: Annotated[str, Field(description=RESULT_DEF + " As a string.")]
    validation_criteria: Annotated[list[str], Field(description=VALIDATION_CRITERIA_DEF)]

    class Config:
        arbitrary_types_allowed = True

    def to_validation_submission(self) -> str:
        validation_criteria_text = "\n".join(self.validation_criteria)
        return f"""Please validate the following result:
Hypothesis: {self.hypothesis}
Validation Criteria:
{validation_criteria_text}
Result:
{self.result_str}

Parameters:

{format_args_kwargs(*self.args, **self.kwargs)}

"""

    def to_function_memory(self, question):
        return f"""
Question:
{question}
Hypothesis:
{self.hypothesis}
Parameters:
{format_args_kwargs(*self.args, **self.kwargs)}
Result:
{self.result_str}
"""

    def to_background_memory(self, question):
        return f"""
Question:
{question}
Result:
{self.result_str}
"""


CRITERION_DEF = "A criterion for which to evaluate the response on."
CRITERION_RESULT_DEF = "The result of evaluating the criterion."
CRITERION_JUSTIFICATION_DEF = "The justification for evaluation result for the criterion."
QUESTION_RESPONSE_QUESTION_DEF = "The question that the response is trying to answer."
QUESTION_RESPONSE_RESULT_DEF = "The result of the evaluation of the response to the question."
QUESTION_RESPONSE_JUSTIFICATION_DEF = (
    "The justification for the result of the evaluation of the response to the question."
)
VALIDATION_RESULT_RESULT_DEF = (
    "Whether or not the response sufficiently answered the question and passed all validation criteria."
)
QUESTION_RESPONSE_DEF = "Results of evaluating the response compared to the submitted question."
CRITERIA_RESULTS_DEF = "A list of CriterionResult evaluations for each of the submitted criteria."
VALIDATION_RESULT_DEF = "Contains the result from validating the response a function call."


class CriterionResult(BaseModel):
    criterion: Annotated[str, Field(description=CRITERION_DEF)]
    result: Annotated[bool, Field(description=CRITERION_RESULT_DEF)]
    justification: Annotated[str, Field(description=CRITERION_JUSTIFICATION_DEF)]

    def __str__(self) -> str:
        return (
            f"Criterion: {self.criterion}\n"
            f"Result: {'Passed' if self.result else 'Failed'}\n"
            f"Justification: {self.justification}\n"
        )


class QuestionResponseResult(BaseModel):
    question: Annotated[str, Field(description=QUESTION_RESPONSE_QUESTION_DEF)]
    result: Annotated[bool, Field(description=QUESTION_RESPONSE_RESULT_DEF)]
    justification: Annotated[str, Field(description=QUESTION_RESPONSE_JUSTIFICATION_DEF)]

    def __str__(self) -> str:
        return (
            f"Question: {self.question}\n"
            f"Result: {'Passed' if self.result else 'Failed'}\n"
            f"Justification: {self.justification}\n"
        )


class ValidationResult(BaseModel):
    result: Annotated[bool, Field(description=VALIDATION_RESULT_RESULT_DEF)]
    question_response_result: Annotated[QuestionResponseResult, Field(description=QUESTION_RESPONSE_DEF)]
    criteria_results: Annotated[list[CriterionResult], Field(description=CRITERIA_RESULTS_DEF)]

    def __str__(self) -> str:
        return (
            f"Validation Result: {'Passed' if self.result else 'Failed'}\n\n"
            f"Question Response Evaluation:\n"
            f"{str(self.question_response_result)}\n"
            f"Criteria Evaluations:\n" + "\n".join(f"- {str(criterion)}" for criterion in self.criteria_results)
        )


QUESTION_DEF = "The question to answer by invoking the function."


class AutoGenResult(BaseModel):
    result: Annotated[Any, Field(description="The actual result of the function invocation.")]
    question: Annotated[str, Field(description=QUESTION_DEF)]
    function_response: Annotated[FunctionResponse, Field(description=FUNCTION_RESPONSE_DEF)]
    validation_result: Annotated[ValidationResult, Field(description=VALIDATION_RESULT_DEF)]

    def to_function_memory(self):
        return self.function_response.to_function_memory(self.question)

    def to_background_memory(self):
        return self.function_response.to_background_memory(self.question)
