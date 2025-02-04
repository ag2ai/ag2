from typing import Annotated, Callable, List, Optional
from pydantic import BaseModel, Field
from autogen.agentchat.contrib.swarm_agent import SwarmAgent


###############################################################################
# FUNCTION RESPONSE
###############################################################################

class FunctionResponse(BaseModel):
    """
    Represents the response from invoking a tool or function.
    
    Attributes:
        function (Callable): The function that was called.
        hypothesis (str): A hypothesis tested by the function, expressed as "if X then Y".
        params (dict): Parameters passed to the function during invocation.
        before_context_variables (dict): Context variables before function invocation.
        after_context_variables (dict): Context variables after function invocation.
        result (Optional[str]): The result produced by the function call.
        exception (Optional[str]): An exception message, if the function raised one.
        outer_agent_override (Optional[SwarmAgent]): Optional override for the next agent to handle the result.
    """
    function: Annotated[Callable, Field(description="The function that was called.")]
    hypothesis: Annotated[str, Field(description="Hypothesis tested, in the form 'if X then Y'.")]
    params: Annotated[dict, Field(description="Parameters passed to the function.")]
    before_context_variables: Annotated[dict, Field(
        description="Context variables before the function invocation."
    )]
    after_context_variables: Annotated[dict, Field(
        description="Context variables after the function invocation."
    )]
    result: Annotated[Optional[str], Field(description="The function's return value.")]
    exception: Annotated[Optional[str], Field(description="Exception raised during the function call, if any.")]
    outer_agent_override: Annotated[Optional[SwarmAgent], Field(
        description="Override for the next agent to handle the result."
    )]

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return self.to_string()

    def to_string(self, indent=0) -> str:
        """
        Converts the function response into a multi-line formatted string.

        Args:
            indent (int): Number of spaces to indent the string.
        
        Returns:
            str: A formatted string describing the function response.
        """
        indent_space = " " * indent
        return (
            f"{indent_space}Function: {self.function.__name__}\n"
            f"{indent_space}Hypothesis: {self.hypothesis}\n"
            f"{indent_space}Parameters: {self.params}\n"
            f"{indent_space}Before Context Variables: {self.before_context_variables}\n"
            f"{indent_space}After Context Variables: {self.after_context_variables}\n"
            f"{indent_space}Result: {self.result or 'N/A'}\n"
            f"{indent_space}Exception: {self.exception or 'None'}\n"
            f"{indent_space}Outer Agent Override: {self.outer_agent_override.name if self.outer_agent_override else 'None'}"
        )


###############################################################################
# FINDINGS, RESEARCH, NOTES, AND LEARNING
###############################################################################

class FindingModel(BaseModel):
    """
    Represents an individual piece of research information ("finding").
    
    Attributes:
        finding (str): The information discovered during research.
        relevance (Optional[str]): The relevance of the finding to the research topic.
        validation (Optional[str]): An explanation of how the finding is valid.
        source (Optional[str]): The source of the information.
    """
    finding: Annotated[str, Field(description="The discovered information.")]
    relevance: Annotated[Optional[str], Field(description="Relevance to the research topic.")]
    validation: Annotated[Optional[str], Field(description="Validation details for the finding.")]
    source: Annotated[Optional[str], Field(description="Source of the finding.")]

    def to_string(self, indent=0) -> str:
        """
        Converts the finding into a multi-line formatted string.

        Args:
            indent (int): Number of spaces to indent the string.
        
        Returns:
            str: A formatted string describing the finding.
        """
        indent_space = " " * indent
        return (
            f"{indent_space}Finding: {self.finding}\n"
            f"{indent_space}Relevance: {self.relevance or 'N/A'}\n"
            f"{indent_space}Validation: {self.validation or 'N/A'}\n"
            f"{indent_space}Source: {self.source or 'N/A'}"
        )


class ResearchModel(BaseModel):
    """
    Represents comprehensive research details.
    
    Attributes:
        findings (List[FindingModel]): List of findings during the research.
        result_summary (str): A summary of the research results.
        feedback (str): Feedback for improving the research question.
        follow_up_ideas (List[str]): Suggestions for follow-up research.
    """
    findings: Annotated[List[FindingModel], Field(description="List of findings from the research.")]
    result_summary: Annotated[str, Field(description="Summary of the research results.")]
    feedback: Annotated[str, Field(description="Feedback on improving the research question.")]
    follow_up_ideas: Annotated[List[str], Field(description="Ideas for follow-up research.")]

    def to_string(self, indent=0) -> str:
        """
        Converts research details into a multi-line formatted string.

        Args:
            indent (int): Number of spaces to indent the string.

        Returns:
            str: A formatted string describing the research.
        """
        indent_space = " " * indent
        findings_str = "\n".join(
            finding.to_string(indent=indent + 4) for finding in self.findings
        )
        return (
            f"{indent_space}Result Summary: {self.result_summary}\n"
            f"{indent_space}Feedback: {self.feedback}\n"
            f"{indent_space}Follow-Up Ideas: {', '.join(self.follow_up_ideas)}\n"
            f"{indent_space}Findings:\n{findings_str}"
        )


class NoteModel(BaseModel):
    """
    Represents a cautionary note or guidance for improving processes.

    Attributes:
        note (str): The note content.
        issue (Optional[str]): The issue it helps prevent or address.
    """
    note: Annotated[str, Field(description="Guidance or caution note.")]
    issue: Annotated[Optional[str], Field(description="The issue this note helps prevent.")]

    def to_string(self, indent=0) -> str:
        """
        Converts the note into a multi-line formatted string.

        Args:
            indent (int): Number of spaces to indent the string.

        Returns:
            str: A formatted string describing the note.
        """
        indent_space = " " * indent
        return (
            f"{indent_space}Note: {self.note}\n"
            f"{indent_space}Issue: {self.issue or 'N/A'}"
        )


class LearningModel(BaseModel):
    """
    Holds learning insights, including notes and feedback.

    Attributes:
        notes (List[NoteModel]): List of notes related to the research process.
        feedback (str): Feedback on improving the research process.
    """
    notes: Annotated[List[NoteModel], Field(description="Notes about the research process.")]
    feedback: Annotated[str, Field(description="Feedback for improving the research process.")]

    def to_string(self, indent=0) -> str:
        """
        Converts learning details into a multi-line formatted string.

        Args:
            indent (int): Number of spaces to indent the string.

        Returns:
            str: A formatted string describing the learning insights.
        """
        indent_space = " " * indent
        notes_str = "\n".join(note.to_string(indent=indent + 4) for note in self.notes)
        return (
            f"{indent_space}Feedback: {self.feedback}\n"
            f"{indent_space}Notes:\n{notes_str}"
        )


###############################################################################
# RESULT MODEL
###############################################################################

class ResultModel(BaseModel):
    """
    Represents the result of an agent's research, containing all relevant information.

    Attributes:
        name (str): Name of the agent producing the result.
        question (str): The input question that triggered this result.
        value (str): The final result or answer.
        research (ResearchModel): Research details, including findings and feedback.
        learning (LearningModel): Learning insights, including notes and feedback.
        function_responses (List[FunctionResponse]): Responses from function calls involved in generating the result.
    """
    name: Annotated[str, Field(description="Name of the agent.")]
    question: Annotated[str, Field(description="Input question leading to this result.")]
    value: Annotated[str, Field(description="The final result or answer.")]
    research: Annotated[ResearchModel, Field(description="Details of the research.")]
    learning: Annotated[LearningModel, Field(description="Insights and learning from the process.")]
    function_responses: Annotated[List[FunctionResponse], Field(description="Responses from involved function calls.")]

    def to_string(self, indent=0) -> str:
        """
        Converts the result into a multi-line formatted string.

        Args:
            indent (int): Number of spaces to indent the string.

        Returns:
            str: A formatted string describing the result.
        """
        indent_space = " " * indent
        functions_str = "\n".join(
            function_response.to_string(indent=indent + 4) for function_response in self.function_responses
        )
        return (
            # Result is likely multi-line
            f"{indent_space}Final Result: \n{self.value}\n"
            f"{indent_space}Research:\n{self.research.to_string(indent=indent + 4)}\n"
            f"{indent_space}Learning:\n{self.learning.to_string(indent=indent + 4)}\n"
            f"{indent_space}Function Responses:\n{functions_str}"
        )


###############################################################################
# KNOWLEDGE MODEL
###############################################################################

class KnowledgeModel(BaseModel):
    """
    A container for an agent's accumulated results.

    Attributes:
        agent_name (str): Name of the agent owning the knowledge.
        results (List[ResultModel]): List of all results produced by the agent.
    """
    agent_name: Annotated[str, Field(description="Name of the agent.")]
    results: Annotated[List[ResultModel], Field(description="List of results produced by the agent.")] = []
