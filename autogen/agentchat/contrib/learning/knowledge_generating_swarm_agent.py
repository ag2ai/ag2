# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from inspect import signature, Signature, Parameter

from autogen.agentchat.agent import Agent
from autogen.agentchat.chat import ChatResult
from autogen.agentchat.contrib.learning.learning_models import (
    FunctionResponse, LearningModel, NoteModel, ResearchModel, ResultModel
)
from autogen.agentchat.contrib.learning.knowledge_sharing_swarm_agent import KnowledgeSharingSwarmAgent
from autogen.agentchat.groupchat import GroupChat
from autogen.agentchat.contrib.swarm_agent import (
    AFTER_WORK, AfterWorkOption, SwarmAgent, SwarmResult, initiate_swarm_chat
)

class KnowledgeGeneratingSwarmAgent(KnowledgeSharingSwarmAgent):
    def __init__(
        self,
        name: str,
        researcher_llm_config: dict,
        result_analyzer_llm_config: dict,
        note_taker_llm_config: dict,
        functions: List[Callable],
        user_notes: List[str] = [],
        agent_invocation_notes: List[str] = [],
        description: str = "",
        max_rounds: int = 10,
        use_own_knowledge: bool = True,
        verbose: bool = True,
        knowledge_sources: Optional[List[SwarmAgent]] = None,
        **kwargs,
    ) -> None:
        
        super().__init__(
            name=name,
            llm_config=None,  # None for now, it shouldn't do LLM things
            system_message=f"",
            knowledge_sources=knowledge_sources,
            use_own_knowledge=use_own_knowledge,
            **kwargs
        )

        self._verbose = verbose
        self.user_notes = user_notes
        self.description = description
        self._agent_invocation_notes = agent_invocation_notes
        self.max_rounds = max_rounds
        self._after_work_override = None

        # TODO Can probably remove these and replace it with something that scans for the agent from the message history
        self._pending_research = None
        self._pending_learning = None
        self._pending_function_results = []

        self._result_analyzer = SwarmAgent(
            name="ResultAnalyzer-"+name,
            llm_config=_configure_structured_output(result_analyzer_llm_config, ResearchModel),
        )
        # TODO Same as above
        self._result_analyzer.register_hook(hookable_method="process_message_before_send", hook=self._result_analyzer_structured_output_hook)
        self._note_taker = SwarmAgent(
            name="NoteTaker-"+name,
            llm_config=_configure_structured_output(note_taker_llm_config, LearningModel),
        )
        self._note_taker.register_hook(hookable_method="process_message_before_send", hook=self._note_taker_structured_output_hook)

        self._functions:list[FunctionWithNotes] = [FunctionWithNotes(SubSwarmFunctionWrapper(function, self._function_response_callback, self._result_analyzer), self.user_notes) for function in functions]

        self._researcher = SwarmAgent(
            name="Researcher-"+name,
            llm_config=researcher_llm_config,
            functions=self._functions,
        )

        # Researcher passes to Result Analyzer
        self._researcher.register_hand_off(AFTER_WORK(self._result_analyzer))
        # Result Analyzer passes to Note Taker
        self._result_analyzer.register_hand_off(AFTER_WORK(self._note_taker))

        self.register_reply([Agent, None], KnowledgeGeneratingSwarmAgent._reply)

        # We need to keep a reference to a single one of these because we update its signature
        self.invocation_function = self._init_invocation_function()

    def _init_invocation_function(self):
        def invoke(context_variables:dict, question:str) -> SwarmResult:
            return SwarmResult(agent=self, context_variables=context_variables, values=question)

        invoke.__doc__ = f"""
Use this function to ask a question to the scientist {self.name}
{self.description}

:param context_variables: dict 
:param question: str A question for the scientist to answer.
:returns SwarmAgent:
"""
        return FunctionWithNotes(invoke, self._agent_invocation_notes)

    def get_invocation_function(self):
        return self.invocation_function

    def _result_analyzer_structured_output_hook(self, sender:Agent, message: Union[dict, str], recipient:SwarmAgent, silent:bool) -> Union[dict, str]:
        message_text = message if isinstance(message, str) else message.get("content", message)
        
        self._pending_research = ResearchModel.parse_raw(message_text)
        # Don't send it, it was just for storing
        return ""

    def _note_taker_structured_output_hook(self, sender:Agent, message: Union[dict, str], recipient:Agent, silent:bool) -> Union[dict, str]:
        message_text = message if isinstance(message, str) else message.get("content", message)
        
        self._pending_learning = LearningModel.parse_raw(message_text)
        # Don't send it, it was just for storing
        return ""
    
    # TODO Function results should be in a dictionary keyed by function for multi-function
    def _function_response_callback(self, function_response:FunctionResponse):
        self._pending_function_results.append(function_response)

        # Force this agent to route to the different agent when one is returned
        if function_response.outer_agent_override:
            self._after_work_override = function_response.outer_agent_override

    # This is what allows a function which is invoked by the sub-swarm to bubble up and influence where this agent will respond to
    # So someone can set whatever after work conditions like normal, but they can also have their functions return SwarmResult still
    # This is for other agents which are using this agent more like a function/invocation but not via the invocation style
    def _after_work_callable(self, last_speaker: SwarmAgent, messages: List[Dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, SwarmAgent, str]:
        override = self._after_work_override
        if override:
            self._after_work_override = None
            return override
        if isinstance(self.after_work, Callable):
            return self.after_work(last_speaker, messages, groupchat)
        return self.after_work if self.after_work else AfterWorkOption.TERMINATE

    def _reply(self, messages: Optional[List[Dict]] = None,
                        sender: Optional[Agent] = None,
                        config: Optional[Any] = None,
                    ) -> Tuple[bool, Union[str, Dict, None]]:
        
        # Just to be safe for now, I haven't dug through to see what messages it returns last
        task = _get_last_non_empty_message(messages)
        # We do 
        sender_context = sender._context_variables
        if sender_context is None:
            sender_context = {}

        return self._run_learning_swarm(sender_context, task)

    def _run_learning_swarm(self, context_variables:dict, question: str) -> Tuple[ChatResult, dict[str, Any], "SwarmAgent"]:
        # Ensure these are reset for each invocation, need to figure out a solution that works concurrently too
        self._pending_function_results = []
        self._pending_learning = None
        self._pending_research = None

        # Must use findings and also must be run before every invocation
        self._update_prompts(question)

        # Must use notes to update functions before every invocation
        #TODO This puts all notes on all functions, haven't thought of how to split the notes up yet sadly
        self._update_function_notes()

        chat_result, final_context_variables, last_active_agent = initiate_swarm_chat(
            initial_agent=self._researcher,
            agents=[self._researcher, self._result_analyzer, self._note_taker],
            messages=question,
            max_rounds=self.max_rounds,
            # I think this works as a reference everywhere basically 
            context_variables=context_variables
        )

        #TODO Maybe have a flag for this, maybe this is what silent is, haven't looked yet
        value = _get_last_non_empty_message(chat_result.chat_history)

        result = ResultModel(name=self.name, question=question, value=value, research=self._pending_research, learning=self._pending_learning, function_responses=self._pending_function_results) 

        # This will improve how other SwarmAgents invoke this agent.
        self.invocation_function.add_notes([result.research.feedback])
        return True, result.value
    
    def collect_notes(self) -> list[NoteModel]:
        notes = []
        for result in self.knowledge.results:
            notes += result.learning.notes
        return notes

    def _update_prompts(self, question):
        knowledge_text = self.collect_knowledge_from_sources_for_prompt()
        self._researcher.update_system_message(get_researcher_prompt(question, knowledge_text))
        self._result_analyzer.update_system_message(get_result_analyzer_prompt(question, knowledge_text))
        self._note_taker.update_system_message(get_note_taker_prompt(self.collect_notes()))

    def _update_function_notes(self):
        notes = self.collect_notes()
        for function in self._functions:
            function.set_notes(notes)

# This is needed to manage the collection of data to store as state for knowledge
# and to bubble up swarm things from the sub-swarm like if the function returns a SwarmResult for the outer swarm
# TODO How can these two wrapper things be pulled out and made re-usable to something thats not this
class SubSwarmFunctionWrapper:
    def __init__(self, tool_function, function_response_callback, result_analyzer):
        self.tool_function = tool_function
        self.function_response_callback = function_response_callback
        self.result_analyzer = result_analyzer

    def __call__(self, context_variables, hypothesis, *args, **kwargs):
        # Capture parameters for FunctionResponse
        params = {"args": args, "kwargs": kwargs}
        # Record the initial state of context_variables
        # TODO: Decide if this should be kept, or flagged off
        
        before_context_variables = copy.deepcopy(context_variables)
        result = None
        exception = None
        outer_agent_override = None
        next_agent = None
        try:
            # Call the tool function
            result = self.tool_function(context_variables=context_variables, *args, **kwargs)
            # When the function returns a SwarmResult, then expect that this function call terminates the research and the LearningAgent passes to the result.
            if isinstance(result, SwarmResult):
                outer_agent_override = result.agent
                next_agent = self.result_analyzer
            
            text_output = get_experiment_output_message_format(hypothesis, params, result)
            return SwarmResult(agent=next_agent, context_variables=context_variables, values=text_output)
        except Exception as e:
            # Capture the exception and debug it, not sure if I need to do this actually?
            exception = str(e)
            return str(e)
        finally:
            # Record the final state of context_variables
            after_context_variables = copy.deepcopy(context_variables)

            # Store the function results
            self.function_response_callback(FunctionResponse(
                function=self.tool_function,
                hypothesis=hypothesis,
                params=params,
                result=result,
                exception=exception,
                before_context_variables=before_context_variables,
                after_context_variables=after_context_variables,
                outer_agent_override=outer_agent_override
            ))

    @property
    def __signature__(self):
        """
        Generate a custom function signature for the wrapper.

        Includes context_variables and hypothesis as parameters.
        """
        tool_sig = signature(self.tool_function)
        params = list(tool_sig.parameters.values())

        # So I add on the context_variables parameter no matter what because I wasn't sure 
        # what happens if a function doesn't return a SwarmResult, if it would get all the way back up
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

        # Insert the new parameters before args/kwargs
        params = [context_variables_param, hypothesis_param] + [
            p for p in params if p.name not in ("context_variables", "hypothesis")
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
            config['response_format'] = structured_output_type
        return llm_config_copy

# TODO Check if there is an actual util for this
def _get_last_non_empty_message(messages):
    for message in reversed(messages):
        if message['content']:
            return message['content']
    return None

# By using the docstring as a place to keep notes that gets into the schema of a function, 
# this is like applying memory to manually modifying docstrings to tune function use. It makes it your docstring + new findings
# TODO Double check if descriptions on functions is considered strongly/actually pushed to the LLM in a useable way.
# TODO Also adding things at the bottom of the docstring, maybe they stop reading at return?
class FunctionWithNotes:
    def __init__(self, function:Callable, initial_notes:list[str]=[]):
        self.initial_notes = initial_notes
        self.notes = initial_notes
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
    
    def add_notes(self, new_notes:list[str]):
        self.notes += new_notes

    def set_notes(self, new_notes_set):
        self.notes = self.initial_notes + new_notes_set

    @property
    def __doc__(self):
        # Retrieve the original docstring
        original_doc = self.function.__doc__ or ""
        
        return get_invocation_function_description(original_doc, self.notes)

    @property
    def __name__(self):
        # Forward the name of the tool function
        return self.function.__name__ 
    
    @property
    def __signature__(self):
        return signature(self.function)
    

# Big TODO list
# TODO I'm still having problems with getting findings and notes to not just be the results but rephrased.
# TODO Figure out how to stop the sub-swarm 2 messages before it hits the limit and get it to learn about that
# TODO Implementation of getting structured output out of result_analyzer and note_taker can be improved. I want to still try and not use context_variables though
# TODO Function results need to be stored per-function and in a dictionary. This is necessary because notes should be per-function too. How do you even do per-function? Maybe this also has to be iterative or really good prompting.
# TODO Currently this will force the invocation function to always return to the caller, could pass it in when the function is generated and have multiple functions to update the signature of, which is honestly fine
# TODO Probably need a method naming scheme to differentiate things for agent class and for sub-swarm
# TODO Check if ON_CONDITION still works when the agent has no LLM?
# TODO Lots of concurrency issues to support async, seems doable though for sure
# TODO I need to look into what exactly is passed to the next LLM for function invocation
# TODO Have not considered code execution at all, but could be cool actually, learning + writing code sounds interesting?
# TODO What do I need to change to make this serilaizable so it can be in context_variables and gain those benefits (I think there are? but haven't really checked)
# Possible features
# TODO Consider removing all reliance on messages for the sub-swarm. Then I could craft the information passed in for findings/notes
# TODO Have finding generation run after each tool call or figure out how to structure a really good prompt
# TODO Introduce a separate step before the researcher to filter down the knowledge
# TODO So thinking about how to implement Tree of Thought but for swarms, I think you pull out the concept of sub-swarms from this and make it a more native thing, and thene this could use it. Just a note, needs lots of thinking through 
# TODO Make it so you can pass in structured output for the result value
# TODO Maybe it is better to just have the note taker evaluate functions and responses
# TODO Maybe it is better to have the findings thing run every single time



############## NEW GENERAL IDEAS ##############
# For all successful function calls, store them in a graph database or RAG database to be looked up after?
# Maybe they're associated with the question its given?
# All findings get associated with the original query its asking for
# As well as sub-things maybe?
# TODO when I was using claude, I had a really cool aha moment where a random finding from a step
# I asked asked it to find when X thing happened in the timeseries dataset, and it gave me a secondary interesting observation at the same time.
###############################################

# TODO I need a lot of help with prompting, I'm not familiar with how to do good generic prompting, and really only have experience with claude sonnet. 
# so be warned that these are ugly
def get_researcher_prompt(question, knowledge_text) -> str:
    return f"""
Role: Researcher
Task: You are responsible for following the instructions below to execute the scientific method to answer the question provided in the Question block below.

Do not ask anything of the user, continue researching until completion.
Do not ask for permission for anything, continue researching until completion. 
 
Instructions:
- Conduct background research: Analyze the knowledge provided to you. 
- Iteratively gather new information by executing the following process: 
    - Conduct an experiment by executing one of the provided functions.
        - On failure: Troubleshoot, adjust function use strategy and continue iterating
    - Analyze the results of the experiment and draw conclusions from it.

Question:
{question}

{knowledge_text}
"""

def get_result_analyzer_prompt(question, knowledge_text) -> str:
    return f"""
Role: Result Analyzer

Task: You are responsible for analyzing all of the information gathered during the research process.

Do not ask anything of the user, continue researching until completion.
Do not ask for permission for anything, continue researching until completion. 

Instructions:
- Provide a list of findings from the research.
- Findings should NOT be taken from the final response. 
- Findings should come from the results of successful function calls.
- Findings can be of intermediate peices of information as well as collected information.  

Question:
{question}

{knowledge_text}
"""

def get_note_taker_prompt(notes) -> str:
    introduction = "Below are notes previously gathered on how best to use the function."
    explanation = f"There are {len(notes)} total notes, labeled N1 through n{len(notes)}."
    lines = [introduction, explanation]
    for i, note in enumerate(notes):
        lines.append(f"N{i}: {note}")
    lines_text = "\n".join(lines)
    return f"""
Role: Note Taker

Task: You are responsible for writing notes on how to make use of tools more effective.

Do not ask anything of the user, continue researching until completion.
Do not ask for permission for anything, continue researching until completion. 

Instructions:
- Review the message history focusing on tool usage.
- Construct a set of notes based on tool usage intended to prevent errors and get better results from the tool.

{lines_text}
"""

def get_experiment_output_message_format(hypothesis, params, result):
    return f"""Experiment:
Hypothesis: {hypothesis}
Parameters: {params}
Result:

{result}            
"""

def get_invocation_function_description(original_doc, notes): 
    introduction = "Below are notes on how to most effectively use this function."
    explanation = f"There are {len(notes)} total notes, labeled N1 through N{len(notes)}."

    lines = [original_doc, "\n", introduction, explanation]
    for i, note in enumerate(notes):
        lines.append(f"N{i}: {note}")

    return "\n".join(lines)