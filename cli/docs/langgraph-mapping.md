# LangGraph/LangChain to AG2 Conversion Mapping

> **Design reference only** — `ag2 convert` is not yet implemented.
> This document maps the full API surface of LangGraph (v1.1.x) and
> LangChain (v1.x) as of March 2026 for planning purposes.

---

## 1. Package & Import Mapping

### LangGraph Packages

| LangGraph Package | Purpose | AG2 Equivalent |
|---|---|---|
| `langgraph` (v1.1.3) | Core: StateGraph, Pregel engine | `autogen` (ag2) |
| `langgraph-prebuilt` | create_react_agent, ToolNode | `autogen` ConversableAgent + tools |
| `langgraph-checkpoint` | BaseCheckpointSaver, BaseStore | No direct equivalent (AG2 has no built-in checkpointing) |
| `langgraph-checkpoint-postgres` | PostgreSQL persistence | No equivalent |
| `langgraph-checkpoint-sqlite` | SQLite persistence | No equivalent |
| `langgraph-supervisor` | create_supervisor, handoff tools | `autogen` GroupChat or Swarm |
| `langgraph-sdk` | HTTP client for remote graphs | No equivalent (use ag2 serve REST API) |
| `langgraph-cli` | Dev server, build, deploy | `ag2` CLI |

### Core Import Translations

```python
# LangGraph imports                          # AG2 equivalents
from langgraph.graph import StateGraph    # No direct equivalent (AG2 is not graph-based)
from langgraph.graph import START, END    # No equivalent (implicit in AG2 conversation flow)
from langgraph.graph import MessagesState # No equivalent (AG2 manages messages internally)
from langgraph.graph.message import add_messages  # Built into ConversableAgent
from langgraph.types import Command       # Closest: OnCondition / AfterWork handoffs
from langgraph.types import Send          # No equivalent
from langgraph.types import interrupt     # human_input_mode="ALWAYS" or "TERMINATE"
from langgraph.types import CachePolicy   # No equivalent
from langgraph.prebuilt import create_react_agent  # ConversableAgent with tools
from langgraph.prebuilt import ToolNode   # Built into ConversableAgent tool execution
from langgraph.checkpoint.memory import MemorySaver  # No equivalent

# LangChain imports
from langchain_core.messages import HumanMessage    # dict: {"role": "user", "content": "..."}
from langchain_core.messages import AIMessage       # dict: {"role": "assistant", "content": "..."}
from langchain_core.messages import SystemMessage   # ConversableAgent(system_message="...")
from langchain_core.messages import ToolMessage     # Internal to AG2 tool execution
from langchain_core.tools import tool               # from autogen import register_function / Tool
from langchain_core.tools import BaseTool           # autogen.tools.Tool
from langchain_core.tools import StructuredTool     # autogen.tools.Tool
from langchain_openai import ChatOpenAI             # LLMConfig(api_type="openai", model="...")
from langchain_anthropic import ChatAnthropic       # LLMConfig(api_type="anthropic", model="...")

# LangChain 1.0 agents
from langchain.agents import create_agent           # ConversableAgent with tools
from langchain.agents import AgentState             # Not needed (AG2 manages state internally)
from langchain.agents.middleware import ...          # register_hook / register_reply patterns
```

---

## 2. Core Concepts Mapping

### 2.1 Graph Definition vs Agent Definition

| Concept | LangGraph | AG2 | Conversion Notes |
|---|---|---|---|
| **Workflow unit** | `StateGraph(state_schema)` | `ConversableAgent(...)` or `GroupChat(...)` | Fundamental paradigm shift: graph nodes become agents |
| **State schema** | `TypedDict` with `Annotated` reducers | No explicit state; messages are the state | AG2 uses conversation history as implicit state. For custom state, use `context_variables` |
| **Node** | `graph.add_node("name", func)` | `ConversableAgent(name="name", ...)` | Each node becomes an agent. Node function logic goes into system_message + tools |
| **Edge (fixed)** | `graph.add_edge("a", "b")` | Implicit via `initiate_chat()` or GroupChat ordering | AG2 doesn't have explicit fixed edges; flow is determined by conversation |
| **Conditional edge** | `graph.add_conditional_edges(src, func, map)` | `OnCondition(target=agent, condition="...")` in Swarm, or `speaker_selection_method` in GroupChat | Closest mapping is Swarm's `OnCondition` for LLM-driven routing |
| **Entry point** | `graph.add_edge(START, "first")` | `initial_agent` in swarm, or first agent in `initiate_chat()` | Implicit in AG2 |
| **End point** | `graph.add_edge("last", END)` | `is_termination_msg` or `AfterWork(AfterWorkOption.TERMINATE)` | AG2 uses termination conditions rather than explicit end nodes |
| **Compile** | `graph.compile(checkpointer=...)` | Not needed (AG2 agents are ready to use) | No compilation step in AG2 |
| **Invoke** | `app.invoke({"messages": [...]})` | `agent.initiate_chat(recipient, message="...")` | Different invocation model |
| **Stream** | `app.stream(input, stream_mode="values")` | No built-in streaming equivalent | MANUAL: AG2 has no graph-level streaming |

### 2.2 State Management

| LangGraph | AG2 | Conversion Notes |
|---|---|---|
| `TypedDict` state schema | No explicit schema | AG2 has no typed state. Convert to `context_variables` dict |
| `Annotated[list, add_messages]` reducer | Built into agent message handling | Messages are automatically accumulated in AG2 |
| `Annotated[list, operator.add]` custom reducer | No equivalent | MANUAL: Must implement accumulation logic in agent hooks |
| `MessagesState` (prebuilt) | Built into every ConversableAgent | AG2 agents inherently manage message lists |
| State read in node: `state["key"]` | `context_variables["key"]` in Swarm | Only available in Swarm pattern |
| State update from node: `return {"key": val}` | Modify `context_variables` in tool functions | Tools can accept `ContextVariables` via dependency injection |
| `context_schema` (immutable runtime data) | No equivalent | MANUAL: Pass as closure variables or config |

### 2.3 Node Functions to Agent Logic

LangGraph node functions receive state and return state updates. In AG2, this logic is distributed across agent configuration and tool functions.

**LangGraph pattern:**
```python
def researcher_node(state: AgentState) -> dict:
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}
```

**AG2 equivalent:**
```python
researcher = ConversableAgent(
    name="researcher",
    system_message="You are a researcher...",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
```

### 2.4 Conditional Routing

**LangGraph pattern:**
```python
def route(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
```

**AG2 Swarm equivalent:**
```python
agent = SwarmAgent(name="agent", ...)
agent.register_hand_off([
    OnCondition(target=tool_agent, condition="When tools need to be called"),
    AfterWork(AfterWorkOption.TERMINATE)
])
```

**AG2 GroupChat equivalent:**
```python
groupchat = GroupChat(
    agents=[agent, tool_agent],
    speaker_selection_method="auto",  # LLM-driven routing
    allowed_or_disallowed_speaker_transitions={
        agent: [tool_agent],
        tool_agent: [agent],
    },
    speaker_transitions_type="allowed",
)
```

---

## 3. Prebuilt Components Mapping

### 3.1 create_react_agent

| Parameter | LangGraph | AG2 Equivalent |
|---|---|---|
| `model` | `BaseChatModel` or string like `"openai:gpt-4o"` | `llm_config=LLMConfig(model="gpt-4o", api_type="openai")` |
| `tools` | `list[BaseTool \| Callable \| dict]` | `register_function(func, caller=agent, executor=agent)` or `functions=[...]` on SwarmAgent |
| `prompt` | `str \| SystemMessage \| Callable` | `system_message="..."` |
| `name` | `str` | `name="..."` |
| `response_format` | `BaseModel` (Pydantic) for structured output | No direct equivalent; use tool-based structured output |
| `state_schema` | Custom `TypedDict` | Not applicable |
| `checkpointer` | `BaseCheckpointSaver` | No equivalent |
| `interrupt_before` | `list[str]` (node names) | `human_input_mode="ALWAYS"` (coarser control) |
| `interrupt_after` | `list[str]` (node names) | `human_input_mode="TERMINATE"` (coarser control) |
| `pre_model_hook` | `Callable` | `register_hook("process_all_messages_before_reply", func)` |
| `post_model_hook` | `Callable` | `register_hook("process_last_received_message", func)` |
| `version` | `"v1"` or `"v2"` | Not applicable |

**Full translation example:**

```python
# LangGraph
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="openai:gpt-4o",
    tools=[search, calculator],
    prompt="You are a helpful assistant.",
    name="assistant",
)
result = agent.invoke({"messages": [("user", "What is 2+2?")]})
```

```python
# AG2
from autogen import ConversableAgent, LLMConfig, register_function

llm_config = LLMConfig(api_type="openai", model="gpt-4o")

assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)
executor = ConversableAgent(
    name="executor",
    human_input_mode="NEVER",
    llm_config=False,
)
register_function(search, caller=assistant, executor=executor, description="Search the web")
register_function(calculator, caller=assistant, executor=executor, description="Calculate math")

executor.initiate_chat(assistant, message="What is 2+2?", max_turns=10)
```

### 3.2 ToolNode

| Feature | LangGraph ToolNode | AG2 |
|---|---|---|
| Tool execution | `ToolNode(tools=[...])` | Built into ConversableAgent via `register_for_execution` |
| Error handling | `handle_tool_errors=True` returns error as ToolMessage | AG2 returns error messages automatically |
| Tool validation | `ValidationNode` | No equivalent; use `register_input_guardrail` |

### 3.3 create_supervisor (langgraph-supervisor)

| Parameter | LangGraph | AG2 Equivalent |
|---|---|---|
| `agents` | `list[CompiledGraph \| Runnable]` | `agents=[agent1, agent2, ...]` in GroupChat or Swarm |
| `model` | `BaseChatModel` | `llm_config` on GroupChatManager |
| `prompt` | `str` (supervisor instructions) | `system_message` on manager or `select_speaker_message_template` |
| `supervisor_name` | `str` | `admin_name` in GroupChat |
| `output_mode` | `"last_message"` or `"full_history"` | No direct equivalent; GroupChat always has full history |
| `handoff_tool_prefix` | `str` (e.g., `"transfer_to"`) | `OnCondition(target=agent, condition="...")` in Swarm |
| `tools` | Extra supervisor tools | Register tools on manager agent |

**Full translation:**

```python
# LangGraph Supervisor
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

math_agent = create_react_agent(model=model, tools=[add], name="math")
search_agent = create_react_agent(model=model, tools=[search], name="search")

workflow = create_supervisor(
    [math_agent, search_agent],
    model=model,
    prompt="Route to the appropriate expert.",
)
app = workflow.compile()
result = app.invoke({"messages": [{"role": "user", "content": "What is 2+2?"}]})
```

```python
# AG2 GroupChat (supervisor pattern)
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig, register_function

llm_config = LLMConfig(api_type="openai", model="gpt-4o")

math_agent = ConversableAgent(
    name="math", system_message="You are a math expert.", llm_config=llm_config,
    human_input_mode="NEVER",
)
register_function(add, caller=math_agent, executor=math_agent, description="Add two numbers")

search_agent = ConversableAgent(
    name="search", system_message="You are a search expert.", llm_config=llm_config,
    human_input_mode="NEVER",
)
register_function(search, caller=search_agent, executor=search_agent, description="Search the web")

groupchat = GroupChat(
    agents=[math_agent, search_agent],
    messages=[],
    speaker_selection_method="auto",  # LLM picks next speaker (supervisor-like)
    max_round=20,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
math_agent.initiate_chat(manager, message="What is 2+2?")
```

---

## 4. Tools Mapping

### 4.1 Tool Definition

| LangChain/LangGraph | AG2 | Notes |
|---|---|---|
| `@tool` decorator | `@tool` decorator (different import) or plain function + `register_function` | LangChain `@tool` uses docstring as description; AG2 uses `description` param |
| `BaseTool` subclass | `autogen.tools.Tool(name, description, func)` | Subclass pattern has no direct equivalent |
| `StructuredTool.from_function(func, name, description)` | `Tool(name=name, description=description, func=func)` | Clean mapping |
| `args_schema` (Pydantic model) | Type hints on function parameters | AG2 infers schema from type hints |
| Tool with `Annotated` params | `Annotated[str, "description"]` on function params | Direct mapping |

**LangChain tool:**
```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Max results")

@tool(args_schema=SearchInput)
def search(query: str, limit: int = 10) -> str:
    """Search the web for information."""
    return f"Results for {query}"
```

**AG2 tool:**
```python
from typing import Annotated
from autogen import ConversableAgent, register_function

def search(
    query: Annotated[str, "Search query"],
    limit: Annotated[int, "Max results"] = 10,
) -> str:
    """Search the web for information."""
    return f"Results for {query}"

register_function(
    search,
    caller=assistant,
    executor=executor,
    description="Search the web for information.",
)
```

### 4.2 Tool Registration Patterns

| Pattern | LangGraph | AG2 |
|---|---|---|
| Bind to model | `model.bind_tools([tool1, tool2])` | `register_function(func, caller=agent, ...)` |
| Bind at agent creation | `create_react_agent(tools=[...])` | `ConversableAgent(functions=[func1, func2])` on SwarmAgent |
| Separate caller/executor | Not applicable (ToolNode handles both) | `register_function(func, caller=llm_agent, executor=exec_agent)` |
| Dynamic tool selection | `tool_chooser` on ToolNode | `func_call_filter=True` in GroupChat |

---

## 5. Multi-Agent Patterns Mapping

### 5.1 Pattern: Sequential Chain

**LangGraph:**
```python
graph = StateGraph(State)
graph.add_node("step1", step1_func)
graph.add_node("step2", step2_func)
graph.add_node("step3", step3_func)
graph.add_edge(START, "step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)
```

**AG2:**
```python
# Option 1: Sequential chat
agent1.initiate_chats([
    {"recipient": agent2, "message": "...", "max_turns": 2},
    {"recipient": agent3, "message": "...", "max_turns": 2},
])

# Option 2: GroupChat with round_robin
groupchat = GroupChat(
    agents=[agent1, agent2, agent3],
    speaker_selection_method="round_robin",
)
```

### 5.2 Pattern: Router / Conditional Branching

**LangGraph:**
```python
def router(state):
    category = state["category"]
    if category == "billing":
        return "billing_agent"
    elif category == "technical":
        return "tech_agent"
    return "general_agent"

graph.add_conditional_edges("classifier", router)
```

**AG2 Swarm:**
```python
classifier = SwarmAgent(name="classifier", ...)
classifier.register_hand_off([
    OnCondition(target=billing_agent, condition="Route to billing for payment/invoice questions"),
    OnCondition(target=tech_agent, condition="Route to tech support for technical issues"),
    AfterWork(agent=general_agent),  # Default fallback
])
```

### 5.3 Pattern: Supervisor / Hierarchical

**LangGraph:**
```python
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    [researcher, writer, reviewer],
    model=model,
    prompt="Coordinate the research and writing team.",
)
```

**AG2:**
```python
groupchat = GroupChat(
    agents=[researcher, writer, reviewer],
    speaker_selection_method="auto",
    select_speaker_message_template="Coordinate the team: {roles}. Select: {agentlist}",
    max_round=30,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
```

### 5.4 Pattern: Collaborative (Shared Scratchpad)

**LangGraph:**
```python
class SharedState(TypedDict):
    messages: Annotated[list, add_messages]
    scratchpad: Annotated[list[str], operator.add]

# Both agents read/write the shared scratchpad
```

**AG2:**
```python
# All agents in GroupChat share the full conversation history (messages are the scratchpad)
groupchat = GroupChat(agents=[agent_a, agent_b], messages=[])
```

> Note: AG2 doesn't have a typed shared scratchpad. The conversation itself serves
> as the shared context. For structured shared state, use `context_variables` in Swarm.

### 5.5 Pattern: Hierarchical Teams (Nested Subgraphs)

**LangGraph:**
```python
research_team = create_supervisor([searcher, analyst], model=model).compile(name="research")
writing_team = create_supervisor([writer, editor], model=model).compile(name="writing")
top_supervisor = create_supervisor([research_team, writing_team], model=model).compile()
```

**AG2:**
```python
# AG2 supports nested GroupChats via nested chats on ConversableAgent
research_manager = GroupChatManager(
    groupchat=GroupChat(agents=[searcher, analyst], ...),
    llm_config=llm_config,
    name="research_team",
)
writing_manager = GroupChatManager(
    groupchat=GroupChat(agents=[writer, editor], ...),
    llm_config=llm_config,
    name="writing_team",
)
top_groupchat = GroupChat(
    agents=[research_manager, writing_manager],
    speaker_selection_method="auto",
)
top_manager = GroupChatManager(groupchat=top_groupchat, llm_config=llm_config)
```

### 5.6 Pattern: Fan-out / Parallel Execution

**LangGraph:**
```python
# Using Send for dynamic fan-out
def fan_out(state):
    return [Send("process", {"item": item}) for item in state["items"]]

graph.add_conditional_edges("splitter", fan_out)
```

**AG2:** No direct equivalent. AG2 does not support parallel agent execution natively.
Must be implemented manually with asyncio or sequential processing.

### 5.7 Pattern: Evaluator-Optimizer Loop

**LangGraph:**
```python
graph.add_node("generator", generate)
graph.add_node("evaluator", evaluate)
graph.add_conditional_edges("evaluator", lambda s: "generator" if s["score"] < 0.8 else END)
graph.add_edge("generator", "evaluator")
```

**AG2:**
```python
# Two-agent conversation naturally creates a feedback loop
generator = ConversableAgent(name="generator", system_message="Generate content...", ...)
evaluator = ConversableAgent(
    name="evaluator",
    system_message="Evaluate and request improvements until quality > 0.8...",
    is_termination_msg=lambda msg: "APPROVED" in msg.get("content", ""),
    ...
)
generator.initiate_chat(evaluator, message="Write an essay about...", max_turns=10)
```

---

## 6. Execution & Runtime Mapping

### 6.1 Invocation

| Feature | LangGraph | AG2 |
|---|---|---|
| Sync invoke | `app.invoke(input)` | `agent.initiate_chat(recipient, message=msg)` |
| Async invoke | `app.ainvoke(input)` | `await agent.a_initiate_chat(recipient, message=msg)` |
| Sync stream | `app.stream(input, stream_mode=...)` | No equivalent |
| Async stream | `app.astream(input, stream_mode=...)` | No equivalent |
| Max iterations | `config={"recursion_limit": N}` | `max_turns=N` or `max_round=N` (GroupChat) |
| Thread ID | `config={"configurable": {"thread_id": "..."}}` | No equivalent |

### 6.2 Streaming Modes

| LangGraph Mode | Description | AG2 Equivalent |
|---|---|---|
| `"values"` | Full state after each node | No equivalent |
| `"updates"` | State delta per node | No equivalent |
| `"messages"` | Token-level LLM output | No equivalent (AG2 prints to IOStream) |
| `"custom"` | User-defined events | No equivalent |
| `"debug"` | Detailed trace | `verbose` mode on agents |

### 6.3 Persistence & Checkpointing

| Feature | LangGraph | AG2 |
|---|---|---|
| In-memory checkpointer | `MemorySaver()` | No equivalent |
| PostgreSQL checkpointer | `PostgresSaver(...)` | No equivalent |
| SQLite checkpointer | `SqliteSaver(...)` | No equivalent |
| State snapshots | `app.get_state(config)` | `agent.chat_messages` (current messages only) |
| State history | `app.get_state_history(config)` | No equivalent |
| Update state | `app.update_state(config, values)` | No equivalent |
| Resume from checkpoint | Automatic with thread_id | No equivalent |

> MANUAL intervention required: LangGraph checkpointing has no AG2 equivalent.
> Converted code that relies on persistence, resume, or state history will need
> significant rearchitecting or a custom persistence layer.

### 6.4 Human-in-the-Loop

| Feature | LangGraph | AG2 |
|---|---|---|
| Interrupt before node | `compile(interrupt_before=["node"])` | `human_input_mode="ALWAYS"` on target agent |
| Interrupt after node | `compile(interrupt_after=["node"])` | `human_input_mode="TERMINATE"` |
| Dynamic interrupt | `interrupt(value)` in node function | `human_input_mode="ALWAYS"` |
| Resume with input | `Command(resume=value)` | User types response when prompted |
| Approve/reject | Check interrupt value, route accordingly | `human_input_mode="TERMINATE"` with termination check |
| Breakpoints | Specific node-level breakpoints | Coarser: per-agent `human_input_mode` setting |

> AG2's human-in-the-loop is coarser-grained. LangGraph allows node-level breakpoints;
> AG2 applies human input mode at the agent level. Fine-grained interrupt/resume
> patterns require MANUAL conversion.

---

## 7. Message Types Mapping

| LangChain Message | AG2 Message Format |
|---|---|
| `HumanMessage(content="...")` | `{"role": "user", "content": "..."}` |
| `AIMessage(content="...", tool_calls=[...])` | `{"role": "assistant", "content": "...", "tool_calls": [...]}` |
| `SystemMessage(content="...")` | `ConversableAgent(system_message="...")` (not a message, it's config) |
| `ToolMessage(content="...", tool_call_id="...")` | `{"role": "tool", "content": "...", "tool_call_id": "..."}` |
| `FunctionMessage(content="...", name="...")` | `{"role": "function", "name": "...", "content": "..."}` |
| `RemoveMessage(id="...")` | No equivalent |

---

## 8. LangChain 1.0 Agent API Mapping

### 8.1 create_agent (LangChain 1.0)

| Parameter | LangChain | AG2 |
|---|---|---|
| `model` | `str` or `BaseChatModel` | `llm_config=LLMConfig(...)` |
| `tools` | `list[BaseTool \| Callable]` | `register_function(...)` |
| `system_prompt` | `str` | `system_message="..."` |
| `response_format` | Pydantic model for structured output | No direct equivalent |
| `state_schema` | Custom TypedDict | Not applicable |
| `middleware` | `list[AgentMiddleware]` | `register_hook(...)` and `register_reply(...)` |
| `name` | `str` | `name="..."` |
| `context_schema` | Runtime context type | Not applicable |

### 8.2 Middleware to AG2 Hooks

| LangChain Middleware | AG2 Hook/Pattern |
|---|---|
| `@before_model` | `register_hook("process_all_messages_before_reply", func)` |
| `@after_model` | `register_hook("process_last_received_message", func)` |
| `@wrap_model_call` | `register_reply(trigger, custom_reply_func)` |
| `@wrap_tool_call` | Override via `register_function` with wrapper |
| `@dynamic_prompt` | `update_agent_state_before_reply=[UpdateSystemMessage(func)]` |
| `SummarizationMiddleware` | No built-in equivalent |
| `HumanInTheLoopMiddleware` | `human_input_mode="ALWAYS"` |

### 8.3 Legacy AgentExecutor (deprecated)

| Feature | LangChain AgentExecutor | AG2 |
|---|---|---|
| Agent + tools execution | `AgentExecutor(agent=agent, tools=tools)` | `ConversableAgent` with registered tools |
| Max iterations | `max_iterations=N` | `max_consecutive_auto_reply=N` |
| Early stopping | `early_stopping_method="generate"` | `is_termination_msg=lambda msg: ...` |
| Return intermediate | `return_intermediate_steps=True` | Messages are always available in `chat_messages` |
| Error handling | `handle_parsing_errors=True` | Built-in error handling in tool execution |

---

## 9. LangGraph Functional API Mapping

| LangGraph Functional | AG2 | Notes |
|---|---|---|
| `@entrypoint(checkpointer=...)` | Top-level function calling `initiate_chat` | No decorator equivalent |
| `@task` | Individual agent or tool call | Tasks map to tool functions or sub-conversations |
| `task().result()` | Synchronous result of `initiate_chat` | Direct mapping |
| `interrupt(value)` inside entrypoint | `human_input_mode="ALWAYS"` | Coarser control |
| `Command(resume=val)` | User input in response to prompt | Manual handling |

---

## 10. Advanced Features Gap Analysis

### Features with NO AG2 Equivalent (Require Manual Intervention)

| LangGraph Feature | Why No Equivalent | Suggested Workaround |
|---|---|---|
| **Typed state with reducers** | AG2 is conversation-based, not state-machine-based | Use `context_variables` dict for shared state in Swarm |
| **Checkpointing / persistence** | AG2 has no built-in persistence | Implement custom persistence with DB/file storage |
| **State snapshots & history** | No checkpoint system | Log messages manually |
| **Graph visualization** | AG2 isn't graph-based | Use AG2's logging and monitoring tools |
| **Stream modes (values, updates, messages, custom, debug)** | AG2 doesn't have structured streaming | Use IOStream and verbose mode |
| **Fan-out / parallel Send** | AG2 agents run sequentially | Use asyncio for parallel execution |
| **Node-level caching (CachePolicy, InMemoryCache)** | No equivalent | Implement caching in tool functions |
| **Super-step execution model** | Different execution model | N/A — conceptual mismatch |
| **Compiled graph validation** | AG2 validates at runtime | Use `ag2 doctor` for static analysis |
| **RemoveMessage** | AG2 doesn't support message removal | Use `clear_history()` for full reset |
| **RemainingSteps managed value** | No step tracking | Use `max_consecutive_auto_reply` |
| **context_schema (immutable runtime context)** | No equivalent | Pass data via closure or global config |
| **Graph topology migration** | No graph topology concept | N/A |
| **add_sequence() helper** | No equivalent | Use sequential `initiate_chats()` |
| **Structured response_format** | AG2 doesn't enforce output schema | Use tool-based structured output or post-processing |

### Features that Map Cleanly (Automatable)

| LangGraph Feature | AG2 Equivalent | Confidence |
|---|---|---|
| Tool definition (`@tool`) | `register_function` / `Tool` | HIGH |
| System prompt | `system_message` | HIGH |
| Model configuration | `LLMConfig` | HIGH |
| Agent naming | `name` parameter | HIGH |
| Max iterations | `max_turns` / `max_round` | HIGH |
| Termination condition | `is_termination_msg` | HIGH |
| Two-agent ReAct loop | `ConversableAgent` pair with tools | HIGH |
| Supervisor pattern | `GroupChat` with `speaker_selection_method="auto"` | MEDIUM |
| Sequential chain | `initiate_chats` or `round_robin` GroupChat | MEDIUM |
| Human-in-the-loop (basic) | `human_input_mode` | MEDIUM |

### Features that Require Significant Rework

| LangGraph Feature | Difficulty | Notes |
|---|---|---|
| Complex conditional routing | MEDIUM | Convert to Swarm `OnCondition` or GroupChat transitions |
| Subgraph composition | MEDIUM | Convert to nested GroupChat |
| State-dependent logic | HIGH | Rewrite as tool functions or agent system messages |
| Checkpoint-dependent workflows | HIGH | Requires custom persistence layer |
| Streaming pipelines | HIGH | Fundamental architecture difference |
| Cross-graph Command routing | HIGH | Convert to Swarm handoffs |

---

## 11. AST Detection Patterns

For the `ag2 convert` command, these are the AST patterns to detect in source files:

### Import Detection

```python
# Detect these import patterns to identify LangGraph/LangChain code
LANGGRAPH_IMPORTS = [
    "langgraph.graph",           # StateGraph, START, END, MessagesState
    "langgraph.prebuilt",        # create_react_agent, ToolNode
    "langgraph.types",           # Command, Send, interrupt, CachePolicy
    "langgraph.checkpoint",      # MemorySaver, BaseCheckpointSaver
    "langgraph.cache",           # InMemoryCache
    "langgraph_supervisor",      # create_supervisor
    "langgraph.func",            # entrypoint, task
]

LANGCHAIN_IMPORTS = [
    "langchain_core.tools",      # tool, BaseTool, StructuredTool
    "langchain_core.messages",   # HumanMessage, AIMessage, etc.
    "langchain.agents",          # create_agent, AgentExecutor, create_react_agent
    "langchain.agents.middleware", # Middleware classes
    "langchain_openai",          # ChatOpenAI
    "langchain_anthropic",       # ChatAnthropic
    "langchain_core.runnables",  # RunnableConfig
]
```

### Class/Function Detection

```python
# Key constructs to detect and transform
CONSTRUCTS = {
    "StateGraph(": "Convert to GroupChat or Swarm pattern",
    "create_react_agent(": "Convert to ConversableAgent + register_function",
    "create_supervisor(": "Convert to GroupChat with auto speaker selection",
    "ToolNode(": "Remove (tool execution built into ConversableAgent)",
    ".add_node(": "Convert to ConversableAgent creation",
    ".add_edge(": "Convert to GroupChat or initiate_chat ordering",
    ".add_conditional_edges(": "Convert to OnCondition or speaker_transitions",
    ".compile(": "Remove (no compilation needed in AG2)",
    ".invoke(": "Convert to initiate_chat()",
    ".stream(": "WARN: no streaming equivalent",
    "MemorySaver(": "WARN: no persistence equivalent",
    "@tool": "Convert import path, adjust decorator usage",
    "BaseTool": "Convert to Tool class",
    "StructuredTool.from_function(": "Convert to Tool()",
    "HumanMessage(": "Convert to dict format",
    "AIMessage(": "Convert to dict format",
    "SystemMessage(": "Convert to system_message parameter",
    "interrupt(": "WARN: convert to human_input_mode",
    "Command(": "Convert to OnCondition/AfterWork handoff",
    "Send(": "WARN: no parallel execution equivalent",
    "@entrypoint": "Convert to top-level function with initiate_chat",
    "@task": "Convert to tool function or agent",
    "AgentExecutor(": "Convert to ConversableAgent (legacy pattern)",
    "create_agent(": "Convert to ConversableAgent (LangChain 1.0)",
}
```

---

## 12. Model Provider Mapping

| LangChain Model | AG2 LLMConfig |
|---|---|
| `ChatOpenAI(model="gpt-4o")` | `LLMConfig(api_type="openai", model="gpt-4o")` |
| `ChatAnthropic(model="claude-sonnet-4-20250514")` | `LLMConfig(api_type="anthropic", model="claude-sonnet-4-20250514")` |
| `ChatGoogleGenerativeAI(model="gemini-2.0-flash")` | `LLMConfig(api_type="google", model="gemini-2.0-flash")` |
| `ChatOllama(model="llama3")` | `LLMConfig(api_type="ollama", model="llama3", base_url="http://localhost:11434/v1")` |
| `"openai:gpt-4o"` (string format) | `LLMConfig(api_type="openai", model="gpt-4o")` |

---

## 13. Conversion Priority & Strategy

### Phase 1: High-confidence automatic conversion
1. Import rewriting (LangChain/LangGraph imports to AG2 imports)
2. Tool definitions (`@tool` decorator and `BaseTool` to `register_function`)
3. Model configuration (`ChatOpenAI` etc. to `LLMConfig`)
4. Simple `create_react_agent` to `ConversableAgent` + tools
5. Message type conversion (LangChain messages to dicts)
6. System prompt extraction

### Phase 2: Pattern-based conversion (may need user review)
7. `StateGraph` with simple linear edges to `initiate_chats`
8. `StateGraph` with conditional edges to Swarm `OnCondition`
9. `create_supervisor` to `GroupChat` with auto selection
10. Human-in-the-loop patterns

### Phase 3: AI-assisted conversion (complex cases)
11. Complex state management with custom reducers
12. Subgraph composition
13. Checkpoint-dependent workflows
14. Custom streaming pipelines
15. Middleware chains

### Warning Generation
The converter should emit warnings for:
- `MemorySaver` / checkpointer usage (no equivalent)
- `stream()` / `astream()` calls (no equivalent)
- `Send()` for parallel execution (no equivalent)
- `CachePolicy` usage (no equivalent)
- Complex `TypedDict` state with custom reducers (partial equivalent)
- `response_format` structured output (no direct equivalent)
- `interrupt()` calls (coarser equivalent available)

---

## Sources

- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph Workflows and Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LangGraph Choosing APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis)
- [LangGraph Multi-Agent Collaboration](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
- [LangGraph GitHub Releases](https://github.com/langchain-ai/langgraph/releases)
- [LangGraph Supervisor](https://github.com/langchain-ai/langgraph-supervisor-py)
- [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [LangGraph Overview (DeepWiki)](https://deepwiki.com/langchain-ai/langgraph/2-core-architecture)
- [LangGraph create_react_agent (DeepWiki)](https://deepwiki.com/langchain-ai/langgraph/8.1-react-agent-(create_react_agent))
- [LangGraph StateGraph Reference](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)
- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain Messages](https://docs.langchain.com/oss/python/langchain/messages)
- [AG2 ConversableAgent](https://docs.ag2.ai/latest/docs/api-reference/autogen/ConversableAgent/)
- [AG2 GroupChat](https://docs.ag2.ai/latest/docs/api-reference/autogen/GroupChat/)
- [AG2 Swarm Orchestration](https://docs.ag2.ai/latest/docs/use-cases/notebooks/notebooks/agentchat_swarm/)
- [AG2 Quickstart](https://docs.ag2.ai/latest/docs/home/quickstart/)
- [AG2 Swarm Deep-Dive](https://docs.ag2.ai/0.8.3/docs/user-guide/advanced-concepts/swarm/deep-dive/)
- [LangGraph PyPI](https://pypi.org/project/langgraph/)
- [LangGraph Conditional Edges](https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn)
- [LangChain 1.0 vs LangGraph 1.0](https://www.clickittech.com/ai/langchain-1-0-vs-langgraph-1-0/)
- [LangGraph Breaking Changes](https://github.com/langchain-ai/langgraph/issues/6363)
- [LangChain 1.0 Agents Guide](https://www.leanware.co/insights/langchain-agents-complete-guide-in-2025)
