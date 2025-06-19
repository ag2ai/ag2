# [RFC] - Simplifying Workflow Design with AG2's High-Level API

Author: Davor Runje, Tvrtko Sternak, Davorin Ruševljan
Status: Discussion

# 1. **Introduction**

In today’s fast-paced development environment, building workflows that integrate across different platforms—be it a console application, a REST API server, or a web UI—can be a complex and time-consuming task. Developers often need to juggle multiple agents, tools, and interaction patterns, all while ensuring that workflows remain efficient and manageable. These challenges can result in high overhead, especially when trying to implement sophisticated multi-agent systems with minimal boilerplate code.

This RFC proposes a **high-level API** within AG2 to simplify the creation and execution of workflows, providing an easy-to-use framework for integrating multi-agent systems across various runtimes. The goal is to make it easier for developers to define agents, attach tools, and execute workflows in a way that works seamlessly across environments, whether you're building a console app, a REST server, or a web-based UI.

To illustrate the power and simplicity of this new API, consider the following examples where two agents collaborate to answer a user query. In these examples, one agent gathers information, while the other validates the facts. We will explore how this can be done both in a **WebSocket environment** and a **CLI application**.

---

# 2 Motivational Examples

## 2.1 **Example: WebSocket-Based Workflow**

Let's first consider the WebSocket example, where agents work together in a dynamic, interactive environment. The WebSocket server handles asynchronous communication between the agents and a client, allowing real-time interaction.

### **WebSocket Server Example with `websockets` Library**

```python
import asyncio
import websockets
import json
from typing import Any
from autogen import ConversableAgent, LLMConfig
from autogen.tools import tool

# Set up LLMConfig as a context manager for agent configuration
llm_config = LLMConfig({"config_list": [{"api_key": "your-api-key", "model": "gpt-4o-mini"}]})

# Define tools for the agents
@tool(description="Search Wikipedia for the given query.")
def search_wikipedia(query: str) -> str:
    return "Leonardo da Vinci was an Italian polymath."

@tool(description="Send the answer to the user.")
def send_answer(answer: str) -> str:
    return f"Answer sent: {answer}"

async def run_agents_and_process_events():
    with llm_config:
        alice = ConversableAgent(system_message="You are a helpful assistant.")
        bob = ConversableAgent(system_message="You are a fact-checker.")

    alice.add_tool(search_wikipedia)
    bob.add_tool(send_answer)

    response = await a_run(alice, bob, message="Who is Leonardo da Vinci?")
    return response

async def process_event_loop(response: AsyncRunResponseProtocol, websocket: websockets.WebSocketServerProtocol):
    async for event in response.events:
        if event.type == "input_request":
            await websocket.send(json.dumps({"type": "input_request", "prompt": event.prompt}))
            user_input = await websocket.recv()
            event.respond(InputResponseEvent(value=user_input))

        elif event.type == "output":
            await websocket.send(json.dumps({"type": "output", "value": event.value}))

        elif event.type == "agent_message":
            await websocket.send(json.dumps({"type": "agent_message", "message": event.message}))

        elif event.type == "system":
            await websocket.send(json.dumps({"type": "system", "value": event.value}))

        elif event.type == "error":
            await websocket.send(json.dumps({"type": "error", "error": event.error}))

async def handler(websocket: websockets.WebSocketServerProtocol, path: str):
    response = await run_agents_and_process_events()
    await process_event_loop(response, websocket)

async def main():
    server = await websockets.serve(handler, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

asyncio.run(main())

```

### Explanation:

- The WebSocket server listens for incoming client connections and processes events from agents Alice and Bob.
- **Events** are processed asynchronously, with **input requests** sent to the client, where the client responds with input. The server then sends back output, agent messages, and system messages as they occur in the workflow.

---

## 2.2 **Example: Synchronous CLI Application Using `typer`**

The second example demonstrates a synchronous version of the same workflow, but in a CLI application built using the `typer` library.

### **CLI Application Code with `typer` and Synchronous Event Handling**

```python
import typer
from os import environ
from autogen import ConversableAgent, LLMConfig
from autogen.tools import tool

# Set up the typer app
app = typer.Typer()

# Set up LLMConfig as a context manager for agent configuration
llm_config = LLMConfig({"config_list": [{"api_key": environ["OPENAI_API_KEY"], "model": "gpt-4o-mini"}]})

# Define tools for the agents
@tool(description="Search Wikipedia for the given query.")
def search_wikipedia(query: str) -> str:
    return "Leonardo da Vinci was an Italian polymath."

@tool(description="Send the answer to the user.")
def send_answer(answer: str) -> str:
    return f"Answer sent: {answer}"

def run_agents_and_process_events():
    with llm_config:
        alice = ConversableAgent(system_message="You are a helpful assistant.")
        bob = ConversableAgent(system_message="You are a fact-checker.")

    alice.add_tool(search_wikipedia)
    bob.add_tool(send_answer)

    response = run(alice, bob, message="Who is Leonardo da Vinci?")
    return response

def process_event_loop(response):
    s = None
    for event in response.events:
        try:
            if event.type == "input_request":
                print(f"Input Request: {event.prompt}")
                s = input("Your response: ")
                event.respond(InputResponseEvent(value=s))

            elif event.type == "input_response":
                print(f"Input Response: {event.value}")

            elif event.type == "output":
                print(f"Output: {event.value}")

            elif event.type == "agent_message":
                print(f"Agent Message: {event.message}")

            elif event.type == "system":
                print(f"System: {event.value}")

            elif event.type == "error":
                print(f"Error occurred: {event.error}")

            else:
                print(f"Unrecognized event type: {event.type}")

        except Exception as e:
            print(f"Error processing event {event}: {str(e)}")

@app.command()
def run_workflow():
    response = run_agents_and_process_events()
    process_event_loop(response)

if __name__ == "__main__":
    app()

```

### Explanation:

- The `typer` library is used to build a simple CLI that prompts the user for input and displays agent messages.
- This is a **synchronous** process where the program waits for user input before continuing the workflow and handling the next event.
- The program processes each event step-by-step in the console, making it easy for developers to create interactive workflows in a synchronous environment.

## 2.3 Example: Group Chat Manager for Handling Transitions Between Multiple Agents

### **Example Code with `NewGroupChatManager`**:

```python
from autogen import ConversableAgent, LLMConfig
from autogen.tools import tool
from autogen.chat_manager import NewGroupChatManager, Keyword

# Set up LLMConfig for agent configuration
llm_config = LLMConfig({"api_type": "openai", "model": "gpt-4o-mini"})

# Define a tool to submit the plan
@tool
def submit_plan(plan: str) -> str:
    return f"Plan submitted: {plan}"

# Create the agents for the workflow
with llm_config:
    planner = ConversableAgent("You are a planner. Collaborate with teacher and reviewer to create lesson plans.")

    reviewer = ConversableAgent("You are a reviewer. Review lesson plans against 4th grade curriculum. Provide max 3 changes.")

    teacher = ConversableAgent(
        "You are a teacher. Choose topics and work with planner and reviewer. Say DONE! when finished.",
        tools=[submit_plan]
    )

    # Set up the Group Chat Manager with a termination condition (when teacher says DONE!)
    chat_manager = GroupChatManager(terminate_on=Keyword("DONE!"))

# Run the workflow with the three agents and the group chat manager
response = run(
    planner, reviewer, teacher,
    message="Create lesson plans for 4th grade.",
    chat_manager=chat_manager,
)

# Process the events generated during the workflow
def process_event_loop(response):
    s = None
    for event in response.events:
        try:
            if event.type == "input_request":
                print(f"Input Request: {event.prompt}")
                s = input("Your response: ")
                event.respond(InputResponseEvent(value=s))  # Responding to the request

            elif event.type == "input_response":
                print(f"Input Response: {event.value}")

            elif event.type == "output":
                print(f"Output: {event.value}")

            elif event.type == "agent_message":
                print(f"Agent Message: {event.message}")

            elif event.type == "system":
                print(f"System: {event.value}")

            elif event.type == "error":
                print(f"Error occurred: {event.error}")

            else:
                print(f"Unrecognized event type: {event.type}")

        except Exception as e:
            print(f"Error processing event: {str(e)}")

# Start the workflow and process events
process_event_loop(response)
```

### **Explanation**:

1. **Setting Up the Agents**:
    - **Planner**: The planner is responsible for collaborating with the teacher and reviewer to create the lesson plan.
    - **Reviewer**: The reviewer checks the lesson plan against a set curriculum and provides suggested changes (up to 3).
    - **Teacher**: The teacher selects topics and works with the planner and reviewer. Once the lesson plan is ready, the teacher will signal completion by saying "DONE!".
2. **The `GroupChatManager`**:
    - The `GroupChatManager` is used to manage the transitions between agents in this group workflow. It listens for the `Keyword("DONE!")` from the teacher to terminate the workflow. This ensures that the process stops once the teacher is satisfied with the lesson plan.
3. **Event Handling**:
    - The workflow progresses by processing events such as **input requests**, **input responses**, **output**, **agent messages**, and **system messages**. These events trigger interactions between the agents, with the teacher eventually signaling the end of the workflow by saying "DONE!".
4. **Run and Workflow Execution**:
    - The `run()` function starts the workflow, passing the three agents (planner, reviewer, teacher) and the `chat_manager`. The agents communicate through the `chat_manager`, ensuring that transitions between them are handled automatically. The workflow will stop when the teacher says "DONE!".
5. **Event Processing**:
    - The `process_event_loop` function handles each event generated during the workflow. For example, if an **input request** event is triggered, the program prompts the user for input and responds to the agent with the provided answer.

### **How It Works**:

1. The **planner** and **teacher** begin the lesson planning process.
2. The **reviewer** provides feedback on the plan.
3. The **teacher** interacts with both the planner and reviewer, making changes and finalizing the plan.
4. The **teacher** signals the completion of the workflow by saying "DONE!".
5. The **Group Chat Manager** ensures that the workflow ends when the teacher finishes, and the process terminates cleanly.

---

## 3. **Proposed High-Level API**

### 3.1 **Overview of the High-Level API**

To support such use cases, we propose a **high-level API** for AG2 that simplifies the process of defining agents, attaching tools, handling events, and integrating with various runtime environments. The API abstracts much of the complexity of workflow orchestration while maintaining flexibility for different platforms.

### 3.2 **Key Components**:

1. **Agent Creation**:
    - Agents can be created easily, configured with a system message, and equipped with tools. A context manager (`LLMConfig`) is used to configure agent settings like the LLM model and API keys. Names would be generated automatically if not specified.

    ```python
    with llm_config:
        alice = ConversableAgent(system_message="...")
        bob = ConversableAgent(system_message="...")

    agent.add_tool(some_tool)

    ```

2. **Event Handling**:
    - The event loop will process different types of events such as `input_request`, `input_response`, `output`, and `agent_message`. The events are processed synchronously or asynchronously, depending on the runtime.
    - For synchronous environments, the `run()` function and `process_event_loop` work together, while for asynchronous environments, the `a_run()` function and `process_event_loop` use async patterns like `async for`.
3. **Tool Integration**:
    - Tools are integrated directly with agents using decorators, making it easy to add functionality like querying a database, calling external APIs, or processing user input.
4. **Execution Control**:
    - The `run()` function starts the workflow and returns a response containing events that need to be processed. The event-handling mechanism allows for smooth transitions between different types of workflow stages.

---

### 3.3 **Event Classes**

To handle events, we propose defining a base `Event` class, with different subclasses for each type of event. These classes will include both synchronous and asynchronous processing methods.

### 3.3.1 **Base Event Class**

```python
from typing import Protocol
from uuid import UUID, uuid4
from abc import ABC, abstractmethod

EventType = Literal["input_request", "input_response", "agent_message", "output", "system", "error"]

class Event(ABC):
    uuid: UUID = uuid4()
    type: EventType

    @abstractmethod
    def process(self, event_processor: "EventProcessorProtocol"): ...

    @abstractmethod
    async def a_process(self, event_processor: "AsyncEventProcessorProtocol"): ...

```

### 3.3.2 **Event Subclasses (e.g., InputRequestEvent)**

```python
class InputRequestEvent(Event):
    prompt: str
    type: EventType = "input_request"

    def process(self, event_processor: "EventProcessorProtocol"):
        event_processor.process_input_request_event(self)

    async def a_process(self, event_processor: "AsyncEventProcessorProtocol"):
        await event_processor.a_process_input_request_event(self)

```

---

### 3.4 **Sync and Async Versions of `RunResponseProtocol`**

The `RunResponseProtocol` will serve as the return type for both synchronous and asynchronous execution functions (`run()` and `a_run()`). It will contain the events and messages generated during workflow execution.

### 3.4.1 **Synchronous Version: `RunResponseProtocol`**

```python
from typing import Iterable

class RunResponseProtocol(Protocol):
    @property
    def events(self) -> Iterable[Event]: ...

    @property
    def messages(self) -> Iterable[str]: ...

    @property
    def summary(self) -> str: ...

```

### 3.4.2 **Asynchronous Version: `AsyncRunResponseProtocol`**

```python
from typing import AsyncIterable

class AsyncRunResponseProtocol(Protocol):
    @property
    def events(self) -> AsyncIterable[Event]: ...

    @property
    async def summary(self) -> str: ...

    @property
    def messages(self) -> AsyncIterable[str]: ...

```

---

## 4. **Conclusion**

The proposed high-level API for AG2 simplifies the creation and management of multi-agent workflows, making it easy for developers to integrate these workflows into various runtime environments such as console applications, REST servers, and web UIs. The API abstracts away much of the complexity, providing an intuitive interface for creating agents, handling tools, and processing events.

By using this API, developers can easily build interactive, multi-agent systems in both synchronous and asynchronous environments, reducing boilerplate code and improving productivity.

We encourage community feedback on the proposed API to refine it further and ensure it meets the needs of all users.

---

This concludes the RFC document. Let me know if you'd like any further modifications or additions!
