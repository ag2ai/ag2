# [RFC] - AgentBus: Inter Agent Communication Model

Author: Davor Runje, Tvrtko Sternak, Davorin Ruševljan

Status: Discussion

## Introduction

Current ag2 model has enabled fast experimentation and development of proof of concept implementations of various
agentic patterns. As such it provided solid ground to spawn many new agentic concepts and patterns. However, now when our focus broadens to put those concept and patterns in production environment we are facing new challenges with:
- support for agents with different prerequisites, like different versions of some package, or even python versions
- interoperability with non ag2 agents
- ability to split agent workflow into several processes, possibly running on different machines
- development of new agent flow patterns while definitely possible (groupchat, swarm) has somewhat high barrier, since
  framework does not provide some common building blocks, so there is small commonality between differing flow patterns even when there could be.
- incorporation of non LLM agents


## Pain points in current model

Most of the challenges described in introduction seem to have following causes in common:
- agents are referred to as direct python object reference
- communication between agents is performed with direct method calls

Additionally new agentic flow patterns are developend basically from the scratch, because there are no
common building blocks from the framework.

## What we would like to achieve (Goals)

Inter agent communication model that:
- does not rely on direct python references and method calls
- is able to be split over the process barrier
- lends itself to elegant implementation of current ag2 flow patterns, but also
  provides base for development of new ones
- can incorporate existing ag2 agents (possibly via proxies)
- can incorporate "foreign" agents from different frameworks (possibly via proxy)
- can incorporate non llm agents

## AgentBus - Basic Concept
In order to achieve goals mentioned we propose AgentBus concept, in which Agentic system is represented
as collection of Agents, all connected to the AgentBus.

AgentBus carries events, which themselves are structured objects that are like JSON objects.

Each agent can emit event to the AgentBus. Once emitted, event is presented to all Agents
connected to the AgentBus. Specially, agentic flow could be started by initial event provided by
starter of the flow.

Once presented with the event, Agent can select to be activated by the event and perform action.
Agent can be possibly activated by different events, and perform different actions, during which
Agent can emit new events.

## Agent Activation - Event Selectors

In order to provide elegant way for Agents to activate on certain events, we propose concept
of selectors, that take event and choose whether event should activate agent or not. Attached to
this selector would be code to be executed if agent is activated by the event. Agent can have
several selectors, so that it can be activated by different events, and perform different actions.

Two forms of Event Selectors are envisioned:
- pattern matching selectors
- universal selectors

### Pattern Matching Selectors

Pattern matching selectors (PMS) take the event, and check if event conforms to certain pattern.

Syntax for PMS would be proposed in following RFC, but it should provide means to check in arbitrary depth:
- if event has particular key
- if value under some key has particular value
- if some value is list, dictionary or atomic value
- if some element of list has particular value

Few examples of one possible style:
``` python
@select("{role:'critic', task: 'pedantic_review', _ }")
def pedantic_review(self, event: Event):
  review = self._pedantic_review(event)
  self.emit(review)

@select("{role:'critic', task: 'improvement_suggestions', _ }")
def suggest_imporovements(self, event: Event):
  # do stuff needed..
  ...

@select("{text: content, _}")
def on_text(self, event: Event):
  if self.belongs_to_history(event):
    content = event['content']
    self.add_to_history(content)
  else:
    self.reject(event)

```

The actual capabilities and syntax to be proposed will take inspiration from pattern matching capabilities of
- Common web server routing selectors
- Python
- Prolog
- Elixir
- Rust

### Universal Selectors

Take event and use callable to decide if event is selected or rejected.

### Selectors TBD

- are selectors evaluated:
  - sequentially, and only after some selector is rejected next one is evaluated
  - concurrently

## Common Agentic Flow Patterns
here are some examples how common Agentic flow might be implemented using AgentBus

### Bucket Passing

### Chat Master

## Tools
tbd:
- Agent that executes functions for others?
- Each agent can execute function calls
- Combination?

## State/Stateless
are agents stateless or statefull?


## Interfacing to Outer World
### To Alien Agents
agent that acts as proxy for external agent
### To executors
agent that listens to the ui events and passes them to external world. Also - it accepts responses
from outer world and emits them as events to the AgentBus
### Execution logging
agent that listens everything

## EventBus Possible implementations

- in process emulation
- message queue based system


{'role': 'critic', ...}
