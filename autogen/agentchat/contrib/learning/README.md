# Feedback and Direction for Concept Development

## Overview

This iteration is intended to:

- Prove that the concepts work together cohesively.

There is a lot of future work to be done, many more are included in the code as TODOs, but some examples are:

- Separate these concepts into more defined and manageable components in the future.
- Refine naming conventions, which currently lack consistency due to changes throughout development.

Note: a lot of the conceptual naming/wording in this is work in progress so they may be confusing.

## Core Concept

The general idea is to create a system that:

- **Observes the problem-solving process** (e.g., function/tool usage) and augments it with strategies like the scientific method (as an idea) in order to get data about tool use.
- **Collects, stores and makes that knowledge available to other agents and functions** without relying on message passing.

### Key Features

1. **Information Gathering**:
   - Collect insights from all interactions and results in the conversation.
     - Example: If the agent runs `SHOW TABLES;` early in the process, that information should be recorded.
     - Example: If certain parameters consistently result in errors, that behavior should be logged.
   - Record these insights in a structured and reusable format.

2. **Feedback Loop**:
   - Dynamically update system messages and function docstrings with gathered information.
   - This creates a loop where each iteration becomes faster and more efficient due to improved contextual knowledge.

3. **Knowledge Format**:
   - Store information in a **concrete data format**:
     - Enables analysis by another LLM for accuracy and optimization.
     - Allows permanent storage for future use as a prompt.
   - Effectively acts as a **training framework for functions**.

### Applications

- Enable a system to explore using a tool, gather insights, and provide tips for subsequent runs.
- Build a **domain-specific prompt-tuning framework for functions/agents**:
  - Train a general-purpose agent to specialize in a domain through iterative experimentation.
  - For instance:
    - Use training data to simulate questions.
    - Adjust knowledge dynamically to achieve correct answers or reduce iterations.
      - Can use TOT/MCST to do this, one route decides to gather information A first other route decides info B, one may be better than the other

## Benefits

- Automates the discovery of knowledge required by the agent.
- Reduces reliance on manual intervention for prompt engineering.
- Creates a **self-experimenting system** that improves over time through trial and error.

---

## Components

### 1. **Knowledge Source/Knowledge Generator**

- **Goal**: Share knowledge between agents in a swarm without relying on message history alone.
- **Data Models**:
  - Maintain knowledge as Python objects instead of message history.
  - Register agents with each other to share relevant information:
    - Example: SQL query agent may pull knowledge from a RAG agent.
    - Example: RAG agent might not care about SQL query knowledge.
    - Examplne: Reasoning agent cares about knowledge from the SQL agent but not RAG agent, so you now have a teired memory system where information can stop at certain levels or bubble up.

- **Implementation**:
  - Wrap function use in a "scientific method"-like loop (very loose):
    - Treat functions as "experiments."
    - Generate hypotheses for each tool run.
    - Evaluate results to gain insights.

---

### 2. **Sub-Swarm**

- **Purpose**: Handle complex retry loops when an agent lacks sufficient information or effective prompts.
- **Design**:
  - Similar to how a ReasoningAgent creates a sub-conversation.
  - Sub-swarms allow:
    - Functions to execute within the sub-swarm.
    - Results to bubble back up to the main swarm.
  - Enables state transitions via `SwarmResult` and context variable updates.
  - Enables functions to modify context_variables

- **Advantages**:
  - Reduces overall context size.
  - Allows for validation and corrects facts in the knowledge directly, ensuring reliable information flow.
  - Allows flexible and traceable solutions.

---

## Key Challenges

1. **Structuring Knowledge**:
   - Ensuring that gathered knowledge is accurate, relevant, and trustworthy.
   - Building systems to validate and compact this knowledge effectively.

2. **Reducing Hallucinations**:
   - By trusting only pre-populated and validated knowledge, reasoning agents could potentially reduce hallucination risks.

3. **Flexibility**:
   - Ensuring the system works across different domains and tools while maintaining modularity.

---

## Next Steps

1. **Refine the Structure**:
   - Focus on making the knowledge-sharing process seamless and efficient.
   - Separate concepts (e.g., KnowledgeSource, Sub-Swarm) into well-defined modules.

2. **Experimentation and Testing**:
   - Test the hypothesis-driven loop for tool usage.
   - Evaluate how effectively the system trains itself on domain-specific tasks.

3. **Naming Conventions**:
   - Establish consistent terminology to avoid confusion as concepts evolve.

---

## Closing Thoughts

The key to success lies in making the **knowledge structure robust and trustworthy**. If the reasoning agent can rely entirely on validated knowledge, it could significantly improve accuracy and reduce hallucinations. Additionally, modularizing components like the **Sub-Swarm** and **Knowledge Generator** will make the system more flexible and scalable in various applications.
