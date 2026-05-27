---
name: explain-flow
description: Trace and explain the message flow in an AG2 multi-agent application
---

Analyze the AG2 code in the current file or selection. Trace the complete message flow:

1. **Entry point**: Where does `run_group_chat()`, `a_run()`, or `a_run_group_chat()` get called?
2. **Pattern**: What pattern is used (AutoPattern, RoundRobinPattern, DefaultPattern)?
3. **Agent chain**: Which agents participate and in what order?
4. **Handoffs**: What OnCondition / OnContextCondition handoffs are configured? What are the transition targets?
5. **Tool calls**: What tools are registered and which agents call/execute them?
6. **Speaker selection**: How is the next speaker chosen (AutoPattern LLM, round-robin, handoff, manual)?
7. **Context variables**: Is ContextVariables used? How does shared state flow between agents?
8. **Termination**: What conditions end the conversation (max_rounds, TerminateTarget, is_termination_msg)?
9. **Guardrails**: Are any guardrails (LLMGuardrail, RegexGuardrail) registered?

Draw an ASCII diagram showing the agent interaction pattern. Identify any potential issues like missing tool registration, no termination condition, missing llm_config in AutoPattern, or unclear speaker selection.
