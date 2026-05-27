---
name: review-agent-config
description: Review AG2 agent configuration for common mistakes and anti-patterns
---

Review the AG2 agent configuration in the current file. Check for these common issues:

**Critical:**
- [ ] Every `register_for_llm()` has a matching `register_for_execution()`
- [ ] Agent names have no whitespace
- [ ] `llm_config` uses `LLMConfig()` class, not raw dicts
- [ ] API keys come from environment variables, not hardcoded
- [ ] All agents have unique names
- [ ] Group manager llm_config has no tools/functions
- [ ] Code executor class names use correct capitalization: `LocalCommandLineCodeExecutor`, `DockerCommandLineCodeExecutor`

**Important:**
- [ ] Agents in group chat have meaningful `description` fields
- [ ] Termination condition is defined (`max_rounds` on `run_group_chat` or `is_termination_msg`)
- [ ] `human_input_mode` is set appropriately for the use case
- [ ] Tool parameter types are annotated with `Annotated[type, "description"]`
- [ ] Code execution agents have proper `code_execution_config`
- [ ] `LLMGuardrail` instances include the `llm_config` parameter
- [ ] When using `AutoPattern`, `group_manager_args` includes `llm_config`

**Best Practices:**
- [ ] Uses `run_group_chat()` with patterns instead of legacy `GroupChat + GroupChatManager`
- [ ] System messages are clear and specific about the agent's role
- [ ] Tool descriptions clearly state WHEN the LLM should use the tool
- [ ] Handoffs use `OnContextCondition` for simple state checks (cheaper than `OnCondition`)
- [ ] Async workflows use `a_run_group_chat` / `a_run`

Report findings as a checklist with pass/fail for each item and specific fix suggestions for any failures.
