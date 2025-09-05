import os

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group.guardrails import RegexGuardrail
from autogen.remote import HTTPAgentBus

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

PYTHON_CODER_PROMPT = (
    "You are an expert Python developer. "
    "When asked to make changes to a code file, "
    "you should update the code to reflect the requested changes. "
    "Do not provide explanations or context; just return the updated code."
    "You should work in a single file. Just a code listing, without markdown markup."
    "Do not generate trailing whitespace or extra empty lines. "
    "Strongly follow provided recommendations for code quality. "
    "Do not generate code comments, unless required by linter. "
)


code_agent = ConversableAgent(
    name="coder",
    system_message=PYTHON_CODER_PROMPT,
    llm_config=llm_config,
    is_termination_msg=lambda x: "LGTM" in x.get("content", ""),
    human_input_mode="NEVER",
    silent=True,
)

sensitive_info_guardrail = RegexGuardrail(
    name="sensitive_info_detector",
    condition=r".*(ssn|social security|\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}).*",
    target=AgentTarget(code_agent),
    activation_message="Sensitive information detected - routing to compliance review",
)

code_agent.register_output_guardrail(sensitive_info_guardrail)
code_agent.register_input_guardrail(sensitive_info_guardrail)

app = HTTPAgentBus(agents=[code_agent])
