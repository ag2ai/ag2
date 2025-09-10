import os

from dotenv import load_dotenv

from autogen import LLMConfig
from autogen.agents.experimental.document_agent import DocAgent

load_dotenv()

OAI_KEY = os.getenv("OPENAI_API_KEY")

llm_config = LLMConfig({"api_type": "openai", "model": "gpt-5", "api_key": OAI_KEY})

document_agent = DocAgent(llm_config=llm_config)

# Use the run method to process a message
run_response = document_agent.run(
    message="Could you ingest /test/agentchat/contrib/graph_rag/Toast_financial_report.pdf and tell me the fiscal year 2024 financial summary?",
    max_turns=1,
)
run_response.process()

print("Document agent processing completed")
