import os
import random
import json
from typing import Annotated, Any, List, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern

from dotenv import load_dotenv
load_dotenv()

# Mock database of previous transactions
def get_previous_transactions() -> list[dict[str, Any]]:
    today = datetime.now()
    return [
        {
            "vendor": "Staples",
            "amount": 500,
            "date": (today - timedelta(days=3)).strftime("%Y-%m-%d"),  # 3 days ago
            "memo": "Quarterly supplies",
        },
        {
            "vendor": "Acme Corp",
            "amount": 1500,
            "date": (today - timedelta(days=10)).strftime("%Y-%m-%d"),  # 10 days ago
            "memo": "NDA services",
        },
        {
            "vendor": "Globex",
            "amount": 12000,
            "date": (today - timedelta(days=5)).strftime("%Y-%m-%d"),  # 5 days ago
            "memo": "Confidential",
        },
    ]

# Simple duplicate detection function
def check_duplicate_payment(
    vendor: Annotated[str, "The vendor name"],
    amount: Annotated[float, "The transaction amount"],
    memo: Annotated[str, "The transaction memo"]
) -> dict[str, Any]:
    """Check if a transaction appears to be a duplicate of a recent payment"""
    previous_transactions = get_previous_transactions()

    today = datetime.now()

    for tx in previous_transactions:
        tx_date = datetime.strptime(tx["date"], "%Y-%m-%d")
        date_diff = (today - tx_date).days

        # If vendor, memo and amount match, and transaction is within 7 days
        if (
            tx["vendor"] == vendor and
            tx["memo"] == memo and
            tx["amount"] == amount and
            date_diff <= 7
        ):
            return {
                "is_duplicate": True,
                "reason": f"Duplicate payment to {vendor} for ${amount} on {tx['date']}"
            }

    return {
        "is_duplicate": False,
        "reason": "No recent duplicates found"
    }

# Configure the LLM for finance bot (standard configuration)
finance_llm_config = LLMConfig(config_list={
    "api_type": "openai",
    "model": "gpt-5-nano",
    "api_key": os.environ.get("OPENAI_API_KEY"),
})

# Define the system message for our finance bot
finance_system_message = """
You are a financial compliance assistant. You will be given a set of transaction descriptions.

For each transaction:
1. First, extract the vendor name, amount, and memo
2. Check if the transaction is a duplicate using the check_duplicate_payment tool
3. If the tool identifies a duplicate, automatically reject the transaction
4. If not a duplicate, continue with normal evaluation:
    - If it seems suspicious (e.g., amount > $10,000, vendor is unusual, memo is vague), ask the human agent for approval
    - Otherwise, approve it automatically

Provide clear explanations for your decisions, especially for duplicates or suspicious transactions.
When all transactions are processed, summarize the results and say "You can type exit to finish".
"""

# Create the agents with respective LLM configurations
finance_bot = ConversableAgent(
    name="finance_bot",
    system_message=finance_system_message,
    functions=[check_duplicate_payment],
    llm_config=finance_llm_config,
)

# Create the human agent for oversight
human = ConversableAgent(
    name="human",
    human_input_mode="ALWAYS",  # Always ask for human input
)

# Define structured output for audit logs
class TransactionAuditEntry(BaseModel):
    vendor: str
    amount: float
    memo: str
    status: str  # "approved" or "rejected"
    reason: str

class AuditLogSummary(BaseModel):
    total_transactions: int
    approved_count: int
    rejected_count: int
    transactions: List[TransactionAuditEntry]
    summary_generated_time: str

# Configure the LLM for summary bot with structured output
summary_llm_config = LLMConfig(
    config_list = {
        "api_type": "openai",
        "model": "gpt-5-nano",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "response_format": AuditLogSummary,  # Using the Pydantic model for structured output
    },
)

# Define the system message for the summary agent
# Note: No formatting instructions needed!
summary_system_message = """
You are a financial summary assistant that generates audit logs.
Analyze the transaction details and their approval status from the conversation.
Include each transaction with its vendor, amount, memo, status and reason.
"""

summary_bot = ConversableAgent(
    name="summary_bot",
    system_message=summary_system_message,
    llm_config=summary_llm_config,
)

def is_termination_msg(x: dict[str, Any]) -> bool:
    content = x.get("content", "")
    try:
        # Try to parse the content as JSON (structured output)
        data = json.loads(content)
        return isinstance(data, dict) and "summary_generated_time" in data
    except:
        # If not JSON, check for text marker
        return (content is not None) and "==== SUMMARY GENERATED ====" in str(content)

# Generate new transactions including a duplicate
transactions = [
    "Transaction: $500 to Staples. Memo: Quarterly supplies.",  # Duplicate
    "Transaction: $4000 to Unicorn LLC. Memo: Reimbursement.",
    "Transaction: $12000 to Globex. Memo: Confidential.",  # Duplicate
    "Transaction: $22000 to Initech. Memo: Urgent request."
]

# Format the initial message
initial_prompt = (
    "Please process the following transactions one at a time, checking for duplicates:\n\n" +
    "\n".join([f"{i+1}. {tx}" for i, tx in enumerate(transactions)])
)

pattern = AutoPattern(
    initial_agent=finance_bot,
    agents=[finance_bot, summary_bot],
    user_agent=human,
    group_manager_args = {
        "llm_config": finance_llm_config,
        "is_termination_msg": is_termination_msg
    },
)

result, _, _ = initiate_group_chat(
    pattern=pattern,
    messages=initial_prompt,
)

# Pretty-print the transaction_data
final_message = result.chat_history[-1]["content"]
structured_data = json.loads(final_message)
print(json.dumps(structured_data, indent=4))