import os
import duckdb
from autogen.agentchat.contrib.learning.knowledge_sharing_swarm_agent import KnowledgeSharingSwarmAgent
from autogen.agentchat.contrib.learning.knowledge_generating_swarm_agent import KnowledgeGeneratingSwarmAgent
from autogen.agentchat.contrib.swarm_agent import AFTER_WORK, SwarmAgent, initiate_swarm_chat
from autogen.agentchat.user_proxy_agent import UserProxyAgent

# LLM Configuration
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.environ.get("OPENAI_API_KEY")
    }
]

llm_config = {
    "cache_seed": 42,  # Change the cache_seed for different trials
    "temperature": 1,
    "config_list": config_list,
    "timeout": 120,
}

def run_query(context_variables: dict, query: str) -> str:
    """
    Executes a DuckDB query and returns the results as a markdown table.

    Args:
        context_variables (dict): Contextual variables (unused in this function).
        query (str): The DuckDB-compatible SQL query to execute.

    Returns:
        str: Markdown-formatted query results, or an error message if too many rows are returned.
    """
    with duckdb.connect("./autogen/agentchat/contrib/learning/animal_crossing.duckdb") as conn:
        df = conn.execute(query).fetchdf()
        if len(df) > 50:
            return f"Error: Query returned {len(df)} rows. Results must be below 50 rows."
        print(query)  # Log the query for debugging
        return df.to_markdown()

# Scientist Agent to handle queries
queryer = KnowledgeGeneratingSwarmAgent(
    name="queryer",
    researcher_llm_config=llm_config,
    result_analyzer_llm_config=llm_config,
    note_taker_llm_config=llm_config,
    max_rounds=50,
    functions=[run_query],
    user_notes=[
        'You are connecting to a DuckDB database, ensure all queries are compatible with DuckDB.',
        'Begin by gathering information about what data is available.',
        'Get sample rows to understand the shape of the data.'
    ]
)

# Reasoner Prompt
REASONER_PROMPT = """Role: Reasoner

Instructions:
- You will be given a question from a user.
- Step by step, gather information by invoking the provided functions.
- Reason through the information provided to you as knowledge.
- Respond to the user once you are able to answer the question using your knowledge.
- You must cite sources for claims by referencing the findings from your provided knowledge
"""

# Knowledge-Sharing Agent to reason through gathered knowledge
reasoner = KnowledgeSharingSwarmAgent(
    name="reasoner",
    llm_config=llm_config,
    functions=[queryer.get_invocation_function()],
    system_message=REASONER_PROMPT
)

# User agent to simulate human input
user = SwarmAgent(
    name="user",
    human_input_mode="ALWAYS"
)

# Register handoffs between agents
user.register_hand_off(AFTER_WORK(reasoner))
reasoner.register_hand_off(AFTER_WORK(user))
queryer.register_hand_off(AFTER_WORK(reasoner))

# Start the swarm chat
chat_result, final_context_variables, last_active_agent = initiate_swarm_chat(
    initial_agent=reasoner,
    agents=[queryer, user, reasoner],
    # This is intentionally being very vague and not at all related to writing sql queries
    messages="What conditions are correct to catch a golden trout?",
    context_variables={}  # Initialize with an empty context
)

# Print the accumulated knowledge from the queryer agent
print(queryer.knowledge)
