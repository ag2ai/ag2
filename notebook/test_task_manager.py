import os
from pathlib import Path
from dotenv import load_dotenv

from autogen import LLMConfig
from autogen.agents.experimental.document_agent import task_manager
from autogen.agents.experimental.document_agent.task_manager_agent import TaskManagerAgent
from autogen.agents.experimental.document_agent.document_utils import Ingest, Query
from autogen.agents.experimental.document_agent.inmemory_query_engine import InMemoryQueryEngine
from autogen.agentchat.group.context_variables import ContextVariables

load_dotenv()

OAI_KEY = os.getenv("OPENAI_API_KEY")

def test_task_manager_individual():
    """Test TaskManagerAgent individually with context variables."""
    
    # Create LLM config
    llm_config = LLMConfig({"api_type": "openai", "model": "gpt-4o-mini", "api_key": OAI_KEY})
    
    # Create query engine
    query_engine = InMemoryQueryEngine(llm_config=llm_config)
    
    # Create TaskManagerAgent
    task_manager = TaskManagerAgent(
        llm_config=llm_config,
        query_engine=query_engine,
        parsed_docs_path="./test_parsed_docs"
    )
    
    # Create context variables with test data
#     context_variables = ContextVariables(
#         data={
#             "CompletedTaskCount": 0,
#             "DocumentsToIngest": [
#                 Ingest(path_or_url="./test_parsed_docs/test_document.txt")
#             ],
#             "DocumentsIngested": [],
#             "QueriesToRun": [
#                 Query(query_type="RAG_QUERY", query="ingest the document and then answer the question: What is the fiscal year 2024 financial summary?")
#             ],
#             "QueryResults": [],
#         }
#     )
    
#     # Set context variables on the agent
#     task_manager.context_variables = context_variables
    
#     print("=== Testing TaskManagerAgent individually ===")
#     print(f"Initial context variables: {context_variables}")
#     print(f"Agent context variables: {task_manager.context_variables}")
    
#     # Test the tools directly
#     print("\n=== Testing ingest_documents tool ===")
#     try:
#         # Get the registered tools
#         tools = task_manager._tools
#         print(f"Registered tools: {[tool.name for tool in tools]}")
        
#         # Find the ingest_documents tool
#         ingest_tool = None
#         for tool in tools:
#             if tool.name == "ingest_documents":
#                 ingest_tool = tool
#                 break
        
#         if ingest_tool:
#             print(f"Found ingest_documents tool: {ingest_tool}")
#             # Call the tool function directly
#             result = ingest_tool.func()
#             print(f"Tool result: {result}")
#         else:
#             print("ingest_documents tool not found")
            
#     except Exception as e:
#         print(f"Error testing ingest_documents: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n=== Testing execute_query tool ===")
#     try:
#         # Find the execute_query tool
#         query_tool = None
#         for tool in tools:
#             if tool.name == "execute_query":
#                 query_tool = tool
#                 break
        
#         if query_tool:
#             print(f"Found execute_query tool: {query_tool}")
#             # Call the tool function directly
#             result = query_tool.func()
#             print(f"Tool result: {result}")
#         else:
#             print("execute_query tool not found")
            
#     except Exception as e:
#         print(f"Error testing execute_query: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     test_task_manager_individual()

context_variables = ContextVariables(
        data={
            "CompletedTaskCount": 0,
            "DocumentsToIngest": [
                Ingest(path_or_url="./test_parsed_docs/test_document.txt")
            ],
            "DocumentsIngested": [],
            "QueriesToRun": [
                Query(query_type="RAG_QUERY", query="ingest the document and then answer the question: What is the fiscal year 2024 financial summary?")
            ],
            "QueryResults": [],
        }
    )

task_manager.ingest_documents(context_variables)