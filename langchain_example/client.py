# Create server parameters for stdio connection
import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

model = ChatOpenAI(model="gpt-4o")

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["math_server.py"],
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Print original MCP tools
            print("Original MCP tools:")
            tools = await session.list_tools()
            print(tools)
            print("*" * 100)

            # Get tools
            tools = await load_mcp_tools(session)
            print(f"Langchain Tools: {tools}")
            print("*" * 100)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            # agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            # return agent_response


# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)
