import asyncio
import tempfile

from dotenv import load_dotenv

from autogen import ConversableAgent
from autogen.coding import DockerCommandLineCodeExecutor
from autogen.llm_config.config import LLMConfig
from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper, setup_instrumentation

load_dotenv()

# Create a temporary directory to store the code files.
temp_dir = tempfile.TemporaryDirectory()

# Create a Docker command line code executor.
executor = DockerCommandLineCodeExecutor(
    image="python:3.12-slim",  # Execute code using the given docker image name.
    timeout=10,  # Timeout for each code execution in seconds.
    work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
    # (Optional): forward any extra kwargs directly to docker.containers.create().
    container_create_kwargs={
        # Example: pass an env-var and request GPU 0
        "environment": {"TOKEN": "<your-token>"},
        # "device_requests": [docker.types.DeviceRequest(device_ids=["0"], capabilities=[["gpu"]])],
    },
)

# Create an agent with code executor configuration that uses docker.
code_executor_agent_using_docker = ConversableAgent(
    "code_executor_agent_docker",
    llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={"executor": executor},  # Use the docker command line code executor.
    human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
)

# When the code executor is no longer used, stop it to release the resources.
# executor.stop()

# The code writer agent's system message is to instruct the LLM on how to use
# the code executor in the code executor agent.
code_writer_system_message = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply 'TERMINATE' in the end when everything is done.
"""

# Configure the LLM
llm_config = LLMConfig(
    config_list={
        "api_type": "openai",
        "model": "gpt-5-nano",
    }
)

# Create an agent with code executor configuration.
code_executor_agent = ConversableAgent(
    "code_executor_agent",
    llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={"executor": executor},  # Use the local command line code executor.
    human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
)

code_writer_agent = ConversableAgent(
    "code_writer_agent",
    system_message=code_writer_system_message,
    llm_config=llm_config,
    code_execution_config=False,  # Turn off code execution for this agent.
)


async def run_code_execution(prompt: str) -> str:
    # Start the conversation from the code writer agent
    chat_result = code_writer_agent.initiate_chat(
        code_executor_agent,
        message=prompt,
    )

    return chat_result


if __name__ == "__main__":
    tracer = setup_instrumentation("local-code-execution", "http://127.0.0.1:14317")
    instrument_llm_wrapper(tracer)
    instrument_agent(code_executor_agent, tracer)
    instrument_agent(code_writer_agent, tracer)

    code = asyncio.run(run_code_execution("Write Python code to calculate the 14th Fibonacci number."))

    print(code)
