# # from autogen import ConversableAgent, LLMConfig
# # from autogen.tools.apply_patch_tool import ApplyPatchTool, WorkspaceEditor
# # import os
# # from pathlib import Path
# # from dotenv import load_dotenv

# # load_dotenv()

# # # Set your desired workspace directory
# # workspace_dir = Path("./my_project")  # or any path you want
# # workspace_dir.mkdir(exist_ok=True)

# # # Configure LLM
# # llm_config = LLMConfig(
# #     config_list={
# #         "api_type": "responses",
# #         "model": "gpt-5.1",
# #         "api_key": os.getenv("OPENAI_API_KEY"),
# #         "built_in_tools": ["apply_patch"],
# #     }
# # )

# # # Create workspace editor and tool
# # # editor = WorkspaceEditor(workspace_dir=str(workspace_dir))
# # # patch_tool = ApplyPatchTool(editor=editor)

# # # Create agent
# # coding_agent = ConversableAgent(
# #     name="coding_assistant",
# #     llm_config=llm_config,
# #     system_message="""You are a helpful coding assistant...""",
# # )

# # # Register the tool with the agent
# # # patch_tool.register_tool(coding_agent)

# # # Create a new project structure
# # buggy_code = """
# # def calculate_average(numbers):
# #     total = 0
# #     for num in numbers:
# #         total += num
# #     return total / len(numbers)  # Bug: doesn't handle empty list

# # def divide(a, b):
# #     return a / b  # Bug: doesn't handle division by zero
# # """

# # # Write the buggy file manually (or use the agent)
# # Path("calculator/main.py").write_text(buggy_code)

# # # Now ask the agent to fix the bugs
# # result = coding_agent.initiate_chat(
# #     recipient=coding_agent,
# #     message="""
# #     I have a file called buggy_math.py with some bugs:
# #     1. The calculate_average function doesn't handle empty lists
# #     2. The divide function doesn't handle division by zero
# #     Please fix both bugs by adding proper error handling.
# #     """,
# #     max_turns=2,
# #     clear_history=True,
# # )

# # # print("Chat Summary:")
# # # print(result.summary)

# # print("Chat Summary:")
# # print(result.summary)

# # import os

# # from autogen import ConversableAgent, LLMConfig

# # # Configure LLM with workspace_dir - ApplyPatchTool will be auto-registered
# # llm_config = LLMConfig(
# #     config_list={
# #         "api_type": "responses",
# #         "model": "gpt-5.1",
# #         "api_key": os.getenv("OPENAI_API_KEY"),
# #         "built_in_tools": ["apply_patch"],
# #     },
# #     workspace_dir="./my_project_folder",  # NEW: Just specify workspace_dir here!
# # )

# # # Create agent - no need to manually create editor or patch_tool
# # coding_agent = ConversableAgent(
# #     name="coding_assistant",
# #     llm_config=llm_config,
# #     system_message="""You are a helpful coding assistant...""",
# # )

# # # Tool is automatically registered! Just use it:
# # result = coding_agent.initiate_chat(
# #     recipient=coding_agent,
# #     message="Create app.py in the workspace directory",
# #     max_turns=2,
# # )

# import os

# from autogen import ConversableAgent, LLMConfig

# # Configure LLM with workspace_dir and allowed_paths
# llm_config = LLMConfig(
#     config_list={
#         "api_type": "responses",
#         "model": "gpt-5.1",
#         "api_key": os.getenv("OPENAI_API_KEY"),
#         "built_in_tools": ["apply_patch"],
#     },
#     workspace_dir="./my_project_folder",
#     allowed_paths=["src/**", "*.py", "tests/**"],  # Only allow operations in these paths
# )

# # Create agent - no need to manually create editor or patch_tool
# coding_agent = ConversableAgent(
#     name="coding_assistant",
#     llm_config=llm_config,
#     system_message="""You are a helpful coding assistant...""",
# )

# # Test 1: Try to create file in allowed path (should work)
# print("=== Test 1: Allowed path ===")
# result1 = coding_agent.initiate_chat(
#     recipient=coding_agent,
#     message="Create src/main.py with a simple hello world function",
#     max_turns=2,
# )

# # Test 2: Try to create file in NOT allowed path (should fail with clear error)
# # print("\n=== Test 2: NOT allowed path ===")
# result2 = coding_agent.initiate_chat(
#     recipient=coding_agent,
#     message="Create config/settings.json in the config directory (outside of src/ and tests/)",
#     max_turns=2,
# )

# # Test 3: Try to update file in NOT allowed path
# # print("\n=== Test 3: Update file in NOT allowed path ===")
# result3 = coding_agent.initiate_chat(
#     recipient=coding_agent,
#     message="Create a file called root_file.py in the root of the workspace (not in src/ or tests/)",
#     max_turns=2,
# )
