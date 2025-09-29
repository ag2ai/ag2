import asyncio
import json
import os
from datetime import datetime

# ✨ NEW: Add Union and Optional imports
from typing import Any, Literal

import autogen
from autogen import UserProxyAgent

today_date = datetime.now()
from collections.abc import Awaitable, Callable

from dependencies import get_websocket_manager
from helpers import async_to_sync
from managers import prompts

from autogen import LLMConfig


class CustomGroupChatManager(autogen.GroupChatManager):
    def __init__(
        self,
        groupchat: Any,
        llm_config: Any,
        chat_id: str | None = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        queue: asyncio.Queue[Any] | None = None,
    ) -> None:
        super().__init__(groupchat=groupchat, llm_config=llm_config, human_input_mode=human_input_mode)
        self.queue = queue
        self.chat_id = chat_id

    async def send_websocket(self, message: Any) -> None:
        ws_manager = await get_websocket_manager()
        await ws_manager.send_chat_message(session_id=self.chat_id, content=message, source="AgentChat")
        return

    def _print_received_message(
        self, message: dict[str, Any] | str, sender: "autogen.Agent", skip_head: bool = True
    ) -> Any:
        super()._print_received_message(message, sender)
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        db_content = {"role": "user", "name": sender.name, "content": content}
        db_content["raw_message"] = message
        async_to_sync(self.send_websocket(db_content))


class CustomUserProxyAgent(UserProxyAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def set_input_function(self, input_function: Callable[[str], Awaitable[str]]) -> Callable[[str], Awaitable[str]]:
        self.input_function = input_function
        return self.input_function

    async def a_get_human_input(self, prompt: str) -> str:
        print("🔁 Custom logic before input...")

        prompt = json.dumps({"agent": self.name, "prompt": prompt})
        return await self.input_function(prompt)


class AgentChat:
    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id
        self.cancellation_token = None
        self.llm_config = LLMConfig(api_type="openai", model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

        self.agents = self.get_agents()
        self.get_chat_manager()

    def set_input_function(self, input_function: Any) -> Any:
        self.input_function = input_function
        return self.input_function

    def _set_cancellation_token(self, cancellation_token: Any) -> None:
        self.cancellation_token = cancellation_token

    def _check_cancellation(self) -> None:
        """Check if operations should be cancelled"""
        if self.cancellation_token and self.cancellation_token.is_cancelled():
            raise asyncio.CancelledError(f"OMNI operation cancelled for session {self.chat_id}")

    def get_planner(self) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="planner_agent",
            llm_config=self.llm_config,
            system_message=prompts.planner_agent.format(chart_location=os.getcwd()),
        )

    def get_code_writer(self) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="code_writer_agent", llm_config=self.llm_config, system_message=prompts.code_writer
        )

    def get_code_executor(self) -> UserProxyAgent:
        return UserProxyAgent(
            name="code_executor_agent",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": os.getcwd(),
                "use_docker": False,
            },
            llm_config=self.llm_config,
        )

    def get_code_debugger(self) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="code_debugger_agent",
            llm_config=self.llm_config,
            max_consecutive_auto_reply=5,
            system_message=prompts.debugger.format(chart_location=os.getcwd()),
        )

    def get_agent_aligner(self) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="agent_aligner",
            llm_config=self.llm_config,
            description="This agent will align the whole process and will share which agent will work next. ",
            system_message=prompts.agent_aligner,
        )

    def get_user_acceptance(self) -> CustomUserProxyAgent:
        return CustomUserProxyAgent(
            name="get_user_acceptance",
            llm_config=self.llm_config,
            human_input_mode="ALWAYS",
            code_execution_config=False,
        )

    def get_process_completion(self) -> autogen.AssistantAgent:
        return autogen.AssistantAgent(
            name="process_completion_agent",
            llm_config=self.llm_config,
            system_message=prompts.process_completion,
        )

    def custom_speaker_selection_func(self, last_speaker: Any, groupchat: Any) -> Any | None:
        try:
            self._check_cancellation()

            messages = groupchat.messages

            if len(messages) == 1 or last_speaker.name == "chat_manager":
                return groupchat.agent_by_name("agent_aligner")
            if last_speaker.name == "agent_aligner":
                if "planner_agent" in messages[-1]["content"]:
                    return groupchat.agent_by_name("planner_agent")
                elif "code_writer_agent" in messages[-1]["content"]:
                    return groupchat.agent_by_name("code_writer_agent")
                elif "code_executor_agent" in messages[-1]["content"]:
                    return groupchat.agent_by_name("code_executor_agent")
                elif "process_completion_agent" in messages[-1]["content"]:
                    return groupchat.agent_by_name("process_completion_agent")

            if last_speaker.name in ["planner_agent", "code_debugger_agent"]:
                return groupchat.agent_by_name("agent_aligner")
            if last_speaker.name == "code_writer_agent":
                return groupchat.agent_by_name("code_executor_agent")
            if last_speaker.name == "code_executor_agent":
                if "exitcode: 0" in messages[-1]["content"]:
                    return groupchat.agent_by_name("process_completion_agent")
                else:
                    return groupchat.agent_by_name("code_debugger_agent")

            if last_speaker.name == "code_debugger_agent":
                return groupchat.agent_by_name("code_writer_agent")

            return None
        except asyncio.CancelledError:
            raise

    def get_agents(self) -> list[Any]:
        self.planner = self.get_planner()
        self.agent_aligner = self.get_agent_aligner()
        self.code_writer = self.get_code_writer()
        self.code_executor = self.get_code_executor()
        self.code_debugger = self.get_code_debugger()
        self.process_completion = self.get_process_completion()
        return [
            self.planner,
            self.agent_aligner,
            self.code_writer,
            self.code_executor,
            self.code_debugger,
            self.process_completion,
        ]

    def get_chat_manager(self) -> tuple[CustomGroupChatManager, autogen.GroupChat]:
        self.groupchat = autogen.GroupChat(
            agents=self.agents,
            messages=[],
            speaker_selection_method=self.custom_speaker_selection_func,
            max_round=500,
        )
        self.manager = CustomGroupChatManager(
            groupchat=self.groupchat, llm_config=self.llm_config, human_input_mode="ALWAYS", chat_id=self.chat_id
        )
        return self.manager, self.groupchat
