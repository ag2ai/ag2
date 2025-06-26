# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os
import pickle
from typing import Any, Optional, Union

from .... import Agent, code_utils
from ....formatting_utils import colored
from ....import_utils import optional_import_block, require_optional_import
from ....llm_config import LLMConfig
from ...assistant_agent import ConversableAgent
from ..text_analyzer_agent import TextAnalyzerAgent
from .agent_capability import AgentCapability

with optional_import_block():
    import chromadb
    from chromadb.config import Settings


class CustomizedTeachability(AgentCapability):
    """Teachability uses a vector database to give an agent the ability to remember user teachings,
    where the user is any caller (human or not) sending messages to the teachable agent.
    Teachability is designed to be composable with other agent capabilities.
    To make any conversable agent teachable, instantiate both the agent and the Teachability class,
    then pass the agent to teachability.add_to_agent(agent).
    Note that teachable agents in a group chat must be given unique path_to_db_dir values.

    When adding Teachability to an agent, the following are modified:
    - The agent's system message is appended with a note about the agent's new ability.
    - A hook is added to the agent's `process_last_received_message` hookable method,
    and the hook potentially modifies the last of the received messages to include earlier teachings related to the message.
    Added teachings do not propagate into the stored message history.
    If new user teachings are detected, they are added to new memos in the vector database.
    """

    static_task_id_prefix = "ag2_customized_teachability_metadata"

    def __init__(
        self,
        verbosity: Optional[int] = 0,
        reset_db: Optional[bool] = False,
        path_to_db_dir: Optional[str] = "./tmp/teachable_agent_db",
        recall_threshold: Optional[float] = 1.5,
        max_num_retrievals: Optional[int] = 10,
        llm_config: Optional[Union[LLMConfig, dict[str, Any], bool]] = None,
    ):
        """Args:
        verbosity (Optional, int): # 0 (default) for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
        reset_db (Optional, bool): True to clear the DB before starting. Default False.
        path_to_db_dir (Optional, str): path to the directory where this particular agent's DB is stored. Default "./tmp/teachable_agent_db"
        recall_threshold (Optional, float): The maximum distance for retrieved memos, where 0.0 is exact match. Default 1.5. Larger values allow more (but less relevant) memos to be recalled.
        max_num_retrievals (Optional, int): The maximum number of memos to retrieve from the DB. Default 10.
        llm_config (LLMConfig or dict or False): llm inference configuration passed to TextAnalyzerAgent.
            If None, TextAnalyzerAgent uses llm_config from the teachable agent.
        """
        self.verbosity = verbosity
        self.path_to_db_dir = path_to_db_dir
        self.recall_threshold = recall_threshold
        self.max_num_retrievals = max_num_retrievals
        self.llm_config = llm_config

        self.analyzer = None
        self.teachable_agent = None

        # Create the memo store.
        self.memo_store = CustomizedMemoStore(self.verbosity, reset_db, self.path_to_db_dir)

    def add_to_agent(self, agent: ConversableAgent):
        """Adds teachability to the given agent."""
        self.teachable_agent = agent

        # Register a hook for processing the last message.
        agent.register_hook(hookable_method="process_last_received_message", hook=self.process_last_received_message)

        agent.register_reply(
            [Agent, None], self._customized_teachability_gen_reply
        )  # , position=self._register_reply_position)

        # Was an llm_config passed to the constructor?
        if self.llm_config is None:
            # No. Use the agent's llm_config.
            self.llm_config = agent.llm_config
        assert self.llm_config, "Teachability requires a valid llm_config."

        # Create the analyzer agent.
        self.analyzer = TextAnalyzerAgent(llm_config=self.llm_config)

        # Append extra info to the system message.
        agent.update_system_message(
            agent.system_message
            + "\nYou've been given the special ability to remember user tasks from prior conversations."
        )

    def process_last_received_message(self, text: Union[dict[str, Any], str]) -> list[dict[str, Any]]:
        """Appends any relevant memos to the message text, and stores any new tasks in new memos.
        Uses TextAnalyzerAgent to make decisions about memo storage and retrieval.
        """

        # Try to retrieve relevant memos from the DB.
        expanded_text = text
        expanded_text, nearest_memo = self._consider_memo_retrieval(text)
        # print("[test] process_last_received_message()\n")

        # Try to store any user teachings in new memos to be used in the future.
        task_id = self._consider_memo_storage(text, expanded_text, nearest_memo)

        # Return the (possibly) expanded message text.
        return f"{CustomizedTeachability.static_task_id_prefix}={task_id} {expanded_text}"

    def _customized_teachability_gen_reply(
        self,
        recipient: ConversableAgent,
        messages: Optional[list[dict[str, Any]]],
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
        if messages is None:
            return False, None

        # print(f"_customized_teachability_gen_reply() messages: {messages}")

        last_message = code_utils.content_str(messages[-1]["content"])
        # print(f"_customized_teachability_gen_reply() last_message: {last_message}")
        task_id, last_message = self._separate_taskid_from_message(last_message)
        # print(f"_customized_teachability_gen_reply() task_id: {task_id}\n    last_message: {last_message}")

        reply = self.teachable_agent.generate_oai_reply(sender=sender, messages=messages)
        # print(f"_customized_teachability_gen_reply() reply (type(reply)={type(reply)}): {reply}")
        if (reply[0]) and (task_id != -1):
            # Merge the new reply with the latest state

            new_context = reply[1]
            memo = self.memo_store._get_memo_by_id(task_id)
            if ("context" in memo) and (memo["context"] != ""):
                new_context = f"{memo['context']} \n# [Latest Change]:\n{new_context}"
                response = self._analyze(
                    new_context,
                    f"Please merge the above TEXT, prioritizing and fully respecting the content under '[Latest Change]', as a single plan (or a single solution) for the task (or the problem) below: {memo['detailed_task']}",
                )
                new_context = response

            self.memo_store.update_task_context(task_id, new_context)
        return reply

    def _separate_taskid_from_message(self, message) -> (int, str):
        task_id = -1
        st = str(message)
        if st.startswith(CustomizedTeachability.static_task_id_prefix):
            prefix_len = len(CustomizedTeachability.static_task_id_prefix)
            space_loc = st.find(" ")
            if (space_loc >= prefix_len + 2) and (len(st) > prefix_len and st[prefix_len] == "="):
                task_id_st = st[prefix_len + 1 : space_loc]
                try:
                    task_id = int(task_id_st)
                    st = st[space_loc:]
                except ValueError:
                    if self.verbosity >= 1:
                        print(colored(f"Could not parse int from {task_id_st}.", "red"))
        return (task_id, st)

    def _consider_memo_storage(self, comment: Union[dict[str, Any], str], expanded_text, nearest_memo: dict) -> int:
        """Decides whether to store something from one user comment in the DB."""

        # print(f'[test] _consider_memo_storage() comment: {comment}\n')

        task_id = -1

        if "task_id" in nearest_memo:
            # Extract the detailed task.
            detailed_task = self._analyze(
                expanded_text,
                "Briefly summarize this task from the TEXT and include all requirements and constraints if any.",
            )
            task_id = nearest_memo["task_id"]
            self.memo_store.update_task(
                task_id, nearest_memo["summarized_task"], detailed_task, nearest_memo["context"]
            )
        else:
            response = self._analyze(
                comment,
                "Does any part of the TEXT ask the agent to perform a task or solve a problem? Answer with just one word, yes or no.",
            )
            if "yes" in response.lower():
                # Extract the detailed task.
                detailed_task = self._analyze(
                    comment,
                    "Briefly summarize this task from the TEXT and include all requirements and constraints if any.",
                )
                # Summarize the task.
                summarized_task = self._analyze(
                    detailed_task,
                    "Summarize very briefly the task described in the TEXT. Leave out the detailed requirements and constraints.",
                )
                context = ""

                if self.verbosity >= 1:
                    print(colored("\nREMEMBER THIS TASK", "light_yellow"))
                task_id = self.memo_store.add_task(summarized_task, detailed_task, context)

        self.memo_store._save_memos()

        return task_id

    def _consider_memo_retrieval(self, comment: Union[dict[str, Any], str]):
        """Decides whether to retrieve memos from the DB, and add them to the chat context."""

        if self.verbosity >= 1:
            print(colored("\nLOOK FOR NEAREST MEMO", "light_yellow"))
        nearest_memo = self.memo_store.get_nearest_memo(comment, self.recall_threshold)
        # TODO: Add keyword-based rag
        if "detailed_task" in nearest_memo:
            comment = comment + self._concatenate_memo_texts([(nearest_memo["detailed_task"], nearest_memo["context"])])

        return comment, nearest_memo

    def _concatenate_memo_texts(self, memo_list: list) -> str:
        """Concatenates the memo texts into a single string for inclusion in the chat context."""
        memo_texts = ""
        if len(memo_list) > 0:
            info = "\n# Memories that might help\n"
            for memo in memo_list:
                detailed_task = memo[0]
                context = memo[1]
                info = info + "- " + detailed_task + "\n"
                info = info + "\n# Latest Output and Context that you should respect below\n" + context + "\n"
            if self.verbosity >= 1:
                print(colored("\nMEMOS APPENDED TO LAST MESSAGE...\n" + info + "\n", "light_yellow"))
            memo_texts = memo_texts + "\n" + info
        return memo_texts

    def _analyze(self, text_to_analyze: Union[dict[str, Any], str], analysis_instructions: Union[dict[str, Any], str]):
        """Asks TextAnalyzerAgent to analyze the given text according to specific instructions."""
        self.analyzer.reset()  # Clear the analyzer's list of messages.
        self.teachable_agent.send(
            recipient=self.analyzer, message=text_to_analyze, request_reply=False, silent=(self.verbosity < 2)
        )  # Put the message in the analyzer's list.
        self.teachable_agent.send(
            recipient=self.analyzer, message=analysis_instructions, request_reply=True, silent=(self.verbosity < 2)
        )  # Request the reply.
        return self.teachable_agent.last_message(self.analyzer)["content"]


@require_optional_import("chromadb", "teachable")
class CustomizedMemoStore:
    """Provides memory storage and retrieval for a teachable agent, using a vector database.
    Each DB entry (called a memo) is a pair of strings: a task text and its detailed constraints in text.
    Vector embeddings are currently supplied by Chroma's default Sentence Transformers.
    """

    def __init__(
        self,
        verbosity: Optional[int] = 0,
        reset: Optional[bool] = False,
        path_to_db_dir: Optional[str] = "./tmp/teachable_agent_db",
    ):
        """Args:
        - verbosity (Optional, int): 1 to print memory operations, 0 to omit them. 3+ to print memo lists.
        - reset (Optional, bool): True to clear the DB before starting. Default False.
        - path_to_db_dir (Optional, str): path to the directory where the DB is stored.
        """
        self.verbosity = verbosity
        self.path_to_db_dir = path_to_db_dir

        # Load or create the vector DB on disk.
        settings = Settings(
            anonymized_telemetry=False, allow_reset=True, is_persistent=True, persist_directory=path_to_db_dir
        )
        self.db_client = chromadb.Client(settings)
        self.vec_db = self.db_client.create_collection("memos", get_or_create=True)  # The collection is the DB.

        # Load or create the associated memo dict on disk.
        self.path_to_dict = os.path.join(path_to_db_dir, "uid_text_dict.pkl")
        self.uid_text_dict = {}
        self.last_memo_id = 0
        if (not reset) and os.path.exists(self.path_to_dict):
            print(colored("\nLOADING MEMORY FROM DISK", "light_green"))
            print(colored(f"    Location = {self.path_to_dict}", "light_green"))
            with open(self.path_to_dict, "rb") as f:
                self.uid_text_dict = pickle.load(f)
                self.last_memo_id = len(self.uid_text_dict)
                if self.verbosity >= 3:
                    self.list_memos()

        # Clear the DB if requested.
        if reset:
            self.reset_db()

    def list_memos(self):
        """Prints the contents of CustomizedMemoStore."""
        print(colored("LIST OF MEMOS", "light_green"))
        for uid, text_tuple in self.uid_text_dict.items():
            summarized_task, detailed_task, context = text_tuple
            print(
                colored(
                    f"  ID: {uid}\n    summarized_task: {summarized_task}\n    detailed_task: {detailed_task}\n    context: {context}",
                    "light_green",
                )
            )

    def _save_memos(self):
        """Saves self.uid_text_dict to disk."""
        with open(self.path_to_dict, "wb") as file:
            pickle.dump(self.uid_text_dict, file)

    def reset_db(self):
        """Forces immediate deletion of the DB's contents, in memory and on disk."""
        print(colored("\nCLEARING MEMORY", "light_green"))
        self.db_client.delete_collection("memos")
        self.vec_db = self.db_client.create_collection("memos")
        self.uid_text_dict = {}
        self._save_memos()

    def update_task(self, task_id, summarized_task: str, detailed_task: str, context: str):
        """Update a task in the vector DB."""
        st_tid = str(task_id)
        self.vec_db.add(documents=[summarized_task], ids=[st_tid])
        self.uid_text_dict[st_tid] = summarized_task, detailed_task, context
        if self.verbosity >= 1:
            print(
                colored(
                    f"\nTASK UPDATED IN VECTOR DATABASE:\n  ID\n    {task_id}\n  TASK\n    {summarized_task}\n  DETAIL\n    {detailed_task}\n  CONTEXT\n    {context}\n",
                    "light_yellow",
                )
            )
        if self.verbosity >= 3:
            self.list_memos()

    def update_task_context(self, task_id, context: str):
        """Update a task's context in the vector DB."""
        st_tid = str(task_id)
        if st_tid not in self.uid_text_dict:
            if self.verbosity >= 1:
                print(colored(f"\nTASK NOT FOUND IN uid_text_dict:\n  ID\n    {task_id}\n", "red"))
            return
        summarized_task, detailed_task, old_context = self.uid_text_dict[st_tid]
        self.uid_text_dict[st_tid] = summarized_task, detailed_task, context
        if self.verbosity >= 1:
            print(
                colored(
                    f"\nCONTEXT UPDATED IN uid_text_dict:\n  ID\n    {task_id}\n  TASK\n    {summarized_task}\n  DETAIL\n    {detailed_task}\n  CONTEXT\n    {context}\n",
                    "light_yellow",
                )
            )
        if self.verbosity >= 3:
            self.list_memos()

    def add_task(self, summarized_task: str, detailed_task: str, context: str):
        """Adds a task to the vector DB."""
        self.last_memo_id += 1
        self.vec_db.add(documents=[summarized_task], ids=[str(self.last_memo_id)])
        self.uid_text_dict[str(self.last_memo_id)] = summarized_task, detailed_task, context
        if self.verbosity >= 1:
            print(
                colored(
                    f"\nTASK ADDED TO VECTOR DATABASE:\n  ID\n    {self.last_memo_id}\n  TASK\n    {summarized_task}\n  DETAIL\n    {detailed_task}\n  CONTEXT\n    {context}\n",
                    "light_yellow",
                )
            )
        if self.verbosity >= 3:
            self.list_memos()
        return self.last_memo_id

    def _get_memo_by_id(self, task_id: int):
        st_tid = str(task_id)
        if st_tid not in self.uid_text_dict:
            if self.verbosity >= 1:
                print(colored(f"\nTASK NOT FOUND IN uid_text_dict:\n  ID\n    {task_id}\n", "red"))
            return None
        summarized_task, detailed_task, context = self.uid_text_dict[st_tid]
        if self.verbosity >= 1:
            print(
                colored(
                    f"\nGET MEMO BY ID IN uid_text_dict:\n  ID\n    {task_id}\n  TASK\n    {summarized_task}\n  DETAIL\n    {detailed_task}\n  CONTEXT\n    {context}\n",
                    "light_yellow",
                )
            )
        return {
            "task_id": task_id,
            "summarized_task": summarized_task,
            "detailed_task": detailed_task,
            "context": context,
        }

    def get_nearest_memo(self, query_text: str, recall_threshold: float):
        """Retrieves the nearest memo to the given query text."""
        results = self.vec_db.query(query_texts=[query_text], n_results=1)
        num_results = len(results["ids"][0])
        if num_results == 0:
            return {}
        uid, summarized_task, distance = results["ids"][0][0], results["documents"][0][0], results["distances"][0][0]
        if self.verbosity >= 1:
            print(
                colored(
                    f"\nTask RETRIEVED FROM VECTOR DATABASE:\n  Task\n    {summarized_task}\n  DISTANCE\n    {distance}\n  THRESHOLD\n    {recall_threshold}",
                    "light_yellow",
                )
            )
        if distance > recall_threshold:
            return {}
        summarized_task_2, detailed_task, context = self.uid_text_dict[uid]
        assert summarized_task == summarized_task_2
        if self.verbosity >= 1:
            print(
                colored(
                    f"\nTask RETRIEVED FROM VECTOR DATABASE:\n  Task\n    {summarized_task}\n  Detail\n    {detailed_task}\n  DISTANCE\n    {distance}",
                    "light_yellow",
                )
            )
        return {
            "task_id": uid,
            "summarized_task": summarized_task,
            "detailed_task": detailed_task,
            "distance": distance,
            "context": context,
        }
