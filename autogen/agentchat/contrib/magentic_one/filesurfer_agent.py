from typing import Any, Callable, Dict, List, Optional, Union
from typing_extensions import Literal
from autogen.agentchat import ConversableAgent
from .markdown_browser import RequestsMarkdownBrowser  # type: ignore
from . import file_tools
from .file_tools import FILE_TOOL_SCHEMA


class FileSurferAgent(ConversableAgent):
    """An agent that can navigate and read local files using a text-based browser."""

    DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI Assistant that can navigate and read local files."
    
    def __init__(
        self,
        name: str = "file_surfer",
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        browser: Optional[RequestsMarkdownBrowser] = None,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
        )

        # Initialize browser
        self._browser = browser or RequestsMarkdownBrowser(viewport_size=1024 * 5, downloads_folder="coding")

        # Register file navigation functions
        self.register_function({
            "open_local_file": lambda path: file_tools.open_local_file(self._browser, path),
            "page_up": lambda: file_tools.page_up(self._browser),
            "page_down": lambda: file_tools.page_down(self._browser),
            "find_on_page": lambda search_string: file_tools.find_on_page(self._browser, search_string),
            "find_next": lambda: file_tools.find_next(self._browser),
            "get_browser_state": lambda: file_tools.get_browser_state(self._browser)
        })

        # Register function schemas for LLM
        if self.llm_config:
            
            self.update_tool_signature(FILE_TOOL_SCHEMA[0], is_remove=False)
            self.update_tool_signature(FILE_TOOL_SCHEMA[1], is_remove=False) 
            self.update_tool_signature(FILE_TOOL_SCHEMA[2], is_remove=False)
            self.update_tool_signature(FILE_TOOL_SCHEMA[3], is_remove=False)
            self.update_tool_signature(FILE_TOOL_SCHEMA[4], is_remove=False)
            self.update_tool_signature(FILE_TOOL_SCHEMA[5], is_remove=False)

    @property
    def browser(self) -> RequestsMarkdownBrowser:
        """Get the browser instance."""
        return self._browser
