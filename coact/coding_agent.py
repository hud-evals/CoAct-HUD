import asyncio
from typing import Any, Callable, Optional

from hud.env.environment import Environment
from hud.adapters.common.types import CustomAction

from autogen.llm_config import LLMConfig
from autogen.code_utils import PYTHON_VARIANTS
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

# Import the helper function from async_utils
from .async_utils import run_async_in_sync


CODER_SYSTEM_MESSAGE = """# Your role
- You are a programmer, you need to solve a task step-by-step given by the user. 
- You can write code in ```bash...``` code blocks for bash scripts, and ```python...``` code blocks for python code. 
- Your linux username is "user".
- If you want to use sudo, follow the format: "echo password | sudo -S [YOUR COMMANDS]" (no quotes for the word "password").

# Requirements
- You MUST verify the result before save the changes.
- When you write code, you must identify the language (whether it is python or bash) of the code.
- Wrap all your code in ONE code block. DO NOT let user save the code as a file and execute it for you.
- Do not include __main__ in your python code.
- When you modify a spreadsheet, **make sure every value is in the expected cell**.
- When importing a package, you need to check if the package has been installed. If not, you need to install it yourself.
- You need to print the progressive and final result.
- If you met execution error, you need to analyze the error message and try to fix the error.
"""

class TerminalProxyAgent(MultimodalConversableAgent):
    def __init__(
            self, 
        name: str, 
        env: Environment,
        llm_config: LLMConfig = False, 
        system_message: str = "",
        human_input_mode: str = "NEVER",
        code_execution_config = {},
        is_termination_msg: Optional[Callable[[dict[str, Any]], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        default_auto_reply: Optional[str] = None,
        description: Optional[str] = None,
        silent: bool = False
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
            silent=silent
        )
        self.env = env

    def run_code(self, code: str, lang: str = "python", **kwargs):
        exitcode = 1
        logs = ""
        image = None
        
        if lang in ["bash", "shell", "sh"]:
            # Use custom action for bash execution with proper CLA format
            action = CustomAction(
                action="bash" + code
            )
            obs, _, _, info = run_async_in_sync(self.env.step(action))
            
            # Extract results - check both info and obs for output
            # The environment might return results in different ways
            if info and isinstance(info, dict):
                if info.get("status") == "success":
                    exitcode = 0
                    logs = info.get("script_output", "")
                else:
                    exitcode = 1 if info.get("status") == "error" else 0
                    logs = info.get("script_output", info.get("error", info.get("message", "")))
            else:
                # Fallback: assume success if we got an observation
                exitcode = 0 if obs else 1
                logs = str(info) if info else "Code executed"
                
        elif lang in PYTHON_VARIANTS:
            # Use custom action for python execution with proper CLA format
            action = CustomAction(
                action="python" + code
            )
            obs, _, _, info = run_async_in_sync(self.env.step(action))
            
            # Extract results - check both info and obs for output
            # The environment might return results in different ways
            if info and isinstance(info, dict):
                if info.get("status") == "success":
                    exitcode = 0
                    logs = info.get("script_output", info.get("message", ""))
                else:
                    exitcode = 1 if info.get("status") == "error" else 0
                    logs = info.get("script_output", info.get("error", info.get("message", "")))
            else:
                # Fallback: assume success if we got an observation
                exitcode = 0 if obs else 1
                logs = str(info) if info else "Code executed"
                
        else:
            exitcode = -1
            logs = f"unknown language {lang}"
            
        # Get screenshot after code execution
        # if obs and obs.screenshot:
            # image = obs.screenshot
            
        return exitcode, logs, image
