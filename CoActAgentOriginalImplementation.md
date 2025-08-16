# CoAct Agent Original Implementation

## Overview
The CoAct agent is a multi-agent system that orchestrates between different specialized agents to complete desktop automation tasks. It uses a hierarchical approach with an orchestrator that delegates tasks to either a GUI agent or a coding agent.

## Architecture

### 1. Entry Point: `run_coact.py`
- Main script that sets up and runs the coact agent system
- Processes tasks from a test configuration file
- Uses multiprocessing to handle multiple tasks in parallel
- Key function: `process_task()` which:
  - Creates an `OrchestratorAgent` with system instructions
  - Creates an `OrchestratorUserProxyAgent` that manages the environment
  - Initiates a chat between them to solve the task

### 2. Core Components

#### `OrchestratorAgent` (in `operator_agent.py`)
- Main decision-making agent that analyzes tasks
- Has two tools available:
  - `call_gui_agent`: For UI interaction tasks
  - `call_coding_agent`: For file operations and programming tasks
- System message instructs it to:
  - Describe screenshots
  - Create detailed plans
  - Verify results
  - Prefer coding agent for file operations

#### `OrchestratorUserProxyAgent` (in `operator_agent.py`)
- Executes the orchestrator's decisions
- Manages the `DesktopEnv` environment
- Key methods:
  - `_call_gui_agent()`: Runs the CUA (Computer Use API) agent
  - `_call_coding_agent()`: Runs the terminal proxy agent
- Handles environment setup with AWS AMI configurations
- Tracks execution history and costs

### 3. Specialized Agents

#### GUI Agent (`cua_agent.py`)
- Uses OpenAI's Computer Use API (computer-use-preview model)
- Converts CUA actions to PyAutoGUI commands
- Key function: `run_cua()` which:
  - Takes screenshots
  - Sends them to the CUA API
  - Executes returned actions
  - Handles reasoning and message processing
- Action mapping: `_cua_to_pyautogui()` converts CUA format to PyAutoGUI

#### Coding Agent (`coding_agent.py`)
- `TerminalProxyAgent` class for executing code
- Supports bash and Python script execution
- Uses environment's controller methods:
  - `env.controller.run_bash_script()`
  - `env.controller.run_python_script()`
- Returns execution results and error messages

## Environment Integration

### DesktopEnv
- All agents interact with `DesktopEnv` from the `desktop_env` package
- Key environment methods used:
  - `env.reset()`: Initialize environment
  - `env.step()`: Execute actions
  - `env.controller.get_screenshot()`: Get current screen
  - `env.controller.run_bash_script()`: Execute bash commands
  - `env.controller.run_python_script()`: Execute Python code
  - `env.evaluate()`: Get task completion score
  - `env.close()`: Cleanup environment

### Action Flow
1. Orchestrator receives task and screenshot
2. Decides whether to use GUI or coding agent
3. Executes chosen agent's actions through environment
4. Captures results and screenshots
5. Continues until task completion or failure

## Key Features

### Multi-Step Planning
- Orchestrator creates detailed plans before execution
- Tracks requirements like file paths and names
- Verifies results after each major step

### Error Handling
- Retry logic for API calls
- Graceful handling of infeasible tasks
- Error capturing and reporting

### Cost Tracking
- Tracks token usage and costs for CUA API calls
- Logs costs per step and total costs

### History Management
- Saves chat histories, screenshots, and results
- Truncates history to manage API context limits
- Preserves call_id pairs for proper API communication

## Configuration
- Uses OAI_CONFIG_LIST for LLM configurations
- Supports multiple models (gpt-4o, o3, o4-mini, etc.)
- Configurable parameters:
  - Max steps for each agent type
  - Screen dimensions
  - Sleep duration after actions
  - AWS region and AMI selection

## Limitations
- Synchronous execution model
- Tight coupling with DesktopEnv
- Direct dependency on AWS for environment provisioning
- Fixed action mapping to PyAutoGUI
