# CoAct-HUD

This is an implementation of the CoAct agent for the hud platform
A multi-agent system that orchestrates GUI and coding agents to complete computer-use tasks using the HUD SDK.


## Overview

The CoAct agent system consists of:
- **Orchestrator Agent**: Plans and coordinates task execution
- **GUI Agent (CUA)**: Performs GUI interactions (clicking, typing, scrolling)
- **Coding Agent**: Executes Python and Bash scripts for programmatic tasks

## Installation

1. Install dependencies:
```bash
pip install hud-python openai>=1.30
```

2. Set up OpenAI API configuration in `coact/OAI_CONFIG_LIST`:
```json
[
    {
        "api_key": "your-api-key",
        "model": "o3"
    },
    {
        "api_key": "your-api-key", 
        "model": "o4-mini"
    }
]
```

## Running Tasksets

### Quick Start

Run the OSWorld-Verified taskset with the simple runner:
```bash
cd coact_agent
python run_osworld.py
```

Or specify options:
```bash
python run_osworld.py --parallel --verbose
```

### Run a single task

You can target a single task from the taskset without modifying code:
```bash
# By task id (preferred when you know the id)
python coact_taskset_runner.py --taskset OSWorld-Verified --task-id TASK_ID

# By index in the taskset (0-based)
python coact_taskset_runner.py --taskset OSWorld-Verified --task-index 0

# Or limit to the first N tasks
python coact_taskset_runner.py --taskset OSWorld-Verified --limit 1
```
`run_osworld.py` forwards all CLI args to the runner, so these also work:
```bash
python run_osworld.py --task-id TASK_ID --verbose
```

### Using the Full Runner

For more control, use the taskset runner directly:
```bash
python coact_taskset_runner.py --taskset OSWorld-Verified --name "My Evaluation"
```

### Full Command Options

```bash
python coact_taskset_runner.py \
    --taskset OSWorld-Verified \
    --name "CoAct OSWorld Evaluation" \
    --model o4-mini \
    --orchestrator-model o3 \
    --config coact/OAI_CONFIG_LIST \
    --orchestrator-max-steps 15 \
    --cua-max-steps 25 \
    --coding-max-steps 20 \
    --save-dir ./coact_results \
    --parallel \
    --max-concurrent 5 \
    --verbose
```

### Parameters

- `--taskset`: HUD taskset ID to evaluate (e.g., "OSWorld-Verified")
- `--name`: Name for the evaluation job
- `--model`: Model for coding/CUA agents (default: o4-mini)
- `--orchestrator-model`: Model for orchestrator (default: o3)
- `--config`: Path to OpenAI config file (default: `coact/OAI_CONFIG_LIST`)
- `--orchestrator-max-steps`: Max orchestrator conversation turns
- `--cua-max-steps`: Max GUI agent steps per task
- `--coding-max-steps`: Max coding agent steps per task
- `--save-dir`: Directory to save results and logs
- `--parallel`: Run tasks in parallel (faster but uses more resources)
- `--max-concurrent`: Max concurrent tasks when parallel
- `--verbose`: Print detailed progress

### Single Task Options

- `--task-id`: Run a single task by its ID (preferred)
- `--task-index`: Run a single task by 0-based index within the taskset
- `--limit`: Limit number of tasks to run from the start of the filtered list

## Architecture

### Action Flow

1. **Task Loading**: Loads tasks from HUD taskset
2. **Environment Creation**: Creates HUD environment for each task
3. **Orchestration**: Orchestrator agent plans approach
4. **Execution**: Delegates to GUI or coding agents as needed
5. **Evaluation**: HUD SDK evaluates task completion

### CLA Actions

All actions follow the Common Language Actions (CLA) specification:

#### GUI Actions
- `ClickAction`: Mouse clicks with point and button
- `TypeAction`: Text input
- `ScrollAction`: Scrolling with direction
- `DragAction`: Drag operations with path
- `PressAction`: Keyboard shortcuts
- `MoveAction`: Mouse movement
- `WaitAction`: Delays

#### Custom Actions
- `CustomAction(action="bash")`: Execute bash scripts
- `CustomAction(action="python")`: Execute Python code
- `CustomAction(action="FAIL")`: Mark task as infeasible
- `ResponseAction`: Final text response

Screenshots are fetched explicitly via `ScreenshotFetch` to ensure portability across HUD environments.

## Results

Results are saved in the specified save directory with structure:
```
coact_results/
├── {job_name}/
│   ├── {task_id}/
│   │   ├── initial_screenshot.png
│   │   ├── chat_history.json
│   │   ├── result.txt (reward score)
│   │   ├── cua_output_*/  (GUI agent logs)
│   │   └── coding_output_*/ (Coding agent logs)
```

## Supported Tasksets

- **OSWorld-Verified**: Web and desktop automation tasks
- Any HUD-compatible taskset can be evaluated

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure HUD SDK is installed and in Python path
2. **API Key Issues**: Check OAI_CONFIG_LIST has valid API keys
3. **Environment Errors**: Verify HUD environment servers are running
4. **Memory Issues**: Reduce `--max-concurrent` for parallel runs

### Debugging

Enable verbose mode for detailed logs:
```bash
python coact_taskset_runner.py --taskset OSWorld-Verified --verbose
```

Check individual task logs in the save directory for specific errors.

## Performance Tips

1. **Parallel Execution**: Use `--parallel` for faster evaluation
2. **Adjust Steps**: Reduce max steps if tasks complete quickly
3. **Model Selection**: Use faster models for simple tasks
4. **Resource Management**: Monitor memory/CPU with parallel runs

## Development

### Adding Custom Actions

To add custom actions, modify the adapter in `coact/coact_adapter.py`:

```python
elif action_type == "your_custom_action":
    converted_action = CustomAction(
        action="your_action",
        args={"key": "value"}
    )
```

### Extending Agents

New agent types can be added by:
1. Creating agent class inheriting from `MultimodalConversableAgent`
2. Registering with orchestrator in `operator_agent.py`
3. Adding tool definition for orchestrator to call

## License

See LICENSE file in the root directory.
