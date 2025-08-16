# HUD SDK Implementation Guide

## Overview
The HUD SDK provides a standardized interface for creating and interacting with various environments (browser, desktop, custom) through agents. It uses an async/await pattern and a common action language (CLA) for cross-environment compatibility.

## Core Concepts

### 1. Environment Creation and Management
```python
# Load a task
taskset = await load_taskset("OSWorld-Verified-XLang", system_prompt=SYSTEM_PROMPT)

# Create environment from task
env = await gym.make(test)

# Environment lifecycle
obs, _ = await env.reset()  # Initialize
obs, reward, terminated, info = await env.step(action)  # Execute action
result = await env.evaluate()  # Check completion
await env.close()  # Cleanup
```

### 2. Agent Structure (`claude.py`)
- **Async Pattern**: All methods use async/await
- **Client Management**: Uses AsyncAnthropic client
- **Adapter Pattern**: Uses adapter to convert actions
- **Key Method**: `fetch_response(observation) -> tuple[list[Any], bool]`
  - Returns: (actions, done_flag)
  - Handles multimodal input (text + screenshot)

### 3. Action Adaptation (`adapter.py`)
The ClaudeAdapter converts agent-specific actions to Common Language Actions (CLA):

#### Supported CLA Actions:
- **ClickAction**: Mouse clicks (left, right, middle, double, triple)
- **TypeAction**: Text input
- **MoveAction**: Mouse movement
- **DragAction**: Click and drag
- **ScrollAction**: Scroll operations
- **PressAction**: Keyboard shortcuts
- **WaitAction**: Delays
- **ScreenshotFetch**: Request screenshot
- **PositionFetch**: Get cursor position
- **ResponseAction**: Text responses
- **CustomAction**: Environment-specific actions

#### Adapter Features:
- Maintains action memory for context-dependent actions
- Maps agent-specific formats to CLA
- Validates action parameters
- Preserves reasoning and logs

### 4. Custom Actions and Screenshot
For environment-specific operations:
```python
# Task infeasible
{"type": "custom", "action": "FAIL"}

# Custom bash execution (proposed)
{"type": "custom", "action": "bash", "code": "ls -la"}

# Custom python execution (proposed)
{"type": "custom", "action": "python", "code": "print('Hello')"}
```

For screenshots, explicitly request one with:
```python
from hud.adapters.common.types import ScreenshotFetch
obs, *_ = await env.step(ScreenshotFetch())
```

## Key Differences from CoAct Implementation

### 1. Async vs Sync
- **HUD SDK**: Fully async with await/async patterns
- **CoAct**: Synchronous execution model

### 2. Environment Interface
- **HUD SDK**: 
  - `gym.make()` for environment creation
  - `env.step(action)` for all actions
  - Standardized observation format
- **CoAct**: 
  - Direct DesktopEnv instantiation
  - Separate methods for different action types
  - Custom controller methods

### 3. Action Format
- **HUD SDK**: 
  - Single action format through adapter
  - All actions go through `env.step()`
  - CLA provides cross-environment compatibility
- **CoAct**: 
  - PyAutoGUI commands for GUI
  - Direct controller calls for code execution
  - Environment-specific action handling

### 4. Agent Architecture
- **HUD SDK**: 
  - Single agent with adapter
  - Clean separation of concerns
  - Stateless action conversion
- **CoAct**: 
  - Multiple specialized agents
  - Complex orchestration logic
  - Stateful interaction management

## Requirements for HUD-SDK Compatible Agent

### 1. Async Implementation
- Convert all methods to async/await
- Use AsyncAnthropic or similar async clients
- Handle async environment lifecycle

### 2. Adapter Creation
- Implement adapter to convert agent actions to CLA
- Map all action types to appropriate CLA actions
- Handle custom actions for code execution

### 3. Environment Integration
- Replace DesktopEnv with HUD env
- Use `env.step()` for all actions
- Follow standard observation/action flow

### 4. Action Handling
- Convert PyAutoGUI commands to CLA actions
- Map code execution to custom actions
- Preserve reasoning and logging

### 5. Error Handling
- Handle async exceptions
- Manage environment lifecycle properly
- Implement graceful failure modes
