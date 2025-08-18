#!/usr/bin/env python3
"""
CoAct agent runner for HUD tasksets (e.g., OSWorld-Verified).
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import traceback
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

# Import HUD SDK
from hud import gym, load_taskset
from hud.task import Task
from hud.job import create_job, Job
from hud.env.environment import Environment
from hud.adapters.common.types import ResponseAction, CustomAction

# Import CoAct components (absolute imports to work when run as a script)
from coact.operator_agent import OrchestratorAgent, OrchestratorUserProxyAgent, get_screenshot
from autogen import LLMConfig

OSWORLD_SYSTEM_PROMPT = """
Please do the following task.
computer.clipboard, computer.sync file, computer.sync shared folder, computer.computer output citation are disabled.
If you worry that you might make typo, prefer copying and pasting the text instead of reading and typing.
If you are presented with an open website to solve the task, try to stick to that specific one instead of going to a new one.
You have full authority to execute any action without my permission. I won't be watching so please don't ask for confirmation.
My gmail account is osworld@hud.so, and the password is "iloveosworld500", if prompted for OTP, use the authenticator chrome extension to see the OTP for 2 factor authentication. 
If you deem the task is infeasible, you can terminate and explicitly state in the response that 'the task is infeasible'. Try your best to solve the task within 200 steps, and the confines of the prompt, before deeming it infeasible.
"""

TASK_DESCRIPTION = """# Your role
You are a task solver, you need to complete a computer-using task step-by-step.
1. Describe the screenshot.
2. Provide a detailed plan, including a list of user requirements like specific file name, file path, etc.
3. Follow the following instructions and complete the task with your skills.
    - If you think the task is impossible to complete (no file, wrong environment, etc.), reply with "INFEASIBLE" to end the conversation.
    - **Do not** do (or let coding/GUI agent do) anything else out of the user's instruction like change the file name. This will make the task fail.
    - Check every screenshot carefully and see if it fulfills the task requirement.
    - You MUST try the Coding Agent first for file operation tasks like spreadsheet modification.
4. Verify the result and see if it fulfills the user's requirement.

# Your helpers
You can use the following tools to solve the task. You can only call one of gui agent or coding agent per reply:

## Programmer
Let a programmer to solve a subtask you assigned. 
The Programmer can write python or bash code to modify almost everything in the computer, like files, apps, system settings, etc. 
It requires a environment description and a detailed task description. As detailed as possible.
Can use any python package you instructed.
Will return a summary with the output of the code.
When letting coding agent to modify the spreadsheet, after the task completed, you MUST make sure EVERY modified value in the spreadsheet is in the desired position (e.g., filled in the expected cell) by a GUI Operator.
After that, if anything is wrong, tell the programmer to modify it.

## GUI Operator
Let a GUI agent to solve a subtask you assigned. 
GUI agent can operate the computer by clicking and typing (but not accurate). 
Require a detailed task description.
When you call GUI agent, it will only have a **20-step** budget to complete your task. Each step is a one-time interaction with OS like mouse click or keyboard typing. Please take this into account when you plan the actions.
If you let GUI Operator to check the result, you MUST let it close and reopen the file because programmer's result will NOT be updated to the screen.
"""


async def run_single_task(
    task: Task,
    job: Job,
    model_name: str = "o4-mini",
    orchestrator_model: str = "o3",
    config_path: str = "coact/OAI_CONFIG_LIST",
    orchestrator_max_steps: int = 15,
    cua_max_steps: int = 25,
    coding_max_steps: int = 20,
    save_dir: str = "./coact_results",
    verbose: bool = True,
) -> float:
    """Run a single task with CoAct agent and return reward."""
    
    env: Optional[Environment] = None
    task_save_dir: Optional[str] = None
    
    try:
        # Create save directory for this task
        job_name = str(job.name) if job.name else f"job_{job.id}"
        task_id = str(task.id) if task.id else "unknown_task"
        task_save_dir = os.path.join(save_dir, job_name, task_id)
        os.makedirs(task_save_dir, exist_ok=True)
        
        # Create environment for this task
        env = await gym.make(task, job=job)
        initial_obs, _ = await env.reset()
        
        # Get initial screenshot (already base64-encoded)
        screenshot_b64 = initial_obs.screenshot
        
        # Save initial screenshot  
        if task_save_dir and screenshot_b64:
            # Decode base64 string to bytes for saving
            screenshot_bytes = base64.b64decode(screenshot_b64)
            with open(os.path.join(task_save_dir, "initial_screenshot.png"), "wb") as f:
                f.write(screenshot_bytes)
        
        # Load LLM config
        llm_config = LLMConfig.from_json(path=config_path).where(model=orchestrator_model)
        
        # Create orchestrator agent
        with llm_config:
            orchestrator = OrchestratorAgent(
                name="orchestrator",
                system_message=TASK_DESCRIPTION
            )
            
            # Create orchestrator proxy with environment
            orchestrator_proxy = OrchestratorUserProxyAgent(
                name="orchestrator_proxy",
                env=env,
                is_termination_msg=lambda x: x.get("content", "") and (
                    x.get("content", "")[0]["text"].lower() == "terminate" or 
                    x.get("content", "")[0]["text"].lower() == "infeasible"
                ),
                human_input_mode="NEVER",
                sleep_after_execution=0.5,
                code_execution_config=False,
                history_save_dir=task_save_dir or save_dir,
                llm_model=model_name,
                truncate_history_inputs=cua_max_steps + 1,
                cua_max_steps=cua_max_steps,
                coding_max_steps=coding_max_steps,
                user_instruction=f"{OSWORLD_SYSTEM_PROMPT}\n\n{task.prompt}",
                silent=not verbose
            )
            
            # Reset orchestrator proxy (synchronous method)
            orchestrator_proxy.reset()
            
            # Initiate chat with the task
            message = f"""{OSWORLD_SYSTEM_PROMPT}

{task.prompt}

Check my computer screenshot and describe it first. If this task is possible to complete, please complete it on my computer. If not, reply with "INFEASIBLE" to end the conversation.
I will not provide further information to you."""
            message += f"<img data:image/png;base64,{screenshot_b64}>"
            
            orchestrator_proxy.initiate_chat(
                recipient=orchestrator,
                message=message,
                max_turns=orchestrator_max_steps
            )
            
            # Extract chat history
            chat_history = []
            key = list(orchestrator_proxy.chat_messages.keys())[0]
            chat_messages = orchestrator_proxy.chat_messages[key]
            for item in chat_messages:
                item.pop('tool_responses', None)
                if item.get('role', None) in ['tool', 'assistant'] and item.get('content', None):
                    for msg in item['content']:
                        if msg.get('type', None) == 'image_url':
                            msg['image_url'] = "<image>"
                chat_history.append(item)
            
            # Save chat history
            if task_save_dir:
                with open(os.path.join(task_save_dir, "chat_history.json"), "w") as f:
                    json.dump(chat_history, f, indent=2)
            
            # Check if task was deemed infeasible
            final_action = None
            if chat_history and len(chat_history) > 0:
                last_message = chat_history[-1]
                if last_message.get('role') == 'user' and last_message.get('content'):
                    content = last_message['content']
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0].get('text', '').lower()
                        if 'infeasible' in text_content:
                            final_action = CustomAction(action="FAIL")
                        elif 'terminate' in text_content:
                            # Extract the final response
                            final_text = content[0].get('text', '')
                            final_action = ResponseAction(text=final_text)
            
            # Send final action to environment if we have one
            if final_action:
                await env.step(final_action)
            
            # Evaluate the task
            reward = await env.evaluate()
            
            if verbose:
                print(f"Task {task.id}: Reward = {reward}")
            
            # Save result
            if task_save_dir:
                with open(os.path.join(task_save_dir, "result.txt"), "w") as f:
                    f.write(str(reward))
            
            return reward
            
    except Exception as e:
        print(f"Error running task {task.id}: {e}")
        traceback.print_exc()
        # Save error
        if task_save_dir:
            try:
                with open(os.path.join(task_save_dir, "error.txt"), "w") as f:
                    f.write(str(e))
                    f.write("\n\n")
                    f.write(traceback.format_exc())
            except:
                pass
        
                # Attempt to evaluate the task
        try:
            reward = await env.evaluate()
            print(f"Error in task, evaluating anyway: {task.id}: Reward = {reward}")
            return reward
        except Exception as e:
            print(f"Error evaluating task {task.id}: {e}")
            traceback.print_exc()
            reward = 0.0
        
        # Ensure we always return a numeric reward on error paths
        return reward
        
        
    finally:
        # Clean up environment
        if env:
            try:
                await env.close()
            except:
                pass


def run_task_wrapper(args):
    """Wrapper to run async task in thread pool."""
    task, job, model_name, orchestrator_model, config_path, orchestrator_max_steps, \
    cua_max_steps, coding_max_steps, save_dir, verbose = args
    
    return asyncio.run(run_single_task(
        task=task,
        job=job,
        model_name=model_name,
        orchestrator_model=orchestrator_model,
        config_path=config_path,
        orchestrator_max_steps=orchestrator_max_steps,
        cua_max_steps=cua_max_steps,
        coding_max_steps=coding_max_steps,
        save_dir=save_dir,
        verbose=verbose
    ))


async def run_taskset(
    taskset_id: str,
    job_name: str,
    model_name: str = "o4-mini",
    orchestrator_model: str = "o3",
    config_path: str = "coact/OAI_CONFIG_LIST",
    orchestrator_max_steps: int = 15,
    cua_max_steps: int = 25,
    coding_max_steps: int = 20,
    save_dir: str = "./coact_results",
    parallel: bool = False,
    max_concurrent: int = 5,
    verbose: bool = False,
    task_id: Optional[str] = None,
    task_index: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[float]:
    """Load and run a HUD taskset, return list of rewards."""
    
    # Load the taskset
    print(f"Loading taskset: {taskset_id}")
    taskset = await load_taskset(taskset_id, metadata={"partial": True})
    print(f"Loaded {len(taskset.tasks)} tasks")

    # Select subset based on filters
    tasks_to_run: List[Task] = list(taskset.tasks)
    if task_id is not None:
        tasks_to_run = [t for t in tasks_to_run if str(getattr(t, "id", "")) == str(task_id)]
        if len(tasks_to_run) == 0:
            print(f"No task with id '{task_id}' found in taskset '{taskset_id}'.")
            return []
    elif task_index is not None:
        if task_index < 0 or task_index >= len(tasks_to_run):
            print(f"task-index {task_index} is out of range 0..{len(tasks_to_run)-1}")
            return []
        tasks_to_run = [tasks_to_run[task_index]]
    if limit is not None:
        tasks_to_run = tasks_to_run[: max(0, limit)]
    print(f"Running {len(tasks_to_run)} task(s)")
    
    # Create job for tracking
    job = await create_job(job_name, evalset_id=taskset.id)
    print(f"Created job: {job.name} (ID: {job.id})")
    
    rewards = []
    
    if parallel:
        print(f"Running tasks in parallel (max {max_concurrent} concurrent)")
        # Run tasks in parallel using thread pool
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            task_args = [
                (task, job, model_name, orchestrator_model, config_path,
                 orchestrator_max_steps, cua_max_steps, coding_max_steps,
                 save_dir, verbose)
                for task in tasks_to_run
            ]
            rewards = list(executor.map(run_task_wrapper, task_args))
    else:
        print("Running tasks sequentially")
        # Run tasks sequentially
        for i, task in enumerate(tasks_to_run):
            print(f"Running task {i+1}/{len(tasks_to_run)}: {task.id}")
            reward = await run_single_task(
                task=task,
                job=job,
                model_name=model_name,
                orchestrator_model=orchestrator_model,
                config_path=config_path,
                orchestrator_max_steps=orchestrator_max_steps,
                cua_max_steps=cua_max_steps,
                coding_max_steps=coding_max_steps,
                save_dir=save_dir,
                verbose=verbose
            )
            rewards.append(reward)
    
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Run CoAct agent on HUD tasksets")
    parser.add_argument(
        "--taskset",
        type=str,
        default="OSWorld-Verified-XLang",
        help="The taskset ID to evaluate (default: OSWorld-Verified-XLang)"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run a single task by its task id (overrides --task-index)"
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Run a single task by its index in the taskset (0-based)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tasks to run from the start of the (filtered) list"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="CoAct Evaluation",
        help="Name for the evaluation job"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o4-mini",
        help="Model for coding/CUA agents (default: o4-mini)"
    )
    parser.add_argument(
        "--orchestrator-model",
        type=str,
        default="o3",
        help="Model for orchestrator agent (default: o3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="coact/OAI_CONFIG_LIST",
        help="Path to OpenAI config file"
    )
    parser.add_argument(
        "--orchestrator-max-steps",
        type=int,
        default=10,
        help="Max orchestrator steps"
    )
    parser.add_argument(
        "--cua-max-steps",
        type=int,
        default=25,
        help="Max CUA agent steps"
    )
    parser.add_argument(
        "--coding-max-steps",
        type=int,
        default=20,
        help="Max coding agent steps"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./coact_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tasks in parallel"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent tasks when running in parallel"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Run the taskset
    rewards = asyncio.run(run_taskset(
        taskset_id=args.taskset,
        job_name=args.name,
        model_name=args.model,
        orchestrator_model=args.orchestrator_model,
        config_path=args.config,
        orchestrator_max_steps=args.orchestrator_max_steps,
        cua_max_steps=args.cua_max_steps,
        coding_max_steps=args.coding_max_steps,
        save_dir=args.save_dir,
        parallel=args.parallel,
        max_concurrent=args.max_concurrent,
        verbose=args.verbose,
        task_id=args.task_id,
        task_index=args.task_index,
        limit=args.limit,
    ))
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Complete: {args.name}")
    print(f"Taskset: {args.taskset}")
    print(f"Tasks: {len(rewards)}")
    reward_list = [reward.get("reward") for reward in rewards if reward is not None]
    non_null_rewards = [reward for reward in reward_list if reward is not None]
    print(f"Rewards: {reward_list}")
    print(f"Average Reward: {sum(non_null_rewards)/len(non_null_rewards) if non_null_rewards else 0:.3f}")
    tasks_without_reward = [reward for reward in rewards if reward is None]
    if tasks_without_reward:
        print(f"Tasks without reward: {len(tasks_without_reward)}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
