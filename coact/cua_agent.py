import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import openai
from hud.env.environment import Environment
from openai import OpenAI  # pip install --upgrade openai>=1.30

from dotenv import load_dotenv
load_dotenv()

from hud.env.environment import Environment

# Import the helper function from async_utils
from .async_utils import run_async_in_sync
from hud.adapters.common.types import ScreenshotFetch
from hud.adapters.common.types import (
    ClickAction, TypeAction, ScrollAction, DragAction, 
    PressAction, MoveAction, WaitAction, Point, CustomAction
)

logger = logging.getLogger("desktopenv")


async def get_screenshot(env: Environment) -> str:
    """Get a base64 encoded screenshot from the environment."""
    obs, _, _, _ = await env.step(ScreenshotFetch())
    return obs.screenshot if obs and obs.screenshot else ""

GPT4O_INPUT_PRICE_PER_1M_TOKENS = 3.00
GPT4O_OUTPUT_PRICE_PER_1M_TOKENS = 12.00

PROMPT_TEMPLATE = """# Task
{instruction}

# Hints
- Sudo password is "password".
- The username for the gmail account is "osworld@hud.so".
- The password for the gmail account is "iloveosworld500".
- If prompted for One Time Password or 2FA, there's an authenticator Extension installed in Google Chrome with the live OTP for this osworld@hud.so account. 
- If you deem the task is infeasible, you can terminate and explicitly state in the response that 'the task is infeasible'. Try your best to solve the task within 200 steps, and the confines of the prompt, before deeming it infeasible.
- Keep the windows/applications opened at the end of the task.
- Do not use shortcut to reload the application except for the browser, just close and reopen.
- If "The document has been changed by others" pops out, you should click "cancel" and reopen the file.
- If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'.
- If you don't know how to continue the task, reply your concern or question along with 'IDK'.
""".strip()
DEFAULT_REPLY = "Please continue the user task. If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'."


def _cua_to_pyautogui(action) -> str:
    """Convert an Action (dict **or** Pydantic model) into a pyautogui call."""
    def fld(key: str, default: Any = None) -> Any:
        return action.get(key, default) if isinstance(action, dict) else getattr(action, key, default)

    act_type = fld("type")
    if not isinstance(act_type, str):
        act_type = str(act_type).split(".")[-1]
    act_type = act_type.lower()

    if act_type in ["click", "double_click"]:
        button = fld('button', 'left')
        if button == 1 or button == 'left':
            button = 'left'
        elif button == 2 or button == 'middle':
            button = 'middle'
        elif button == 3 or button == 'right':
            button = 'right'

        if act_type == "click":
            return f"pyautogui.click({fld('x')}, {fld('y')}, button='{button}')"
        if act_type == "double_click":
            return f"pyautogui.doubleClick({fld('x')}, {fld('y')}, button='{button}')"
        
    if act_type == "scroll":
        cmd = ""
        if fld('scroll_y', 0) != 0:
            cmd += f"pyautogui.scroll({-fld('scroll_y', 0) / 100}, x={fld('x', 0)}, y={fld('y', 0)});"
        return cmd
    if act_type == "drag":
        path = fld('path', [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
        cmd = f"pyautogui.moveTo({path[0]['x']}, {path[0]['y']}, _pause=False); "
        cmd += f"pyautogui.dragTo({path[1]['x']}, {path[1]['y']}, duration=0.5, button='left')"
        return cmd

    if act_type == 'move':
        return f"pyautogui.moveTo({fld('x')}, {fld('y')})"

    if act_type == "keypress":
        keys = fld("keys", []) or [fld("key")]
        if len(keys) == 1:
            return f"pyautogui.press('{keys[0].lower()}')"
        else:
            return "pyautogui.hotkey('{}')".format("', '".join(keys)).lower()
        
    if act_type == "type":
        text = str(fld("text", ""))
        return "pyautogui.typewrite({:})".format(repr(text))
    
    if act_type == "wait":
        return "WAIT"
    
    return "WAIT"  # fallback



def _to_input_items(output_items: list) -> list:
    """
    Convert `response.output` into the JSON-serialisable items we're allowed
    to resend in the next request.  We drop anything the CUA schema doesn't
    recognise (e.g. `status`, `id`, …) and cap history length.
    
    IMPORTANT: We must preserve reasoning items along with their corresponding
    computer_call items to maintain the required pairing.
    """
    print(f"  _to_input_items: Processing {len(output_items)} output items")
    cleaned: List[Dict[str, Any]] = []
    
    # First pass: collect all reasoning and computer_call items
    reasoning_items = {}
    computer_calls = []
    message_items = []
    
    for item in output_items:
        raw: Dict[str, Any] = item if isinstance(item, dict) else item.model_dump()
        
        # Strip noisy / disallowed keys
        raw.pop("status", None)
        
        item_type = raw.get("type", "")
        if not isinstance(item_type, str):
            item_type = str(item_type).split(".")[-1]
            
        if item_type == "reasoning":
            # Store reasoning items by their ID for pairing
            if "id" in raw:
                reasoning_items[raw["id"]] = raw
                print(f"    Found reasoning item with id: {raw['id']}")
        elif item_type == "computer_call":
            computer_calls.append(raw)
            print(f"    Found computer_call with call_id: {raw.get('call_id', 'N/A')}")
        elif item_type == "message":
            message_items.append(raw)
            print(f"    Found message item")
    
    # Second pass: add items in correct order, ensuring reasoning-computer_call pairs
    for item in output_items:
        raw: Dict[str, Any] = item if isinstance(item, dict) else item.model_dump()
        raw.pop("status", None)
        
        item_type = raw.get("type", "")
        if not isinstance(item_type, str):
            item_type = str(item_type).split(".")[-1]
            
        # Add all items to preserve order and pairing
        # The API will handle the proper pairing based on IDs
        if item_type in ["reasoning", "computer_call", "message"]:
            cleaned.append(raw)

    print(f"  _to_input_items: Returning {len(cleaned)} cleaned items")
    return cleaned


def call_openai_cua(client: OpenAI,
                    history_inputs: list,
                    screen_width: int = 1920,
                    screen_height: int = 1080,
                    environment: str = "linux") -> Tuple[Any, float]:
    retry = 0
    response = None
    while retry < 3:
        try:
            response = client.responses.create(
                model="computer-use-preview",
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": screen_width,
                    "display_height": screen_height,
                    "environment": environment,
                }],
                input=history_inputs,
                reasoning={
                    "summary": "concise"
                },
                tool_choice="required",
                truncation="auto",
            )
            break
        except openai.BadRequestError as e:
            retry += 1
            logger.error(f"Error in response.create: {e}")
            time.sleep(0.5)
        except openai.InternalServerError as e:
            retry += 1
            logger.error(f"Error in response.create: {e}")
            time.sleep(0.5)
    if retry == 3:
        raise Exception("Failed to call OpenAI.")

    cost = 0.0
    if response and hasattr(response, "usage") and response.usage:
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost = (input_tokens / 1_000_000) * GPT4O_INPUT_PRICE_PER_1M_TOKENS
        output_cost = (output_tokens / 1_000_000) * GPT4O_OUTPUT_PRICE_PER_1M_TOKENS
        cost = input_cost + output_cost

    return response, cost


def run_cua(
    env: Environment,
    instruction: str,
    max_steps: int,
    save_path: str = './',
    screen_width: int = 1920,
    screen_height: int = 1080,
    sleep_after_execution: float = 0.3,
    truncate_history_inputs: int = 100,
) -> Tuple[str, float]:
    client = OpenAI()

    # 0 / reset & first screenshot
    logger.info(f"Instruction: {instruction}")
    screenshot_b64 = run_async_in_sync(get_screenshot(env))  # Already base64 encoded
    if screenshot_b64:
        screenshot_bytes = base64.b64decode(screenshot_b64)
        with open(os.path.join(save_path, "initial_screenshot.png"), "wb") as f:
            f.write(screenshot_bytes)
    history_inputs = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": PROMPT_TEMPLATE.format(instruction=instruction)},
            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
        ],
    }]

    response, cost = call_openai_cua(client, history_inputs, screen_width, screen_height)
    total_cost = cost
    logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    step_no = 0
    
    reasoning_list = []
    reasoning = ""

    # 1 / iterative dialogue
    while step_no < max_steps:
        step_no += 1
        print(f"\n=== STEP {step_no} ===")
        print(f"Processing {len(response.output)} output items from previous response")
        
        new_items = _to_input_items(response.output)
        print(f"Adding {len(new_items)} items to history_inputs")
        for item in new_items:
            item_type = item.get('type', 'unknown')
            call_id = item.get('call_id', 'N/A')
            reasoning_id = item.get('id', 'N/A') if item.get('type') == 'reasoning' else 'N/A'
            if reasoning_id != 'N/A':
                print(f"  - {item_type} (id: {reasoning_id})")
            else:
                print(f"  - {item_type} (call_id: {call_id})")
        
        history_inputs += new_items

        # --- robustly pull out computer_call(s) ------------------------------
        calls: List[Dict[str, Any]] = []
        # completed = False
        breakflag = False
        for i, o in enumerate(response.output):
            typ = o["type"] if isinstance(o, dict) else getattr(o, "type", None)
            if not isinstance(typ, str):
                typ = str(typ).split(".")[-1]
            if typ == "computer_call":
                calls.append(o if isinstance(o, dict) else o.model_dump())
            elif typ == "reasoning" and len(o.summary) > 0:
                reasoning = o.summary[0].text
                reasoning_list.append(reasoning)
                logger.info(f"[Reasoning]: {reasoning}")
            elif typ == 'message':
                if 'TERMINATE' in o.content[0].text:
                    reasoning_list.append(f"Final output: {o.content[0].text}")
                    reasoning = "My thinking process\n" + "\n- ".join(reasoning_list) + '\nPlease check the screenshot and see if it fulfills your requirements.'
                    breakflag = True
                    break
                if 'IDK' in o.content[0].text:
                    reasoning = f"{o.content[0].text}. I don't know how to complete the task. Please check the current screenshot."
                    breakflag = True
                    break
                try:
                    json.loads(o.content[0].text)
                    history_inputs.pop(len(history_inputs) - len(response.output) + i)
                    step_no -= 1
                except Exception as e:
                    logger.info(f"[Message]: {o.content[0].text}")
                    if '?' in o.content[0].text:
                        history_inputs += [{
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": DEFAULT_REPLY},
                            ],
                        }]
                    elif "{" in o.content[0].text and "}" in o.content[0].text:
                        history_inputs.pop(len(history_inputs) - len(response.output) + i)
                        step_no -= 1
                    else:
                        logger.info(f"[Message]: {o.content[0].text}")
                        history_inputs.pop(len(history_inputs) - len(response.output) + i)
                        reasoning = o.content[0].text
                        reasoning_list.append(reasoning)
                        step_no -= 1

        if breakflag:
            break

        for action_call in calls:
            print(f"\nProcessing computer_call with call_id: {action_call.get('call_id', 'N/A')}")
            py_cmd = _cua_to_pyautogui(action_call["action"])
            cla_action = CustomAction(action=py_cmd)

            # --- execute in VM ---------------------------------------------------
            obs, *_ = run_async_in_sync(env.step(cla_action))
            if sleep_after_execution > 0:
                print(f"Sleeping for {sleep_after_execution} seconds after executing {py_cmd}")
                time.sleep(sleep_after_execution)

            # --- send screenshot back -------------------------------------------
            screenshot_b64 = obs.screenshot if obs and obs.screenshot else ""  # Already base64 encoded
            if screenshot_b64:
                screenshot_bytes = base64.b64decode(screenshot_b64)
                with open(os.path.join(save_path, f"step_{step_no}.png"), "wb") as f:
                    f.write(screenshot_bytes)
            output_item = {
                "type": "computer_call_output",
                "call_id": action_call["call_id"],
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                },
            }
            print(f"Adding computer_call_output for call_id: {action_call['call_id']}")
            history_inputs += [output_item]
            if "pending_safety_checks" in action_call and len(action_call.get("pending_safety_checks", [])) > 0:
                history_inputs[-1]['acknowledged_safety_checks'] = [
                    {
                        "id": psc["id"],
                        "code": psc["code"],
                        "message": "Please acknowledge this warning if you'd like to proceed."
                    }
                    for psc in action_call.get("pending_safety_checks", [])
                ]
        
        # truncate history inputs while preserving call_id pairs and reasoning items
        if len(history_inputs) > truncate_history_inputs:
            print(f"\n=== TRUNCATION NEEDED: {len(history_inputs)} > {truncate_history_inputs} ===")
            original_history = history_inputs[:]
            history_inputs = [history_inputs[0]] + history_inputs[-truncate_history_inputs:]
            print(f"After initial truncation: {len(history_inputs)} items")
            
            # Find all call_ids and reasoning IDs in the truncated history
            call_ids_in_truncated = set()
            reasoning_ids_in_truncated = set()
            
            print(f"Analyzing truncated history items...")
            for item in history_inputs:
                if isinstance(item, dict):
                    if 'call_id' in item:
                        call_ids_in_truncated.add(item['call_id'])
                        print(f"  Found {item.get('type', 'unknown')} with call_id: {item['call_id']}")
                    if item.get('type') == 'reasoning' and 'id' in item:
                        reasoning_ids_in_truncated.add(item['id'])
                        print(f"  Found reasoning with id: {item['id']}")
            
            # Check if any call_ids are missing their pairs (including reasoning)
            call_id_types = {}  # call_id -> list of types that reference it
            reasoning_for_calls = {}  # Maps computer_call call_ids to their reasoning items (not just IDs)
            computer_calls_by_id = {}  # Maps call_ids to their computer_call items
            
            print(f"Scanning original history for call_ids and reasoning items...")
            # First scan original history to build complete mappings
            for item in original_history:
                if isinstance(item, dict):
                    item_type = item.get('type', '')
                    
                    # Store computer_call items
                    if item_type == 'computer_call' and 'call_id' in item:
                        call_id = item['call_id']
                        computer_calls_by_id[call_id] = item
                        
                        # Find the reasoning item that should precede this computer_call
                        # Reasoning items typically come right before computer_calls
                        orig_idx = original_history.index(item)
                        if orig_idx > 0:
                            prev_item = original_history[orig_idx - 1]
                            if isinstance(prev_item, dict) and prev_item.get('type') == 'reasoning':
                                # Store the full reasoning item, not just its ID
                                reasoning_for_calls[call_id] = prev_item
                                print(f"  Mapped reasoning {prev_item.get('id', 'N/A')} to computer_call {call_id}")
                    
                    if 'call_id' in item:
                        call_id = item['call_id']
                        if call_id not in call_id_types:
                            call_id_types[call_id] = []
                        call_id_types[call_id].append(item_type)
            
            # Build a map of what's in the truncated history
            truncated_call_id_types = {}
            for item in history_inputs:
                if isinstance(item, dict) and 'call_id' in item:
                    call_id = item['call_id']
                    item_type = item.get('type', '')
                    if call_id not in truncated_call_id_types:
                        truncated_call_id_types[call_id] = []
                    truncated_call_id_types[call_id].append(item_type)
            
            # Find unpaired call_ids (should have computer_call, computer_call_output, and reasoning)
            unpaired_call_ids = []
            missing_reasoning_ids = set()
            
            # Check each call_id in the TRUNCATED history to see if it's missing its pair
            for call_id, types_in_truncated in truncated_call_id_types.items():
                # Get the full set of types from original history
                original_types = call_id_types.get(call_id, [])
                
                # Check if we have all required parts in truncated history
                has_call_in_truncated = 'computer_call' in types_in_truncated
                has_output_in_truncated = 'computer_call_output' in types_in_truncated
                
                # If we have output but no call, or call but no output, it's unpaired
                if has_output_in_truncated and not has_call_in_truncated:
                    print(f"  Found orphaned computer_call_output: {call_id}")
                    unpaired_call_ids.append(call_id)
                elif has_call_in_truncated and not has_output_in_truncated:
                    print(f"  Found orphaned computer_call: {call_id}")
                    unpaired_call_ids.append(call_id)
                
                # Check if reasoning is missing for this call
                if has_call_in_truncated and call_id in reasoning_for_calls:
                    reasoning_item = reasoning_for_calls[call_id]
                    reasoning_id = reasoning_item.get('id') if isinstance(reasoning_item, dict) else None
                    if reasoning_id and reasoning_id not in reasoning_ids_in_truncated:
                        missing_reasoning_ids.add(reasoning_id)
            
            # Add missing pairs and reasoning items from original history
            if unpaired_call_ids or missing_reasoning_ids:
                print(f"Found unpaired call_ids: {unpaired_call_ids}")
                print(f"Found missing reasoning_ids: {missing_reasoning_ids}")
                # Find missing items in their original order
                missing_items = []
                
                # For each unpaired call_id, we need to add back the missing parts
                for call_id in unpaired_call_ids:
                    # Check what's missing for this call_id
                    types_in_truncated = truncated_call_id_types.get(call_id, [])
                    
                    # If we have output but no call, add back BOTH reasoning and call
                    if 'computer_call_output' in types_in_truncated and 'computer_call' not in types_in_truncated:
                        # Add the reasoning item first (if it exists)
                        if call_id in reasoning_for_calls:
                            reasoning_item = reasoning_for_calls[call_id]
                            if reasoning_item not in history_inputs and reasoning_item not in missing_items:
                                missing_items.append(reasoning_item)
                                print(f"  Adding missing reasoning for call_id {call_id}")
                        
                        # Add the computer_call item
                        if call_id in computer_calls_by_id:
                            call_item = computer_calls_by_id[call_id]
                            if call_item not in history_inputs and call_item not in missing_items:
                                missing_items.append(call_item)
                                print(f"  Adding missing computer_call for call_id {call_id}")
                    
                    # If we have call but no output, add back the output
                    elif 'computer_call' in types_in_truncated and 'computer_call_output' not in types_in_truncated:
                        # Find the output in original history
                        for item in original_history:
                            if (isinstance(item, dict) and 
                                item.get('type') == 'computer_call_output' and 
                                item.get('call_id') == call_id and 
                                item not in history_inputs and 
                                item not in missing_items):
                                missing_items.append(item)
                                print(f"  Adding missing computer_call_output for call_id {call_id}")
                                break
                
                # Add any additional missing reasoning items (not covered by unpaired calls)
                for item in original_history:
                    if isinstance(item, dict):
                        if (item.get('type') == 'reasoning' and 
                            item.get('id') in missing_reasoning_ids and 
                            item not in history_inputs and 
                            item not in missing_items):
                            missing_items.append(item)
                            print(f"  Adding missing reasoning item with id {item.get('id')}")
                
                # Insert missing items back, preserving their original order
                for missing_item in missing_items:
                    # Find the best insertion point based on original history order
                    original_index = original_history.index(missing_item)
                    
                    # Find insertion point in truncated history
                    insert_pos = len(history_inputs)  # default to end
                    for i, existing_item in enumerate(history_inputs[1:], 1):  # skip first item (initial prompt)
                        if existing_item in original_history:
                            existing_original_index = original_history.index(existing_item)
                            if existing_original_index > original_index:
                                insert_pos = i
                                break
                    
                    history_inputs.insert(insert_pos, missing_item)
                    print(f"  Inserted missing {missing_item.get('type')} at position {insert_pos}")

        # Debug: Show what we're sending to the API
        print(f"\n=== CALLING OPENAI CUA API ===")
        print(f"Sending {len(history_inputs)} history items")
        
        # Count different types of items
        type_counts = {}
        call_id_map = {}  # Maps call_ids to their types
        for idx, item in enumerate(history_inputs):
            if isinstance(item, dict):
                item_type = item.get('type', 'user' if item.get('role') == 'user' else 'unknown')
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
                
                if 'call_id' in item:
                    call_id = item['call_id']
                    if call_id not in call_id_map:
                        call_id_map[call_id] = []
                    call_id_map[call_id].append((item_type, idx))
        
        print(f"Item type counts: {type_counts}")
        
        # Check for orphaned call_ids
        orphaned_outputs = []
        for call_id, type_positions in call_id_map.items():
            types = [tp[0] for tp in type_positions]
            if 'computer_call_output' in types and 'computer_call' not in types:
                print(f"WARNING: Found orphaned computer_call_output for call_id: {call_id}")
                print(f"  Types present for this call_id: {types}")
                print(f"  Positions: {type_positions}")
                orphaned_outputs.append(call_id)
        
        if orphaned_outputs:
            print(f"\n!!! CRITICAL: About to send orphaned computer_call_outputs: {orphaned_outputs}")
            print("This will likely cause an API error!")
        
        # Debug: Check computer_call and reasoning pairing
        print(f"\nVerifying computer_call/reasoning pairs:")
        for idx, item in enumerate(history_inputs):
            if isinstance(item, dict) and item.get('type') == 'computer_call':
                call_id = item.get('call_id', 'N/A')
                # Check if there's a reasoning item before this
                has_reasoning = False
                if idx > 0:
                    prev_item = history_inputs[idx - 1]
                    if isinstance(prev_item, dict) and prev_item.get('type') == 'reasoning':
                        has_reasoning = True
                        print(f"  ✓ computer_call {call_id} has preceding reasoning {prev_item.get('id', 'N/A')}")
                if not has_reasoning:
                    print(f"  ✗ computer_call {call_id} MISSING preceding reasoning!")
        
        response, cost = call_openai_cua(client, history_inputs, screen_width, screen_height)
        total_cost += cost
        logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    
    logger.info(f"Total cost for the task: ${total_cost:.4f}")
    history_inputs[0]['content'][1]['image_url'] = "<image>"
    for item in history_inputs:
        if item.get('type', None) == 'computer_call_output':
            item['output']['image_url'] = "<image>"
    return history_inputs, reasoning, total_cost

