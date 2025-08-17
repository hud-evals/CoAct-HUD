import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import openai
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



def _cua_to_cla_action(action) -> Any:
    """Convert a CUA Action (dict **or** Pydantic model) into a proper CLA action for HUD env."""
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

        x = fld('x')
        y = fld('y')
        
        if act_type == "click":
            return ClickAction(point=Point(x=x, y=y), button=button)
        if act_type == "double_click":
            # Double click is a click with pattern
            return ClickAction(point=Point(x=x, y=y), button="left", pattern=[100])
        
    if act_type == "scroll":
        x = fld('x', 0)
        y = fld('y', 0)
        scroll_y = fld('scroll_y', 0)
        # Convert CUA scroll format to CLA scroll
        return ScrollAction(
            point=Point(x=x, y=y),
            scroll=Point(x=0, y=-scroll_y // 100)  # CUA uses percentages
        )
        
    if act_type == "drag":
        path = fld('path', [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
        return DragAction(
            path=[Point(x=p["x"], y=p["y"]) for p in path]
        )

    if act_type == 'move':
        x = fld('x')
        y = fld('y')
        return MoveAction(point=Point(x=x, y=y))

    if act_type == "keypress":
        keys = fld("keys", []) or [fld("key")]
        if not isinstance(keys, list):
            keys = [keys]
        # Convert to lowercase for CLA compatibility
        keys = [k.lower() if isinstance(k, str) else str(k).lower() for k in keys]
        return PressAction(keys=keys)  # type: ignore[arg-type]
        
    if act_type == "type":
        text = str(fld("text", ""))
        return TypeAction(text=text, enter_after=False)
    
    if act_type == "wait":
        duration = fld("duration", 1)
        # Convert to milliseconds if needed
        return WaitAction(time=duration * 1000 if duration < 100 else duration)
    
    # Default fallback
    return WaitAction(time=1000)


def _to_input_items(output_items: list) -> list:
    """
    Convert `response.output` into the JSON-serialisable items we're allowed
    to resend in the next request.  We drop anything the CUA schema doesn't
    recognise (e.g. `status`, `id`, …) and cap history length.
    """
    cleaned: List[Dict[str, Any]] = []

    for item in output_items:
        raw: Dict[str, Any] = item if isinstance(item, dict) else item.model_dump()

        # ---- strip noisy / disallowed keys ---------------------------------
        raw.pop("status", None)
        cleaned.append(raw)

    return cleaned  # keep just the most recent 50 items


def call_openai_cua(client: OpenAI,
                    input_items: list,
                    screen_width: int = 1920,
                    screen_height: int = 1080,
                    environment: str = "ubuntu",
                    previous_response_id: str | None = None,
                    include_reasoning_summary: bool = False) -> Tuple[Any, float]:
    retry = 0
    response = None
    while retry < 3:
        try:
            # Build common payload
            payload: Dict[str, Any] = {
                "model": "computer-use-preview",
                "tools": [{
                    "type": "computer_use_preview",
                    "display_width": screen_width,
                    "display_height": screen_height,
                    "environment": environment,
                }],
                "truncation": "auto",
            }

            if previous_response_id:
                # Subsequent calls: link conversation by previous_response_id
                payload["previous_response_id"] = previous_response_id
                payload["input"] = input_items
            else:
                # Initial call: send initial prompt (and optional screenshot)
                payload["input"] = input_items
                if include_reasoning_summary:
                    payload["reasoning"] = {"summary": "concise"}

            response = client.responses.create(**payload)
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
) -> Tuple[List[Dict[str, Any]], str, float]:
    client = OpenAI()

    # 0 / reset & first screenshot
    logger.info(f"Instruction: {instruction}")
    screenshot_b64 = run_async_in_sync(get_screenshot(env))
    with open(os.path.join(save_path, "initial_screenshot.png"), "wb") as f:
        f.write(base64.b64decode(screenshot_b64) if screenshot_b64 else b"")
    history_inputs = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": PROMPT_TEMPLATE.format(instruction=instruction)},
            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
        ],
    }]

    # Initial request per docs: send prompt (and optional screenshot), include reasoning summary
    response, cost = call_openai_cua(
        client,
        input_items=history_inputs,
        screen_width=screen_width,
        screen_height=screen_height,
        environment="ubuntu",
        previous_response_id=None,
        include_reasoning_summary=True,
    )
    total_cost = cost
    logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    step_no = 0
    
    reasoning_list = []
    reasoning = ""

    # 1 / iterative dialogue
    while step_no < max_steps:
        step_no += 1
        history_inputs += _to_input_items(response.output)

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
            py_cmd = _cua_to_pyautogui(action_call["action"])
            cla_action = CustomAction(action=py_cmd)

            # --- execute in VM ---------------------------------------------------
            obs, *_ = run_async_in_sync(env.step(cla_action))
            if sleep_after_execution > 0:
                time.sleep(sleep_after_execution)

            # --- send screenshot back -------------------------------------------
            screenshot_b64 = obs.screenshot if obs and obs.screenshot else ""
            with open(os.path.join(save_path, f"step_{step_no}.png"), "wb") as f:
                f.write(base64.b64decode(screenshot_b64) if screenshot_b64 else b"")
            # Build next input strictly per docs: only the computer_call_output referencing last call
            next_input: Dict[str, Any] = {
                "type": "computer_call_output",
                "call_id": action_call["call_id"],
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                },
            }
            # Acknowledge any pending safety checks returned
            if "pending_safety_checks" in action_call and len(action_call.get("pending_safety_checks", [])) > 0:
                next_input["acknowledged_safety_checks"] = [
                    {
                        "id": psc["id"],
                        "code": psc["code"],
                        "message": "Please acknowledge this warning if you'd like to proceed.",
                    }
                    for psc in action_call.get("pending_safety_checks", [])
                ]
            history_inputs.append(next_input)
            if "pending_safety_checks" in action_call and len(action_call.get("pending_safety_checks", [])) > 0:
                history_inputs[-1]['acknowledged_safety_checks'] = [
                    {
                        "id": psc["id"],
                        "code": psc["code"],
                        "message": "Please acknowledge this warning if you'd like to proceed."
                    }
                    for psc in action_call.get("pending_safety_checks", [])
                ]
        
        # truncate history inputs while preserving required pairs and dependencies
        if len(history_inputs) > truncate_history_inputs:
            original_history = history_inputs[:]
            history_inputs = [history_inputs[0]] + history_inputs[-truncate_history_inputs:]
            
            # Find all call_ids in the truncated history
            call_ids_in_truncated = set()
            for item in history_inputs:
                if isinstance(item, dict) and 'call_id' in item:
                    call_ids_in_truncated.add(item['call_id'])
            
            # Check if any call_ids are missing their pairs
            call_id_types = {}  # call_id -> list of types that reference it
            for item in history_inputs:
                if isinstance(item, dict) and 'call_id' in item:
                    call_id = item['call_id']
                    item_type = item.get('type', '')
                    if call_id not in call_id_types:
                        call_id_types[call_id] = []
                    call_id_types[call_id].append(item_type)
            
            # Find unpaired call_ids (should have both computer_call and computer_call_output)
            unpaired_call_ids = []
            for call_id, types in call_id_types.items():
                # Check if we have both call and output
                has_call = 'computer_call' in types
                has_output = 'computer_call_output' in types
                if not (has_call and has_output):
                    unpaired_call_ids.append(call_id)
            
            # Add missing pairs from original history while preserving order
            if unpaired_call_ids:
                # Find missing paired items in their original order
                missing_items = []
                for item in original_history:
                    if (isinstance(item, dict) and 
                        item.get('call_id') in unpaired_call_ids and 
                        item not in history_inputs):
                        missing_items.append(item)
                
                # Insert missing items back, preserving their original order
                # We need to find appropriate insertion points to maintain chronology
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

            # Ensure each computer_call has a preceding reasoning item.
            # Some OpenAI responses require the reasoning item (type 'reasoning')
            # referenced by the computer_call to be present in the input history.
            # To be safe, for any computer_call without a prior reasoning item in
            # the truncated history, include the most recent reasoning item from
            # the original history prior to that call.
            # Build list of indices of reasoning items in original history
            reasoning_indices = [idx for idx, it in enumerate(original_history)
                                 if isinstance(it, dict) and it.get('type') == 'reasoning']

            if reasoning_indices:
                # Scan truncated history to find computer_call entries
                i = 0
                while i < len(history_inputs):
                    item = history_inputs[i]
                    if isinstance(item, dict) and item.get('type') == 'computer_call':
                        # Check if there is any reasoning item before this index in truncated history
                        has_prior_reasoning = any(
                            isinstance(prev, dict) and prev.get('type') == 'reasoning'
                            for prev in history_inputs[:i]
                        )
                        if not has_prior_reasoning:
                            # Find the nearest prior reasoning in original history relative to this item
                            try:
                                orig_idx = original_history.index(item)
                            except ValueError:
                                orig_idx = -1
                            # Find the last reasoning index less than orig_idx
                            prior_reasoning_idx = -1
                            for ridx in reasoning_indices:
                                if ridx < orig_idx:
                                    prior_reasoning_idx = ridx
                                else:
                                    break
                            if prior_reasoning_idx != -1:
                                reasoning_item = original_history[prior_reasoning_idx]
                                # Insert reasoning item just before the computer_call in truncated history
                                if reasoning_item not in history_inputs:
                                    history_inputs.insert(i, reasoning_item)
                                    i += 1  # Skip over the inserted reasoning
                    i += 1

        # Subsequent requests should use previous_response_id and only send the computer_call_output
        response, cost = call_openai_cua(
            client,
            input_items=history_inputs[-1:],  # send only the latest output (and safety acks)
            screen_width=screen_width,
            screen_height=screen_height,
            environment="ubuntu",
            previous_response_id=getattr(response, "id", None),
            include_reasoning_summary=False,
        )
        total_cost += cost
        logger.info(f"Cost: ${cost:.6f} | Total Cost: ${total_cost:.6f}")
    
    logger.info(f"Total cost for the task: ${total_cost:.4f}")
    history_inputs[0]['content'][1]['image_url'] = "<image>"
    for item in history_inputs:
        if item.get('type', None) == 'computer_call_output':
            item['output']['image_url'] = "<image>"
    return history_inputs, reasoning, total_cost

