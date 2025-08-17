# ruff: noqa: S101

from __future__ import annotations

from typing import Any, ClassVar

from hud.adapters.common import CLA, Adapter
from hud.adapters.common.types import (
    CLAKey,
    ClickAction,
    CustomAction,
    DragAction,
    MoveAction,
    Point,
    PositionFetch,
    PressAction,
    ResponseAction,
    ScreenshotFetch,
    ScrollAction,
    TypeAction,
    WaitAction,
)


class CoActAdapter(Adapter):
    """Adapter to convert CoAct agent actions (CUA and coding) to HUD SDK CLA format."""
    
    KEY_MAP: ClassVar[dict[str, CLAKey]] = {
        "return": "enter",
        "super": "win",
        "super_l": "win",
        "super_r": "win",
        "right shift": "shift",
        "left shift": "shift",
        "down shift": "shift",
        "windows": "win",
        "page_down": "pagedown",
        "page_up": "pageup",
        # Note: meta, cmd, command, option are not valid CLAKey values
        # They will be handled by the default mapping
    }

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        super().__init__()
        self.agent_width = width
        self.agent_height = height

    def _map_key(self, key: str) -> CLAKey:
        """Map a key to its standardized form."""
        return self.KEY_MAP.get(key.lower(), key.lower())  # type: ignore

    def convert(self, data: Any) -> CLA:
        """Convert CoAct action to CLA format.
        
        Handles both CUA actions (from OpenAI Computer Use API) and 
        custom coding actions (bash/python execution).
        """
        try:
            # Validate input data
            if not isinstance(data, dict):
                raise ValueError(f"Invalid action: {data}")

            action_type = data.get("action") or data.get("type")
            
            # Handle CUA actions
            if action_type == "click" or action_type == "left_click":
                assert "coordinate" in data or ("x" in data and "y" in data)
                if "coordinate" in data:
                    coord = data["coordinate"]
                    x, y = coord if isinstance(coord, list) else (coord[0], coord[1])
                else:
                    x, y = data["x"], data["y"]
                
                button = data.get("button", "left")
                if button == 1:
                    button = "left"
                elif button == 2:
                    button = "middle"
                elif button == 3:
                    button = "right"
                    
                converted_action = ClickAction(point=Point(x=x, y=y), button=button)

            elif action_type == "double_click":
                assert "coordinate" in data or ("x" in data and "y" in data)
                if "coordinate" in data:
                    coord = data["coordinate"]
                    x, y = coord if isinstance(coord, list) else (coord[0], coord[1])
                else:
                    x, y = data["x"], data["y"]
                    
                converted_action = ClickAction(
                    point=Point(x=x, y=y), button="left", pattern=[100]
                )

            elif action_type == "right_click":
                assert "coordinate" in data or ("x" in data and "y" in data)
                if "coordinate" in data:
                    coord = data["coordinate"]
                    x, y = coord if isinstance(coord, list) else (coord[0], coord[1])
                else:
                    x, y = data["x"], data["y"]
                    
                converted_action = ClickAction(point=Point(x=x, y=y), button="right")

            elif action_type == "middle_click":
                assert "coordinate" in data or ("x" in data and "y" in data)
                if "coordinate" in data:
                    coord = data["coordinate"]
                    x, y = coord if isinstance(coord, list) else (coord[0], coord[1])
                else:
                    x, y = data["x"], data["y"]
                    
                converted_action = ClickAction(point=Point(x=x, y=y), button="middle")

            elif action_type == "triple_click":
                assert "coordinate" in data or ("x" in data and "y" in data)
                if "coordinate" in data:
                    coord = data["coordinate"]
                    x, y = coord if isinstance(coord, list) else (coord[0], coord[1])
                else:
                    x, y = data["x"], data["y"]
                    
                converted_action = ClickAction(
                    point=Point(x=x, y=y),
                    button="left",
                    pattern=[100, 100],
                )

            elif action_type == "type":
                assert "text" in data
                converted_action = TypeAction(
                    text=data["text"],
                    enter_after=False,
                )

            elif action_type == "key" or action_type == "keypress":
                keys = []
                if "text" in data:
                    if "+" in data["text"]:
                        keys = [self._map_key(k) for k in data["text"].split("+")]
                    else:
                        keys = [self._map_key(data["text"])]
                elif "keys" in data:
                    keys = data["keys"] if isinstance(data["keys"], list) else [data["keys"]]
                    keys = [self._map_key(k) for k in keys]
                elif "key" in data:
                    keys = [self._map_key(data["key"])]
                    
                assert len(keys) > 0
                # Cast to list[CLAKey] to satisfy type checker
                converted_action = PressAction(keys=keys)  # type: ignore[arg-type]

            elif action_type == "move" or action_type == "mouse_move":
                assert "coordinate" in data or ("x" in data and "y" in data)
                if "coordinate" in data:
                    coord = data["coordinate"]
                    x, y = coord if isinstance(coord, list) else (coord[0], coord[1])
                else:
                    x, y = data["x"], data["y"]
                    
                converted_action = MoveAction(point=Point(x=x, y=y))

            elif action_type == "drag" or action_type == "left_click_drag":
                if "path" in data:
                    path = data["path"]
                    converted_action = DragAction(
                        path=[Point(x=p["x"], y=p["y"]) for p in path]
                    )
                elif "coordinate" in data:
                    # For CUA-style drag, use last position from memory
                    coord = data["coordinate"]
                    x, y = coord if isinstance(coord, list) else (coord[0], coord[1])
                    if (
                        len(self.memory) == 0
                        or (
                            not isinstance(self.memory[-1], MoveAction)
                            and not isinstance(self.memory[-1], ClickAction)
                        )
                        or self.memory[-1].point is None
                    ):
                        raise ValueError("Drag must be preceded by a move or click action")
                    else:
                        converted_action = DragAction(
                            path=[self.memory[-1].point, Point(x=x, y=y)]
                        )
                else:
                    raise ValueError("Drag action requires path or coordinate")

            elif action_type == "scroll":
                assert "coordinate" in data or ("x" in data and "y" in data)
                if "coordinate" in data:
                    coord = data["coordinate"]
                    x, y = coord if isinstance(coord, list) else (coord[0], coord[1])
                else:
                    x, y = data["x"], data["y"]
                
                # Handle different scroll formats
                if "scroll_direction" in data:
                    direction = data["scroll_direction"]
                    amount = data.get("scroll_amount", 5)
                    
                    if direction == "up":
                        scroll = Point(x=0, y=-amount)
                    elif direction == "down":
                        scroll = Point(x=0, y=amount)
                    elif direction == "left":
                        scroll = Point(x=-amount, y=0)
                    elif direction == "right":
                        scroll = Point(x=amount, y=0)
                    else:
                        raise ValueError(f"Unsupported scroll direction: {direction}")
                elif "scroll_y" in data:
                    # CUA format - note the sign inversion
                    scroll = Point(x=0, y=-data["scroll_y"] / 100)
                else:
                    scroll = Point(x=0, y=5)  # Default scroll down
                    
                converted_action = ScrollAction(
                    point=Point(x=x, y=y),
                    scroll=scroll,
                )

            elif action_type == "wait":
                duration = data.get("duration", data.get("time", 1))
                converted_action = WaitAction(time=duration)

            elif action_type == "screenshot":
                converted_action = ScreenshotFetch()

            elif action_type == "cursor_position":
                converted_action = PositionFetch()

            elif action_type == "response":
                converted_action = ResponseAction(text=data.get("text", ""))

            # Handle custom coding actions
            elif action_type == "bash":
                assert "code" in data
                # CustomAction takes action concatenated with code
                converted_action = CustomAction(
                    action="bash" + data["code"]
                )

            elif action_type == "python":
                assert "code" in data
                # CustomAction takes action concatenated with code
                converted_action = CustomAction(
                    action="python" + data["code"]
                )

            elif action_type == "custom":
                # Generic custom action
                action_name = data.get("action", "unknown")
                custom_data = {k: v for k, v in data.items() if k not in ["type", "action", "reasoning", "logs"]}
                converted_action = CustomAction(
                    action=action_name,
                    **custom_data
                )

            else:
                raise ValueError(f"Unsupported action type: {action_type}")

            # Preserve metadata
            converted_action.reasoning = data.get("reasoning", None)
            converted_action.logs = data.get("logs", None)

            return converted_action
            
        except AssertionError:
            raise ValueError(f"Invalid action: {data}") from None
