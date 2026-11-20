import random
from datetime import datetime
from zoneinfo import ZoneInfo


# Tool implementations

def get_current_time(timezone: str = "America/New_York") -> dict:
    """Return the current date and time in the requested timezone."""
    now = datetime.now(ZoneInfo(timezone))
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "timezone": timezone,
    }


def random_number(min: int = 0, max: int = 100) -> dict:
    """Return a random integer, including both endpoints."""
    if type(min) is not int or type(max) is not int:
        raise TypeError("min and max must be integers.")

    if min > max:
        raise ValueError("min must be less than or equal to max.")

    return {"result": random.randint(min, max)}


# Signatures shown to the LLM through apply_chat_template(tools=...).

TOOLS_SIGNATURE = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取指定时区的当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "时区名称，例如 Asia/Shanghai、America/New_York"
                        ),
                        "default": "America/New_York",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "random_number",
            "description": "生成指定范围内的随机整数，包含最小值和最大值",
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "最小值（包含）",
                        "default": 0,
                    },
                    "max": {
                        "type": "integer",
                        "description": "最大值（包含）",
                        "default": 100,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


# Registry used by the agent to execute a tool by name.

TOOLS_FUNCTIONS = {
    "get_current_time": get_current_time,
    "random_number": random_number,
}
