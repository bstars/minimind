import re
import sys
import random
from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time(timezone: str = "America/New_York") -> dict:
    return {"datetime": datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds"),
            "timezone": timezone}

def random_number(min: int = 0, max: int = 100) -> dict:
    return {"result": random.randint(min, max)}

TOOLS_SIGNATURE = [
    {
        "type": "function", 
        "function": {
            "name": "get_current_time", 
            "description": "获取当前日期和时间，支持指定时区", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}
                }, 
                "required": []
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "random_number", 
            "description": "生成指定范围内的随机数", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "min": {"type": "integer", "description": "最小值", "default": 0}, 
                    "max": {"type": "integer", "description": "最大值", "default": 100}
                }, 
                "required": []
            }
        }
    },
]


TOOLS_FUNCTIONS = {
    fn.__name__: fn 
    for fn in (get_current_time, random_number)
}

TOOL_SCHEMAS = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLS_SIGNATURE}
