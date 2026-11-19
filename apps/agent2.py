import argparse
import ast
import json
import math
import operator
from pathlib import Path
import random
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
from transformers import AutoTokenizer, TextStreamer
from model.model import MinimindCausalLLM
from model.model_config import MinimindConfig
from apps.tools import *


class ColoredStreamer(TextStreamer):
    RESET = "\033[0m"

    COLORS = [
        ("<think>", "</think>"), "\033[90m", # Gray; use "\033[36m" for cyan.
        ("<tool_call>", "</tool_call>"),"\033[90m" # Gray; use "\033[36m" for cyan.
    ]

    def on_finalized_text(self, text, stream_end=False):
        for (begin_tag, end_tag), color in self.COLORS:
            text = text.replace(
                begin_tag, color + begin_tag
            ).replace(
                end_tag, end_tag + self.RESET
            )

        if stream_end:
            # Restore the color even if generation stopped mid-thinking.
            text += self.RESET

        super().on_finalized_text(text, stream_end=stream_end)


def parse_tool_calls(text: str) -> list[dict]:

    # remove thinking block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL) 
    
    # detect tool call
    pattern = r"<tool_call>(.*?)</tool_call>"
    bodies = re.findall(pattern, text, flags=re.DOTALL)
    remainder = re.sub(pattern, "", text, flags=re.DOTALL)
    if "<tool_call>" in remainder or "</tool_call>" in remainder:
        raise ValueError("Incomplete tool call; close each <tool_call> with </tool_call>.")

    calls = []
    for body in bodies:
        call = json.loads(body)
        arguments = call.get("arguments", {})
        if isinstance(arguments, str): arguments = json.loads(arguments)
        calls.append({"name": call["name"], "arguments": arguments})
    return calls


def parse_tool_calls_test():
    texts = [
        """
        <tool_call>
        {"name": "calculate", "arguments": {"x": 2, "y": 3}}
        </tool_call>
        """,
        """
        <tool_call>{"name": "search", "arguments": {"query": "PPO"}}</tool_call>
        <tool_call>{"name": "calculate", "arguments": {"x": 5}}</tool_call>
        """
    ]

    for text in texts:
        print("=="*20)
        print(text)
        print("--"*20)
        print(parse_tool_calls(text))
        print("=="*20)
        print()

def execute_tools(call:dict) -> dict:
    name, arguments = call["name"], call["arguments"]

    if name not in TOOLS_FUNCTIONS: raise ValueError(f"Unknown tool: {name}")
    try:
        return TOOLS_FUNCTIONS[name](**arguments)
    except Exception as exc:
        raise "Tool exection error"


class Agent:
    def __init__(self, model: MinimindCausalLLM, tokenizer: AutoTokenizer, device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        # sampling options
        self.temperature = 1.0
        self.top_p = 0.95
        self.top_k = 20
        self.messages = [{"role": "system", "content": ""}]

    def _sample(self, logits):
        if self.temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits.float() / self.temperature
        if self.top_k:
            cutoff = torch.topk(logits, min(self.top_k, logits.shape[-1]), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, -torch.inf)
        if self.top_p < 1:
            sorted_logits, indices = logits.sort(dim=-1, descending=True)
            remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > self.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            logits = logits.masked_fill(torch.zeros_like(remove).scatter(1, indices, remove), -torch.inf)
        return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


    # def generate(self, messages: list[dict]) -> str:
    #     prompt = self.tokenizer.apply_chat_template(
    #         messages, tools=TOOLS_SIGNATURE, tokenize=False, add_generation_prompt=False
    #     )

    def run(self):

        pass
        
if __name__ == "__main__":
    # parse_tool_calls_test()

    device = torch.device("cuda:0")
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    
    # model
    config = MinimindConfig()
    model = MinimindCausalLLM(config).to(device)
    ckpt_path = "ckpts/sft.pth"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    agent = Agent(model, tokenizer, device)