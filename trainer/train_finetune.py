import os
import argparse
import math
from contextlib import nullcontext

import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import json
import random
from transformers import AutoTokenizer
from datasets import load_dataset, Features, Sequence, Value



from model.model_config import MinimindConfig
from model.model import MinimindCausalLLM

"""

sft_t2t(_mini).jsonl has format
    {
        "conversations":[
            {
                "role": "user",
                "content": "xxxx"
            },
            {
                "role": "assistant",
                "content": "xxxx"
            },
            {
                "role": "user",
                "content": "xxxx"
            },
            {
                "role": "assistant",
                "content": "xxxx",
                "reasoning_content": "xxxx"
            }
        ],
        "conversations": [
            {
                "role": "system",
                "content": "",
                "tools": "......"
            },
            {
                "role": "user",
                "content": "xxx", 
            },
            {
                "role": "assistant",
                "content": "xxxx",
                "tool_calls": "xxxx"
            },
            {
                "role": "tool",
                "content": "xxx"
            }
        ]
    }
"""

DATA_PATH = '/home/jiarui/Desktop/Dataset/Minimind/sft_t2t_mini.jsonl'


def preprocess_chat(conversations: list[dict], add_system_ratio=0.2):
    # data with tool is unchanged
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]

    if conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{
            "role": "system",
            "content": random.choice(SYSTEM_PROMPTS)
        }] + conversations
    else:
        return conversations

def postprocess_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # remove think tag
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class SFTDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str, max_length=512, subset=10000):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features(
            {
                'conversations': [
                    {
                        'role': Value('string'), 
                        'content': Value('string'), 
                        'reasoning_content': Value('string'), 
                        'tools': Value('string'), 
                        'tool_calls': Value('string')
                    }
                ]
            }
        )
        self.samples = load_dataset("json", data_files=data_path, split="train", features=features)
        if subset > 0:
            self.samples = self.samples.select(
                range(min(subset, len(self.samples)))
            )
        self.bos_id = tokenizer("%sassistant\n" % tokenizer.bos_token, add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"])

            if message.get("tool_calls"):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        # Calculate loss only on assistant-response tokens; ignore system, user, tool-result, and padding tokens.
        # TODO: shift label, and compute attention mask
        labels = np.array([-100] * len(input_ids))
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                end_ = min(end + len(self.eos_id), self.max_length)
                labels[start:end_] = input_ids[start:end_]

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)

            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = preprocess_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = postprocess_chat(prompt)
        print(prompt)
        # input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # labels = self.generate_labels(input_ids)
        # return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # path parameters
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--ckpt_dir", type=str, default="ckpts")
    parser.add_argument("--log_dir", type=str, default="trainer/log_pretrain")
    parser.add_argument("--from_weight", type=str, default=None)
    parser.add_argument("--save_weight", type=str, default="pretrain.pth")

    # training params
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--subset", type=int, default=-1)

    # infrus
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--print_every", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--compile", action="store_true")

    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    config = MinimindConfig(vocab_size=len(tokenizer))

    # data
    dataset = SFTDataset(tokenizer=tokenizer, data_path=args.data_path, max_length=args.max_length, subset=args.subset)

    # TODO: collate_fn