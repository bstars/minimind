
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

import sys
sys.path.append("..")
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

def pre_process_chat(conversations: list[dict], add_system_ratio=0.2):
    # keep data with tool unchanged
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

def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    # remove think tag
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class SFTDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str, max_length=512):
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
        self.bos_id = tokenizer("%sassistant\n" % tokenizer.bos_token, add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations: list[dict]):
        messages = []
        tools = None

        for message in conversations:
            message = dict(message)

            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]

            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
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
        conversations = pre_process_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        non_pad_mask = input_ids != self.tokenizer.pad_token_id
        return input_ids, labels, non_pad_mask

        

def get_lr(
    current_step: int,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
    min_lr: float
):
    current_step = min(max(current_step, 1), total_steps)

    # linear warmup
    if current_step <= warmup_steps:
        return peak_lr * current_step / warmup_steps

    # cos decay
    decay_progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    cos_factor = 0.5 * (1. + math.cos(math.pi * decay_progress))

    return min_lr + cos_factor * (peak_lr - min_lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind SFT")

    # path parameters
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--tokenizer_dir", type=str, default="../tokenizer")
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts")
    parser.add_argument("--log_dir", type=str, default="../log_sft")
    parser.add_argument("--from_weight", type=str, default="pretrain.pth")
    parser.add_argument("--save_weight", type=str, default="sft.pth")

    # training params
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--subset", type=int, default=None)

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


    # data
    dataset = SFTDataset(tokenizer=tokenizer, data_path=args.data_path, max_length=args.max_seq_len)


    # visualize an example
    # sample = dataset.samples[-5]
    # conversations = pre_process_chat(sample["conversations"])
    # print(conversations)
    # prompt = dataset.create_chat_prompt(conversations)
    # print(prompt)

    if args.subset: dataset = Subset(dataset, range(args.subset))
        
    dataloader = DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # model & optimizer
    config = MinimindConfig(vocab_size=len(tokenizer))
    model = MinimindCausalLLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    if args.from_weight is not None:
        ckpt_path = os.path.join(args.ckpt_dir, args.from_weight)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    # learning rate scheduling
    updates_per_epoch = math.ceil(len(dataloader) / args.accumulation_steps)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = min(max(warmup_steps, 1), total_steps)
    min_lr = args.learning_rate * args.min_lr_ratio

    # mixed precision
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    use_grad_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            

    # compile
    if args.compile:
        model = torch.compile(model)

    # logging
    writer = SummaryWriter(log_dir=args.log_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device: {device}")
    print(f"Training samples: {len(dataset):}")
    print(f"Model parameters: {parameter_count:}")
    print(f"Optimizer steps: {total_steps:}")
    print(f"Mixed-precision dtype: {amp_dtype if use_amp else 'float32'}")


    # training loop
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    past_kv = [None] * config.num_hidden_layers

    for epoch in range(args.epochs):
        accumulated_cross_ent_loss = 0.0
        accumulated_aux_loss = 0.0

        for batch_idx, (input_ids, label_ids, non_pad_mask) in enumerate(dataloader):

            input_ids = input_ids.to(device, non_blocking=True)
            label_ids = label_ids.to(device, non_blocking=True)
            non_pad_mask = non_pad_mask.to(device, non_blocking=True)

            # accumulate gradient
            group_start = (batch_idx // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, len(dataloader) - group_start)
            with amp_context:
                logits, _, aux_loss = model(
                    input_ids[:, :-1],
                    past_kv,
                    attention_mask=None,
                    non_pad_mask=non_pad_mask[:, :-1],
                    return_cache=False
                )

                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), 
                    label_ids[:, 1:].reshape(-1), 
                    ignore_index=-100
                )
                loss = ce_loss + config.moe_aux_loss_coef * aux_loss
                scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()
            accumulated_cross_ent_loss += ce_loss.detach().item()
            accumulated_aux_loss += aux_loss.detach().item()

            # accumulated gradient step
            is_accumulation_boundary = (batch_idx + 1) % args.accumulation_steps == 0
            is_last_batch = batch_idx + 1 == len(dataloader)
            if not (is_accumulation_boundary or is_last_batch): continue

            global_step += 1
            current_lr = get_lr(
                current_step=global_step,
                total_steps=total_steps,
                peak_lr=args.learning_rate,
                warmup_steps=warmup_steps,
                min_lr=min_lr
            )

            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


            
            mean_cross_ent_loss = accumulated_cross_ent_loss/ group_size
            mean_aux_loss = accumulated_aux_loss / group_size
            accumulated_cross_ent_loss = 0.
            accumulated_aux_loss = 0.

            writer.add_scalar("train/cross_entropy", mean_cross_ent_loss, global_step)
            writer.add_scalar("train/MoE_aux", mean_aux_loss, global_step)
            writer.add_scalar("train/gradient_norm", grad_norm.item(), global_step)
            writer.add_scalar("train/learning_rate", current_lr, global_step)


            if global_step % args.print_every == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} | "
                    f"step {global_step}/{total_steps} | "
                    f"ent loss {mean_cross_ent_loss:.4f} | "
                    f"aux loss {mean_aux_loss:.4f} | "
                    f"grad_norm {grad_norm.item():.4f} | "
                    f"lr {current_lr:.2e}"
                )

            if global_step % args.save_every == 0:
                save_path = os.path.join(args.ckpt_dir, "%d.pth" % (global_step))
                
                if args.compile:
                    raw_model = getattr(model, "_orig_mod", model)
                    state_dict = raw_model.state_dict()
                    
                else:
                    state_dict = model.state_dict()
            
                state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
                torch.save(state_dict, save_path)


    save_path = os.path.join(args.ckpt_dir, args.save_weight)

    if args.compile:
        raw_model = getattr(model, "_orig_mod", model)
        state_dict = raw_model.state_dict()
        
    else:
        state_dict = model.state_dict()

    state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
    torch.save(state_dict, save_path)
    
    writer.flush()
    writer.close()