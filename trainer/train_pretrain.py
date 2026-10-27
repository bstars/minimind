import os
import argparse

from contextlib import nullcontext
from transformers import AutoTokenizer
from datasets import load_dataset

import math
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


import sys
sys.path.append("..")
from model.model_config import MinimindConfig
from model.model import MinimindCausalLLM




DATA_PATH = '/home/jiarui/Desktop/Dataset/Minimind/pretrain_t2t_mini.jsonl'

class PretrainDataset(Dataset):
    def __init__(self, tokenizer, data_path: str, max_length=512, split="train"):
        super().__init__()

        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split=split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        At this point, non_pad_mask is used for MoE load computation, 
        and attention_mask is left as is for future.
        """
        # sample = self.samples[index]
        # tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # input_ids = torch.tensor(input_ids, dtype=torch.long)
        # labels = input_ids.clone()
        # labels[input_ids == self.tokenizer.pad_token_id] = -100
        # non_pad_mask = input_ids != self.tokenizer.pad_token_id
        # return input_ids, labels, non_pad_mask

        sample = self.samples[index]
        content = self.tokenizer(str(sample["text"]), add_special_tokens=False, truncation=False).input_ids
        if len(content) <= self.max_length - 2:
            tokens = [self.tokenizer.bos_token_id, *content, self.tokenizer.eos_token_id]
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            # Fill the sequence, but do not label the artificial cutoff as EOS.
            full_tokens = [self.tokenizer.bos_token_id, *content, self.tokenizer.eos_token_id]

            max_start = len(full_tokens) - self.max_length
            start = torch.randint(0, max_start + 1, (1,)).item()
            tokens = full_tokens[start : start + self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        non_pad_mask = input_ids != self.tokenizer.pad_token_id
        return input_ids, labels, non_pad_mask


class PackedPretrainDataset(Dataset):
    def __init__(self, tokenizer, data_path: str, max_length=512):
        super().__init__()

        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

        samples = load_dataset("json", data_files=data_path, split="train")

        self.packs = []
        current_pack = []
        current_length = 0

        for sample in samples:
            content = tokenizer(
                str(sample["text"]),
                add_special_tokens=False,
                truncation=False,
            ).input_ids
    