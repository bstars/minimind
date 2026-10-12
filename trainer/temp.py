import os
import argparse

from contextlib import nullcontext
from transformers import AutoTokenizer
from datasets import load_dataset

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


import sys
sys.path.append("..")
from model.model_config import MinimindConfig
from model.model import MinimindCausalLLM




DATA_PATH = '/home/jiarui/Desktop/Dataset/Minimind/pretrain_t2t_mini.jsonl'


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../tokenizer")
    samples = load_dataset("json", data_files=DATA_PATH, split="train")
    lengths = []
    for i in range(100):
        sample = samples[i]
        tokens = tokenizer(str(sample['text']), add_special_tokens=False).input_ids
        lengths.append(len(tokens))

    lengths = np.asarray(lengths)
    print("mean:", lengths.mean())
    print("median:", np.median(lengths))
    print("p90/p95/p99:", np.percentile(lengths, [90, 95, 99]))
    print("max:", lengths.max())
    print("truncated at 340:", np.mean(lengths > 338))
    print("truncated at 512:", np.mean(lengths > 510))