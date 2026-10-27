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

class PackedPretrainDataset(Dataset):
    """
    Each item is a fixed-length packed row:

      input_ids:      [L]
      labels:         [L], with BOS and padding set to -100
      position_ids:   [L], reset to 0 at each sample's BOS
      attention_mask: [L, L], bool; True means the query may attend to the key
      non_pad_mask:   [L], bool

    Rows are causal and isolated by sample. The final row is padded if needed.
    """

    def __init__(self, tokenizer, data_path: str, max_length=512, split="train"):
        super().__init__()

        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        samples = load_dataset("json", data_files=data_path, split=split)

        self.packs = []
        current_input_ids = []
        current_labels = []
        current_position_idxs = []
        current_segment_ids = []
        

        segment_id = 0
        for sample in samples:
            sample_input_ids = tokenizer(str(sample["text"]), add_special_tokens=False, truncation=False).input_ids
            sample_input_ids = [tokenizer.bos_token_id, *sample_input_ids, tokenizer.eos_token_id]

            sample_labels = sample_input_ids.copy()
            sample_labels[0] = -100 # Do not predict BOS


            # fill up current pack
            offset = 0
            while offset < len(sample_input_ids):
                take = min(
                    self.max_length - len(current_input_ids),
                    len(sample_input_ids) - offset
                )

                current_input_ids.extend(sample_input_ids[offset: offset+take])
                current_labels.extend(sample_labels[offset: offset+take])
                current_position_idxs.extend(range(offset, offset+take))
                current_segment_ids.extend([segment_id] * take)

                offset += take

                if len(current_input_ids) == self.max_length:
                    self.packs.append(self._make_pack(current_input_ids, current_labels, current_position_idxs, current_segment_ids))

                    current_input_ids = []
                    current_labels = []
                    current_position_idxs = []
                    current_segment_ids = []
            segment_id += 1

        # Ignoring the last data
        # Essentialy no data padded
        # if len(current_input_ids) > 0:
        #     self.packs.append(
        #         self._make_pack(
        #             current_input_ids,
        #             current_labels,
        #             current_position_idxs,
        #             current_segment_ids,
        #         )
        #     )

    def _make_pack(self, input_ids, labels, position_idxs, segment_ids):
        """
        :param input_ids: List[int], shape [valid_length]
        :param labels: List[int], shape [valid_length], BOS label already be -100
        :param position_idxs: List[int], shape [valid_length]
        :param segment_ids: List[int], shape [valid_length], source sample id

        :return: A dict containing:
            input_ids: torch.Tensor[int], [max_length]
            labels: torch.Tensor[int], [max_length]
            position_idxs: torch.Tensor[int] [max_length]
            attention_mask: torch.Tensor[bool], [max_length, max_length]
            non_pad_mask: torch.Tensor[bool], [max_length]
        """

        valid_length = len(input_ids)
        if valid_length > self.max_length:
            raise ValueError(f"Pack length {valid_length} exceeds max_length {self.max_length}")
        pad_length = self.max_length - valid_length


        input_ids = torch.tensor(input_ids + [self.pad_token_id] * pad_length, dtype=torch.long)
        labels = torch.tensor(labels + [-100] * pad_length)
        position_idxs = torch.tensor(position_idxs + [0] * pad_length, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids + [-1] * pad_length, dtype=torch.long)
        non_pad_mask = torch.arange(self.max_length) < valid_length

        positions = torch.arange(self.max_length)
        causal_mask = positions[:, None] >= positions[None, :]
        same_segment = segment_ids[:, None] == segment_ids[None, :]
        attention_mask = (
            causal_mask
            & same_segment
            & non_pad_mask[None, :]
        )

        # Numerical safeguard such that each pad token attend to itself
        pad_positions = torch.arange(valid_length, self.max_length)
        attention_mask[pad_positions, pad_positions] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_idxs": position_idxs,
            "attention_mask": attention_mask,
            "non_pad_mask": non_pad_mask,
        }

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, index):
        return self.packs[index]

        

