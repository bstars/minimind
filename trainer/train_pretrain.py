import os
import argparse

from contextlib import nullcontext
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

import math
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F



from model.model_config import MinimindConfig
# from model.model import MinimindCausalLLM


DATA_PATH = '/home/jiarui/Desktop/Dataset/Minimind/pretrain_t2t_mini.jsonl'


def tokenize_batch(batch, tokenizer):
    return {
        "input_ids": tokenizer(
            [str(text) for text in batch["text"]],
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    }

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
        # samples = samples.select(range(min(100_000, len(samples))))

        tokenized = samples.map(
            tokenize_batch,
            batched=True,
            batch_size=1024,
            num_proc=8,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=samples.column_names,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        self.packs = []
        current_input_ids = []
        current_labels = []
        current_position_idxs = []
        current_segment_ids = []
        

        segment_id = 0
        for sample in tqdm(tokenized, desc="Packing"):
            sample_input_ids = [
                tokenizer.bos_token_id,
                *sample["input_ids"],
                tokenizer.eos_token_id,
            ]

            sample_labels = sample_input_ids.copy()
            sample_labels[0] = -100  # Do not predict BOS

            # Existing packing logic continues here.
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
                    # self.packs.append(self._make_pack(current_input_ids, current_labels, current_position_idxs, current_segment_ids))
                    self.packs.append((current_input_ids, current_labels, current_position_idxs, current_segment_ids))

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

        # return {
        #     "input_ids": input_ids,
        #     "labels": labels,
        #     "position_idxs": position_idxs,
        #     "attention_mask": attention_mask,
        #     "non_pad_mask": non_pad_mask,
        # }

        return input_ids, labels, position_idxs, attention_mask, non_pad_mask

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, index):
        input_ids, labels, position_idxs, segment_ids = self.packs[index]
        return self._make_pack(input_ids, labels, position_idxs, segment_ids)


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
    dataset = PackedPretrainDataset(tokenizer=tokenizer, data_path=args.data_path, max_length=args.max_length)
    dataloader = DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print(len(dataset))

    # for input_ids, labels, position_idxs, attention_mask, non_pad_mask in dataloader:
    #     print(input_ids.shape, labels.shape, position_idxs.shape, attention_mask.shape, non_pad_mask.shape)

