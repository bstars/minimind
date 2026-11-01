import os
import argparse

from contextlib import nullcontext
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F



from model.model_config import MinimindConfig
from model.model import MinimindCausalLLM


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

    Build one packed row on demand:
      1. Start with samples[index].
      2. Fill remaining space with randomly selected samples.
      3. Pass max_length + 1 tokens to _make_pack for label shifting.
    """

    def __init__(
        self,
        tokenizer,
        data_path: str,
        max_length=512,
        max_pos=32768,
        split="train",
        subset=10000,
    ):
        super().__init__()

        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if max_pos < 2:
            raise ValueError("max_pos must be at least 2")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_pos = max_pos
        self.pad_token_id = tokenizer.pad_token_id

        self.samples = load_dataset(
            "json", data_files=data_path, split=split
        )

        if subset > 0:
            self.samples = self.samples.select(
                range(min(subset, len(self.samples)))
            )

        if len(self.samples) == 0:
            raise ValueError("Dataset contains no samples")


    def __len__(self):
        return len(self.samples)

    def _make_pack(self, input_ids, label_ids, position_idxs, segment_ids):
        """
        :param input_ids: List[int], shape [valid_length]
        :param label_ids: List[int], shape [valid_length], BOS label already be -100
        :param position_idxs: List[int], shape [valid_length]
        :param segment_ids: List[int], shape [valid_length], source sample id

        :return: A dict containing:
            input_ids: torch.Tensor[int], [max_length]
            label_ids: torch.Tensor[int], [max_length]
            position_idxs: torch.Tensor[int] [max_length]
            attention_mask: torch.Tensor[bool], [max_length, max_length]
            non_pad_mask: torch.Tensor[bool], [max_length]
        """

        valid_length = len(input_ids) - 1
        if valid_length > self.max_length:
            raise ValueError(f"Pack length {valid_length} exceeds max_length {self.max_length}")
        pad_length = self.max_length - valid_length


        input_ids = torch.tensor(input_ids[:-1] + [self.pad_token_id] * pad_length, dtype=torch.long)
        label_ids = torch.tensor(label_ids[1:] + [-100] * pad_length)
        position_idxs = torch.tensor(position_idxs[:-1] + [0] * pad_length, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids[:-1] + [-1] * pad_length, dtype=torch.long)
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


        return input_ids, label_ids, position_idxs, attention_mask, non_pad_mask


    def __getitem__(self, index):

        current_input_ids = []
        current_label_ids = []
        current_position_idxs = []
        current_segment_ids = []

        target_length = self.max_length + 1
        sample_index = index
        segment_id = 0

        while len(current_input_ids) < target_length:
            sample = self.samples[sample_index]

            token_ids = self.tokenizer(
                str(sample["text"]),
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            sample_input_ids = [self.tokenizer.bos_token_id, *token_ids, self.tokenizer.eos_token_id]
            sample_input_ids = sample_input_ids[:self.max_pos]

            sample_label_ids = sample_input_ids.copy()
            sample_label_ids[0] = -100  # Do not predict BOS

            offset = 0
            take = min(
                target_length - len(current_input_ids),
                len(sample_input_ids) - offset,
            )

            current_input_ids.extend(
                sample_input_ids[offset:offset + take]
            )
            current_label_ids.extend(
                sample_label_ids[offset:offset + take]
            )
            current_position_idxs.extend(
                range(offset, offset + take)
            )
            current_segment_ids.extend([segment_id] * take)

            segment_id += 1

            if len(current_input_ids) < target_length:
                sample_index = torch.randint(
                    len(self.samples), size=(1,)
                ).item()

        return self._make_pack(
            current_input_ids,
            current_label_ids,
            current_position_idxs,
            current_segment_ids,
        )

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
    dataset = PackedPretrainDataset(tokenizer=tokenizer, data_path=args.data_path, max_length=args.max_length, max_pos=config.max_position_embeddings, subset=args.subset)
    dataloader = DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print(len(dataset))


    # model & optimizer
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


    # for input_ids, labels, position_idxs, attention_mask, non_pad_mask in dataloader:
    #     # print(input_ids.shape, labels.shape, position_idxs.shape, attention_mask.shape, non_pad_mask.shape)
    #     input_ids = input_ids.to(device, non_blocking=True)
    #     labels = labels.to(device, non_blocking=True)
    #     position_idxs = position_idxs.to(device, non_blocking=True)
    #     attention_mask = attention_mask.to(device, non_blocking=True)
    #     non_pad_mask = non_pad_mask.to(device, non_blocking=True)

    #     past_kv = [None] * config.num_hidden_layers
    #     logits, _, aux_loss = model(input_ids, position_idxs, past_kv, attention_mask, non_pad_mask, return_cache=False, logits_to_keep=0)
    #     # print(logits.shape)


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


        for batch_idx, (input_ids, label_ids, position_idxs, attention_mask, non_pad_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            label_ids = label_ids.to(device, non_blocking=True)
            position_idxs = position_idxs.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            non_pad_mask = non_pad_mask.to(device, non_blocking=True)

            # accumulate gradient
            group_start = (batch_idx // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, len(dataloader) - group_start)
            with amp_context:
                logits, _, aux_loss = model(
                    input_ids, 
                    position_idxs, 
                    past_kv, 
                    attention_mask, 
                    non_pad_mask, 
                    return_cache=False, 
                    logits_to_keep=0
                )

                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), 
                    label_ids.reshape(-1), 
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


