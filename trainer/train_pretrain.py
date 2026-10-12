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
            tokens = [self.tokenizer.bos_token_id, *content[:self.max_length - 1]]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
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
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # path parameters
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--tokenizer_dir", type=str, default="../tokenizer")
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts")
    parser.add_argument("--log_dir", type=str, default="../log_pretrain")
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
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--subset", type=int, default=None)

    # infrus
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--print_every", type=int, default=8)
    parser.add_argument("--compile", action="store_true")


    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)


    # data
    dataset = PretrainDataset(tokenizer=tokenizer, data_path=args.data_path, max_length=args.max_seq_len, split="train")
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
                    f"ce loss {mean_cross_ent_loss:.4f} | "
                    f"aux loss {mean_aux_loss:.4f} | "
                    f"grad_norm {grad_norm.item():.4f} | "
                    f"lr {current_lr:.2e}"
                )


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







