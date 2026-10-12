import os
import torch
from torch import nn
import torch.nn.functional as F
import einops
from typing import Tuple, List
from transformers import AutoTokenizer, TextStreamer

import sys
sys.path.append("..")
from model.model_config import MinimindConfig
from model.model import MinimindCausalLLM

@torch.inference_mode()
def generate(
    prefix: str, 
    tokenizer: AutoTokenizer,
    model: MinimindCausalLLM,
    max_new_token=512,
    temperature=0.85, 
    top_p=0.85, 
    top_k=50
):

    token_ids = [tokenizer.bos_token_id] + tokenizer(prefix, add_special_tokens=False, truncation=False).input_ids
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    streamer.put(input_ids.cpu())

    past_kv = [None] * model.config.num_hidden_layers
    model.eval()

    for _ in range(max_new_token):
        logits, past_kv, _ = model(
            input_ids, 
            past_kv=past_kv, 
            attention_mask=None, 
            non_pad_mask=None, 
            return_cache=True,
            logits_to_keep=1
        )

        # logits: [1, 1, vocab_size]

        next_token_logits = logits[:, -1, :]

        if temperature == 0:
            # Greedy decoding.
            next_token = torch.argmax(next_token_logits,dim=-1, keepdim=True)

        else:
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                top_k_values, top_k_idx = torch.topk(next_token_logits, k=top_k, dim=-1) # top_k_values [1, k]
                next_token_logits[next_token_logits < top_k_values[:, -1, None]] = -float('inf')

            if top_p is not None:

                # [1, vocab_size], [1, vocab_size]
                sorted_logits, sorted_indices = torch.sort(next_token_logits, dim=-1, descending=True) # sorted_indices # [1, vocab_size]
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) # [1, vocab_size]
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                masked_indices = sorted_indices[0, sorted_mask[0]]
                next_token_logits[0, masked_indices] = float("-inf")

                
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1,)

        next_token_id = next_token.item()
        token_ids.append(next_token_id)

        streamer.put(next_token.cpu())
        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = next_token
    streamer.end()   
    return tokenizer.decode(token_ids, skip_special_tokens=True)
if __name__ == "__main__":
    device = torch.device("cuda:0")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../tokenizer")
    
    # model
    config = MinimindConfig()
    model = MinimindCausalLLM(config).to(device)
    ckpt_path = "../ckpts/pretrain.pth"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


    ret = generate(
        "Machine Learning is",
        tokenizer=tokenizer,
        model=model, 
        max_new_token=350,
        temperature=0.9,
        top_p=None,
        top_k=50
    )
