import os
import numpy as np
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

class ThinkingStreamer(TextStreamer):
    THINK_COLOR = "\033[90m"  # Gray; use "\033[36m" for cyan.
    RESET = "\033[0m"

    def on_finalized_text(self, text, stream_end=False):
        text = text.replace(
            "<think>", self.THINK_COLOR + "<think>"
        ).replace(
            "</think>", "</think>" + self.RESET
        )

        if stream_end:
            # Restore the color even if generation stopped mid-thinking.
            text += self.RESET

        super().on_finalized_text(text, stream_end=stream_end)


@torch.inference_mode()
def generate(
    device,
    token_ids,
    streamer: TextStreamer,
    past_kv: List[None] | List[Tuple[torch.Tensor]],
    tokenizer: AutoTokenizer,
    model: MinimindCausalLLM,
    max_new_token=512,
    temperature=0.99, 
    top_p=0.95, 
    top_k=20,
):

    """
    Query LLM to respond to user input
    """

    # start_ids = tokenizer("%sassistant\n<think>\n\n</think>\n\n" % tokenizer.bos_token, add_special_tokens=False).input_ids
    start_ids = tokenizer("%sassistant\n" % tokenizer.bos_token, add_special_tokens=False).input_ids
    end_ids = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    token_ids.extend(start_ids)
    input_ids = torch.tensor([start_ids], dtype=torch.long, device=device)
    

    for step in range(max_new_token):
        past_len = 0 if past_kv[0] is None else past_kv[0][0].shape[1]
        position_idxs = torch.arange(
            past_len,
            past_len + input_ids.shape[1],
            dtype=torch.long,
            device=device,
        )[None, :]
        logits, past_kv, _ = model(
            input_ids,
            position_idxs,
            past_kv,
            attention_mask=None, 
            non_pad_mask=None,
            return_cache=True,
            logits_to_keep=1
        )

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

        #
        # When hit the end, update the KV cache and includ the final token
        #
        if next_token_id == tokenizer.eos_token_id:

            end_input_ids = torch.tensor([end_ids], dtype=torch.long, device=device)
            token_ids.extend(end_ids[1:])
            streamer.put(end_input_ids.cpu())

            past_len = 0 if past_kv[0] is None else past_kv[0][0].shape[1]
            position_idxs = torch.arange(
                past_len,
                past_len + end_input_ids.shape[1],
                dtype=torch.long,
                device=device,
            )[None, :]

            _, past_kv, _ = model(
                end_input_ids,
                position_idxs,
                past_kv,
                attention_mask=None, 
                non_pad_mask=None,
                return_cache=True,
                logits_to_keep=1
            )
            streamer.end()
            return token_ids, past_kv

        #
        # if over maximum length, force termination
        #
        if step == max_new_token - 1:
            token_ids.pop()

            end_input_ids = torch.tensor(
                [end_ids],
                dtype=torch.long,
                device=device,
            )
            past_len = 0 if past_kv[0] is None else past_kv[0][0].shape[1]
            position_idxs = torch.arange(
                past_len,
                past_len + end_input_ids.shape[1],
                dtype=torch.long,
                device=device,
            )[None, :]
            token_ids.extend(end_ids)


            # Stream the forced ending.
            streamer.put(end_input_ids.cpu())

            _, past_kv, _ = model(
                end_input_ids,
                position_idxs,
                past_kv,
                attention_mask=None, 
                non_pad_mask=None,
                return_cache=True,
                logits_to_keep=1
            )
            streamer.end()
            return token_ids, past_kv
        
        # Ordinary token
        streamer.put(next_token.cpu())
        input_ids = next_token
            

@torch.inference_mode()
def chat(
    device,
    tokenizer: AutoTokenizer,
    model: MinimindCausalLLM,
):

    #
    # initialize system prompt and kv cache
    #
    input_str = "<|im_start|>system\n你是一个专业的AI助手，请提供有价值的回答。<|im_end|>\n"
    token_ids = tokenizer(input_str, add_special_tokens=False, truncation=False).input_ids
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    position_idxs = torch.arange(input_ids.shape[1], dtype=torch.long, device=device)[None, :]

    past_kv = [None] * model.config.num_hidden_layers
    model.eval()


    _, past_kv, _ = model(
        input_ids,
        position_idxs,
        past_kv,
        attention_mask=None, 
        non_pad_mask=None,
        return_cache=True,
        logits_to_keep=1
    )


    streamer = ThinkingStreamer(
        tokenizer,
        skip_prompt=False,
        skip_special_tokens=True,
    )
    streamer.put(input_ids.cpu())

    while True:
        print("--" * 20)

        # append user input to chat history
        input_str = "<|im_start|>user\n" + input() + "<|im_end|>\n"
        input_ids = tokenizer(input_str, add_special_tokens=False, truncation=False).input_ids
        token_ids.extend(input_ids)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        past_len = 0 if past_kv[0] is None else past_kv[0][0].shape[1]
        position_idxs = torch.arange(
            past_len,
            past_len + input_ids.shape[1],
            dtype=torch.long,
            device=device,
        )[None, :]

        _, past_kv, _ = model(
            input_ids,
            position_idxs,
            past_kv,
            attention_mask=None, 
            non_pad_mask=None,
            return_cache=True,
            logits_to_keep=1
        )

        # query LLM and update chat KV history
        token_ids, past_kv = generate(
            device,
            token_ids=token_ids,
            streamer=streamer,
            past_kv=past_kv,
            tokenizer=tokenizer,
            model=model
        )



if __name__ == '__main__':
    device = torch.device("cuda:0")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    
    # model
    config = MinimindConfig()
    model = MinimindCausalLLM(config).to(device)
    ckpt_path = "ckpts/sft.pth"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


    chat(device, tokenizer, model)