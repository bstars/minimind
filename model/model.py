import math
import torch
from torch import nn
import torch.nn.functional as F
import einops
from typing import Tuple, List

import sys
sys.path.append("..")
from model.model_config import MinimindConfig
from model.modules import precompute_freqs_cis, Attention, FeedForward, MoEFeedForward


class MinimindBlock(nn.Module):
    def __init__(self, layer_id:int, config:MinimindConfig):
        super().__init__()

        self.attn = Attention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = MoEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(
        self, 
        x, 
        position_embedding, 
        past_kv, 
        attention_mask=None, 
        non_pad_mask=None,
        return_cache=False
    ):
        """
        :param x: [batch_size, seq_len, hidden_dim]
        :param position_embedding: (cos, sin)
            cos: [seq_len, head_dim]
            sin: [seq_len, head_dim]
        :param past_kv: (k, v)
            k: [batch_size, past_seq_len, num_kv_heads, head_dim]
            v: [batch_size, past_seq_len, num_kv_heads, head_dim]
        :param attention_mask: [batch_size, seq_len]
        :param non_pad_mask: [batch_size, seq_len]
        :param return_cache: bool
        

        :return output: [batch_size, seq_len, hidden_dim]
        :return kv_cache: [k, v]
            k: [batch_size, total_seq_len, head_dim]
            v: [batch_size, total_seq_len, head_dim]
        """
        residual = x

        # Attention
        x, kv_cache = self.attn(
            self.input_layernorm(x),
            position_embedding,
            past_kv, 
            attention_mask,
            return_cache
        )
        x += residual

        # FeedForward
        x = x + self.mlp(self.post_attention_layernorm(x), non_pad_mask)
        return x, kv_cache

class MinimindModel(nn.Module):
    def __init__(self, config:MinimindConfig):
        super().__init__()

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            MinimindBlock(l, config) for l in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        cos, sin = precompute_freqs_cis(
            head_dim=config.head_dim,
            max_len=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )

        self.register_buffer("cos", cos, persistent=False) # [max_len, head_dim]
        self.register_buffer("sin", sin, persistent=False) # [max_len, head_dim]

    def forward(
        self, 
        input_ids: torch.Tensor, 
        past_kv: List[Tuple[torch.Tensor]] | List[None], 
        attention_mask: torch.Tensor | None = None, 
        non_pad_mask: torch.Tensor | None = None,
        return_cache=False
    ):
        """
        :param input_ids: [batch_size, seq_len]
        :param past_kv: 
            [(k1,v1), (k2,v2), ... ]
            ki: [batch_size, past_seq_len, head_dim]
            vi: [batch_size, past_seq_len, head_dim]
        :param attention_mask: [batch_size, seq_len]
        :param non_pad_mask: [batch_size, seq_len]
        :param return_cache: bool

        :return x: [batch_size, seq_len, hidden_dim]
        :return kv_cache: 
            [(k1,v1), (k2,v2), ... ]
            ki: [batch_size, past_seq_len + seq_len, head_dim]
            vi: [batch_size, past_seq_len + seq_len, head_dim]

        """
        batch_size, seq_len = input_ids.shape

        if past_kv[0] is None: 
            start_pos = 0
        else:
            start_pos = past_kv[0][0].shape[1]

        x = self.dropout(self.word_embedding(input_ids))
        position_embeddings = (
            self.cos[start_pos: start_pos+seq_len],
            self.sin[start_pos: start_pos+seq_len]
        )

        kv_cache = []
        aux_loss = 0.
        
        for layer, past_kv_step in zip(self.layers, past_kv):
            x, kv_cache_step = layer(x, position_embeddings, past_kv_step, attention_mask, non_pad_mask, return_cache)
            kv_cache.append(kv_cache_step)

            
            aux_loss += layer.mlp.aux_loss

        x = self.norm(x)

        return x, kv_cache, aux_loss


class MinimindCausalLLM(nn.Module):
    def __init__(self, config:MinimindConfig):
        super().__init__()
        self.config = config
        self.model = MinimindModel(self.config)
        self.llm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.llm_head.weight = self.model.word_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        past_kv: List[Tuple[torch.Tensor]] | List[None], 
        attention_mask: torch.Tensor | None = None, 
        non_pad_mask: torch.Tensor | None = None,
        return_cache=False,
        logits_to_keep:int=0
    ):
        """
        :param input_ids: [batch_size, seq_len]
        :param past_kv: 
            [(k1,v1), (k2,v2), ... ]
            ki: [batch_size, past_seq_len, head_dim]
            vi: [batch_size, past_seq_len, head_dim]
        :param attention_mask: [batch_size, seq_len]
        :param non_pad_mask: [batch_size, seq_len]
        :param return_cache: bool
        :param logits_to_keep: int

        :return logits: [batch_size, ret_len, vocab_size]
        :return kv_cache: 
            [(k1,v1), (k2,v2), ... ]
            ki: [batch_size, past_seq_len + seq_len, head_dim]
            vi: [batch_size, past_seq_len + seq_len, head_dim]
        """
        x, kv_cache, aux_loss = self.model(input_ids, past_kv, attention_mask, non_pad_mask, return_cache)

        # x: [batch_size, seq_len, hidden_dim]
        logits = self.llm_head(x)
        return logits[:, -logits_to_keep:, :], kv_cache, aux_loss