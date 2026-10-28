import math
import torch
from torch import nn
import torch.nn.functional as F
import einops
from typing import Tuple, List


from model.model_config import MinimindConfig
from model.modules import precompute_freqs_cis, Attention, FeedForward

class MinimindBlock(nn.Module):
    def __init__(self, layer_id:int, config:MinimindConfig):
        super().__init__()

        self.attn = Attention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.mlp = FeedForward(config)
        # self.mlp = MoEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(
        self, 
        x, 
        position_embedding, 
        past_kv=None, 
        attention_mask=None, 
        non_pad_mask=None,
        return_cache=False
    ):
        """
        In training mode, past_kv should be None, and attention_mask is not None, attention_mask should already take care of causality
        In inference mode, past_kv might not be None, attention_mask is None, causality is taken care by this function

        :param x: [batch_size, seq_len, hidden_dim]
        :param position_embedding: (cos, sin)
            cos: [batch, seq_len, head_dim]
            sin: [batch, seq_len, head_dim]
        :param past_kv: (k, v)
            k: [batch_size, past_len, num_kv_heads, head_dim]
            v: [batch_size, past_len, num_kv_heads, head_dim]
        :param attention_mask: [batch_size, seq_len, seq_len]
        :param return_cache: bool

        :return output: [batch_size, seq_len, hidden_dim]
        :return kv_cache: [k, v]
            k: [batch_size, past_len + seq_len, num_kv_heads, head_dim]
            v: [batch_size, past_len + seq_len, num_kv_heads, head_dim]
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
        position_idxs: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None], 
        attention_mask: torch.Tensor | None = None, 
        non_pad_mask: torch.Tensor | None = None,
        return_cache=False
    ):
        """            
        In training mode, past_kv should be None, and attention_mask is not None, attention_mask should already take care of causality
        In inference mode, past_kv might not be None, attention_mask is None, causality is taken care by this function

        :param x: [batch_size, seq_len, hidden_dim]
        :param past_kv: [(k1,v1), (k2,v2), ... ]
                ki: [batch_size, past_seq_len, head_dim]
                vi: [batch_size, past_seq_len, head_dim]


        :param attention_mask: [batch_size, seq_len, seq_len]
        :param non_pad_mask: [batch_size, seq_len]
        :param return_cache: bool

        :return output: [batch_size, seq_len, hidden_dim]
        :return kv_cache: [(k1,v1), (k2,v2), ... ]
            ki: [batch_size, past_len + seq_len, head_dim]
            vi: [batch_size, past_len + seq_len, head_dim]
        """
        if len(past_kv) != len(self.layers): raise ValueError("past_kv must contain one entry per layer")

        x = self.dropout(self.word_embedding(input_ids))

        cos = self.cos[position_idxs]  # [batch_size, seq_len, head_dim]
        sin = self.sin[position_idxs]  # [batch_size, seq_len, head_dim]
        position_embedding = (cos, sin)

        kv_cache = []
        aux_loss = 0.

        for layer, past_kv_layer in zip(self.layers, past_kv):
            x, kv_cache_layer = layer(x, position_embedding, past_kv_layer, attention_mask, non_pad_mask, return_cache)
            kv_cache.append(kv_cache_layer)
            aux_loss += layer.mlp.aux_loss

        x = self.norm(x)
        return x, kv_cache, aux_loss
