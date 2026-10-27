import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List

from model.model_config import MinimindConfig


def precompute_freqs_cis(
    head_dim: int,
    max_len: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    inv_freq = 1.0 / (
        rope_base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
    ) # [head_dim//2,]

    if rope_scaling is not None:
        # TODO: implement yarn for long-context inference
        raise NotImplementedError(
            "TODO: Implement YaRN for long-context inference."
            "Use rope_scaling=None for now."
        )

    attn_factor = 1.0
    position = torch.arange(max_len, dtype=torch.float32) # [max_seq_len]
    angles = torch.outer(position, inv_freq)    # [max_seq_legnth, head_dim//2]
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attn_factor   # [max_seq_length, head_dim]
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attn_factor   # [max_seq_length, head_dim]
    return cos, sin

def rot_half(x):
    dim = x.shape[-1]
    return torch.cat([-x[..., dim//2:], x[..., :dim//2]], dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim=1):
    """
    :param q: [batch_size, seq_len, num_query_heads, head_dim]
    :param k: [batch_size, seq_len, num_key_value_heads, head_dim]
    :param sin, cos: [seq_len, head_dim//2]

    :return q_rot: same shape as q
    :return k_rot: same shape as k

    [cos  -sin] * [x1] = [cos * x1 - sin * x2] = cos * x + sin * rot_half(x)
    [sin   cos]   [x2]   [cos * x2 + sin * x1]
    """

    _cos = torch.unsqueeze(cos, dim=unsqueeze_dim)  # [seq_len, 1, head_dim//2]
    _sin = torch.unsqueeze(sin, dim=unsqueeze_dim)  # [seq_len, 1, head_dim//2]

    q_rot = (q * _cos + rot_half(q) * _sin).to(q.dtype)
    k_rot = (k * _cos + rot_half(k) * _sin).to(k.dtype)
    return q_rot, k_rot

def repeat_kv(x: torch.Tensor, n_rep: int):
    """
    :param x: [batch_size, seq_len, num_kv_heads, head_dim]
    :return: [batch_size, seq_len, num_kv_heads * n_rep, head_dim]
    """
    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1: return x
    return x[:, :, :, None, :].expand(
        batch_size, seq_len, num_kv_heads, n_rep, head_dim
    ).reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, config: MinimindConfig):
        super().__init__()

        """
        Query heads:    Q0    Q1    Q2    Q3    Q4    Q5    Q6    Q7
                         \    /      \    /      \    /      \    /
        KV heads:          KV0         KV1         KV2         KV3
        """
        self.num_kv_heads = config.num_kv_heads
        self.num_query_heads = config.num_query_heads
        self.n_rep = self.num_query_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.input_dim, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.input_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.input_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_query_heads * self.head_dim, self.output_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        position_embedding: Tuple[torch.Tensor, torch.Tensor], 
        past_kv: Tuple[torch.Tensor, torch.Tensor] | None = None, 
        attention_mask: torch.Tensor | None = None,
        return_cache: bool = False
    ):
        """
        :param x: [batch_size, seq_len, hidden_dim]
        :param position_embedding: (cos, sin)
            cos: [seq_len, head_dim]
            sin: [seq_len, head_dim]
        :param past_kv: (k, v)
            k: [batch_size, past_seq_len, num_kv_heads, head_dim]
            v: [batch_size, past_seq_len, num_kv_heads, head_dim]
        :param attention_mask: [batch_size, total_seq_len]
        :param return_cache

        :return output: [batch_size, seq_len, hidden_dim]
        :return kv_cache: [k, v]
            k: [batch_size, total_seq_len, head_dim]
            v: [batch_size, total_seq_len, head_dim]
        """

        batch_size, seq_len, _ = x.shape
        xq = self.q_proj(x).view(batch_size, seq_len, self.num_query_heads, self.head_dim) # [batch_size, seq_len, num_query_heads, head_dim]
        xk = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim) # [batch_size, seq_len, num_kv_heads, head_dim]
        xv = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim) # [batch_size, seq_len, num_kv_heads, head_dim]

        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=1)

        if past_kv is not None:
            past_k, past_v = past_kv
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)

        # xq: [batch, seq_len, num_query_heads, head_dim]
        # xk: [batch_size, total_seq_len, num_kv_heads, head_dim]
        # xv: [batch_size, total_seq_len, num_kv_heads, head_dim]