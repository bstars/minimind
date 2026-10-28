import math
from dataclasses import dataclass



@dataclass
class MinimindConfig:
    # Model
    hidden_dim: int = 768
    num_hidden_layers: int = 8
    vocab_size: int = 6400

    # Attention
    num_query_heads: int = 8
    num_kv_heads: int = 4
    flash_attn: bool = True
    
    rope_theta: float = 1e6
    rope_scaling: dict | None = None

    # MoE
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_token: int = 2
    moe_aux_loss_coef: float = 5e-4


    # Other
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0


    # post-init
    head_dim: int | None = None
    intermediate_dim: int | None = None


    def __post_init__(self):
        self.head_dim = self.hidden_dim // self.num_query_heads
        self.intermediate_dim = math.ceil(self.hidden_dim * math.pi / 64) * 64