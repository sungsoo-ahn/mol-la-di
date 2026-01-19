"""DiT (Diffusion Transformer) block with AdaLN-Zero conditioning."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP."""

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        # MLP to project sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) integer timesteps

        Returns:
            (B, d_model) timestep embeddings
        """
        half_dim = self.d_model // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero for conditioning.

    Produces scale, shift for pre-norm and gate for residual.
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, d_model * 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple:
        """
        Args:
            c: Conditioning (B, cond_dim)

        Returns:
            Tuple of (shift1, scale1, gate1, shift2, scale2, gate2)
            Each has shape (B, d_model)
        """
        out = self.linear(c)
        return out.chunk(6, dim=-1)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN-Zero conditioning."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms (without affine - AdaLN provides scale/shift)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # AdaLN-Zero modulation
        self.adaLN = AdaLNZero(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, N, d_model)
            c: Conditioning from timestep embedding (B, d_model)

        Returns:
            Updated features (B, N, d_model)
        """
        B, N, _ = x.shape

        # Get AdaLN modulation parameters
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c)
        # Each is (B, d_model)

        # Expand for broadcasting
        shift1 = shift1.unsqueeze(1)  # (B, 1, d_model)
        scale1 = scale1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1

        Q = self.q_proj(h).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        attn_out = self.out_proj(attn_out)

        # Gated residual
        x = x + gate1 * attn_out

        # Feedforward with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(h))))

        # Gated residual
        x = x + gate2 * ff_out

        return x


class DiTBlockWithCrossAttention(nn.Module):
    """DiT block with optional cross-attention for conditioning."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        cross_attention: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.cross_attention = cross_attention

        # Self-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Cross-attention (optional)
        if cross_attention:
            self.cross_q_proj = nn.Linear(d_model, d_model)
            self.cross_k_proj = nn.Linear(d_model, d_model)
            self.cross_v_proj = nn.Linear(d_model, d_model)
            self.cross_out_proj = nn.Linear(d_model, d_model)
            self.norm_cross = nn.LayerNorm(d_model, elementwise_affine=False)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # AdaLN-Zero modulation
        n_params = 9 if cross_attention else 6
        self.adaLN = nn.Linear(d_model, d_model * n_params)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, N, d_model)
            c: Conditioning from timestep (B, d_model)
            context: Optional context for cross-attention (B, M, d_model)

        Returns:
            Updated features (B, N, d_model)
        """
        B, N, _ = x.shape

        # Get modulation parameters
        params = self.adaLN(c)
        if self.cross_attention:
            chunks = params.chunk(9, dim=-1)
            shift1, scale1, gate1 = chunks[0], chunks[1], chunks[2]
            shift_c, scale_c, gate_c = chunks[3], chunks[4], chunks[5]
            shift2, scale2, gate2 = chunks[6], chunks[7], chunks[8]
        else:
            shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(6, dim=-1)

        # Expand for broadcasting
        shift1 = shift1.unsqueeze(1)
        scale1 = scale1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        # Self-attention
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1

        Q = self.q_proj(h).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = x + gate1 * attn_out

        # Cross-attention (if enabled and context provided)
        if self.cross_attention and context is not None:
            shift_c = shift_c.unsqueeze(1)
            scale_c = scale_c.unsqueeze(1)
            gate_c = gate_c.unsqueeze(1)

            h = self.norm_cross(x)
            h = h * (1 + scale_c) + shift_c

            M = context.size(1)
            Q = self.cross_q_proj(h).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
            K = self.cross_k_proj(context).view(B, M, self.nhead, self.head_dim).transpose(1, 2)
            V = self.cross_v_proj(context).view(B, M, self.nhead, self.head_dim).transpose(1, 2)

            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1)
            cross_out = torch.matmul(attn_weights, V)
            cross_out = cross_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
            cross_out = self.cross_out_proj(cross_out)

            x = x + gate_c * cross_out

        # Feedforward
        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(h))))

        x = x + gate2 * ff_out

        return x
