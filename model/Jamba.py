from torch import Tensor, nn
from zeta import MambaBlock, MultiQueryAttention
from zeta.nn import FeedForward
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
from moe import MoE


# -------------------------------------------------
# Transformer Block (continuous)
# -------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = SimpleRMSNorm(dim)
        self.attn = MultiQueryAttention(dim, heads)

        self.norm2 = SimpleRMSNorm(dim)
        self.ffn = FeedForward(
            dim,
            dim,
            mult=4,
            swish=True,
            post_act_ln=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x, _, _ = self.attn(self.norm1(x))
        x = x + res

        res = x
        x = self.ffn(self.norm2(x))
        return x + res


# -------------------------------------------------
# Mamba + MoE Block
# -------------------------------------------------
class MambaMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        num_experts: int = 8,
    ):
        super().__init__()

        self.norm1 = SimpleRMSNorm(dim)
        self.mamba = MambaBlock(
            dim,
            depth=1,
            d_state=d_state,
            d_conv=d_conv,
        )

        self.norm2 = SimpleRMSNorm(dim)
        self.moe = MoE(
            dim,
            num_experts=num_experts,
            hidden_dim=dim * 4,
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.mamba(self.norm1(x))
        x = x + res

        res = x
        moe_out, _ = self.moe(self.norm2(x))
        return moe_out + res


# -------------------------------------------------
# Jamba Block (continuous, graph-safe)
# -------------------------------------------------
class JambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        heads: int,
        num_experts: int = 8,
    ):
        super().__init__()

        self.mamba1 = MambaMoEBlock(dim, d_state, d_conv, num_experts)
        self.attn = TransformerBlock(dim, heads)
        self.mamba2 = MambaMoEBlock(dim, d_state, d_conv, num_experts)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mamba1(x)
        x = self.attn(x)
        x = self.mamba2(x)
        return x


# -------------------------------------------------
# Jamba Stack (NO embedding, NO logits)
# -------------------------------------------------
class Jamba(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int,
        d_conv: int,
        heads: int,
        num_experts: int = 8,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                JambaBlock(
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    heads=heads,
                    num_experts=num_experts,
                )
                for _ in range(depth)
            ]
        )

        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (N, D) or (B, N, D) continuous features
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
