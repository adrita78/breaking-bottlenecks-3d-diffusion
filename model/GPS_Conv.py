import argparse
import os.path as osp
from typing import Any, Dict, Optional
import math
import inspect
import torch as th
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
)
import torch_sparse
from torch_geometric.utils import degree, sort_edge_index
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
import sys
#sys.path.append("E:/featurization")
from utils.permute_within_batch import permute_within_batch
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from mamba_ssm import Mamba
from mamba_ssm import Mamba2
from .hydra import Hydra
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Sequential, Linear, Dropout
from torch_geometric.utils import degree, to_dense_batch
from typing import Optional, Dict, Any
import inspect

class GPSConv(nn.Module):
    def __init__(
        self,
        channels: int,
        conv: Optional[nn.Module],
        heads: int = 1,
        depth: int = 1,
        dim_head: int = 1,
        num_tokens: int = 128,
        dim_d: int = 1,
        dim_k: int = 1,
        dim_ff: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        att_type: str = 'transformer',
        order_by_degree: bool = False,
        shuffle_ind: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        assert (self.order_by_degree and self.shuffle_ind == 0) or (not self.order_by_degree), \
            f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'

        # --- Attention Modules ---
        if att_type == 'transformer':
            self.attn = nn.MultiheadAttention(channels, heads, dropout=attn_dropout, batch_first=True)
        elif att_type == 'mamba':
            self.self_attn = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        elif att_type == 'mamba-2':
            self.self_attn_1 = Mamba2(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        elif att_type == 'hydra':
            self.self_attn_2 = Hydra(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        elif att_type == 'jamba':
            self.self_attn_j = Jamba(
                dim=channels,
                depth=depth,
                num_tokens=num_tokens,
                d_state=d_state,
                d_conv=d_conv,
                heads=heads,
                num_experts=16,
                num_experts_per_token=4,
            )

        # --- MLP Block ---
        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        # --- Normalizations ---
        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        # Check if norm supports batch argument
        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()
        if hasattr(self, 'attn'):
            self.attn._reset_parameters()
        reset(self.mlp)
        for norm in (self.norm1, self.norm2, self.norm3):
            if norm is not None:
                norm.reset_parameters()

    def forward(self, h: Tensor, x: Tensor, edge_index: Optional[Tensor] = None,
                batch: Optional[Tensor] = None, **kwargs) -> Tensor:

        # --- Move all tensors & submodules to the same device as input ---
        device = h.device
        self.to(device)
        h = h.to(device)
        x = x.to(device)
        if batch is not None:
            batch = batch.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)

        hs = []

        # --- Local MPNN ---
        if self.conv is not None:
            y, x = self.conv(h, x, **kwargs)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = y + h
            if self.norm1 is not None:
                y = self.norm1(y, batch=batch) if self.norm_with_batch else self.norm1(y)
            hs.append(y)

        # --- Global Attention ---
        if self.att_type == 'transformer':
            y, mask = to_dense_batch(h, batch)
            y, _ = self.attn(y, y, y, key_padding_mask=~mask, need_weights=False)
            y = y[mask]
        elif self.att_type == 'hydra':
            y, mask = to_dense_batch(h, batch)
            y = self.self_attn_2(y)[mask]
        elif self.att_type == 'jamba':
            if self.order_by_degree:
                deg = degree(edge_index[0], h.size(0), dtype=torch.long)
                order_tensor = torch.stack([batch, deg], 1).T
                _, h = sort_edge_index(order_tensor, edge_attr=h)

            if self.shuffle_ind == 0:
                y, mask = to_dense_batch(h, batch)
                y = self.self_attn_j(y)[mask]
            else:
                mamba_arr = []
                for _ in range(self.shuffle_ind):
                    y_ind_perm = permute_within_batch(h, batch).to(device)
                    y_i, mask = to_dense_batch(h[y_ind_perm], batch)
                    y_i = self.self_attn_j(y_i)[mask][y_ind_perm]
                    mamba_arr.append(y_i)
                y = sum(mamba_arr) / self.shuffle_ind

        y = F.dropout(y, p=self.dropout, training=self.training)
        y = y + h  # Residual
        if self.norm2 is not None:
            y = self.norm2(y, batch=batch) if self.norm_with_batch else self.norm2(y)
        hs.append(y)

        # --- Combine local + global ---
        out = sum(hs)
        out = out + self.mlp(out)
        if self.norm3 is not None:
            out = self.norm3(out, batch=batch) if self.norm_with_batch else self.norm3(out)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.channels}, conv={self.conv}, heads={self.heads})'