import argparse
import os.path as osp
from typing import Any, Dict, Optional
import math
import torch as th
import egnn_clean as eg
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
import torch as th
import gc
from torch_geometric.nn import global_add_pool
import inspect
import torch_sparse
from typing import Any, Dict, Optional
import torch.nn.functional as F
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, Sequential
from utils.GPS_Conv import GPSConv
from utils.ESLapPE import EquivStableLapPENodeEncoder
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config
from torch_geometric.graphgym.config import cfg
from utils.ESLapPE import set_cfg_posenc
import os
import argparse
#from featurization import construct_loader
from torch_geometric.data import Batch
import torch
from torch_geometric.nn import global_add_pool
import numpy as np
import torch as th


set_cfg_posenc(cfg)

@th.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        th.ones_like(tensor) for _ in range(th.distributed.get_world_size())
    ]
    th.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = th.cat(tensors_gather, dim=0)
    return output

class GraphModel(nn.Module):
    def __init__(self, hparams):
        super(GraphModel, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.channels       = hparams.channels
        self.heads          = hparams.heads
        self.attn_dropout   = hparams.attn_dropout
        self.in_node_nf     = hparams.in_node_nf
        self.out_node_nf    = hparams.out_node_nf
        self.in_edge_nf     = hparams.in_edge_nf
        self.hidden_nf      = hparams.hidden_nf
        self.pe_dim         = hparams.pe_dim
        self.context_dim    = hparams.context_dim
        self.time_embed_dim = hparams.time_embed_dim
        self.num_layers     = hparams.num_layers
        self.model_type     = hparams.model_type
        self.shuffle_ind    = hparams.shuffle_ind
        self.d_state        = hparams.d_state
        self.d_conv         = hparams.d_conv
        self.order_by_degree= hparams.order_by_degree

        # Embedding layers
        self.node_emb    = Linear(74, self.channels - self.pe_dim - self.time_embed_dim - self.context_dim)
        self.context_emb = Linear(74, self.context_dim)
        #self.pe_lin     = Linear(20, self.pe_dim)
        #self.pe_norm     = BatchNorm1d(20)
        self.pos_encoder = EquivStableLapPENodeEncoder(dim_emb = self.pe_dim)
        self.edge_emb    = Linear(4, self.channels - self.time_embed_dim)


        self.convs = ModuleList()
        for _ in range(self.num_layers):
            if self.model_type == 'egnn':
                conv = eg.EGNN(self.in_node_nf, self.hidden_nf, self.out_node_nf, self.in_edge_nf)
            elif self.model_type == 'mamba-2':
                conv = GPSConv(
                    self.channels,
                    eg.EGNN(self.in_node_nf, self.hidden_nf, self.out_node_nf, self.in_edge_nf),
                    self.heads,
                    self.attn_dropout,
                    att_type='mamba-2',
                    shuffle_ind=self.shuffle_ind,
                    order_by_degree=self.order_by_degree,
                    d_state=self.d_state,
                    d_conv=self.d_conv
                )
            elif self.model_type == 'transformer':
                conv = GPSConv(
                    self.channels,
                    eg.EGNN(self.in_node_nf, self.hidden_nf, self.out_node_nf, self.in_edge_nf),
                    self.heads,
                    self.attn_dropout,
                    att_type='transformer'
                )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
            self.convs.append(conv)

        # Output MLP
        self.mlp = Sequential(
            Linear(self.channels, self.channels // 2),
            ReLU(),
            Linear(self.channels // 2, self.channels // 4),
            ReLU(),
            Linear(self.channels // 4, 1)
        )

    def forward(self, t, context, batch, device=None):
        
        #h_pe = self.pe_norm(pe)
        batch = self.pos_encoder(batch)
        time_emb = timestep_embedding(t, self.time_embed_dim)
        time_emb = time_emb[batch.batch] 
        context = self.context_emb(context)
        context = context[batch.batch]   

        h_node = self.node_emb(batch.x.squeeze(-1)) 
        # Node feature concat
        h = torch.cat((
            h_node,
            batch.pe_EquivStableLapPE,     
            time_emb.squeeze(1),
            context
        ), dim=1)

        # Edge feature concat
        edge_attr = self.edge_emb(batch.edge_attr.squeeze(-1))
        src_nodes = batch.edge_index[0]
        edge_batch = batch.batch[src_nodes]    
        time_emb_edge = time_emb[edge_batch].squeeze(1)
        edge_attr = torch.cat((edge_attr, time_emb_edge), dim=-1)

        # Convolution layers
        for conv in self.convs:
            if self.model_type == 'egnn':
                h, _ = conv(h, batch.pos, edges=batch.edges, edge_attr=edge_attr)
            else:
                h = conv(h, batch.pos, batch.edge_index, batch.batch, edges=batch.edges, edge_attr=edge_attr)
        
        #print("h before global_add_pool", h.shape)
        # Global pooling and MLP head
        #h = global_add_pool(h, batch.batch)
        #print("shape of h after global_add_pool", h.shape)
        output = h
        output = self.mlp(h)
        return output
   
    @th.no_grad()
    def update_xbar(self,  micro, indices: th.Tensor):
        x = micro.x
        data = concat_all_gather(x)         # [total_N, D]
        indices = concat_all_gather(indices)  # [total_N]

        bz = data.shape[0]
        assert indices.shape[0] == bz, \
        f"Mismatch: data has {bz} entries, but indices has {indices.shape[0]}"
        self.x_bar[indices] = data.detach()

    

def timestep_embedding(timesteps, dim, max_period=10000):
      """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
      half = dim // 2
      freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
      ).to(device=timesteps.device)
      args = timesteps[:, None].float() * freqs[None]
      embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
      if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
      return embedding    


