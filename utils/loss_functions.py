import numpy as np
import torch.nn.functional as F
import torch as th
import torch
from torch_scatter import scatter

def ph_loss(input,target,c):
    assert input.shape == target.shape, "Input and target must have the same shape"
    return th.sqrt((input - target) ** 2 + c**2) - c


def mean_flat(tensor, batch):
    sum_per_graph = scatter(tensor, batch, dim=0, reduce="sum")
    count_per_graph = scatter(
        torch.ones(batch.size(0), 1, dtype=tensor.dtype, device=tensor.device),
        batch, dim=0, reduce="sum"
    )
    return sum_per_graph / count_per_graph