from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


@functional_transform('add_custom_lap_pe')
class AddCustomLaplacianEigenPE(BaseTransform):
    r"""Adds Laplacian eigenvalues and eigenvectors to the graph as
    attributes (`data.EigVals` and `data.EigVecs` by default, but can be
    customized via `eigval_attr` and `eigvec_attr`)."""

    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        attr_names: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.is_undirected = is_undirected
        self.attr_names = attr_names or {
        "eigvecs": "laplacian_eigenvector_pe",
        "eigvals": "laplacian_eigenvalue_pe",
    }
        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization="sym",
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        if num_nodes < self.SPARSE_THRESHOLD:
            from numpy.linalg import eig, eigh
            eig_fn = eig if not self.is_undirected else eigh
            eig_vals, eig_vecs = eig_fn(L.todense())
        else:
            from scipy.sparse.linalg import eigs, eigsh
            eig_fn = eigs if not self.is_undirected else eigsh
            eig_vals, eig_vecs = eig_fn(
                L,
                k=min(self.k + 1, num_nodes - 1),
                which="SR" if not self.is_undirected else "SA",
                return_eigenvectors=True,
                **self.kwargs,
            )

        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])

        eig_vecs = torch.from_numpy(eig_vecs)
        eig_vals = torch.from_numpy(eig_vals)
        eig_vecs = eig_vecs[:, 1:self.k + 1]
        eig_vals = eig_vals[1:self.k + 1]

        num_computed = eig_vecs.size(1)
        if num_computed < self.k:
            pad_size = self.k - num_computed
            eig_vecs = torch.cat(
                [eig_vecs, eig_vecs.new_zeros((num_nodes, pad_size))], dim=1
            )
            eig_vals = torch.cat([eig_vals, eig_vals.new_zeros(pad_size)])

        # Random sign flip for stability
        sign = -1 + 2 * torch.randint(0, 2, (self.k,))
        eig_vecs = eig_vecs * sign

        data = add_node_attr(data, eig_vecs, attr_name=self.attr_names["eigvecs"])
        data = add_node_attr(data, eig_vals, attr_name=self.attr_names["eigvals"])

        return data
