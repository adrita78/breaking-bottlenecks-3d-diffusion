import torch

def permute_within_batch(x, batch):
    unique_batches = torch.unique(batch)
    permuted_indices = []

    for batch_index in unique_batches:
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]
        permuted_indices.append(permuted_indices_in_batch)
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices
