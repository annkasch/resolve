from typing import NamedTuple

import torch


class ContextSet(NamedTuple):
    theta: torch.Tensor
    phi: torch.Tensor
    y: torch.Tensor | None
    idx: torch.Tensor
    file_indices: torch.Tensor


class QuerySet(NamedTuple):
    theta: torch.Tensor
    phi: torch.Tensor
    idx: torch.Tensor
    file_indices: torch.Tensor


class BatchCollection(NamedTuple):
    context: ContextSet
    query: QuerySet
    target_y: torch.Tensor | None
