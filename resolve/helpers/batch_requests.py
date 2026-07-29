from __future__ import annotations

from typing import NamedTuple

import torch
from torch.utils.data import Sampler


class BatchRequest(NamedTuple):
    indices: torch.Tensor
    context_size: int
    context_is_subset: bool


class BatchSpan(NamedTuple):
    start: int
    stop: int
    context_size: int
    context_is_subset: bool


class BatchRequestSampler(Sampler):
    """Main-process sampler that sends complete batch requests to workers."""

    def __init__(self):
        self._order = torch.empty(0, dtype=torch.long)
        self._spans = ()

    def set_plan(self, order, spans):
        self._order = order
        self._spans = tuple(spans)

    def __iter__(self):
        order = self._order
        spans = self._spans
        return (
            BatchRequest(
                indices=order[span.start:span.stop],
                context_size=span.context_size,
                context_is_subset=span.context_is_subset,
            )
            for span in spans
        )

    def __len__(self):
        return len(self._spans)
