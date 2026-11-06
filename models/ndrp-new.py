from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor


class NodeDropPooling(torch.nn.Module):
    r"""Uniform random node pooling with a TopK-compatible interface.

    Samples nodes without replacement separately within each graph and returns
    the sparse induced subgraph. No dense adjacency matrix is constructed.
    Features are not scored or gated; returned scores are all ones.

    Args:
        in_channels: Number of input features per node.
        ratio: Fraction of nodes to retain, rounded up, in (0, 1], or a
            positive integer number of nodes to retain per graph. Counts are
            capped at the graph size and every nonempty graph keeps a node.
        min_score: Must be None; score thresholds do not apply to random pooling.
        multiplier: Coefficient applied to retained node features.
        nonlinearity: Accepted for TopK compatibility, but unused.

    Sampling is random in both training and evaluation. Use torch.manual_seed
    to reproduce a sequence of samples. This module has no learnable parameters.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.,
        nonlinearity: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError('in_channels must be a positive integer')
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError('ratio must be a float in (0, 1] or a positive integer')
        if isinstance(ratio, int):
            if ratio < 1:
                raise ValueError('integer ratio must be positive')
        elif not 0. < ratio <= 1.:
            raise ValueError('float ratio must be in (0, 1]')
        if min_score is not None:
            raise ValueError('min_score is not supported by random node pooling')

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

    def reset_parameters(self):
        r"""No parameters to reset; random samples are drawn on each forward pass."""
        pass

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        r"""Return features, edges, edge attributes, batch, indices, and scores.

        edge_index is sparse COO with shape [2, E]. Optional edge_attr may
        contain scalar weights or multi-dimensional edge features. If batch
        is omitted, all nodes belong to a single graph. attn is accepted for
        TopK compatibility but ignored, since selection is feature-independent.
        perm indexes the original feature rows in pooled-node order; score
        contains one for every retained node. Input tensors must share a device.
        """
        if x.dim() != 2 or x.size(1) != self.in_channels:
            raise ValueError('x must have shape [N, in_channels]')
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError('edge_index must have shape [2, E]')
        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            raise ValueError('edge_attr must contain one entry per edge')

        num_nodes = x.size(0)
        if batch is None:
            batch = edge_index.new_zeros(num_nodes)
        if batch.dim() != 1 or batch.numel() != num_nodes:
            raise ValueError('batch must contain one graph id per node')

        # Per-graph equivalent of n = X.size(0) in the single-graph version.
        counts = torch.bincount(batch)
        if isinstance(self.ratio, int):
            keep_counts = counts.clamp(max=self.ratio)
        else:
            # Same role as p = max(1, int(round(pr * n))), but uses ceil
            # rather than round to match TopK's retained-node counts.
            keep_counts = (counts * self.ratio).ceil().to(torch.long)
            keep_counts = torch.minimum(keep_counts.clamp_min(1), counts)

        # A uniform global permutation induces a uniform permutation within
        # each graph. Stable grouping preserves these random within-graph orders.
        shuffled = torch.randperm(num_nodes, device=x.device)
        order = shuffled[torch.argsort(batch[shuffled], stable=True)]
        offsets = counts.cumsum(0) - counts
        ranks = torch.arange(num_nodes, device=x.device)
        ranks = ranks - torch.repeat_interleave(offsets, counts)
        # Take each graph's first p entries: the batched equivalent of
        # S = torch.randperm(n, device=X.device)[:p]. perm concatenates these S
        # values as original node indices. For the same p, the sampling law is
        # identical; batched RNG consumption need not match separate calls.
        perm = order[ranks < keep_counts[batch[order]]]

        # mapping[S[j]] = j translates original indices into pooled indices;
        # dropped nodes map to -1. This matches the row/column order of A[S, S].
        mapping = edge_index.new_full((num_nodes,), -1)
        mapping[perm] = torch.arange(perm.numel(), device=x.device)
        remapped_edges = mapping[edge_index]
        # Sparse equivalent of A[S[:, None], S]: keep an edge exactly when
        # both endpoints are in S, then use their remapped indices. No dense
        # adjacency is needed; existing edge weights/features remain unchanged.
        edge_mask = (remapped_edges >= 0).all(dim=0)
        pooled_edges = remapped_edges[:, edge_mask]
        pooled_attr = edge_attr[edge_mask] if edge_attr is not None else None

        # Exactly X[S] from the original function, concatenated across graphs.
        pooled_x = x[perm]
        # Optional TopK-compatible scaling; multiplier=1 preserves X[S] exactly.
        if self.multiplier != 1:
            pooled_x = self.multiplier * pooled_x
        # Unit scores prevent TopK-style feature gating; selection alone pools.
        score = x.new_ones(perm.numel())

        # batch[perm] preserves the graph identity of every selected node.
        return pooled_x, pooled_edges, pooled_attr, batch[perm], perm, score

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'ratio={self.ratio}, multiplier={self.multiplier})')
