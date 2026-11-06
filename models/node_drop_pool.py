import torch


def node_drop_pool(X, A, pr):
    """
    Node dropping graph pooling.

    Given node features X and a dense adjacency matrix A, this function
    uniformly samples p nodes without replacement and returns the induced
    subgraph on the sampled node set.

    Args:
        X: Node feature matrix of shape (n, d), where n is the number of nodes
           and d is the feature dimension.
        A: Dense adjacency matrix of shape (n, n).
        pr: Pooling ratio, where pr = p / n and 0 < pr < 1.

    Returns:
        X[S]: Pooled node feature matrix of shape (p, d).
        A[S, S]: Induced adjacency matrix of shape (p, p).
    """

    # Number of nodes in the original graph.
    n = X.size(0)
    p = max(1, int(round(pr * n)))

    # Uniformly sample a subset S of p distinct node indices from {0, ..., n - 1}.
    # torch.randperm(n) generates a random permutation of all node indices;
    # taking the first p entries gives sampling without replacement.
    # device=X.device keeps S on the same device as X, avoiding CPU-GPU transfer.
    S = torch.randperm(n, device=X.device)[:p]

    # X[S] selects the rows of X indexed by S, i.e., X_{S, :}.
    #
    # A[S[:, None], S] selects the principal submatrix A_{S, S}.
    # Here S[:, None] has shape (p, 1), while S has shape (p,).
    # PyTorch broadcasts them to all index pairs (S_i, S_j), producing
    # a dense pooled adjacency matrix of shape (p, p).
    return X[S], A[S[:, None], S]
