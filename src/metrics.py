"""Alignment metrics, implemented from Appendix E of Yashwante & Yu (ICML 2026, arXiv 2602.19367).

All functions take paired, L2-normalised embeddings X, Y of shape (N, d) as numpy arrays.
Geometric metrics (Procrustes, CKA, mutual kNN) use a random subset of at most 2000 pairs
(seed 42), as the paper does. Cosine statistics and retrieval use the full test set.
"""
from __future__ import annotations

import numpy as np

GEOM_SUBSET = 2000
GEOM_SEED = 42


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def subset_idx(n: int, max_n: int = GEOM_SUBSET, seed: int = GEOM_SEED) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_n, replace=False))


def cosine_stats(X: np.ndarray, Y: np.ndarray) -> dict:
    """Matched mean, mismatched mean and margin (Eq. in App. E.1)."""
    S = X @ Y.T
    n = S.shape[0]
    diag = np.diag(S)
    off_row_mean = (S.sum(1) - diag) / (n - 1)
    return {
        "cos_matched": float(diag.mean()),
        "cos_mismatched": float((S.sum() - diag.sum()) / (n * (n - 1))),
        "cos_margin": float((diag - off_row_mean).mean()),
    }


def procrustes_disparity(X: np.ndarray, Y: np.ndarray) -> float:
    """Normalised orthogonal-Procrustes disparity ||X~R - Y~||^2 / ||Y~||^2 (App. E.2).

    R is the Kabsch solution with explicit reflection correction (det(R) = +1).
    """
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(U.shape[1])
    D[-1] = d
    R = (U * D) @ Vt
    return float(np.linalg.norm(Xc @ R - Yc) ** 2 / np.linalg.norm(Yc) ** 2)


def _sq_dists(A: np.ndarray) -> np.ndarray:
    sq = (A * A).sum(1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0.0)


def cka_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    """RBF-kernel CKA (App. E.3).

    Bandwidth: median heuristic over the pooled off-diagonal pairwise distances of both
    modalities, gamma = 1 / (2 * median^2). The paper does not give the exact form of its
    median heuristic; this is the standard one (Kornblith et al. 2019 use sigma = median).
    """
    DX, DY = _sq_dists(X), _sq_dists(Y)
    iu = np.triu_indices(X.shape[0], k=1)
    med = np.median(np.sqrt(np.concatenate([DX[iu], DY[iu]])))
    gamma = 1.0 / (2.0 * med ** 2)
    K, L = np.exp(-gamma * DX), np.exp(-gamma * DY)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float((Kc * Lc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Lc)))


def mutual_knn(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """Mean |N_k^X(i) ∩ N_k^Y(i)| / k under cosine distance, excluding self (App. E.4)."""
    def knn(A):
        S = A @ A.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]
    nx, ny = knn(X), knn(Y)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nx, ny)]))


def retrieval(X: np.ndarray, Y: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Bidirectional Recall@k and MRR over the full evaluation set (App. E.5).

    Reports each direction and the macro average of the two, as in the paper's Table 2.
    """
    S = X @ Y.T
    out = {}
    for name, M in (("x2y", S), ("y2x", S.T)):
        correct = np.diag(M)[:, None]
        rank = (M > correct).sum(1)  # 0 = top-1
        for k in ks:
            out[f"R@{k}_{name}"] = float((rank < k).mean())
        out[f"MRR_{name}"] = float((1.0 / (rank + 1)).mean())
    for k in ks:
        out[f"R@{k}"] = (out[f"R@{k}_x2y"] + out[f"R@{k}_y2x"]) / 2
    out["MRR"] = (out["MRR_x2y"] + out["MRR_y2x"]) / 2
    return out


def all_metrics(X: np.ndarray, Y: np.ndarray) -> dict:
    X, Y = l2n(X), l2n(Y)
    out = cosine_stats(X, Y)
    out.update(retrieval(X, Y))
    idx = subset_idx(len(X))
    Xs, Ys = X[idx], Y[idx]
    out["procrustes"] = procrustes_disparity(Xs, Ys)
    out["cka_rbf"] = cka_rbf(Xs, Ys)
    out["mutual_knn@5"] = mutual_knn(Xs, Ys, k=5)
    return out
