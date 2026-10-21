"""Step B: alignment metrics that handle what the paper's metrics don't.

1. Duplicate-aware retrieval. The paper counts only the exact paired item as a hit. With duplicate
   captions (51% of PTB-XL test reports), that's partly unwinnable. Here a retrieved text counts as
   a hit if its *content* equals the query's paired text (group ids passed in), and we report the
   ceiling an oracle could reach under the paper's exact-index rule.
2. Scale-symmetric Procrustes. The paper's disparity ||X~R - Y~||^2 / ||Y~||^2 normalizes by the
   target only, so it moves when the target cloud's spread changes, even with alignment fixed. The
   symmetric version scales both centred clouds to unit Frobenius norm first (as in
   scipy.spatial.procrustes), giving a value in [0, 1] that is invariant to either cloud's scale.
   We also report the spread ratio ||X~|| / ||Y~|| that drives the difference.
3. Deduplicated evaluation. Every metric is recomputed on a subset with one record per unique
   caption, so duplicate structure can't drive the numbers.

All inputs: paired embeddings X, Y (N, d) as numpy arrays (L2-normalized inside).
"""
from __future__ import annotations

import numpy as np

from metrics import GEOM_SEED, GEOM_SUBSET, cka_rbf, cosine_stats, l2n, mutual_knn, \
    procrustes_disparity, subset_idx


def procrustes_symmetric(X: np.ndarray, Y: np.ndarray) -> dict:
    """Scale-symmetric Procrustes (in [0, 1]) plus the paper's version and the spread ratio."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    nx, ny = np.linalg.norm(Xc), np.linalg.norm(Yc)
    s = np.linalg.svd((Xc / nx).T @ (Yc / ny), compute_uv=False)
    # reflections allowed here, as in scipy; with the paper's reflection fix the difference is tiny
    return {"procrustes_sym": float(max(0.0, 1.0 - s.sum() ** 2)),
            "procrustes_paper": procrustes_disparity(X, Y),
            "spread_ratio_x_over_y": float(nx / ny)}


def retrieval_groups(X: np.ndarray, Y: np.ndarray, gy: np.ndarray, ks=(1, 5, 10)) -> dict:
    """X -> Y retrieval where any Y with the same group id as the true pair counts as a hit.

    gy: integer group id per row of Y (e.g. identical caption text -> same id). Also returns the
    exact-index R@k and the exact-index ceiling (expected R@1 of a perfect model breaking ties at
    random among same-group items).
    """
    S = X @ Y.T
    n = len(S)
    order = np.argsort(-S, axis=1)
    hit_group = gy[order] == gy[:, None]
    d = np.diag(S)[:, None]
    above = (S > d).sum(1)                 # items strictly better than the true pair
    ties = (S == d).sum(1) - 1             # other items with exactly the same score
    sizes = np.bincount(gy)[gy]
    out = {}
    for k in ks:
        # exact-index rule with ties broken uniformly at random (expected value), as the paper's
        # index-based scoring effectively does; identical captions give exact ties
        p_hit = np.clip(k - above, 0, ties + 1) / (ties + 1)
        out[f"R@{k}_exact"] = float(p_hit.mean())
        out[f"R@{k}_dupaware"] = float(hit_group[:, :k].any(1).mean())
        out[f"R@{k}_exact_ceiling"] = float(np.minimum(1.0, k / sizes).mean())
    out["n_queries"] = int(n)
    out["n_unique_groups"] = int(len(np.unique(gy)))
    return out


def dedup_index(groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """One random row per unique group id."""
    rng = np.random.default_rng(seed)
    idx = []
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        idx.append(rng.choice(rows))
    return np.sort(np.array(idx))


def all_metrics_v2(X: np.ndarray, Y: np.ndarray, gy: np.ndarray | None = None) -> dict:
    """Paper metrics + fixed metrics, on the full set and (if groups given) a deduplicated set."""
    X, Y = l2n(X), l2n(Y)
    out = {}
    out.update(cosine_stats(X, Y))
    idx = subset_idx(len(X), GEOM_SUBSET, GEOM_SEED)
    out.update(procrustes_symmetric(X[idx], Y[idx]))
    out["cka_rbf"] = cka_rbf(X[idx], Y[idx])
    out["mutual_knn@5"] = mutual_knn(X[idx], Y[idx], k=5)
    if gy is not None:
        out.update(retrieval_groups(X, Y, gy))
        d = dedup_index(gy)
        Xd, Yd = X[d], Y[d]
        dd = subset_idx(len(d), GEOM_SUBSET, GEOM_SEED)
        out["dedup_n"] = int(len(d))
        out["dedup_cos_margin"] = cosine_stats(Xd, Yd)["cos_margin"]
        p = procrustes_symmetric(Xd[dd], Yd[dd])
        out["dedup_procrustes_sym"] = p["procrustes_sym"]
        out["dedup_procrustes_paper"] = p["procrustes_paper"]
        out["dedup_spread_ratio"] = p["spread_ratio_x_over_y"]
        out["dedup_cka_rbf"] = cka_rbf(Xd[dd], Yd[dd])
        r = retrieval_groups(Xd, Yd, np.arange(len(d)))
        out["dedup_R@1"] = r["R@1_exact"]
        out["dedup_R@10"] = r["R@10_exact"]
    return out


def text_groups(reports) -> np.ndarray:
    """Group id per report: identical after lowercasing and collapsing whitespace."""
    norm = [" ".join(str(r).lower().split()) for r in reports]
    _, inv = np.unique(norm, return_inverse=True)
    return inv
