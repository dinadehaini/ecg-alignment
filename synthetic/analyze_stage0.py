"""Analyze stage-0 per-example predictions (preds_*.npz).

Primary quantity: exact excess Brier E[(p - q)^2], q = oracle posterior (no outcome sampling noise);
checked against the sampled Brier difference.
  1. Learning curves per method and group (sigma 1.0, balanced prior).
  2. Paired differences between methods (same test examples), with bootstrap 95% CIs over test
     examples, pooled over seeds.
  3. Ambiguity specificity: within bins of oracle difficulty (q's distance from 0.5), the
     contrastive-minus-predictor gap on structurally non-identifiable vs identifiable examples.
  4. Trap seen vs held-out; seen vs unseen views.
Usage: python analyze_stage0.py --res results
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [("contrastive_cal", "predictor_cal"), ("contrastive_cal", "distill_cal"),
         ("distill_cal", "predictor_cal"), ("contrastive_raw", "contrastive_cal")]
METHODS = ["predictor_raw", "predictor_cal", "ensemble", "contrastive_raw", "contrastive_cal",
           "distill_raw", "distill_cal"]


def load(res):
    out = []
    for f in sorted(Path(res).glob("preds_*.npz")):
        n, s, z, seed = re.match(r"preds_n(\d+)_s([\d.]+)_z([\d.]+)_seed(\d+)", f.stem).groups()
        d = dict(np.load(f, allow_pickle=False))
        d["groups"] = {str(g): d[f"g_{i}"].astype(bool) for i, g in enumerate(d["group_names"])}
        out.append(((int(n), float(s), float(z), int(seed)), d))
    return out


def boot_ci(x, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return np.nan, np.nan
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    a = ap.parse_args()
    runs = load(a.res)
    parts = ["# Stage 0 analysis (exact excess Brier = E[(p - q)^2] vs the oracle posterior q)\n"]

    # 1. learning curves + sanity check
    rows = []
    for (n, s, z, seed), d in runs:
        q = d["oracle"]
        for m in METHODS:
            if m not in d:
                continue
            p = d[m]
            for g, idx in d["groups"].items():
                if idx.sum() == 0:
                    continue
                rows.append({"n": n, "sigma": s, "z_prior": z, "seed": seed, "method": m, "group": g,
                             "exact": float(np.mean((p[idx] - q[idx]) ** 2)),
                             "sampled": float(np.mean((p[idx] - d["truth"][idx]) ** 2) -
                                              np.mean((q[idx] - d["truth"][idx]) ** 2)),
                             "count": int(idx.sum())})
    df = pd.DataFrame(rows)
    main_c = df[(df.sigma == 1.0) & (df.z_prior == 0.5)]
    keyg = ["all", "identifiable, posterior-confident", "identifiable, posterior-ambiguous (noise control)",
            "structurally non-identifiable", "trap, seen view and composition",
            "trap, held-out view or composition", "seen views", "unseen channel masks",
            "held-out (claim, view) compositions"]
    lc = main_c[main_c.group.isin(keyg)].groupby(["group", "method", "n"]).exact.mean().unstack("n")
    parts += ["## 1. Learning curves: exact excess Brier (sigma 1.0, balanced prior; mean over seeds)\n",
              lc.reset_index().to_markdown(index=False, floatfmt=".4f")]
    chk = df.groupby("method")[["exact", "sampled"]].mean()
    parts += ["\nSanity check (all conditions, mean over groups): exact vs sampled excess Brier\n",
              chk.reset_index().to_markdown(index=False, floatfmt=".4f")]

    # 2. paired differences with CIs, per condition, pooled over seeds
    prow = []
    conds = sorted({k[:3] for k, _ in runs})
    for (n, s, z) in conds:
        ds = [d for k, d in runs if k[:3] == (n, s, z)]
        for g in keyg:
            for m1, m2 in PAIRS:
                if m1 not in ds[0] or m2 not in ds[0]:
                    continue
                x = np.concatenate([((d[m1] - d["oracle"]) ** 2 - (d[m2] - d["oracle"]) ** 2)[d["groups"][g]]
                                    for d in ds if g in d["groups"]])
                lo, hi = boot_ci(x)
                prow.append({"n": n, "sigma": s, "z_prior": z, "group": g, "pair": f"{m1} - {m2}",
                             "diff": float(x.mean()) if len(x) else np.nan, "lo": lo, "hi": hi, "count": len(x)})
    pdf = pd.DataFrame(prow)
    parts += ["\n## 2. Paired differences in exact excess Brier (positive = first method worse), 95% bootstrap CI\n",
              pdf.to_markdown(index=False, floatfmt=".4f")]

    # 3. ambiguity specificity, difficulty-matched
    arow = []
    bins = [0, 0.1, 0.25, 0.4, 0.5]
    for (n, s, z) in conds:
        ds = [d for k, d in runs if k[:3] == (n, s, z)]
        q = np.concatenate([d["oracle"] for d in ds])
        st = np.concatenate([d["struct"] for d in ds]).astype(bool)
        gap = np.concatenate([(d["contrastive_cal"] - d["oracle"]) ** 2 - (d["predictor_cal"] - d["oracle"]) ** 2
                              for d in ds])
        dist = np.abs(q - 0.5)
        for lo_b, hi_b in zip(bins[:-1], bins[1:]):
            b = (dist >= lo_b) & (dist < hi_b if hi_b < 0.5 else dist <= hi_b)
            for name, sel in (("non-identifiable", ~st), ("identifiable", st)):
                x = gap[b & sel]
                ci = boot_ci(x)
                arow.append({"n": n, "sigma": s, "z_prior": z, "|q-0.5| bin": f"[{lo_b},{hi_b}]", "structure": name,
                             "gap contrastive-predictor": float(x.mean()) if len(x) else np.nan,
                             "lo": ci[0], "hi": ci[1], "count": int(len(x))})
    parts += ["\n## 3. Ambiguity specificity: contrastive-minus-predictor gap within oracle-difficulty bins\n",
              pd.DataFrame(arow).to_markdown(index=False, floatfmt=".4f")]
    text = "\n".join(parts)
    Path(a.res, "stage0_analysis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
