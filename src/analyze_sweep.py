"""Analyze sweep2_lead<L>.csv: does selecting models by alignment cost diagnostic performance?

1. Spearman rank correlation between each TEST alignment metric and each TEST downstream measure
   (zero-shot AUROC, linear-probe AUROC), overall, within each objective family, and within each
   training temperature. Lower-is-better metrics (Procrustes) are sign-flipped so a positive rho
   always means "better alignment goes with better downstream".
2. Model-selection cost: for each VALIDATION alignment metric, pick the best model by that metric
   and report its TEST downstream scores, compared with the model picked by a prespecified validation
   criterion (validation zero-shot AUROC) and with the best test model (oracle; not a usable
   selector). Repeated within each training temperature.

Usage:  python src/analyze_sweep.py --csv results/emb/sweep2_leadI.csv --out results/sweep_analysis.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOWER_BETTER = {"procrustes_paper", "procrustes_sym", "dedup_procrustes_sym"}
ALIGN = ["cos_margin", "procrustes_paper", "procrustes_sym", "cka_rbf", "mutual_knn@5",
         "R@1_exact", "R@10_exact", "R@1_dupaware", "R@10_dupaware", "dedup_cos_margin",
         "dedup_procrustes_sym", "dedup_cka_rbf", "dedup_R@1"]
DOWN = ["ts_zeroshot_auroc", "ts_probe_auroc"]


def spearman(x, y):
    ok = ~(pd.isna(x) | pd.isna(y))
    x, y = pd.Series(x[ok]), pd.Series(y[ok])
    if len(x) < 4 or x.nunique() < 2:
        return np.nan, len(x)
    return float(x.rank().corr(y.rank())), len(x)


def signed(df, col, metric):
    return -df[col] if metric in LOWER_BETTER else df[col]


def corr_table(df, scope_name):
    rows = []
    for m in ALIGN:
        col = f"ts-txt:{m}"
        if col not in df:
            continue
        r = {"metric": m}
        for d in DOWN:
            rho, n = spearman(signed(df, col, m).values, df[d].values)
            r[d] = rho
            r["n"] = n
        rows.append(r)
    t = pd.DataFrame(rows)
    t.insert(0, "scope", scope_name)
    return t


def selection(df, label):
    out = []
    crit = df.loc[df["val:ts_zeroshot_auroc"].idxmax()]
    oracle_zs, oracle_pr = df["ts_zeroshot_auroc"].max(), df["ts_probe_auroc"].max()
    for m in ALIGN:
        vc = f"val:ts-txt:{m}"
        if vc not in df or df[vc].isna().all():
            continue
        pick = df.loc[(df[vc].idxmin() if m in LOWER_BETTER else df[vc].idxmax())]
        out.append({"scope": label, "selector": f"val {m}",
                    "picked": f"{pick.kind} mix={pick['mix']:.2f} st={pick.soft_tau:.2f} tau={pick.tau:.2f}",
                    "test zs-AUROC": pick.ts_zeroshot_auroc, "test probe-AUROC": pick.ts_probe_auroc,
                    "zs loss vs criterion": crit.ts_zeroshot_auroc - pick.ts_zeroshot_auroc,
                    "probe loss vs criterion": crit.ts_probe_auroc - pick.ts_probe_auroc})
    out.append({"scope": label, "selector": "val zs-AUROC (prespecified criterion)",
                "picked": f"{crit.kind} mix={crit['mix']:.2f} st={crit.soft_tau:.2f} tau={crit.tau:.2f}",
                "test zs-AUROC": crit.ts_zeroshot_auroc, "test probe-AUROC": crit.ts_probe_auroc,
                "zs loss vs criterion": 0.0, "probe loss vs criterion": 0.0})
    out.append({"scope": label, "selector": "oracle (best test; not usable)", "picked": "-",
                "test zs-AUROC": oracle_zs, "test probe-AUROC": oracle_pr,
                "zs loss vs criterion": np.nan, "probe loss vs criterion": np.nan})
    return pd.DataFrame(out)


def fmt(df):
    return df.to_markdown(index=False, floatfmt=".3f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    parts = [f"# Sweep analysis ({len(df)} models, `{Path(a.csv).name}`)\n",
             f"Frozen-encoder probe AUROC (reference): {df['frozen_probe_auroc_ref'].iloc[0]:.3f}\n",
             "## Downstream scores per model (with patient-level 95% bootstrap CIs)\n",
             fmt(df[["kind", "mix", "soft_tau", "tau", "ts-txt:cos_margin", "ts-txt:procrustes_sym",
                     "ts_zeroshot_auroc", "ci_lo:zs_auroc", "ci_hi:zs_auroc", "ts_probe_auroc",
                     "ci_lo:probe_auroc", "ci_hi:probe_auroc"]].sort_values(["tau", "kind", "mix", "soft_tau"])),
             "\n## Spearman: alignment (sign-flipped so + = better alignment) vs downstream\n"]
    tabs = [corr_table(df, "all")]
    for k, g in df.groupby("kind"):
        tabs.append(corr_table(g, f"family={k}"))
    for t, g in df.groupby("tau"):
        tabs.append(corr_table(g, f"tau={t}"))
    parts.append(fmt(pd.concat(tabs)))
    parts.append("\n## Model-selection cost (pick by a validation metric -> test downstream)\n")
    sels = [selection(df, "all")]
    for t, g in df.groupby("tau"):
        sels.append(selection(g.reset_index(drop=True), f"tau={t}"))
    parts.append(fmt(pd.concat(sels)))
    text = "\n".join(parts)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
