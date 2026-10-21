"""Mechanism test: does the alignment-vs-usefulness reversal grow as the text gets more many-to-one?

Compares the 13-model grid (tau 0.2) across text conditions: the original reports (taken from the
full sweep, sweep2_lead<L>.csv) and k-means-coarsened reports (sweep2_lead<L>_k<K>_small.csv).

For each condition: Spearman between each test alignment metric (sign-flipped, + = better alignment)
and test zero-shot AUROC; the model-selection cost of picking by each validation metric (vs. the
prespecified criterion, validation zero-shot AUROC); the zero-shot spread; and the probe range.

Usage: python src/compare_coarsen.py --emb results/emb --ks 500 100 20 --out results/coarsen_leadI.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_sweep import LOWER_BETTER, signed, spearman

METRICS = ["cos_margin", "procrustes_sym", "cka_rbf", "mutual_knn@5", "R@1_exact", "R@1_dupaware"]


def small_subset(df):
    keep = (df.tau == 0.2) & (
        (df.kind == "infonce") |
        ((df.kind == "soft") & df["mix"].isin([0.5, 1.0])) |
        (df.kind == "label"))
    return df[keep].reset_index(drop=True)


def summarize(df, name):
    row = {"condition": name, "n_models": len(df)}
    for m in METRICS:
        rho, _ = spearman(signed(df, f"ts-txt:{m}", m).values, df["ts_zeroshot_auroc"].values)
        row[f"rho {m}"] = rho
    crit = df.loc[df["val:ts_zeroshot_auroc"].idxmax()]
    for m in METRICS:
        vc = f"val:ts-txt:{m}"
        pick = df.loc[df[vc].idxmin() if m in LOWER_BETTER else df[vc].idxmax()]
        row[f"cost {m}"] = crit.ts_zeroshot_auroc - pick.ts_zeroshot_auroc
    base = df[df.kind == "infonce"].iloc[0]
    best = df.loc[df.ts_zeroshot_auroc.idxmax()]
    row["zs baseline [95% CI]"] = f"{base.ts_zeroshot_auroc:.3f} [{base['ci_lo:zs_auroc']:.3f}, {base['ci_hi:zs_auroc']:.3f}]"
    row["zs best [95% CI]"] = f"{best.ts_zeroshot_auroc:.3f} [{best['ci_lo:zs_auroc']:.3f}, {best['ci_hi:zs_auroc']:.3f}]"
    row["best model"] = f"{best.kind} mix={best['mix']:.2f} st={best.soft_tau:.2f}"
    row["zs gain over baseline"] = best.ts_zeroshot_auroc - base.ts_zeroshot_auroc
    row["probe range"] = f"{df.ts_probe_auroc.min():.3f}-{df.ts_probe_auroc.max():.3f}"
    row["baseline margin"] = base["ts-txt:cos_margin"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--ks", type=int, nargs="+", default=[500, 100, 20])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    emb = Path(a.emb)
    rows = [summarize(small_subset(pd.read_csv(emb / f"sweep2_lead{a.lead}.csv")), "original reports")]
    for k in a.ks:
        f = emb / f"sweep2_lead{a.lead}_k{k}_small.csv"
        if f.exists():
            rows.append(summarize(pd.read_csv(f), f"coarsened to {k} groups"))
    t = pd.DataFrame(rows)
    rho_cols = ["condition", "n_models"] + [c for c in t if c.startswith("rho ")]
    cost_cols = ["condition"] + [c for c in t if c.startswith("cost ")]
    other = ["condition", "zs baseline [95% CI]", "zs best [95% CI]", "best model", "zs gain over baseline", "probe range", "baseline margin"]
    text = "\n".join([
        f"# Text-coarsening mechanism test (lead {a.lead}, 13 models at tau 0.2 per condition)\n",
        "## Spearman: alignment (+ = better alignment) vs test zero-shot AUROC\n",
        t[rho_cols].to_markdown(index=False, floatfmt=".2f"),
        "\n## Model-selection cost (test zero-shot AUROC lost vs. the validation zero-shot criterion)\n",
        t[cost_cols].to_markdown(index=False, floatfmt=".3f"),
        "\n## Spread\n",
        t[other].to_markdown(index=False, floatfmt=".3f")])
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
