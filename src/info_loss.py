"""Does within-group information loss explain the alignment-vs-usefulness reversal?

Retrains the 13-model grid at tau 0.2 (same seed and code as the sweep, so the sweep's numbers should
reproduce) and, for each model, measures two things separately on the test set:

  correspondence  how well signals match their own text: the sweep's alignment metrics, incl.
                  duplicate-aware R@1 (credit for any signal-text pair sharing a report).
  preservation    whether the TS embedding keeps distinctions that the paired text does NOT express,
                  within groups of signals that share an identical report (or k-means text cluster):
    collapse          mean within-group cosine distance / mean random-pair distance (1 = no collapse)
    wg_label_auroc    linear-probe superclass scores, centered by group, AUROC on groups whose
                      members disagree on that superclass (the report can't tell them apart)
    age_r2, wg_age_r2 ridge probe for age (in metadata, almost never in the report): global, and
                      within-group (group-centered predictions vs group-centered ages)
    sex_auroc, wg_sex_auroc  same for sex

The frozen TS encoder is scored the same way as a reference. Writes <emb>/infoloss_lead<L><tag>.csv and
a markdown analysis: Spearman across models of preservation vs alignment and vs zero-shot AUROC, and
the rank-partial correlation of alignment with zero-shot AUROC after controlling for each
preservation measure. With 13 models these are descriptive, not tests.

Saves the TS and TXT heads (fp16) to <emb>/heads<tag>/ for later analyses.

Usage (GPU, ~70 s per model): python src/info_loss.py --emb results/emb --ptbxl /content/ptbxl [--coarsen K]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sweep import LOWER_BETTER  # noqa: E402
from data import load_metadata  # noqa: E402
from label_eval import multihot, superclasses  # noqa: E402
from metrics_v2 import all_metrics_v2, text_groups  # noqa: E402
from sweep import coarsen_text, macro_auroc, probe_scores, small_grid, zs_scores  # noqa: E402
from train_heads import load_emb  # noqa: E402
from train_heads_v2 import CONFIG_A, embed_all, train  # noqa: E402

ALIGN = ["cos_margin", "procrustes_sym", "cka_rbf", "mutual_knn@5", "R@1_exact", "R@1_dupaware"]
PRESERVE = ["collapse", "wg_label_auroc", "age_r2", "wg_age_r2", "sex_auroc", "wg_sex_auroc"]


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def collapse_ratio(z, groups, seed=0):
    """Mean cosine distance within multi-member groups (pairs pooled) / mean over random pairs."""
    z = unit(z)
    tot, npairs = 0.0, 0
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        if len(i) < 2:
            continue
        d = 1 - (z[i] @ z[i].T)[np.triu_indices(len(i), 1)]
        tot, npairs = tot + d.sum(), npairs + len(d)
    if npairs == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(z), 20000), rng.integers(0, len(z), 20000)
    ok = a != b
    between = (1 - (z[a[ok]] * z[b[ok]]).sum(1)).mean()
    return float(tot / npairs / between)


def center_by_group(v, groups):
    v = np.asarray(v, dtype=float).copy()
    for g in np.unique(groups):
        i = groups == g
        v[i] -= v[i].mean(0)
    return v


def within_group_auroc(y, s, groups, min_n=20):
    """Per class: AUROC of group-centered scores, only on groups whose members disagree on that class."""
    from sklearn.metrics import roc_auc_score
    vals = []
    for j in range(y.shape[1]):
        mixed = [g for g in np.unique(groups) if len(set(y[groups == g, j])) == 2]
        i = np.isin(groups, mixed)
        if i.sum() < min_n:
            continue
        vals.append(roc_auc_score(y[i, j], center_by_group(s[i, j], groups[i])))
    return float(np.mean(vals)) if vals else float("nan")


def ridge_pred(Xtr, ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]) / 1000, A.T @ (ytr - ym))
    return B @ w + ym


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def preservation(ztr, zte, y_tr, y_te, keep_tr, keep_te, age, sex, groups, device):
    """age/sex: dicts split -> array (NaN = missing). groups: test group ids (all test rows)."""
    out = {"collapse": collapse_ratio(zte, groups)}
    pr = probe_scores(ztr[keep_tr], y_tr, zte[keep_te], device)
    out["wg_label_auroc"] = within_group_auroc(y_te, pr, groups[keep_te])
    ma, mt = ~np.isnan(age["train"]), ~np.isnan(age["test"])
    pa = ridge_pred(ztr[ma], age["train"][ma], zte[mt])
    out["age_r2"] = r2(age["test"][mt], pa)
    g = groups[mt]
    out["wg_age_r2"] = r2(center_by_group(age["test"][mt], g), center_by_group(pa, g))
    from sklearn.metrics import roc_auc_score
    ms, mts = ~np.isnan(sex["train"]), ~np.isnan(sex["test"])
    ps = probe_scores(ztr[ms], sex["train"][ms, None], zte[mts], device)[:, 0]
    out["sex_auroc"] = float(roc_auc_score(sex["test"][mts], ps)) if len(set(sex["test"][mts])) == 2 else float("nan")
    gs = groups[mts]
    mixed = [k for k in np.unique(gs) if len(set(sex["test"][mts][gs == k])) == 2]
    i = np.isin(gs, mixed)
    out["wg_sex_auroc"] = (float(roc_auc_score(sex["test"][mts][i], center_by_group(ps[i], gs[i])))
                           if i.sum() >= 20 else float("nan"))
    return out


def spear(x, y):
    x, y = pd.Series(x).rank(), pd.Series(y).rank()
    return float(x.corr(y))


def partial_spear(x, y, z):
    """Spearman partial correlation of x and y given z."""
    rxy, rxz, ryz = spear(x, y), spear(x, z), spear(y, z)
    return (rxy - rxz * ryz) / np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))


def analyze(df, ref, title):
    sgn = lambda m: -df[f"ts-txt:{m}"] if m in LOWER_BETTER else df[f"ts-txt:{m}"]
    zs = df["ts_zeroshot_auroc"]
    parts = [f"# {title}\n", f"{len(df)} models. Frozen TS encoder reference: " +
             ", ".join(f"{k} {v:.3f}" for k, v in ref.items()) + "\n",
             "## Per model\n",
             df[["kind", "mix", "soft_tau", "ts_zeroshot_auroc", "ts_probe_auroc", "ts-txt:cos_margin",
                 "ts-txt:R@1_dupaware"] + PRESERVE].to_markdown(index=False, floatfmt=".3f"),
             "\n## Spearman across models (alignment sign-flipped so + = better aligned)\n"]
    rows = []
    for p in PRESERVE:
        r = {"preservation measure": p, "vs zero-shot AUROC": spear(df[p], zs),
             "vs probe AUROC": spear(df[p], df["ts_probe_auroc"])}
        for m in ALIGN:
            r[f"vs {m}"] = spear(df[p], sgn(m))
        rows.append(r)
    parts.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"))
    parts.append("\n## Does preservation explain the reversal? Spearman(alignment, zero-shot AUROC): "
                 "raw, and partial given each preservation measure\n")
    rows = []
    for m in ALIGN:
        r = {"alignment metric": m, "raw": spear(sgn(m), zs)}
        for p in PRESERVE:
            r[f"given {p}"] = partial_spear(sgn(m), zs, df[p])
        rows.append(r)
    parts.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--coarsen", type=int, default=0)
    ap.add_argument("--no-save-heads", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[cuda ok] {torch.cuda.get_device_name(0)}", flush=True)
    elif a.allow_cpu:
        device = "cpu"
    else:
        sys.exit("CUDA not available (use a GPU runtime; --allow-cpu is for smoke tests)")

    emb = Path(a.emb)
    split = pd.read_csv(emb / "split.csv", index_col=0)
    meta = load_metadata(a.ptbxl)
    data = load_emb(emb, CONFIG_A, a.lead)
    sc = superclasses(a.ptbxl)
    if a.limit:
        for s in data:
            data[s] = {m: v[:a.limit] for m, v in data[s].items()}
    cids = coarsen_text(data, a.coarsen) if a.coarsen else None
    aux, keep, Y, age, sex = {}, {}, {}, {}, {}
    for s in ("train", "val", "test"):
        rows = split[split.split == s]
        if a.limit:
            rows = rows.head(a.limit)
        sets = sc.loc[rows.index].tolist()
        lab = pd.Series([",".join(sorted(x)) for x in sets]).astype("category").cat.codes.values.copy()
        tg = cids[s] if cids is not None else text_groups(rows.report.tolist())
        aux[s] = {"text_group": torch.as_tensor(tg), "label_group": torch.as_tensor(lab),
                  "txt_raw_n": F.normalize(data[s]["txt"], dim=1).to(device), "sets": sets, "g": np.asarray(tg)}
        keep[s] = np.array([len(x) > 0 for x in sets])
        Y[s] = multihot(sets)[keep[s]]
        ag = meta.loc[rows.index, "age"].astype(float).values
        age[s] = np.where((ag > 0) & (ag < 120), ag, np.nan)  # PTB-XL codes 89+ as 300
        sex[s] = meta.loc[rows.index, "sex"].astype(float).values
    gt = aux["test"]["g"]
    print(f"test groups: {len(np.unique(gt))} unique texts; "
          f"{sum((gt == g).sum() > 1 for g in np.unique(gt))} groups with 2+ signals", flush=True)

    tag = (f"_k{a.coarsen}" if a.coarsen else "") + ("_quick" if a.limit else "")
    ref = preservation(data["train"]["ts"].numpy(), data["test"]["ts"].numpy(), Y["train"], Y["test"],
                       keep["train"], keep["test"], age, sex, gt, device)
    print("frozen TS encoder: " + ", ".join(f"{k} {v:.3f}" for k, v in ref.items()), flush=True)

    out_csv = emb / f"infoloss_lead{a.lead}{tag}.csv"
    hdir = emb / f"heads{tag}"
    hdir.mkdir(exist_ok=True)
    rows_out = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {(r["kind"], r["mix"], r["soft_tau"]) for r in rows_out}
    bs = min(a.bs, len(data["train"]["ts"]))
    for kind, mix, st, tau in small_grid():
        key = (kind, mix, st if st is not None else -1.0)
        if key in done:
            continue
        t0 = time.time()
        targs = SimpleNamespace(epochs=a.epochs, bs=bs, lr=1e-4, wd=1e-5, tau=tau, patience=5)
        for s in aux:
            aux[s]["mix"] = mix
            aux[s]["soft_tau"] = st if st is not None else 0.05
        heads, log = train(kind, a.seed, data, aux, targs, device)
        if not a.no_save_heads:
            torch.save({m: {k: v.half().cpu() for k, v in heads[m].state_dict().items()} for m in ("ts", "txt")},
                       hdir / f"{kind}_mix{mix:.2f}_st{key[2]:.2f}.pt")
        z = {m: v.cpu().numpy() for m, v in embed_all(heads, data["test"], device).items()}
        ztr = {m: v.cpu().numpy() for m, v in embed_all(heads, data["train"], device).items()}
        row = {"kind": kind, "mix": mix, "soft_tau": key[2], "tau": tau, "epochs_run": len(log)}
        mv = all_metrics_v2(z["ts"], z["txt"], gt)
        for k in ALIGN:
            row[f"ts-txt:{k}"] = mv.get(k, np.nan)
        kt = keep["test"]
        row["ts_zeroshot_auroc"] = macro_auroc(Y["test"], zs_scores(z["ts"][kt], ztr["txt"], aux["train"]["sets"]))
        row["ts_probe_auroc"] = macro_auroc(Y["test"], probe_scores(ztr["ts"][keep["train"]], Y["train"],
                                                                    z["ts"][kt], device))
        row.update(preservation(ztr["ts"], z["ts"], Y["train"], Y["test"], keep["train"], kt, age, sex, gt, device))
        row["seconds"] = time.time() - t0
        rows_out.append(row)
        pd.DataFrame(rows_out).to_csv(out_csv, index=False)
        print(f"{kind:7s} mix {mix:.2f} st {key[2]:5.2f} | zs {row['ts_zeroshot_auroc']:.3f} "
              f"R@1dup {row['ts-txt:R@1_dupaware']:.3f} | collapse {row['collapse']:.3f} "
              f"wgLab {row['wg_label_auroc']:.3f} wgAge {row['wg_age_r2']:.3f} "
              f"wgSex {row['wg_sex_auroc']:.3f} ({row['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows_out)
    title = f"Within-group information loss (lead {a.lead}, " + \
            (f"text coarsened to {a.coarsen} groups" if a.coarsen else "original reports") + ", tau 0.2)"
    text = analyze(df, ref, title)
    print(text, flush=True)
    (emb / f"infoloss_lead{a.lead}{tag}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
