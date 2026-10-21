"""Step 1 of the paper plan: do the paper's alignment metrics track downstream usefulness across
many trained models, or move against it?

Trains many head-only models on the cached (frozen) embeddings, varying only the training
objective for the text pairs:
  kind  in {soft, label}            (soft = frozen-text-similarity targets; label = oracle)
  mix   in {0, .25, .5, .75, 1}     T = (1 - mix) * I + mix * T_kind   (mix = 0 is the paper's InfoNCE)
  soft_tau in {0.02, 0.05, 0.1, 0.2} (soft only: how spread out the soft targets are)
  tau   in {0.07, 0.2, 0.5}         contrastive temperature
Each model is scored with the paper's metrics, the step-B metrics, and two usefulness measures
(zero-shot diagnostic AUROC; same-diagnosis P@10). The script then reports Spearman correlations,
across models, between each metric and each usefulness measure (TS-TXT and IMG-TXT pairs).

Writes <emb>/sweep_lead<L>.csv (one row per model) and <emb>/sweep_lead<L>_corr.csv.

Usage (GPU, ~1 min per model):
    python src/sweep.py --emb results/emb --ptbxl /content/ptbxl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_eval import label_retrieval, multihot, superclasses, zero_shot_auroc  # noqa: E402
from metrics_v2 import all_metrics_v2, text_groups  # noqa: E402
from train_heads import load_emb  # noqa: E402
from train_heads_v2 import CONFIG_A, embed_all, train  # noqa: E402

METRICS = ["cos_margin", "procrustes_paper", "procrustes_sym", "cka_rbf", "mutual_knn@5",
           "R@1_exact", "R@10_exact", "R@1_dupaware", "R@10_dupaware",
           "dedup_cos_margin", "dedup_procrustes_sym", "dedup_cka_rbf", "dedup_R@1"]
USEFUL = ["ts_zeroshot_auroc", "ts2txt_label_p@10"]


def grid(quick=False):
    mixes = [0.0, 0.25, 0.5, 0.75, 1.0]
    taus = [0.07, 0.2, 0.5]
    softs = [0.02, 0.05, 0.1, 0.2]
    if quick:
        mixes, taus, softs = [0.0, 1.0], [0.2], [0.05]
    out = []
    for tau in taus:
        out.append(("infonce", 0.0, None, tau))
        for mix in mixes[1:]:
            for st in softs:
                out.append(("soft", mix, st, tau))
            out.append(("label", mix, None, tau))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny grid (smoke test)")
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
    data = load_emb(emb, CONFIG_A, a.lead)
    sc = superclasses(a.ptbxl)
    aux = {}
    for s in ("train", "val", "test"):
        rows = split[split.split == s]
        if a.limit:
            rows = rows.head(a.limit)
            data[s] = {m: v[:a.limit] for m, v in data[s].items()}
        sets = sc.loc[rows.index].tolist()
        lab = pd.Series([",".join(sorted(x)) for x in sets]).astype("category").cat.codes.values.copy()
        aux[s] = {"text_group": torch.as_tensor(text_groups(rows.report.tolist())),
                  "label_group": torch.as_tensor(lab),
                  "txt_raw_n": F.normalize(data[s]["txt"], dim=1).to(device), "sets": sets}
    bs = min(a.bs, len(data["train"]["ts"]))
    gt = aux["test"]["text_group"].numpy()
    keep = np.array([len(x) > 0 for x in aux["test"]["sets"]])
    y_test = multihot(aux["test"]["sets"])[keep]
    sets_keep = [x for x, k in zip(aux["test"]["sets"], keep) if k]

    out_csv = emb / f"sweep_lead{a.lead}{'_quick' if a.quick or a.limit else ''}.csv"
    done = set()
    rows_out = []
    if out_csv.exists():  # resume
        prev = pd.read_csv(out_csv)
        rows_out = prev.to_dict("records")
        done = {(r["kind"], r["mix"], r["soft_tau"], r["tau"]) for r in rows_out}

    g = grid(a.quick)
    print(f"{len(g)} models in the grid, {len(done)} already done", flush=True)
    for kind, mix, st, tau in g:
        key = (kind, mix, st if st is not None else -1.0, tau)
        if key in done:
            continue
        t0 = time.time()
        targs = SimpleNamespace(epochs=a.epochs, bs=bs, lr=1e-4, wd=1e-5, tau=tau, patience=5)
        for s in aux:
            aux[s]["mix"] = mix
            aux[s]["soft_tau"] = st if st is not None else 0.05
        heads, log = train(kind, a.seed, data, aux, targs, device)
        z = {m: v.cpu().numpy() for m, v in embed_all(heads, data["test"], device).items()}
        ztr_txt = embed_all(heads, data["train"], device)["txt"].cpu().numpy()
        row = {"kind": kind, "mix": mix, "soft_tau": key[2], "tau": tau, "epochs_run": len(log)}
        for pair, (x, y) in {"ts-txt": ("ts", "txt"), "img-txt": ("img", "txt")}.items():
            mv = all_metrics_v2(z[x], z[y], gt)
            for k in METRICS:
                row[f"{pair}:{k}"] = mv.get(k, np.nan)
        hit, chance = label_retrieval(z["ts"][keep], z["txt"][keep], sets_keep)
        row["ts2txt_label_p@10"] = hit
        row["ts_zeroshot_auroc"] = zero_shot_auroc(z["ts"][keep], ztr_txt, aux["train"]["sets"], y_test)
        row["seconds"] = time.time() - t0
        rows_out.append(row)
        pd.DataFrame(rows_out).to_csv(out_csv, index=False)
        print(f"{kind:7s} mix {mix:.2f} st {key[2]:5.2f} tau {tau:.2f} ep {len(log):2d} | "
              f"margin {row['ts-txt:cos_margin']:.3f} procS {row['ts-txt:procrustes_sym']:.3f} "
              f"R@1dup {row['ts-txt:R@1_dupaware']:.4f} | AUROC {row['ts_zeroshot_auroc']:.3f} "
              f"labP {row['ts2txt_label_p@10']:.3f} ({row['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows_out)
    corr = []
    for pair in ("ts-txt", "img-txt"):
        for k in METRICS:
            col = f"{pair}:{k}"
            for u in USEFUL:
                ok = df[[col, u]].dropna()
                if len(ok) >= 4 and ok[col].nunique() > 1:
                    rho = ok[col].rank().corr(ok[u].rank())  # Spearman
                    corr.append({"metric": col, "usefulness": u, "spearman": rho, "n_models": len(ok)})
    cdf = pd.DataFrame(corr)
    cdf.to_csv(str(out_csv).replace(".csv", "_corr.csv"), index=False)
    print("\nSpearman correlation across models (metric vs usefulness; for Procrustes, lower = "
          "better, so a positive rho means it moves AGAINST usefulness):", flush=True)
    if len(cdf):
        print(cdf[cdf.usefulness == "ts_zeroshot_auroc"].to_string(index=False), flush=True)
    else:
        print("(fewer than 4 models: no correlations)", flush=True)
    json.dump({"n_models": len(df)}, open(str(out_csv).replace(".csv", "_meta.json"), "w"))


if __name__ == "__main__":
    main()
