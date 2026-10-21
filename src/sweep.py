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
USEFUL = ["ts_zeroshot_auroc", "ts2txt_label_p@10", "ts_probe_auroc"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
N_BOOT = 200


def zs_scores(q, txt_train, sets_train):
    """Zero-shot scores (n, 5): cosine to the mean projected embedding of single-class train reports."""
    P = []
    for c in CLASSES:
        m = np.array([s == frozenset([c]) for s in sets_train])
        p = txt_train[m].mean(0) if m.any() else np.zeros(txt_train.shape[1])
        P.append(p / max(np.linalg.norm(p), 1e-12))
    return q @ np.stack(P).T


def macro_auroc(y, scores):
    from sklearn.metrics import roc_auc_score
    vals = [roc_auc_score(y[:, j], scores[:, j]) for j in range(y.shape[1]) if len(set(y[:, j])) == 2]
    return float(np.mean(vals)) if vals else float("nan")


def probe_scores(Xtr, ytr, Xte, device, steps=300, wd=1e-3):
    """Multi-label logistic-regression probe (one linear layer, BCE), full-batch Adam, standardized inputs."""
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    mu, sd = Xtr_t.mean(0, keepdim=True), Xtr_t.std(0, keepdim=True).clamp(min=1e-6)
    Xtr_t, Xte_t = (Xtr_t - mu) / sd, (Xte_t - mu) / sd
    y = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    torch.manual_seed(0)
    lin = torch.nn.Linear(Xtr_t.shape[1], y.shape[1]).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(lin(Xtr_t), y).backward()
        opt.step()
    with torch.no_grad():
        return lin(Xte_t).cpu().numpy()


def patient_bootstrap(pids, fns, n_boot=N_BOOT, seed=0):
    """Percentile CIs, resampling *patients* with replacement. fns: name -> f(index array) -> float."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(pids)
    rows_of = {p: np.flatnonzero(pids == p) for p in uniq}
    out = {k: [] for k in fns}
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        for k, f in fns.items():
            out[k].append(f(idx))
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def small_grid():
    """13 models at the paper's training temperature (for the text-coarsening mechanism test)."""
    tau = 0.2
    g = [("infonce", 0.0, None, tau)]
    g += [("soft", m, st, tau) for m in (0.5, 1.0) for st in (0.02, 0.05, 0.1, 0.2)]
    g += [("label", m, None, tau) for m in (0.25, 0.5, 0.75, 1.0)]
    return g


def coarsen_text(data, k, seed=0):
    """Mechanism test: replace every report embedding by its k-means centroid (fit on train reports).

    Coarsens the text's meaning by the text model's own similarity, with no diagnostic labels.
    Returns the cluster id per split (used as the duplicate group)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(data["train"]["txt"].numpy())
    C = torch.as_tensor(km.cluster_centers_, dtype=torch.float32)
    ids = {}
    for s in data:
        lab = km.predict(data[s]["txt"].numpy())
        data[s]["txt"] = C[lab].clone()
        ids[s] = lab
    return ids


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
    ap.add_argument("--small", action="store_true", help="13-model grid at tau 0.2 (mechanism test)")
    ap.add_argument("--coarsen", type=int, default=0, help="k-means text coarsening to k groups")
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
    if a.limit:
        for s in ("train", "val", "test"):
            data[s] = {m: v[:a.limit] for m, v in data[s].items()}
    cids = coarsen_text(data, a.coarsen) if a.coarsen else None
    aux = {}
    for s in ("train", "val", "test"):
        rows = split[split.split == s]
        if a.limit:
            rows = rows.head(a.limit)
        sets = sc.loc[rows.index].tolist()
        lab = pd.Series([",".join(sorted(x)) for x in sets]).astype("category").cat.codes.values.copy()
        tg = torch.as_tensor(cids[s]) if cids is not None else torch.as_tensor(text_groups(rows.report.tolist()))
        aux[s] = {"text_group": tg,
                  "label_group": torch.as_tensor(lab),
                  "txt_raw_n": F.normalize(data[s]["txt"], dim=1).to(device), "sets": sets}
    bs = min(a.bs, len(data["train"]["ts"]))
    gt = aux["test"]["text_group"].numpy()
    gv = aux["val"]["text_group"].numpy()
    keep = np.array([len(x) > 0 for x in aux["test"]["sets"]])
    keep_v = np.array([len(x) > 0 for x in aux["val"]["sets"]])
    keep_tr = np.array([len(x) > 0 for x in aux["train"]["sets"]])
    y_test = multihot(aux["test"]["sets"])[keep]
    y_val = multihot(aux["val"]["sets"])[keep_v]
    y_tr = multihot(aux["train"]["sets"])[keep_tr]
    sets_keep = [x for x, k in zip(aux["test"]["sets"], keep) if k]
    test_rows = split[split.split == "test"]
    if a.limit:
        test_rows = test_rows.head(a.limit)
    pids = test_rows.patient_id.values[keep]

    # reference: the same probe on the frozen encoder's own TS features
    ref = macro_auroc(y_test, probe_scores(data["train"]["ts"].numpy()[keep_tr], y_tr,
                                           data["test"]["ts"].numpy()[keep], device))
    print(f"frozen-encoder probe AUROC (reference): {ref:.3f}", flush=True)

    tag = (f"_k{a.coarsen}" if a.coarsen else "") + ("_small" if a.small else "") +           ("_quick" if a.quick or a.limit else "")
    out_csv = emb / f"sweep2_lead{a.lead}{tag}.csv"
    print(f"unique test texts: {len(np.unique(gt))}", flush=True)
    done = set()
    rows_out = []
    if out_csv.exists():  # resume
        prev = pd.read_csv(out_csv)
        rows_out = prev.to_dict("records")
        done = {(r["kind"], r["mix"], r["soft_tau"], r["tau"]) for r in rows_out}

    g = small_grid() if a.small else grid(a.quick)
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
        zs = zs_scores(z["ts"][keep], ztr_txt, aux["train"]["sets"])
        row["ts_zeroshot_auroc"] = macro_auroc(y_test, zs)
        # independent downstream check: linear probe on the projected TS embedding
        ztr_ts = embed_all(heads, data["train"], device)["ts"].cpu().numpy()
        pr = probe_scores(ztr_ts[keep_tr], y_tr, z["ts"][keep], device)
        row["ts_probe_auroc"] = macro_auroc(y_test, pr)
        row["frozen_probe_auroc_ref"] = ref
        # validation-set versions, for the model-selection analysis
        zv = {m: v.cpu().numpy() for m, v in embed_all(heads, data["val"], device).items()}
        mvv = all_metrics_v2(zv["ts"], zv["txt"], gv)
        for k in METRICS:
            row[f"val:ts-txt:{k}"] = mvv.get(k, np.nan)
        row["val:ts_zeroshot_auroc"] = macro_auroc(y_val, zs_scores(zv["ts"][keep_v], ztr_txt,
                                                                    aux["train"]["sets"]))
        # patient-level bootstrap on the test set
        zt = z["ts"][keep] / np.linalg.norm(z["ts"][keep], axis=1, keepdims=True)
        zx = z["txt"][keep] / np.linalg.norm(z["txt"][keep], axis=1, keepdims=True)
        S = zt @ zx.T
        n = len(S)
        marg = np.diag(S) - (S.sum(1) - np.diag(S)) / (n - 1)
        ci = patient_bootstrap(pids, {"cos_margin": lambda i: marg[i].mean(),
                                      "zs_auroc": lambda i: macro_auroc(y_test[i], zs[i]),
                                      "probe_auroc": lambda i: macro_auroc(y_test[i], pr[i])})
        for k, (lo, hi) in ci.items():
            row[f"ci_lo:{k}"], row[f"ci_hi:{k}"] = lo, hi
        row["seconds"] = time.time() - t0
        rows_out.append(row)
        pd.DataFrame(rows_out).to_csv(out_csv, index=False)
        print(f"{kind:7s} mix {mix:.2f} st {key[2]:5.2f} tau {tau:.2f} ep {len(log):2d} | "
              f"margin {row['ts-txt:cos_margin']:.3f} procS {row['ts-txt:procrustes_sym']:.3f} "
              f"R@1dup {row['ts-txt:R@1_dupaware']:.4f} | AUROC {row['ts_zeroshot_auroc']:.3f} "
              f"probe {row['ts_probe_auroc']:.3f} labP {row['ts2txt_label_p@10']:.3f} ({row['seconds']:.0f}s)", flush=True)

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
