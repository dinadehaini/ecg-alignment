"""Cleanup checks on the 13 tau-0.2 models (heads saved by info_loss.py, original reports).

1. Readouts. report-prototype classification (label-informed: mean projected embedding of
   single-superclass TRAIN reports, as in the sweep) vs fixed label-free class descriptions
   (src/prompts_superclass.json, frozen before evaluation; pre-registered readout = ensemble of all
   8 prompts per class; single prompts and per-language ensembles reported as variation).
2. Embedding corrections, fit on TRAIN embeddings only, applied to the pre-normalization head outputs
   before L2 normalization and cosine scoring:
     center / whiten (ZCA, shrinkage 0.1) x  sig (signal side only) | txt (text side only) |
     both (each side its own transform) | shared (one transform fit on pooled signal+text)
3. Unsupervised arms only: the sweep's correlations and selection cost without the label-target arms
   (63-model sweep CSV, and the 13 here).
4. Paired selection regret: resample validation patients, pick each selector's winner, evaluate all
   winners on one shared resample of test patients; report the paired difference to the full
   labeled-validation selector, regret vs the best model in that test resample, and selection
   frequencies. Labeled-budget selectors use 25/50/100 validation patients. Conditional on the
   trained candidates (training-seed uncertainty is separate).
Also: duplicate-aware retrieval chance levels.

Usage (GPU for the text encoder): python src/cleanup.py --emb results/emb --ptbxl /content/ptbxl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sweep import LOWER_BETTER  # noqa: E402
from label_eval import CLASSES, multihot, superclasses  # noqa: E402
from metrics import cka_rbf, cosine_stats, mutual_knn  # noqa: E402
from metrics_v2 import all_metrics_v2, procrustes_symmetric, retrieval_groups, text_groups  # noqa: E402
from sweep import macro_auroc, small_grid  # noqa: E402
from train_heads import Head, load_emb  # noqa: E402
from train_heads_v2 import CONFIG_A  # noqa: E402

ALIGN = ["cos_margin", "procrustes_sym", "cka_rbf", "mutual_knn@5", "R@1_exact", "R@1_dupaware"]
SHRINK = 0.1
N_BOOT = 100
BUDGETS = (25, 50, 100)


# ---------- corrections ----------
def fit_center(Z):
    return {"mu": Z.mean(0), "W": None}


def fit_whiten(Z):
    mu = Z.mean(0)
    C = np.cov(Z - mu, rowvar=False)
    C = (1 - SHRINK) * C + SHRINK * np.trace(C) / len(C) * np.eye(len(C))
    w, V = np.linalg.eigh(C)
    return {"mu": mu, "W": (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T}


def apply(t, Z):
    if t is None:
        return Z
    Z = Z - t["mu"]
    return Z if t["W"] is None else Z @ t["W"]


def _cka_rbf_t(X, Y):
    """Torch version of metrics.cka_rbf (same median-heuristic bandwidth and centering)."""
    def sq(A):
        n2 = (A * A).sum(1)
        return (n2[:, None] + n2[None] - 2 * A @ A.T).clamp(min=0)
    DX, DY = sq(X), sq(Y)
    iu = torch.triu_indices(len(X), len(X), 1, device=X.device)
    med = torch.sqrt(torch.cat([DX[iu[0], iu[1]], DY[iu[0], iu[1]]])).median()
    g = 1.0 / (2.0 * med ** 2)
    K, L = torch.exp(-g * DX), torch.exp(-g * DY)
    Kc = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
    Lc = L - L.mean(0, keepdim=True) - L.mean(1, keepdim=True) + L.mean()
    return float((Kc * Lc).sum() / (Kc.norm() * Lc.norm()))


def selector_metrics(X, Y, g, device="cpu"):
    """The six selector metrics only (all_metrics_v2 also computes unused variants).
    Symmetric Procrustes and CKA run in torch (float64) on `device`; definitions match metrics(_v2)."""
    out = {"cos_margin": cosine_stats(X, Y)["cos_margin"], "mutual_knn@5": mutual_knn(X, Y, k=5)}
    Xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float64, device=device)
    Xc, Yc = Xt - Xt.mean(0), Yt - Yt.mean(0)
    sv = torch.linalg.svdvals((Xc / Xc.norm()).T @ (Yc / Yc.norm()))
    out["procrustes_sym"] = float(max(0.0, 1.0 - float(sv.sum()) ** 2))
    out["cka_rbf"] = _cka_rbf_t(Xt, Yt)
    r = retrieval_groups(X, Y, g, ks=(1,))
    out["R@1_exact"], out["R@1_dupaware"] = r["R@1_exact"], r["R@1_dupaware"]
    return out


def corrections(ztr_sig, ztr_txt):
    """name -> (signal transform, text transform)."""
    out = {"none": (None, None)}
    for kind, fit in (("center", fit_center), ("whiten", fit_whiten)):
        s, t = fit(ztr_sig), fit(ztr_txt)
        sh = fit(np.concatenate([ztr_sig, ztr_txt]))
        out[f"{kind}:sig"] = (s, None)
        out[f"{kind}:txt"] = (None, t)
        out[f"{kind}:both"] = (s, t)
        out[f"{kind}:shared"] = (sh, sh)
    return out


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ---------- readouts ----------
def class_vectors_reports(ztxt_tr, sets_tr):
    P = []
    for c in CLASSES:
        m = np.array([s == frozenset([c]) for s in sets_tr])
        P.append(unit(ztxt_tr[m]).mean(0) if m.any() else np.zeros(ztxt_tr.shape[1]))
    return unit(np.stack(P))


def class_vectors_prompts(zp, idx):
    """zp: (5 classes, n_prompts, d) corrected projected prompts; idx: which prompts to ensemble."""
    return unit(unit(zp[:, idx]).mean(1))


def scores(zsig, P):
    return unit(zsig) @ P.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--heads", default=None, help="default <emb>/heads")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
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
    hdir = Path(a.heads) if a.heads else emb / ("heads_quick" if a.limit else "heads")
    split = pd.read_csv(emb / "split.csv", index_col=0)
    data = load_emb(emb, CONFIG_A, a.lead)
    sc = superclasses(a.ptbxl)
    info = {}
    for s in ("train", "val", "test"):
        rows = split[split.split == s]
        if a.limit:
            rows = rows.head(a.limit)
            data[s] = {m: v[:a.limit] for m, v in data[s].items()}
        sets = sc.loc[rows.index].tolist()
        keep = np.array([len(x) > 0 for x in sets])
        y = multihot(sets)[keep]
        if a.limit:  # smoke test only: random stand-in labels so every code path runs
            keep = np.ones(len(sets), bool)
            y = np.random.default_rng(1).integers(0, 2, (len(sets), len(CLASSES)))
            y[0], y[1] = 0, 1
        info[s] = {"sets": sets, "keep": keep, "y": y,
                   "g": text_groups(rows.report.tolist()), "pid": rows.patient_id.values}

    # duplicate-aware chance levels (random ranking)
    chance = {}
    for s in ("val", "test"):
        g = info[s]["g"]
        n, size = len(g), np.bincount(g)[g]
        chance[s] = {"R@1_dupaware_chance": float((size / n).mean()),
                     "R@10_dupaware_chance": float(np.mean([
                         1.0 if n - sz < 10 else
                         1 - np.exp(np.sum(np.log((n - sz - np.arange(10)) / (n - np.arange(10)))))
                         for sz in size])),
                     "R@1_exact_chance": 1.0 / n}
    print("chance:", json.dumps(chance), flush=True)

    # prompt embeddings (raw text-encoder space)
    pj = json.load(open(Path(__file__).resolve().parent / "prompts_superclass.json", encoding="utf-8"))
    texts = [p for lang in ("en", "de") for c in CLASSES for p in pj[lang][c]]
    if a.limit:  # smoke test: no model download; random stand-ins of the right shape
        raw_p = torch.randn(len(texts), data["train"]["txt"].shape[1])
    else:
        from extract import Qwen3EmbEnc
        enc = Qwen3EmbEnc(device)
        raw_p = enc(texts).float().cpu()
        del enc
        torch.cuda.empty_cache()
    npl = len(pj["en"]["NORM"])  # prompts per class per language
    # (lang, class, i) -> (class, prompt index 0..2*npl-1)
    order = np.array([[li * len(CLASSES) * npl + ci * npl + i for li in range(2) for i in range(npl)]
                      for ci in range(len(CLASSES))])
    readouts = {"reports": None, "prompts:ens8": list(range(2 * npl)),
                "prompts:en4": list(range(npl)), "prompts:de4": list(range(npl, 2 * npl))}
    readouts.update({f"prompts:single{i}": [i] for i in range(2 * npl)})

    rows_out, per_model = [], {}
    for kind, mix, st, tau in small_grid():
        key = f"{kind}_mix{mix:.2f}_st{(st if st is not None else -1.0):.2f}"
        f = hdir / f"{key}.pt"
        if not f.exists():
            print("missing", f, flush=True)
            continue
        sd = torch.load(f, map_location="cpu")
        heads = {}
        for m in ("ts", "txt"):
            h = Head(data["train"][m].shape[1])
            h.load_state_dict({k: v.float() for k, v in sd[m].items()})
            heads[m] = h.eval()
        with torch.no_grad():
            raw = {s: {m: heads[m].net(data[s][m]).numpy() for m in ("ts", "txt")} for s in data}
            zp_raw = heads["txt"].net(raw_p).numpy()
        corr = corrections(raw["train"]["ts"], raw["train"]["txt"])
        test_m = all_metrics_v2(unit(raw["test"]["ts"]), unit(raw["test"]["txt"]), info["test"]["g"])
        base = {"model": key, "kind": kind, "mix": mix, "soft_tau": st if st is not None else -1.0}
        base.update({f"test:{k}": test_m.get(k, np.nan) for k in ALIGN})
        store = {}
        for cname, (ts_t, tx_t) in corr.items():
            zsig = {s: apply(ts_t, raw[s]["ts"]) for s in ("val", "test")}
            ztr_txt = apply(tx_t, raw["train"]["txt"])
            zp = apply(tx_t, zp_raw)[order]  # (5, 8, d)
            for rname, idx in readouts.items():
                P = class_vectors_reports(ztr_txt, info["train"]["sets"]) if idx is None else \
                    class_vectors_prompts(zp, idx)
                r = dict(base, correction=cname, readout=rname)
                for s in ("val", "test"):
                    S = scores(zsig[s], P)
                    r[f"{s}_auroc"] = macro_auroc(info[s]["y"], S[info[s]["keep"]])
                    if rname in ("reports", "prompts:ens8") and cname in ("none", "whiten:shared", "center:shared"):
                        store[(cname, rname, s)] = S
                rows_out.append(r)
        store["zs_val"], store["zt_val"] = unit(raw["val"]["ts"]), unit(raw["val"]["txt"])
        per_model[key] = (base, store)
        print(f"{key}: " + " ".join(f"{r['correction']}/{r['readout']} {r['test_auroc']:.3f}"
                                    for r in rows_out[-len(corr) * len(readouts):]
                                    if r["readout"] in ("reports", "prompts:ens8")
                                    and r["correction"] in ("none", "center:shared", "whiten:shared")), flush=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(emb / f"cleanup_lead{a.lead}.csv", index=False)

    # ---------- summary table (1 + 2) ----------
    sp = lambda x, y: float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    summ = []
    for (cname, rname), g in df.groupby(["correction", "readout"], sort=False):
        if rname.startswith("prompts:single"):
            continue
        for scope, gg in (("all 13", g), ("unsup 9", g[g.kind != "label"])):
            b = gg[gg.kind == "infonce"].iloc[0]
            best = gg.loc[gg.test_auroc.idxmax()]
            summ.append({"readout": rname, "correction": cname, "models": scope,
                         "InfoNCE": b.test_auroc, "best": best.test_auroc, "best model": best.model,
                         "gap best-InfoNCE": best.test_auroc - b.test_auroc,
                         "range": gg.test_auroc.max() - gg.test_auroc.min(),
                         "rho margin": sp(gg["test:cos_margin"], gg.test_auroc),
                         "rho procrustes(+)": sp(-gg["test:procrustes_sym"], gg.test_auroc),
                         "rho R@1 dup": sp(gg["test:R@1_dupaware"], gg.test_auroc)})
    summ = pd.DataFrame(summ)
    single = df[df.readout.str.startswith("prompts:single")]
    var = single.groupby(["correction", "model"]).test_auroc.agg(["mean", "std", "min", "max"]).reset_index()
    var = var.groupby("correction")[["mean", "std", "min", "max"]].mean().reset_index()

    # ---------- 3: unsupervised-only sweep ----------
    sweep_txt = ""
    sw_f = emb / f"sweep2_lead{a.lead}.csv"
    if sw_f.exists():
        sw = pd.read_csv(sw_f)
        rows3 = []
        for scope, g in [("all 63", sw), ("no label arms", sw[sw.kind != "label"])] + \
                [(f"no label, tau={t}", g) for t, g in sw[sw.kind != "label"].groupby("tau")]:
            r = {"scope": scope, "n": len(g)}
            for m in ALIGN:
                v = -g[f"ts-txt:{m}"] if m in LOWER_BETTER else g[f"ts-txt:{m}"]
                r[f"rho {m}"] = sp(v, g.ts_zeroshot_auroc)
            crit = g.loc[g["val:ts_zeroshot_auroc"].idxmax()]
            for m in ("cos_margin", "procrustes_sym", "R@1_dupaware"):
                vc = g[f"val:ts-txt:{m}"]
                pick = g.loc[vc.idxmin() if m in LOWER_BETTER else vc.idxmax()]
                r[f"cost {m}"] = crit.ts_zeroshot_auroc - pick.ts_zeroshot_auroc
            rows3.append(r)
        sweep_txt = pd.DataFrame(rows3).to_markdown(index=False, floatfmt=".3f")

    # ---------- 4: paired selection regret ----------
    rng = np.random.default_rng(0)
    vp, tp = info["val"]["pid"], info["test"]["pid"]
    vrows = {p: np.flatnonzero(vp == p) for p in np.unique(vp)}
    trows = {p: np.flatnonzero(tp == p) for p in np.unique(tp)}
    keys = list(per_model)
    sel_rows = []
    targets = [("none", "reports"), ("none", "prompts:ens8"), ("whiten:shared", "prompts:ens8")]
    kv, kt = info["val"]["keep"], info["test"]["keep"]
    # map keep-filtered label rows: build full-length label arrays (unlabeled rows masked by keep)
    def full_y(s):
        y = np.zeros((len(info[s]["keep"]), len(CLASSES)), int)
        y[info[s]["keep"]] = info[s]["y"]
        return y
    yv, yt = full_y("val"), full_y("test")
    for b in range(a.n_boot):
        vpick = rng.choice(list(vrows), size=len(vrows), replace=True)
        vr = np.concatenate([vrows[p] for p in vpick])
        tr = np.concatenate([trows[p] for p in rng.choice(list(trows), size=len(trows), replace=True)])
        trk = tr[kt[tr]]
        budget_rows = {n: np.concatenate([vrows[p] for p in rng.choice(list(vrows), size=n, replace=False)])
                       for n in BUDGETS if n <= len(vrows)}
        align_vals = {}
        for k in keys:
            st_ = per_model[k][1]
            align_vals[k] = selector_metrics(st_["zs_val"][vr], st_["zt_val"][vr], info["val"]["g"][vr], device)
        cand_sets = {"all 13": keys, "unsup 9": [k for k in keys if not k.startswith("label")]}
        for cname, rname in targets:
            test_all = {k: macro_auroc(yt[trk], per_model[k][1][(cname, rname, "test")][trk]) for k in keys}
            val_auc = lambda rows, ks: {k: macro_auroc(yv[rows[kv[rows]]],
                                                       per_model[k][1][(cname, rname, "val")][rows[kv[rows]]]) for k in ks}
            for cs_name, ks in cand_sets.items():
                test_auc = {k: test_all[k] for k in ks}
                sel = {}
                for mm in ALIGN:
                    vals = {k: align_vals[k][mm] for k in ks}
                    sel[f"val {mm}"] = (min if mm in LOWER_BETTER else max)(vals, key=vals.get)
                full = val_auc(vr, ks)
                sel["val labels (all)"] = max(full, key=full.get)
                for n, rr in budget_rows.items():
                    bv = val_auc(rr, ks)
                    sel[f"val labels ({n} patients)"] = max(bv, key=bv.get)
                best_test = max(test_auc.values())
                for sname, k in sel.items():
                    sel_rows.append({"target": f"{rname} [{cname}]", "candidates": cs_name, "selector": sname,
                                     "boot": b, "picked": k, "test_auroc": test_auc[k],
                                     "diff_vs_labels_all": test_auc[k] - test_auc[sel["val labels (all)"]],
                                     "regret": best_test - test_auc[k]})
        if b % 5 == 0:
            print(f"boot {b}/{a.n_boot}", flush=True)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(emb / f"cleanup_selection_lead{a.lead}.csv", index=False)
    q = lambda x, p: float(np.nanpercentile(x, p))
    selsum = sel.groupby(["target", "candidates", "selector"]).agg(
        test_auroc=("test_auroc", "mean"),
        diff=("diff_vs_labels_all", "mean"),
        diff_lo=("diff_vs_labels_all", lambda x: q(x, 2.5)),
        diff_hi=("diff_vs_labels_all", lambda x: q(x, 97.5)),
        regret=("regret", "mean"),
        regret_hi=("regret", lambda x: q(x, 97.5)),
        top_pick=("picked", lambda x: x.value_counts().index[0]),
        top_pick_freq=("picked", lambda x: x.value_counts(normalize=True).iloc[0])).reset_index()

    keep_cols = ["readout", "correction", "models", "InfoNCE", "best", "gap best-InfoNCE", "range",
                 "rho margin", "rho procrustes(+)", "rho R@1 dup", "best model"]
    main_tab = summ[summ.readout.isin(["reports", "prompts:ens8"])][keep_cols]
    text = "\n".join([
        f"# Cleanup checks (lead {a.lead}; 13 tau-0.2 models, original reports)\n",
        "## 1+2. Test AUROC by readout and embedding correction (corrections fit on train embeddings)\n",
        main_tab.to_markdown(index=False, floatfmt=".3f"),
        "\n## Prompt variation: per-language ensembles\n",
        summ[summ.readout.isin(["prompts:en4", "prompts:de4"]) & (summ.models == "all 13")][keep_cols].to_markdown(index=False, floatfmt=".3f"),
        "\n## Prompt variation: single prompts (mean over models of per-model mean/sd/min/max across the 8 prompts)\n",
        var.to_markdown(index=False, floatfmt=".3f"),
        "\n## 3. Sweep without label-target arms (63-model sweep CSV; zero-shot = report prototypes)\n",
        sweep_txt or "(sweep CSV not found)",
        f"\n## 4. Paired selection over {a.n_boot} validation/test patient resamples (candidates: all 13, or the 9 without label targets)\n",
        "diff = test AUROC of the selector's pick minus that of the full-validation labeled selector "
        "(paired, same test resample); regret = best model in that test resample minus the pick.\n",
        selsum.to_markdown(index=False, floatfmt=".3f"),
        "\n## Duplicate-aware chance levels (random ranking)\n",
        pd.DataFrame(chance).T.to_markdown(floatfmt=".4f")])
    print(text, flush=True)
    (emb / f"cleanup_lead{a.lead}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
