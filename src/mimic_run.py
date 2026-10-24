"""MIMIC-IV-ECG replication under the frozen protocol (PREDICTIONS_MIMIC.md).

1. Train the 9 unsupervised configurations (InfoNCE; soft targets mix 0.5 / 1.0 x soft-tau 0.02 / 0.05 /
   0.1 / 0.2) at tau 0.2 on the MIMIC embeddings, same code and hyperparameters as PTB-XL; heads saved.
2. Readouts per concept (8 report-derived concepts; AUROC on eligible test ECGs only, macro over concepts):
   report prototypes (training reports positive for exactly one concept), hand-written families from
   src/prompts_mimic.json (clinical = PRIMARY, names, report_style = secondary), each uncorrected and
   text-centered (train text mean; pipeline: center each projected text, L2-normalize, average, normalize).
3. Paired selection over 100 validation/test patient resamples: validation alignment metrics, duplicate-aware
   retrieval, 25 / 50 / 100 labeled validation patients, full labeled validation; evaluated per readout.
4. Query source (label-informed): prototypes of size k from training reports vs validation reports whose
   string is seen in training vs unseen strings, matched k, classes eligible in every source.
Usage (GPU): python src/mimic_run.py --emb results/emb_mimic --labels results/mimic/mimic_subset_labels.csv
"""
from __future__ import annotations

import argparse
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
from analyze_sweep import LOWER_BETTER  # noqa: E402
from cleanup import selector_metrics, unit  # noqa: E402
from metrics_v2 import all_metrics_v2, text_groups  # noqa: E402
from train_heads import Head, load_emb  # noqa: E402
from train_heads_v2 import CONFIG_A, embed_all, train  # noqa: E402

ALIGN = ["cos_margin", "procrustes_sym", "cka_rbf", "mutual_knn@5", "R@1_exact", "R@1_dupaware"]
GRID = [("infonce", 0.0, None)] + [("soft", m, st) for m in (0.5, 1.0) for st in (0.02, 0.05, 0.1, 0.2)]
TAU, N_BOOT, BUDGETS, KS, N_DRAWS = 0.2, 100, (25, 50, 100), (5, 10, 25), 20


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(y)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def macro(Y, S, rows=None):
    rows = np.arange(len(Y)) if rows is None else rows
    v = [auroc(Y[rows, j], S[rows, j]) for j in range(Y.shape[1])]
    return float(np.nanmean(v)) if not all(np.isnan(v)) else float("nan")


def cvec(Z, mu, centered):
    Z = Z - mu if centered else Z
    return unit(unit(Z).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--limit", type=int, default=0)
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
    here = Path(__file__).resolve().parent
    split = pd.read_csv(emb / "split.csv", index_col=0)
    lab = pd.read_csv(a.labels).set_index("study_id")
    pj = json.load(open(here / "prompts_mimic.json", encoding="utf-8"))
    C = pj["concepts"]
    data = load_emb(emb, CONFIG_A, a.lead)
    info = {}
    for s in ("train", "val", "test"):
        rows = split[split.split == s]
        if a.limit:
            rows = rows.head(a.limit)
            data[s] = {m: v[:a.limit] for m, v in data[s].items()}
        L = lab.loc[rows.index, C]
        Y = np.where(L.values == "positive", 1.0, np.where(L.values == "negative", 0.0, np.nan))
        if a.limit:  # smoke test only: random stand-in labels
            Y = np.random.default_rng(1).integers(0, 2, Y.shape).astype(float)
            Y[0], Y[1] = 0, 1
        rep = [" ".join(str(r).lower().split()) for r in rows.report]
        info[s] = {"Y": Y, "rep": rep, "g": text_groups(rows.report.tolist()), "pid": rows.patient_id.values}
    print({s: len(info[s]["Y"]) for s in info}, flush=True)

    aux = {s: {"text_group": torch.as_tensor(info[s]["g"]), "label_group": torch.zeros(len(info[s]["g"]), dtype=torch.long),
               "txt_raw_n": F.normalize(data[s]["txt"], dim=1).to(device), "sets": [frozenset()] * len(info[s]["g"])}
           for s in info}
    # hand-written queries
    fams = {f: {c: pj[f][c] for c in C} for f in ("clinical", "names", "report_style")}
    flat = [(f, c, t) for f, d in fams.items() for c in C for t in d[c]]
    if a.limit:
        raw_q = torch.randn(len(flat), data["train"]["txt"].shape[1])
    else:
        from extract import Qwen3EmbEnc
        enc = Qwen3EmbEnc(device)
        raw_q = enc([t for *_, t in flat]).float().cpu()
        del enc
        torch.cuda.empty_cache()
    Ytr = info["train"]["Y"]
    single = {c: np.flatnonzero((Ytr[:, j] == 1) & (np.nansum(Ytr, 1) == 1)) for j, c in enumerate(C)}
    print("single-concept training reports:", {c: len(v) for c, v in single.items()}, flush=True)

    hdir = emb / "heads"
    hdir.mkdir(exist_ok=True)
    per, rows_a = {}, []
    bs = min(256, len(data["train"]["ts"]))
    for kind, mix, st in GRID:
        key = f"{kind}_mix{mix:.2f}_st{(st if st is not None else -1.0):.2f}"
        t0 = time.time()
        targs = SimpleNamespace(epochs=a.epochs, bs=bs, lr=1e-4, wd=1e-5, tau=TAU, patience=5)
        for s in aux:
            aux[s]["mix"], aux[s]["soft_tau"] = mix, (st if st is not None else 0.05)
        heads, log = train(kind, a.seed, data, aux, targs, device)
        torch.save({m: {k: v.half().cpu() for k, v in heads[m].state_dict().items()} for m in ("ts", "txt")},
                   hdir / f"{key}.pt")
        heads = {m: heads[m].eval().cpu() for m in ("ts", "txt")}
        with torch.no_grad():
            raw = {s: {m: heads[m].net(data[s][m]).numpy() for m in ("ts", "txt")} for s in data}
            zq = heads["txt"].net(raw_q).numpy()
        mu = raw["train"]["txt"].mean(0)
        zs = {s: unit(raw[s]["ts"]) for s in ("val", "test")}
        store = {"zs_val": zs["val"], "zt_val": unit(raw["val"]["txt"]), "raw": raw}
        readouts = {"reports": {c: raw["train"]["txt"][single[c]] for c in C}}
        for f in fams:
            readouts[f] = {c: zq[[i for i, (ff, cc, _) in enumerate(flat) if ff == f and cc == c]] for c in C}
        tm = all_metrics_v2(unit(raw["test"]["ts"]), unit(raw["test"]["txt"]), info["test"]["g"])
        for rname, Zc in readouts.items():
            for centered in (False, True):
                P = np.stack([cvec(Zc[c], mu, centered) if len(Zc[c]) else np.zeros(mu.shape) for c in C])
                for s in ("val", "test"):
                    store[(rname, centered, s)] = zs[s] @ P.T
                St = store[(rname, centered, "test")]
                for j, c in enumerate(C):
                    rows_a.append({"model": key, "kind": kind, "readout": rname, "centered": centered, "concept": c,
                                   "auroc": auroc(info["test"]["Y"][:, j], St[:, j])})
        per[key] = store
        print(f"{key}: ep {len(log)} margin {tm['cos_margin']:.3f} | " + " ".join(
            f"{r[:4]}{'C' if ce else ''} {macro(info['test']['Y'], store[(r, ce, 'test')]):.3f}"
            for r in readouts for ce in (False, True)) + f" ({time.time() - t0:.0f}s)", flush=True)

    A = pd.DataFrame(rows_a)
    A.to_csv(emb / "mimic_readouts.csv", index=False)
    I, keys = "infonce_mix0.00_st-1.00", list(per)
    macro_t = A.groupby(["model", "readout", "centered"]).auroc.mean().unstack("model")
    best = macro_t.loc[("clinical", True)].drop(I).idxmax()  # reported, not used for selection
    t1 = pd.DataFrame({"InfoNCE": macro_t[I], "best soft (max over 8)": macro_t.drop(columns=I).max(1)})
    t1["gap"] = t1.iloc[:, 1] - t1.iloc[:, 0]

    # 3. paired selection
    rng = np.random.default_rng(0)
    vp, tp = info["val"]["pid"], info["test"]["pid"]
    vrows = {p: np.flatnonzero(vp == p) for p in np.unique(vp)}
    trows = {p: np.flatnonzero(tp == p) for p in np.unique(tp)}
    targets = [(r, ce) for r in ("clinical", "reports", "names", "report_style") for ce in (False, True)]
    sel_rows = []
    for b in range(a.n_boot):
        vr = np.concatenate([vrows[p] for p in rng.choice(list(vrows), size=len(vrows), replace=True)])
        tr = np.concatenate([trows[p] for p in rng.choice(list(trows), size=len(trows), replace=True)])
        bud = {n: np.concatenate([vrows[p] for p in rng.choice(list(vrows), size=n, replace=False)])
               for n in BUDGETS if n <= len(vrows)}
        al = {k: selector_metrics(per[k]["zs_val"][vr], per[k]["zt_val"][vr], info["val"]["g"][vr], device) for k in keys}
        for r, ce in targets:
            test_auc = {k: macro(info["test"]["Y"], per[k][(r, ce, "test")], tr) for k in keys}
            sel = {f"val {m}": (min if m in LOWER_BETTER else max)(keys, key=lambda k: al[k][m]) for m in ALIGN}
            for nm, rr in [("all", vr)] + [(str(n), x) for n, x in bud.items()]:
                v = {k: macro(info["val"]["Y"], per[k][(r, ce, "val")], rr) for k in keys}
                sel[f"val labels ({nm})"] = max(keys, key=lambda k: -1 if np.isnan(v[k]) else v[k])
            bt = max(test_auc.values())
            for sn, k in sel.items():
                sel_rows.append({"target": f"{r}{' [centered]' if ce else ''}", "selector": sn, "boot": b, "picked": k,
                                 "test_auroc": test_auc[k], "diff": test_auc[k] - test_auc[sel["val labels (all)"]],
                                 "regret": bt - test_auc[k]})
        if b % 10 == 0:
            print(f"boot {b}/{a.n_boot}", flush=True)
    S = pd.DataFrame(sel_rows)
    S.to_csv(emb / "mimic_selection.csv", index=False)
    q = lambda x, p: float(np.nanpercentile(x, p))
    ssum = S.groupby(["target", "selector"]).agg(test_auroc=("test_auroc", "mean"), diff=("diff", "mean"),
                                                  diff_lo=("diff", lambda x: q(x, 2.5)), diff_hi=("diff", lambda x: q(x, 97.5)),
                                                  top=("picked", lambda x: x.value_counts().index[0]),
                                                  top_freq=("picked", lambda x: x.value_counts(normalize=True).iloc[0])).reset_index()

    # 4. query source (label-informed), InfoNCE vs best
    train_str = set(info["train"]["rep"])
    Yv = info["val"]["Y"]
    def pool(s, Y, cond):
        return {c: np.array([i for i in range(len(Y)) if Y[i, j] == 1 and np.nansum(Y[i]) == 1 and cond(info[s]["rep"][i])], int)
                for j, c in enumerate(C)}
    pools = {"train": ("train", pool("train", Ytr, lambda r: True)),
             "val seen string": ("val", pool("val", Yv, lambda r: r in train_str)),
             "val unseen string": ("val", pool("val", Yv, lambda r: r not in train_str))}
    counts = pd.DataFrame([{"source": k, **{c: len(v[1][c]) for c in C}} for k, v in pools.items()])
    rows4 = []
    for k in KS:
        elig = [c for c in C if all(len(p[c]) >= k for _, p in pools.values())]
        if not elig:  # explicit skip: never reduce k or sample with replacement
            rows4.append({"k": k, "classes": "(none eligible: skipped)", "model": I, "source": "-", "centered": False,
                          "auroc": float("nan")})
            continue
        draws ={src: [{c: rng.choice(p[c], size=k, replace=False) for c in elig} for _ in range(N_DRAWS)]
                 for src, (_, p) in pools.items()}
        for mk in (I, best):
            raw = per[mk]["raw"]
            mu = raw["train"]["txt"].mean(0)
            for src, dl in draws.items():
                spl = pools[src][0]
                for ce in (False, True):
                    vals = []
                    for d in dl:
                        P = np.stack([cvec(raw[spl]["txt"][d[c]], mu, ce) for c in elig])
                        Sx = unit(raw["test"]["ts"]) @ P.T
                        cols = [C.index(c) for c in elig]
                        vals.append(np.nanmean([auroc(info["test"]["Y"][:, cj], Sx[:, i]) for i, cj in enumerate(cols)]))
                    rows4.append({"k": k, "classes": ",".join(elig), "model": mk, "source": src, "centered": ce,
                                  "auroc": float(np.mean(vals))})
    R4 = pd.DataFrame(rows4)
    t4 = R4.pivot_table(index=["k", "classes", "source", "centered"], columns="model", values="auroc").reset_index() if len(R4) else R4
    if len(R4) and best in t4 and I in t4:
        t4["gap"] = t4[best] - t4[I]
    text = "\n".join([
        "# MIMIC-IV-ECG replication (frozen protocol; labels are machine-report-derived)\n",
        f"best soft model by test clinical-centered macro AUROC (reported only): {best}\n",
        "## Macro test AUROC by readout (InfoNCE vs best of the 8 soft models)\n", t1.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "\n## Per-concept AUROC, clinical descriptions (primary) and report prototypes\n",
        A[A.model.isin([I, best]) & A.readout.isin(["clinical", "reports"])].pivot_table(
            index=["readout", "centered", "concept"], columns="model", values="auroc").reset_index().to_markdown(index=False, floatfmt=".3f"),
        f"\n## Paired selection ({a.n_boot} resamples; diff vs full labeled validation)\n", ssum.to_markdown(index=False, floatfmt=".3f"),
        "\n## Query source (label-informed; single-concept reports)\n", counts.to_markdown(index=False),
        t4.to_markdown(index=False, floatfmt=".3f") if len(R4) else "(no eligible comparisons)"])
    print(text, flush=True)
    (emb / "mimic_results.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
