"""Mechanism analysis: why does text-side centering close the report-prototype gap but not the prompt gap?

On the 13 saved tau-0.2 heads (original reports; exploratory, PTB-XL already examined extensively):
  A. Per-class test AUROC for each query family, uncorrected vs text-centered (train text mean):
       report prototypes (all single-class train reports), prototype-k (k = 1, 10, 100 random reports,
       20 draws), clinical descriptions (frozen 8-prompt set), class names (EN+DE), report-style
       statements (4 per class); family ensembles plus every single query.
  B. Text geometry. mu = unit(train mean of projected report embeddings, pre-normalization). For each
       class vector c (unit, uncorrected): loading a_c = c . mu. The uncorrected score splits as
       s(x,c) = x.c_perp + a_c (x.mu) with x unit; the second term is a patient-level offset shared by
       all classes, weighted by a_c. Reported: a_c per family, the per-class AUROC of the offset x.mu
       alone, and the across-patient correlation between s(x,c) and x.mu.
  C. Readout vs probe. Logistic-regression probe per class on unit-normalized projected signal
       embeddings (train, no standardization, so directions are comparable); cosine between the probe
       weight w_c and each family's class vector, uncorrected and centered.
Usage (GPU for the text encoder): python src/mechanism.py --emb results/emb --ptbxl /content/ptbxl
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
from cleanup import unit  # noqa: E402
from label_eval import CLASSES, multihot, superclasses  # noqa: E402
from sweep import small_grid  # noqa: E402
from train_heads import Head, load_emb  # noqa: E402
from train_heads_v2 import CONFIG_A  # noqa: E402

KS = (1, 10, 100)
N_DRAWS = 20


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def probe_dirs(X, Y, device, steps=500, wd=1e-4):
    """One logistic regression per class on unit-normalized X (no standardization); returns unit weights."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    torch.manual_seed(0)
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(lin(Xt * 10), Yt).backward()
        opt.step()
    return unit(lin.weight.detach().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--heads", default=None)
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
    hdir = Path(a.heads) if a.heads else emb / ("heads_quick" if a.limit else "heads")
    split = pd.read_csv(emb / "split.csv", index_col=0)
    data = load_emb(emb, CONFIG_A, a.lead)
    sc = superclasses(a.ptbxl)
    info = {}
    for s in ("train", "test"):
        rows = split[split.split == s]
        if a.limit:
            rows = rows.head(a.limit)
            data[s] = {m: v[:a.limit] for m, v in data[s].items()}
        sets = sc.loc[rows.index].tolist()
        keep = np.array([len(x) > 0 for x in sets])
        y = multihot(sets)
        if a.limit:  # smoke test only: random stand-in labels
            keep = np.ones(len(sets), bool)
            y = np.random.default_rng(1).integers(0, 2, (len(sets), len(CLASSES)))
            y[0], y[1] = 0, 1
        info[s] = {"sets": sets, "keep": keep, "y": y}

    here = Path(__file__).resolve().parent
    qv = json.load(open(here / "query_variants.json", encoding="utf-8"))
    pj = json.load(open(here / "prompts_superclass.json", encoding="utf-8"))
    fam_texts = {"clinical": {c: pj["en"][c] + pj["de"][c] for c in CLASSES},
                 "names": {c: qv["names"][c] for c in CLASSES},
                 "report_style": {c: qv["report_style"][c] for c in CLASSES}}
    flat = [(f, c, i, t) for f, d in fam_texts.items() for c in CLASSES for i, t in enumerate(d[c])]
    if a.limit:
        raw_q = torch.randn(len(flat), data["train"]["txt"].shape[1])
    else:
        from extract import Qwen3EmbEnc
        enc = Qwen3EmbEnc(device)
        raw_q = enc([t for *_, t in flat]).float().cpu()
        del enc
        torch.cuda.empty_cache()

    rng = np.random.default_rng(0)
    single_idx = {c: np.flatnonzero([s == frozenset([c]) for s in info["train"]["sets"]]) for c in CLASSES}
    draws = {k: [{c: rng.choice(single_idx[c], size=min(k, len(single_idx[c])), replace=False)
                  if len(single_idx[c]) else np.array([], int) for c in CLASSES} for _ in range(N_DRAWS)]
             for k in KS}

    kt, yt = info["test"]["keep"], info["test"]["y"]
    rows_a, rows_b, rows_c = [], [], []
    for kind, mix, st, tau in small_grid():
        key = f"{kind}_mix{mix:.2f}_st{(st if st is not None else -1.0):.2f}"
        f = hdir / f"{key}.pt"
        if not f.exists():
            continue
        sd = torch.load(f, map_location="cpu")
        heads = {}
        for m in ("ts", "txt"):
            h = Head(data["train"][m].shape[1])
            h.load_state_dict({k: v.float() for k, v in sd[m].items()})
            heads[m] = h.eval()
        with torch.no_grad():
            ztr_txt = heads["txt"].net(data["train"]["txt"]).numpy()
            ztr_ts = heads["ts"].net(data["train"]["ts"]).numpy()
            zte_ts = unit(heads["ts"].net(data["test"]["ts"]).numpy())
            zq = heads["txt"].net(raw_q).numpy()
        mu_raw = ztr_txt.mean(0)
        mu = unit(mu_raw)
        offset = zte_ts @ mu  # patient-level term x . mu

        def classvecs(Z_by_class, centered):
            """Z_by_class: dict class -> (n, d) pre-normalization text vectors; mean of unit vectors."""
            out = []
            for c in CLASSES:
                Z = Z_by_class[c]
                if len(Z) == 0:
                    out.append(np.zeros(zq.shape[1]))
                    continue
                Z = Z - mu_raw if centered else Z
                out.append(unit(unit(Z).mean(0)))
            return np.stack(out)

        # query families -> list of (family, variant, {class: vectors})
        fams = []
        fams.append(("reports", "all", {c: ztr_txt[single_idx[c]] for c in CLASSES}))
        for k in KS:
            for di, dsel in enumerate(draws[k]):
                fams.append((f"proto_k{k}", f"draw{di}", {c: ztr_txt[dsel[c]] for c in CLASSES}))
        for fam, d in fam_texts.items():
            idx = {c: [j for j, (ff, cc, _, _) in enumerate(flat) if ff == fam and cc == c] for c in CLASSES}
            fams.append((fam, "ensemble", {c: zq[idx[c]] for c in CLASSES}))
            for i in range(len(d[CLASSES[0]])):
                fams.append((fam, f"q{i}", {c: zq[[idx[c][i]]] for c in CLASSES}))

        pr_w = probe_dirs(unit(ztr_ts)[info["train"]["keep"]], info["train"]["y"][info["train"]["keep"]], device)
        base = {"model": key, "kind": kind}
        for fam, var, Zc in fams:
            for centered in (False, True):
                P = classvecs(Zc, centered)
                S = zte_ts @ P.T
                for j, c in enumerate(CLASSES):
                    rows_a.append(dict(base, family=fam, variant=var, centered=centered, cls=c,
                                       auroc=auroc(yt[kt, j], S[kt, j])))
                if not centered and var in ("all", "ensemble", "draw0"):
                    Pu = classvecs(Zc, False)
                    for j, c in enumerate(CLASSES):
                        rows_b.append(dict(base, family=fam, cls=c, loading=float(Pu[j] @ mu),
                                           corr_score_offset=float(np.corrcoef(S[:, j], offset)[0, 1])))
                if var in ("all", "ensemble", "draw0"):
                    for j, c in enumerate(CLASSES):
                        rows_c.append(dict(base, family=fam, centered=centered, cls=c,
                                           cos_probe=float(P[j] @ pr_w[j])))
        for j, c in enumerate(CLASSES):
            rows_b.append(dict(base, family="offset x.mu alone", cls=c,
                               offset_auroc=auroc(yt[kt, j], offset[kt])))
        print(f"{key} done", flush=True)

    A, B, C = pd.DataFrame(rows_a), pd.DataFrame(rows_b), pd.DataFrame(rows_c)
    for name, d in (("A", A), ("B", B), ("C", C)):
        d.to_csv(emb / f"mechanism_{name}_lead{a.lead}.csv", index=False)
    unsup = ~A.kind.eq("label")
    # family-level macro AUROC: mean over classes; draws averaged; single queries summarized
    fam_m = A[unsup].groupby(["model", "kind", "family", "variant", "centered"]).auroc.mean().reset_index()
    ens = fam_m[fam_m.variant.isin(["all", "ensemble"]) | fam_m.family.str.startswith("proto_k")]
    ens = ens.groupby(["model", "kind", "family", "centered"]).auroc.mean().reset_index()
    best_key = ens[(ens.family == "reports") & ~ens.centered].sort_values("auroc").model.iloc[-1]

    def gap_table(df, by):
        t = df.pivot_table(index=by, columns="model", values="auroc")
        out = pd.DataFrame({"InfoNCE": t["infonce_mix0.00_st-1.00"], "best soft": t[best_key]})
        out["gap"] = out["best soft"] - out["InfoNCE"]
        return out.reset_index()

    t1 = gap_table(ens, ["family", "centered"])
    singles = fam_m[fam_m.variant.str.startswith("q")]
    t1b = gap_table(singles, ["family", "variant", "centered"])
    sv = t1b.groupby(["family", "centered"])[["InfoNCE", "best soft", "gap"]].agg(["mean", "min", "max"])
    sv.columns = [" ".join(c) for c in sv.columns]
    percls = A[unsup & (A.variant.isin(["all", "ensemble"]))]
    t2 = gap_table(percls, ["family", "centered", "cls"])
    Bf = B[B.family != "offset x.mu alone"]
    t3 = Bf[Bf.model.isin(["infonce_mix0.00_st-1.00", best_key])].groupby(["family", "model"])[
        ["loading", "corr_score_offset"]].mean().reset_index()
    t3b = B[(B.family == "offset x.mu alone") & B.model.isin(["infonce_mix0.00_st-1.00", best_key])][
        ["model", "cls", "offset_auroc"]]
    t4 = C[C.model.isin(["infonce_mix0.00_st-1.00", best_key])].groupby(["family", "centered", "model"]).cos_probe.mean().reset_index()
    text = "\n".join([
        f"# Mechanism analysis (lead {a.lead}; exploratory; best soft = {best_key})\n",
        "## A1. Macro test AUROC by query family (ensembles; proto_k averaged over draws)\n",
        t1.to_markdown(index=False, floatfmt=".3f"),
        "\n## A2. Single queries within each family (mean / min / max over queries)\n",
        sv.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "\n## A3. Per-class AUROC (family ensembles)\n",
        t2.to_markdown(index=False, floatfmt=".3f"),
        "\n## B1. Loading of class vectors on the shared text direction mu, and corr(score, x.mu) across patients\n",
        t3.to_markdown(index=False, floatfmt=".3f"),
        "\n## B2. AUROC of the patient-level offset x.mu alone, per class\n",
        t3b.to_markdown(index=False, floatfmt=".3f"),
        "\n## C. Cosine between each family's class vector and the linear-probe direction (mean over classes)\n",
        t4.to_markdown(index=False, floatfmt=".3f")])
    print(text, flush=True)
    (emb / f"mechanism_lead{a.lead}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
