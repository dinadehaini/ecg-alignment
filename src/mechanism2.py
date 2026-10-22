"""Follow-up mechanism checks on the saved tau-0.2 heads (exploratory; no training).

1. Query source with matched diagnosis and prototype size k (20 random draws): single-class reports from
   TRAIN patients; reports from held-out VALIDATION patients whose exact string also occurs in train;
   validation reports with a string never seen in train. Scored on TEST signals, so no query contains the
   report of a scored ECG. Uncorrected and text-centered (train mean).
2. Centering accounting. For a unit class vector c and the unit shared direction m = unit(train text mean),
   removing the m-component gives the score z.c - (c.m)(z.m) (constant positive rescaling per class does
   not change AUROC). Compared with the actual pipeline (center each report embedding, normalize, average).
3. Probe agreement: across held-out test ECGs, Pearson correlation between each query score z.c and the
   linear probe's logit for that class (probe on unit-normalized train signal embeddings).
4. Secondary: each hand-written query's cosine distance to its nearest training report in the ORIGINAL
   frozen text-encoder space (neither trained head defines it).
Usage (GPU): python src/mechanism2.py --emb results/emb --ptbxl /content/ptbxl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cleanup import unit  # noqa: E402
from label_eval import CLASSES, multihot, superclasses  # noqa: E402
from mechanism import auroc, probe_dirs  # noqa: E402
from metrics_v2 import text_groups  # noqa: E402
from sweep import small_grid  # noqa: E402
from train_heads import Head, load_emb  # noqa: E402
from train_heads_v2 import CONFIG_A  # noqa: E402

KS = (5, 10, 25)
N_DRAWS = 20
UNSUP_KEYS = ("infonce", "soft")


def norm_str(r):
    return " ".join(str(r).lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--lead", default="I")
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
    hdir = emb / ("heads_quick" if a.limit else "heads")
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
        y = multihot(sets)
        if a.limit:
            keep = np.ones(len(sets), bool)
            y = np.random.default_rng(1).integers(0, 2, (len(sets), len(CLASSES)))
            y[0], y[1] = 0, 1
            sets = [frozenset([CLASSES[i % 5]]) for i in range(len(sets))]
        info[s] = {"sets": sets, "keep": keep, "y": y, "rep": [norm_str(r) for r in rows.report]}
    train_strings = set(info["train"]["rep"])

    # query pools per source and class (single-class reports)
    def pool(s, cond):
        return {c: np.array([i for i, (st, r) in enumerate(zip(info[s]["sets"], info[s]["rep"]))
                             if st == frozenset([c]) and cond(r)], int) for c in CLASSES}
    pools = {"train": ("train", pool("train", lambda r: True)),
             "val, string seen in train": ("val", pool("val", lambda r: r in train_strings)),
             "val, unseen string": ("val", pool("val", lambda r: r not in train_strings))}
    counts = [{"source": k, "count": "records", **{c: int(len(v[1][c])) for c in CLASSES}} for k, v in pools.items()]
    counts += [{"source": k, "count": "unique strings",
                **{c: len({info[v[0]]["rep"][i] for i in v[1][c]}) for c in CLASSES}} for k, v in pools.items()]
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    rng = np.random.default_rng(0)
    # a class enters the size-k comparison only if EVERY source has >= k reports for it (no smaller k, no
    # sampling with replacement); otherwise that (k, class) comparison is marked skipped
    eligible = {k: [c for c in CLASSES if all(len(p[c]) >= k for _, p in pools.values())] for k in KS}
    skipped = [{"k": k, "cls": c, "status": "included" if c in eligible[k] else "skipped (a source has < k reports)"}
               for k in KS for c in CLASSES]
    draws = {(src, k): [{c: rng.choice(p[c], size=k, replace=False) for c in eligible[k]} for _ in range(N_DRAWS)]
             for src, (_, p) in pools.items() for k in KS}

    # hand-written queries, raw frozen embeddings
    here = Path(__file__).resolve().parent
    qv = json.load(open(here / "query_variants.json", encoding="utf-8"))
    pj = json.load(open(here / "prompts_superclass.json", encoding="utf-8"))
    fam_texts = {"clinical": {c: pj["en"][c] + pj["de"][c] for c in CLASSES},
                 "names": {c: qv["names"][c] for c in CLASSES},
                 "report_style": {c: qv["report_style"][c] for c in CLASSES}}
    flat = [(f, c, t) for f, d in fam_texts.items() for c in CLASSES for t in d[c]]
    if a.limit:
        raw_q = torch.randn(len(flat), data["train"]["txt"].shape[1])
    else:
        from extract import Qwen3EmbEnc
        enc = Qwen3EmbEnc(device)
        raw_q = enc([t for *_, t in flat]).float().cpu()
        del enc
        torch.cuda.empty_cache()
    # 4. frozen-space distance to the nearest training report (unique strings)
    tr_raw = unit(data["train"]["txt"].numpy())
    uniq = np.unique(text_groups(info["train"]["rep"]), return_index=True)[1]
    qn = unit(raw_q.numpy())
    nn_cos = (qn @ tr_raw[uniq].T).max(1)
    dist = pd.DataFrame([{"family": f, "cls": c, "text": t, "nearest_train_cos": float(v)}
                         for (f, c, t), v in zip(flat, nn_cos)])

    kt, yt = info["test"]["keep"], info["test"]["y"]
    r1, r2, r3 = [], [], []
    for kind, mix, st, tau in small_grid():
        if not kind.startswith(UNSUP_KEYS):
            continue
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
            ztxt = {s: heads["txt"].net(data[s]["txt"]).numpy() for s in ("train", "val")}
            ztr_ts = unit(heads["ts"].net(data["train"]["ts"]).numpy())
            zte = unit(heads["ts"].net(data["test"]["ts"]).numpy())
            zq = heads["txt"].net(raw_q).numpy()
        mu_raw = ztxt["train"].mean(0)
        m_hat = unit(mu_raw)
        zm = zte @ m_hat
        mu_unit = unit(ztxt["train"]).mean(0)  # mean of unit text vectors: same space as the prototypes

        def cvec(Z, centered):
            Z = Z - mu_raw if centered else Z
            return unit(unit(Z).mean(0))

        # 1. query source
        for (src, k), dl in draws.items():
            s_split = pools[src][0]
            for centered in (False, True):
                for di, dsel in enumerate(dl):
                    for j, c in enumerate(CLASSES):
                        if c not in dsel:
                            continue
                        v = cvec(ztxt[s_split][dsel[c]], centered)
                        r1.append({"model": key, "source": src, "k": k, "centered": centered, "draw": di,
                                   "cls": c, "auroc": auroc(yt[kt, j], (zte @ v)[kt])})
        # 2 + 3. accounting and probe agreement for the main query families
        w = probe_dirs(ztr_ts[info["train"]["keep"]], info["train"]["y"][info["train"]["keep"]], device)
        logits = zte @ w.T
        famZ = {"reports (all train)": {c: ztxt["train"][pools["train"][1][c]] for c in CLASSES}}
        for fam in fam_texts:
            famZ[fam] = {c: zq[[i for i, (ff, cc, _) in enumerate(flat) if ff == fam and cc == c]] for c in CLASSES}
        for fam, Zc in famZ.items():
            for j, c in enumerate(CLASSES):
                if len(Zc[c]) == 0:
                    continue
                cu = cvec(Zc[c], False)
                cc_ = cvec(Zc[c], True)
                t_c = unit(Zc[c]).mean(0)  # un-normalized prototype (mean of unit vectors)
                s_unc, s_pipe = zte @ cu, zte @ cc_
                # direct text-mean subtraction: z.(t_c - mu) = z.t_c - z.mu; the per-class norm of
                # (t_c - mu) is constant across patients, so the patient-level term is exactly z.mu
                s_direct = zte @ t_c - zte @ mu_unit
                s_proj = s_unc - (cu @ m_hat) * zm  # projection removal (has a class-specific loading)
                r2.append({"model": key, "family": fam, "cls": c, "loading on m": float(cu @ m_hat),
                           "auroc uncorrected": auroc(yt[kt, j], s_unc[kt]),
                           "auroc direct mean subtraction": auroc(yt[kt, j], s_direct[kt]),
                           "auroc projection removal": auroc(yt[kt, j], s_proj[kt]),
                           "auroc pipeline (center, normalize, average)": auroc(yt[kt, j], s_pipe[kt]),
                           "corr(direct, pipeline)": float(np.corrcoef(s_direct, s_pipe)[0, 1])})
                for lab, sc_ in (("uncorrected", s_unc), ("centered", s_pipe)):
                    r3.append({"model": key, "family": fam, "cls": c, "readout": lab,
                               "corr with probe logit": float(np.corrcoef(sc_[kt], logits[kt, j])[0, 1])})
        print(f"{key} done", flush=True)

    D1, D2, D3 = pd.DataFrame(r1), pd.DataFrame(r2), pd.DataFrame(r3)
    for n, d in (("1", D1), ("2", D2), ("3", D3)):
        d.to_csv(emb / f"mechanism2_{n}_lead{a.lead}.csv", index=False)
    dist.to_csv(emb / f"mechanism2_dist_lead{a.lead}.csv", index=False)
    I, B = "infonce_mix0.00_st-1.00", "soft_mix1.00_st0.20"

    def gap(df, by):
        t = df[df.model.isin([I, B])].groupby(by + ["model"]).auroc.mean().unstack("model")
        t["gap"] = t[B] - t[I]
        return t.rename(columns={I: "InfoNCE", B: "best soft"}).reset_index()

    # macro over classes available in every source at that k
    if len(D1):
        m1 = D1.groupby(["model", "source", "k", "centered", "draw"]).auroc.mean().reset_index()
        t1 = gap(m1, ["source", "k", "centered"])
    else:
        t1 = pd.DataFrame({"note": ["no source had >= k reports for any class"]})
    t2 = D2[D2.model.isin([I, B])].groupby(["family", "model"])[
        ["loading on m", "auroc uncorrected", "auroc direct mean subtraction", "auroc projection removal",
         "auroc pipeline (center, normalize, average)", "corr(direct, pipeline)"]].mean().reset_index()
    t3 = D3[D3.model.isin([I, B])].groupby(["family", "readout", "model"])["corr with probe logit"].mean().unstack("model").reset_index()
    t4 = dist.groupby("family").nearest_train_cos.agg(["mean", "min", "max"]).reset_index()
    text = "\n".join([
        f"# Mechanism follow-up (lead {a.lead}; exploratory; InfoNCE vs {B})\n",
        "Label-informed analyses (sections 1-2 use report prototypes selected by diagnosis); fixed-prompt\n"
        "results are in sections 2-3 under the hand-written families and are reported separately.\n",
        "Single-class report counts per source (records and unique strings):\n", pd.DataFrame(counts).to_markdown(index=False),
        "\nSize-k comparisons included / skipped per class:\n", pd.DataFrame(skipped).to_markdown(index=False),
        "\n## 1. Query source, matched k (macro AUROC over the classes included at that k in ALL sources; 20 draws without replacement)\n",
        t1.to_markdown(index=False, floatfmt=".3f"),
        "\n## 2. Centering accounting (mean over classes)\n", t2.to_markdown(index=False, floatfmt=".3f"),
        "\n## 3. Correlation of query scores with probe logits on test ECGs (mean over classes)\n",
        t3.to_markdown(index=False, floatfmt=".3f"),
        "\n## 4. Hand-written queries: cosine to the nearest training report, frozen encoder space\n",
        t4.to_markdown(index=False, floatfmt=".3f")])
    print(text, flush=True)
    (emb / f"mechanism2_lead{a.lead}.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
