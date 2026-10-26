"""Bounded accounting check of the MIMIC-IV-ECG primary result (post hoc; requested after the frozen run).

InfoNCE vs the best soft model (mix 1, soft-tau 0.2) on the clinical descriptions (primary) and report
prototypes:
1. Per-concept test AUROC: uncorrected, direct text-mean subtraction (z.t_c - z.mu, mu = mean of unit train
   text vectors), and the centering pipeline (center each text, normalize, average, normalize), plus the
   AUROC of the patient-level offset z.mu alone and each class vector's loading on the shared direction.
   Same decomposition as src/mechanism2.py on PTB-XL.
2. Paired uncertainty for the soft - InfoNCE macro gap, all 8 concepts and excluding normal ECG:
   test-patient bootstrap on the seed-0 heads (test-sampling uncertainty), and the same gaps for heads
   retrained with further seeds (training-seed variability; same data, hyperparameters and code).
3. The ECGs excluded by src/mimic_verify.py, by split and by concept prevalence vs the analysed ECGs.
Usage (GPU): python src/mimic_accounting.py --emb <emb_mimic> --labels results/mimic/mimic_subset_labels.csv
             --unavailable results/mimic/mimic_unavailable.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cleanup import unit  # noqa: E402
from metrics_v2 import text_groups  # noqa: E402
from mimic_run import TAU, auroc, cvec, macro  # noqa: E402
from train_heads import Head, load_emb  # noqa: E402
from train_heads_v2 import CONFIG_A, train  # noqa: E402

MODELS = {"infonce_mix0.00_st-1.00": ("infonce", 0.0, None), "soft_mix1.00_st0.20": ("soft", 1.0, 0.2)}
I, B = list(MODELS)


def load_heads(f, data):
    sd = torch.load(f, map_location="cpu")
    heads = {}
    for m in ("ts", "txt"):
        h = Head(data["train"][m].shape[1])
        h.load_state_dict({k: v.float() for k, v in sd[m].items()})
        heads[m] = h.eval()
    return heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--unavailable", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2], help="extra training seeds (seed 0 = saved heads)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available (use a GPU runtime)")
    device = "cuda"
    print(f"[cuda ok] {torch.cuda.get_device_name(0)}", flush=True)
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
        L = lab.loc[rows.index, C]
        Y = np.where(L.values == "positive", 1.0, np.where(L.values == "negative", 0.0, np.nan))
        info[s] = {"Y": Y, "g": text_groups(rows.report.tolist()), "pid": rows.patient_id.values}
    Ytr, Yte = info["train"]["Y"], info["test"]["Y"]
    single = {c: np.flatnonzero((Ytr[:, j] == 1) & (np.nansum(Ytr, 1) == 1)) for j, c in enumerate(C)}
    flat = [(c, t) for c in C for t in pj["clinical"][c]]
    from extract import Qwen3EmbEnc
    enc = Qwen3EmbEnc(device)
    raw_q = enc([t for _, t in flat]).float().cpu()
    del enc
    torch.cuda.empty_cache()
    nn_ = [j for j, c in enumerate(C) if c != "normal_ecg"]

    def scores(heads):
        """test scores per (family, variant): (N_test, 8) arrays; plus per-concept loadings and offset."""
        with torch.no_grad():
            ztr = heads["txt"].net(data["train"]["txt"]).numpy()
            zte = unit(heads["ts"].net(data["test"]["ts"]).numpy())
            zq = heads["txt"].net(raw_q).numpy()
        mu_raw, mu_unit = ztr.mean(0), unit(ztr).mean(0)
        fams = {"clinical": {c: zq[[i for i, (cc, _) in enumerate(flat) if cc == c]] for c in C},
                "reports": {c: ztr[single[c]] for c in C}}
        out, load = {}, {}
        for f, Zc in fams.items():
            cu = np.stack([cvec(Zc[c], mu_raw, False) for c in C])
            cc_ = np.stack([cvec(Zc[c], mu_raw, True) for c in C])
            tc = np.stack([unit(Zc[c]).mean(0) for c in C])  # un-normalized prototype (mean of unit vectors)
            out[(f, "uncorrected")] = zte @ cu.T
            out[(f, "direct z.mu subtraction")] = zte @ tc.T - (zte @ mu_unit)[:, None]
            out[(f, "pipeline")] = zte @ cc_.T
            load[f] = cu @ unit(mu_raw)
        return out, load, zte @ mu_unit

    def gaps(sc, rows=None):
        return {(f, v, sub): macro(Yte[:, cols], sc[B][(f, v)][:, cols], rows) - macro(Yte[:, cols], sc[I][(f, v)][:, cols], rows)
                for (f, v) in sc[B] for sub, cols in (("all 8", list(range(len(C)))), ("excl. normal ECG", nn_))}

    # 1. per-concept decomposition (seed-0 heads)
    sc0, rows1 = {}, []
    for key in MODELS:
        sc0[key], load, off = scores(load_heads(emb / "heads" / f"{key}.pt", data))
        for j, c in enumerate(C):
            rows1.append({"model": key, "family": "offset z.mu alone", "concept": c, "auroc": auroc(Yte[:, j], off)})
            for (f, v), S in sc0[key].items():
                rows1.append({"model": key, "family": f, "variant": v, "concept": c, "auroc": auroc(Yte[:, j], S[:, j]),
                              "loading on mu": float(load[f][j])})
    D1 = pd.DataFrame(rows1)
    t1 = D1[D1.family != "offset z.mu alone"].pivot_table(index=["family", "concept"], columns=["model", "variant"],
                                                         values="auroc").round(3)
    t1l = D1[D1.family != "offset z.mu alone"].groupby(["family", "concept", "model"])["loading on mu"].first().unstack("model").round(3)
    t1o = D1[D1.family == "offset z.mu alone"].pivot_table(index="concept", columns="model", values="auroc").round(3)
    macro_rows = [{"model": k, "family": f, "variant": v, "concepts": sub,
                   "macro auroc": macro(Yte[:, cols], sc0[k][(f, v)][:, cols])}
                  for k in MODELS for (f, v) in sc0[k] for sub, cols in (("all 8", list(range(len(C)))), ("excl. normal ECG", nn_))]
    t1m = pd.DataFrame(macro_rows).pivot_table(index=["family", "concepts", "variant"], columns="model", values="macro auroc")
    t1m["gap"] = t1m[B] - t1m[I]

    # 2a. test-patient bootstrap, seed-0 heads
    rng = np.random.default_rng(0)
    tp = info["test"]["pid"]
    trows = {p: np.flatnonzero(tp == p) for p in np.unique(tp)}
    point = gaps(sc0)
    boots = []
    for b in range(a.n_boot):
        rr = np.concatenate([trows[p] for p in rng.choice(list(trows), size=len(trows), replace=True)])
        boots.append(gaps(sc0, rr))
        if b % 200 == 0:
            print(f"boot {b}/{a.n_boot}", flush=True)
    t2 = pd.DataFrame([{"family": f, "variant": v, "concepts": sub, "gap": point[(f, v, sub)],
                        "lo": np.nanpercentile([x[(f, v, sub)] for x in boots], 2.5),
                        "hi": np.nanpercentile([x[(f, v, sub)] for x in boots], 97.5),
                        "P(gap<=0)": float(np.mean([x[(f, v, sub)] <= 0 for x in boots]))} for (f, v, sub) in point])

    # 2b. training-seed variability: retrain both models with further seeds (heads cached per seed)
    aux = {s: {"text_group": torch.as_tensor(info[s]["g"]), "label_group": torch.zeros(len(info[s]["g"]), dtype=torch.long),
               "txt_raw_n": F.normalize(data[s]["txt"], dim=1).to(device), "sets": [frozenset()] * len(info[s]["g"])}
           for s in info}
    seed_rows = [{"seed": 0, **{f"{f} | {v} | {sub}": g for (f, v, sub), g in point.items()}}]
    for seed in a.seeds:
        hdir = emb / f"heads_seed{seed}"
        hdir.mkdir(exist_ok=True)
        scs = {}
        for key, (kind, mix, st) in MODELS.items():
            f = hdir / f"{key}.pt"
            if not f.exists():
                for s in aux:
                    aux[s]["mix"], aux[s]["soft_tau"] = mix, (st if st is not None else 0.05)
                targs = SimpleNamespace(epochs=a.epochs, bs=256, lr=1e-4, wd=1e-5, tau=TAU, patience=5)
                heads, log = train(kind, seed, data, aux, targs, device)
                torch.save({m: {k: v.half().cpu() for k, v in heads[m].state_dict().items()} for m in ("ts", "txt")}, f)
                print(f"seed {seed} {key}: {len(log)} epochs", flush=True)
            scs[key] = scores(load_heads(f, data))[0]
        seed_rows.append({"seed": seed, **{f"{f} | {v} | {sub}": g for (f, v, sub), g in gaps(scs).items()}})
    t3 = pd.DataFrame(seed_rows).set_index("seed").T
    t3["mean"], t3["min"], t3["max"] = t3.mean(1), t3.min(1), t3.max(1)

    # 3. excluded ECGs (non-finite samples etc.)
    full = pd.read_csv(a.labels)
    bad = pd.read_csv(a.unavailable)
    full["excluded"] = full.study_id.isin(bad.study_id)
    t4a = pd.crosstab(full.split, full.excluded, margins=True)
    t4a.columns = [{False: "analysed", True: "excluded"}.get(c, c) for c in t4a.columns]
    reasons = bad.status.value_counts().to_frame("ECGs")
    prev = []
    for grp, d in full.groupby("excluded"):
        r = {"group": "excluded" if grp else "analysed", "ECGs": len(d)}
        for c in C:
            r[c] = f"{(d[c] == 'positive').sum() / max(1, d[c].isin(['positive', 'negative']).sum()):.3f}"
        prev.append(r)
    t4b = pd.DataFrame(prev)

    D1.to_csv(emb / "mimic_accounting_concepts.csv", index=False)
    pd.DataFrame(boots).to_csv(emb / "mimic_accounting_boot.csv", index=False)
    fmt = lambda d, **k: d.to_markdown(floatfmt=".3f", **k)
    text = "\n".join([
        "# MIMIC-IV-ECG accounting check (post hoc; InfoNCE vs soft mix 1 / soft-tau 0.2)\n",
        "## Macro test AUROC by correction variant\n", fmt(t1m.reset_index(), index=False),
        "\n## Per-concept test AUROC (seed-0 heads)\n", fmt(t1),
        "\n## Loading of each uncorrected class vector on the shared text direction unit(mu)\n", fmt(t1l),
        "\n## AUROC of the patient-level offset z.mu alone, per concept\n", fmt(t1o),
        f"\n## Gap (soft - InfoNCE), test-patient bootstrap ({a.n_boot} resamples, seed-0 heads)\n", fmt(t2, index=False),
        "\n## Gap by training seed (full test set; seed 0 = frozen run's heads)\n", fmt(t3),
        "\n## Excluded ECGs by split\n", fmt(t4a), "\n", fmt(reasons),
        "\n## Concept prevalence (positive / eligible), excluded vs analysed ECGs\n", t4b.to_markdown(index=False)])
    print(text, flush=True)
    (emb / "mimic_accounting.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
