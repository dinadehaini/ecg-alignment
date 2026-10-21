"""Label-level evaluation of the aligned space (beyond the paper's instance-level metrics).

Half of PTB-XL test reports are exact duplicates, so exact-match TS<->TXT retrieval is partly
ill-posed. Here we ask whether the aligned space carries *diagnostic* meaning, using PTB-XL's 5
diagnostic superclasses (NORM, MI, STTC, CD, HYP; aggregated from scp_codes as in Strodthoff et al.):

  * label retrieval: for each test ECG (TS or IMG query), the share of its top-10 retrieved
    reports whose record has the same superclass set, vs. the chance share;
  * zero-shot classification: class prototypes = mean projected text embedding of train reports
    with that single superclass; score = cosine(query, prototype); macro AUROC on test;
  * reference: a supervised logistic-regression probe on the frozen TS encoder features.

Heads are retrained here (same protocol/seeds as train_heads.py) because train_heads.py does not
save weights; the untrained-head floor is included.

Usage:
    python src/label_eval.py --emb results/emb --ptbxl /content/ptbxl --config A --lead I
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_metadata  # noqa: E402
from train_heads import embed, load_emb, train_one  # noqa: E402

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def superclasses(ptbxl_dir) -> pd.Series:
    meta = load_metadata(ptbxl_dir)
    st = pd.read_csv(Path(ptbxl_dir) / "scp_statements.csv", index_col=0)
    st = st[st.diagnostic == 1]
    return meta.scp_codes.apply(
        lambda d: frozenset(st.loc[k, "diagnostic_class"] for k in d if k in st.index))


def multihot(sets) -> np.ndarray:
    return np.array([[c in s for c in CLASSES] for s in sets], dtype=int)


def label_retrieval(q: np.ndarray, g: np.ndarray, sets, k=10) -> tuple[float, float]:
    """Mean share of top-k gallery items with the query's exact superclass set, and chance."""
    sets = list(sets)
    ids = {s: i for i, s in enumerate(set(sets))}
    lab = np.array([ids[s] for s in sets])
    topk = np.argsort(-(q @ g.T), axis=1)[:, :k]
    hit = (lab[topk] == lab[:, None]).mean()
    counts = np.bincount(lab)
    chance = (counts[lab] / len(lab)).mean()
    return float(hit), float(chance)


def zero_shot_auroc(q: np.ndarray, txt_train: np.ndarray, sets_train, y_test: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j, c in enumerate(CLASSES):
        m = np.array([s == frozenset([c]) for s in sets_train])
        if m.sum() == 0 or len(set(y_test[:, j])) < 2:  # only happens on tiny smoke-test data
            continue
        p = txt_train[m].mean(0)
        aucs.append(roc_auc_score(y_test[:, j], q @ (p / np.linalg.norm(p))))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--config", default="A")
    ap.add_argument("--lead", default="I")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    from extract import CONFIGS

    device = "cuda" if torch.cuda.is_available() else ("cpu" if a.allow_cpu else None)
    if device is None:
        sys.exit("CUDA not available (pass --allow-cpu to run on CPU)")
    emb = Path(a.emb)
    split = pd.read_csv(emb / "split.csv", index_col=0)
    sc = superclasses(a.ptbxl)
    sets = {s: sc.loc[split.index[split.split == s]].tolist() for s in ("train", "test")}
    keep = np.array([len(s) > 0 for s in sets["test"]])  # records with >= 1 diagnostic superclass
    y_test = multihot(sets["test"])[keep]
    data = load_emb(emb, CONFIGS[a.config], a.lead)
    assert len(data["test"]["ts"]) == len(sets["test"])
    targs = SimpleNamespace(epochs=a.epochs, bs=min(a.bs, len(data["train"]["ts"])), lr=1e-4,
                            wd=1e-5, tau=0.2, dropout=0.1, patience=5)

    out = {"config": a.config, "lead": a.lead, "n_test_labeled": int(keep.sum()),
           "classes": CLASSES, "runs": {}}
    # supervised reference on frozen TS features
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    ktr = np.array([len(s) > 0 for s in sets["train"]])
    Xtr = data["train"]["ts"].numpy()[ktr]
    Xte = data["test"]["ts"].numpy()[keep]
    ytr = multihot(sets["train"])[ktr]
    sc_ = StandardScaler().fit(Xtr)
    aucs = []
    for j in range(len(CLASSES)):
        if len(set(ytr[:, j])) < 2 or len(set(y_test[:, j])) < 2:  # tiny smoke-test data only
            continue
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc_.transform(Xtr), ytr[:, j])
        aucs.append(roc_auc_score(y_test[:, j], clf.predict_proba(sc_.transform(Xte))[:, 1]))
    out["supervised_probe_frozen_ts_auroc"] = float(np.mean(aucs))
    print(f"supervised linear probe on frozen TS features: macro AUROC {np.mean(aucs):.3f}")

    for name, epochs in (("trained", a.epochs), ("untrained", 0)):
        rows = []
        for seed in a.seeds:
            targs.epochs = epochs
            heads, _ = train_one(data, seed, device, targs)
            zte = {m: v.cpu().numpy() for m, v in embed(heads, data["test"], device).items()}
            ztr_txt = embed(heads, data["train"], device)["txt"].cpu().numpy()
            r = {}
            for qm in ("ts", "img"):
                hit, chance = label_retrieval(zte[qm][keep], zte["txt"][keep],
                                              [s for s, k in zip(sets["test"], keep) if k])
                r[f"{qm}2txt_label_p@10"] = hit
                r["label_p@10_chance"] = chance
                r[f"{qm}_zeroshot_auroc"] = zero_shot_auroc(zte[qm][keep], ztr_txt,
                                                            sets["train"], y_test)
            rows.append(r)
            print(f"  {name} seed {seed}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()),
                  flush=True)
        out["runs"][name] = rows
        print(f"{name} mean: " + "  ".join(
            f"{k} {np.mean([x[k] for x in rows]):.3f}±{np.std([x[k] for x in rows], ddof=1) if len(rows) > 1 else 0:.3f}"
            for k in rows[0]))
    path = Path(a.out) if a.out else emb / f"label_eval_{a.config}_lead{a.lead}.json"
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
