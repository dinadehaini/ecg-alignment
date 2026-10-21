"""CPU smoke test: metric sanity checks + the full pipeline on a handful of PTB-XL records.

Checks that the code runs end to end; the numbers it produces are meaningless.
    python tests/smoke_test.py --work <scratch dir> [--n 10]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import all_metrics, l2n  # noqa: E402


def test_metrics():
    rng = np.random.default_rng(0)
    X = l2n(rng.normal(size=(300, 64)))
    Q, _ = np.linalg.qr(rng.normal(size=(64, 64)))
    if np.linalg.det(Q) < 0:  # paper's Procrustes forbids reflections, so use a proper rotation
        Q[:, 0] *= -1
    same = all_metrics(X, X @ Q)  # a rotated copy is perfectly aligned
    assert same["procrustes"] < 1e-8, same["procrustes"]
    # Procrustes, CKA and mutual kNN are rotation-invariant; retrieval and cosine are not.
    assert same["mutual_knn@5"] > 0.99 and same["cka_rbf"] > 0.999, same
    ident = all_metrics(X, X.copy())
    assert ident["R@1"] == 1.0 and ident["procrustes"] < 1e-8, ident
    rand = all_metrics(X, l2n(rng.normal(size=(300, 64))))
    assert rand["R@10"] < 0.15 and rand["mutual_knn@5"] < 0.1 and rand["procrustes"] > 0.8
    assert abs(rand["cos_margin"]) < 0.05
    print("metrics ok: aligned", {k: round(same[k], 3) for k in ("cka_rbf", "procrustes")},
          "random", {k: round(rand[k], 3) for k in ("cka_rbf", "procrustes", "R@10")})


def run(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--config", default="A")
    args = ap.parse_args()
    test_metrics()

    work = Path(args.work)
    data, emb = work / "ptbxl", work / "emb"
    run("src/download_ptbxl.py", "--out", data, "--only", "0")  # metadata only
    from data import load_metadata, make_splits
    sp = make_splits(load_metadata(data))
    print("split sizes", sp.split.value_counts().to_dict(),
          "patients shared across splits:",
          len(set(sp[sp.split == "train"].patient_id) & set(sp[sp.split == "test"].patient_id)))
    ids = sp.groupby("split", group_keys=False).head(args.n).index.tolist()
    run("src/download_ptbxl.py", "--out", data, "--only", *ids)
    enc = ["--encoders", *args.encoders] if args.encoders else []
    run("src/extract.py", "--ptbxl", data, "--out", emb, "--config", args.config,
        "--limit", args.n, "--bs", 4, "--allow-cpu", *enc)
    run("src/train_heads.py", "--emb", emb, "--config", args.config, "--seeds", 0, 1,
        "--epochs", 3, "--bs", 4, "--allow-cpu")
    run("src/train_heads.py", "--emb", emb, "--config", args.config, "--seeds", 0, 1,
        "--epochs", 0, "--allow-cpu")
    run("src/compare.py", "--emb", emb, "--out", work / "phase1_results.md")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
