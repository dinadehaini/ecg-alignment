"""Step A: does a contrastive objective that stops fighting duplicate/similar reports change what
the aligned space encodes? Trains the paper's heads on cached (frozen) embeddings with:

  infonce : the paper's symmetric InfoNCE (one positive per row)                 [baseline]
  dupmask : identical reports (same normalized text) are all positives for text pairs
  soft    : soft targets from frozen text-embedding similarity (FG-CLEP-style), text pairs
  label   : same diagnostic-superclass set = positives (uses labels: an ORACLE upper bound,
            not a fair alignment method; included to show the ceiling)

Targets are a distribution T over the batch (rows sum to 1); loss = cross-entropy(T, softmax(S/tau)),
both directions, for TS-TXT and IMG-TXT. TS-IMG keeps one positive per row in every variant (plots
are unique to their signal). Everything else (heads, optimizer, schedule, tau = 0.2, batch 256,
early stopping) matches train_heads.py.

Evaluation: paper metrics + step-B metrics (duplicate-aware retrieval, symmetric Procrustes,
deduplicated subset) + label metrics (zero-shot AUROC, same-diagnosis P@10).

Usage:
    python src/train_heads_v2.py --emb <emb dir> --ptbxl <dir with ptbxl_database.csv,
        scp_statements.csv> --loss infonce dupmask soft label --seeds 0 1 2 [--allow-cpu]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_eval import label_retrieval, multihot, superclasses, zero_shot_auroc  # noqa: E402
from metrics_v2 import all_metrics_v2, text_groups  # noqa: E402
from train_heads import Head, load_emb  # noqa: E402

MODS = ("ts", "img", "txt")
CONFIG_A = {"ts": "moment-base", "img": "dinov2-base", "txt": "qwen3-emb-0.6b"}


def soft_ce(logits, T):
    return -(T * F.log_softmax(logits, dim=1)).sum(1).mean()


def pair_loss(a, b, T, tau):
    """Symmetric soft-target contrastive loss; T is (B, B) with rows summing to 1."""
    lg = a @ b.T / tau
    Tt = T.T / T.T.sum(1, keepdim=True)
    return 0.5 * (soft_ce(lg, T) + soft_ce(lg.T, Tt))


def batch_targets(kind, idx, aux, device):
    """Target matrix for text pairs in this batch, optionally mixed with the identity:
    T = (1 - mix) * I + mix * T_kind  (aux["mix"], default 1.0)."""
    T = _targets(kind, idx, aux, device)
    m = aux.get("mix", 1.0)
    if kind != "infonce" and m < 1.0:
        T = (1.0 - m) * torch.eye(len(idx), device=device) + m * T
    return T


def _targets(kind, idx, aux, device):
    B = len(idx)
    if kind == "infonce":
        return torch.eye(B, device=device)
    if kind in ("dupmask", "label"):
        g = aux["text_group" if kind == "dupmask" else "label_group"][idx]
        P = (g[:, None] == g[None, :]).float()
        return (P / P.sum(1, keepdim=True)).to(device)
    if kind == "soft":
        t = aux["txt_raw_n"][idx]
        return F.softmax(t @ t.T / aux["soft_tau"], dim=1)
    raise ValueError(kind)


@torch.no_grad()
def embed_all(heads, X, device, bs=2048):
    for h in heads.values():
        h.eval()
    return {m: torch.cat([heads[m](X[m][i:i + bs].to(device)) for i in range(0, len(X[m]), bs)])
            for m in MODS}


def train(kind, seed, data, aux, a, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    heads = nn.ModuleDict({m: Head(data["train"][m].shape[1]) for m in MODS}).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=a.lr, weight_decay=a.wd)
    tr = {m: data["train"][m].to(device) for m in MODS}
    n = len(tr["ts"])
    spe = n // a.bs
    total = spe * a.epochs
    warm = max(1, int(0.02 * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                            0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, log = float("inf"), None, 0, []
    eye = None
    for ep in range(a.epochs):
        for h in heads.values():
            h.train()
        perm = torch.randperm(n, generator=g)
        run = 0.0
        for i in range(spe):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            idd = idx.to(device)
            z = {m: heads[m](tr[m][idd]) for m in MODS}
            if eye is None or eye.shape[0] != len(idx):
                eye = torch.eye(len(idx), device=device)
            T = batch_targets(kind, idx, aux["train"], device)
            loss = (pair_loss(z["ts"], z["img"], eye, a.tau) + pair_loss(z["ts"], z["txt"], T, a.tau)
                    + pair_loss(z["img"], z["txt"], T, a.tau))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            opt.step()
            sch.step()
            run += loss.item()
        # early stopping on the *standard* InfoNCE val loss, identical for every variant
        zv = embed_all(heads, data["val"], device)
        nv = len(zv["ts"])
        vls = []
        for j in range(0, nv - a.bs + 1, a.bs):
            e = torch.eye(a.bs, device=device)
            s = {m: zv[m][j:j + a.bs] for m in MODS}
            vls.append((pair_loss(s["ts"], s["img"], e, a.tau) + pair_loss(s["ts"], s["txt"], e, a.tau)
                        + pair_loss(s["img"], s["txt"], e, a.tau)).item())
        vl = float(np.mean(vls))
        log.append({"epoch": ep + 1, "train_loss": run / spe, "val_infonce": vl})
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in heads.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    heads.load_state_dict(best_state)
    return heads, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--loss", nargs="+", default=["infonce", "dupmask", "soft", "label"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--soft-tau", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[cuda ok] {torch.cuda.get_device_name(0)}")
    elif a.allow_cpu:
        device = "cpu"
        print("running on CPU (heads only; fine for cached embeddings)")
    else:
        sys.exit("CUDA not available (pass --allow-cpu: this script only trains small heads)")

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
        assert len(rows) == len(data[s]["ts"]), (s, len(rows), len(data[s]["ts"]))
        sets = sc.loc[rows.index].tolist()
        lab_ids = pd.Series([",".join(sorted(x)) for x in sets]).astype("category").cat.codes.values
        txt = data[s]["txt"]
        aux[s] = {"text_group": torch.as_tensor(text_groups(rows.report.tolist())),
                  "label_group": torch.as_tensor(lab_ids),
                  "txt_raw_n": F.normalize(txt, dim=1).to(device),
                  "soft_tau": a.soft_tau, "sets": sets}
    if a.bs > len(data["train"]["ts"]):
        a.bs = len(data["train"]["ts"])
    gt = aux["test"]["text_group"].numpy()
    keep = np.array([len(x) > 0 for x in aux["test"]["sets"]])
    y_test = multihot(aux["test"]["sets"])[keep]
    print(f"test: {len(gt)} reports, {len(np.unique(gt))} unique; {keep.sum()} with a diagnostic label")

    results = {"args": vars(a), "device": device, "runs": []}
    for kind in a.loss:
        for seed in a.seeds:
            t0 = time.time()
            heads, log = train(kind, seed, data, aux, a, device)
            z = {m: v.cpu().numpy() for m, v in embed_all(heads, data["test"], device).items()}
            ztr_txt = embed_all(heads, data["train"], device)["txt"].cpu().numpy()
            r = {"loss": kind, "seed": seed, "epochs_run": len(log), "seconds": time.time() - t0,
                 "ts-img": all_metrics_v2(z["ts"], z["img"]),
                 "ts-txt": all_metrics_v2(z["ts"], z["txt"], gt),
                 "img-txt": all_metrics_v2(z["img"], z["txt"], gt)}
            for qm in ("ts", "img"):
                hit, chance = label_retrieval(z[qm][keep], z["txt"][keep],
                                              [x for x, k in zip(aux["test"]["sets"], keep) if k])
                r[f"{qm}2txt_label_p@10"], r["label_p@10_chance"] = hit, chance
                r[f"{qm}_zeroshot_auroc"] = zero_shot_auroc(z[qm][keep], ztr_txt,
                                                            aux["train"]["sets"], y_test)
            t = r["ts-txt"]
            print(f"{kind:8s} seed {seed} ep {len(log):2d} | TS-TXT margin {t['cos_margin']:.3f} "
                  f"procP {t['procrustes_paper']:.3f} procS {t['procrustes_sym']:.3f} "
                  f"R@1 exact {t.get('R@1_exact', float('nan')):.4f} dup {t.get('R@1_dupaware', float('nan')):.4f} "
                  f"| zs-AUROC {r['ts_zeroshot_auroc']:.3f} labelP@10 {r['ts2txt_label_p@10']:.3f} "
                  f"({r['seconds']:.0f}s)", flush=True)
            results["runs"].append(r)
    out = Path(a.out) if a.out else emb / f"results_v2_lead{a.lead}{'_lim' if a.limit else ''}.json"
    json.dump(results, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
