"""Step 2 of phase 1: train the projection heads on cached embeddings and evaluate.

Protocol from App. G of the paper:
  * head: Linear(d_in, m) -> LayerNorm -> GELU -> Dropout -> Linear(m, 1024), m = max(3*1024, 768)
    (one head per modality, same design);
  * loss: symmetric InfoNCE on TS-IMG, TS-TXT and IMG-TXT, equal weights, fixed tau = 0.2;
  * AdamW lr 1e-4, wd 1e-5, cosine decay with 2% linear warmup, grad clip 1.0;
  * batch 256, up to 50 epochs, early stopping with patience 5 (on val loss here);
  * metrics on the held-out test set (see metrics.py).
Dropout rate is not given in the paper; 0.1 is used here.

Usage:
    python src/train_heads.py --emb <dir with .npy files> --config A --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import all_metrics  # noqa: E402

D_SHARED = 1024
PAIRS = [("ts", "img"), ("ts", "txt"), ("img", "txt")]
MODS = ("ts", "img", "txt")


class Head(nn.Module):
    def __init__(self, d_in, d=D_SHARED, dropout=0.1):
        super().__init__()
        m = max(3 * d, 768)
        self.net = nn.Sequential(nn.Linear(d_in, m), nn.LayerNorm(m), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(m, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def info_nce(a, b, tau):
    logits = a @ b.T / tau
    y = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def total_loss(z, tau, pairs=PAIRS):
    return sum(info_nce(z[x], z[y], tau) for x, y in pairs)


def load_emb(emb_dir: Path, config: dict, lead: str):
    data = {}
    for split in ("train", "val", "test"):
        data[split] = {}
        for mod, name in config.items():
            f = emb_dir / (f"{split}_{mod}_{name}.npy" if mod == "txt"
                           else f"{split}_{mod}_{name}_lead{lead}.npy")
            data[split][mod] = torch.from_numpy(np.load(f)).float()
        n = {len(v) for v in data[split].values()}
        assert len(n) == 1, f"{split}: modality lengths differ {n}"
    return data


@torch.no_grad()
def embed(heads, X, device, bs=2048):
    for h in heads.values():
        h.eval()
    return {m: torch.cat([heads[m](X[m][i:i + bs].to(device)) for i in range(0, len(X[m]), bs)])
            for m in MODS}


def evaluate(z):
    z = {m: v.cpu().numpy() for m, v in z.items()}
    return {f"{x}-{y}": all_metrics(z[x], z[y]) for x, y in PAIRS}


def train_one(data, seed, device, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    heads = nn.ModuleDict({m: Head(data["train"][m].shape[1], dropout=args.dropout)
                           for m in MODS}).to(device)
    if args.epochs == 0:  # untrained-heads floor
        return heads, {"epochs_run": 0}
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.wd)
    n = len(data["train"]["ts"])
    steps_per_epoch = n // args.bs  # drop_last: every batch has 255 negatives
    assert steps_per_epoch > 0, f"batch size {args.bs} > {n} training samples"
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.02 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    tr = {m: data["train"][m].to(device) for m in MODS}
    best, best_state, bad, log = float("inf"), None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(args.epochs):
        heads.train()
        perm = torch.randperm(n, generator=g).to(device)
        run = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.bs:(i + 1) * args.bs]
            z = {m: heads[m](tr[m][idx]) for m in MODS}
            loss = total_loss(z, args.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        # val loss in batches of the training batch size, so it is on the same scale
        zv = embed(heads, data["val"], device)
        nv = len(zv["ts"])
        vl = np.mean([total_loss({m: zv[m][i:i + args.bs] for m in MODS}, args.tau).item()
                      for i in range(0, nv - args.bs + 1, args.bs)] or
                     [total_loss(zv, args.tau).item()])
        log.append({"epoch": ep + 1, "train_loss": run / steps_per_epoch, "val_loss": float(vl)})
        print(f"  seed {seed} ep {ep + 1:02d} train {run / steps_per_epoch:.4f} val {vl:.4f}",
              flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in heads.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    heads.load_state_dict(best_state)
    return heads, {"epochs_run": len(log), "best_val_loss": float(best), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--config", default="A")
    ap.add_argument("--lead", default="I")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out", default=None, help="results json (default: <emb>/results_<config>.json)")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    from extract import CONFIGS
    if torch.cuda.is_available():
        device = "cuda"
    elif args.allow_cpu:
        device = "cpu"
    else:
        sys.exit("CUDA not available (pass --allow-cpu to train the heads on CPU; "
                 "it is slow but works because the encoders are cached).")
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    emb = Path(args.emb)
    data = load_emb(emb, CONFIGS[args.config], args.lead)
    print({s: len(data[s]["ts"]) for s in data})
    runs = []
    for seed in args.seeds:
        t0 = time.time()
        heads, info = train_one(data, seed, device, args)
        if seed == args.seeds[0]:
            assert next(heads.parameters()).device.type == device
        res = evaluate(embed(heads, data["test"], device))
        runs.append({"seed": seed, **info, "test": res, "seconds": time.time() - t0})
        print(f"  seed {seed}: " + "  ".join(
            f"{p} margin {r['cos_margin']:.3f} proc {r['procrustes']:.3f} "
            f"cka {r['cka_rbf']:.3f} mknn {r['mutual_knn@5']:.3f} R@1 {r['R@1']:.4f}"
            for p, r in res.items()), flush=True)
    out = Path(args.out) if args.out else emb / f"results_{args.config}{'_untrained' if args.epochs == 0 else ''}.json"
    json.dump({"config": args.config, "encoders": CONFIGS[args.config], "lead": args.lead,
               "device": device, "args": vars(args), "runs": runs}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
