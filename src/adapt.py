"""Phase-2 pilot: adapt the TS encoder vs. a deeper head, at a matched trainable-parameter budget.

Arms (config A, one lead; IMG and TXT encoders stay frozen and cached):
  head      : the paper's projection heads only (baseline, same training protocol as below);
  deep_head : baseline heads + a residual MLP block on the TS side (frozen MOMENT);
  lora      : baseline heads + LoRA (rank r, q and v of all 12 MOMENT encoder layers).
The deep_head width is chosen so its extra parameters match the LoRA parameter count.

Protocol: the paper's heads/loss/optimizer (see train_heads.py), with one change applied to ALL
arms so the comparison is fair and LoRA fits a free T4: each training step uses the mean of 2
randomly chosen 512-step windows (of the 10) per record; validation/test use all 10 windows.
LoRA training uses GradCache (exact large-batch contrastive gradients with small micro-batches)
and fp16 autocast for the encoder. Checkpoints every epoch, so a disconnected run resumes.

Usage (Colab):
    python src/adapt.py --emb results/emb --ptbxl /content/ptbxl --arm head deep_head lora --seeds 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_signal  # noqa: E402
from extract import HF_IDS, N_WIN, WIN, WIN_STARTS  # noqa: E402
from label_eval import label_retrieval, multihot, superclasses, zero_shot_auroc  # noqa: E402
from train_heads import Head, embed, evaluate, load_emb, total_loss  # noqa: E402

MODS = ("ts", "img", "txt")
D_TS = 768


class DeepHead(nn.Module):
    """Residual MLP block (d -> h -> d) in front of the paper's head."""

    def __init__(self, d_in, h, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_in))
        self.head = Head(d_in, dropout=dropout)

    def forward(self, x):
        return self.head(x + self.block(x))


def lora_param_count(r, n_layers=12, d=D_TS, targets=2):
    return n_layers * targets * r * (d + d)


def deep_width_for(n_params, d=D_TS):
    return max(1, round((n_params - d) / (2 * d + 1)))


# ----------------------------------------------------------------------------- data
def load_signals(emb: Path, ptbxl: str, split_df: pd.DataFrame, lead: str, tag="") -> dict:
    """All signals of one lead, cached on Drive as float16 (N, 5000) per split."""
    out = {}
    for s in ("train", "val", "test"):
        f = emb / f"signals_{s}_lead{lead}{tag}.npy"
        if not f.exists():
            files = split_df[split_df.split == s].filename_hr.tolist()
            arr = np.stack([load_signal(ptbxl, x, lead) for x in files]).astype(np.float16)
            np.save(f, arr)
            print(f"  cached {f.name} {arr.shape}", flush=True)
        out[s] = torch.from_numpy(np.load(f))
    return out


def window_batch(sig: torch.Tensor, wins: torch.Tensor) -> torch.Tensor:
    """sig (B, 5000), wins (B, k) window indices -> (B, k, 512)"""
    starts = torch.as_tensor(WIN_STARTS)[wins]  # (B, k)
    ar = torch.arange(WIN)
    return torch.gather(sig.unsqueeze(1).expand(-1, wins.shape[1], -1), 2,
                        (starts.unsqueeze(-1) + ar).long())


# ----------------------------------------------------------------------------- encoder
class Moment:
    def __init__(self, device, lora_r=0):
        from momentfm import MOMENTPipeline
        self.m = MOMENTPipeline.from_pretrained(HF_IDS["moment-base"],
                                                model_kwargs={"task_name": "embedding"})
        self.m.init()
        for p in self.m.parameters():
            p.requires_grad_(False)
        if lora_r:
            from peft import LoraConfig, inject_adapter_in_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                             target_modules=["q", "v"])
            self.m.encoder = inject_adapter_in_model(cfg, self.m.encoder)
        self.m.to(device).eval()  # eval: no dropout, so GradCache's two passes agree
        self.device = device

    def lora_params(self):
        return [p for n, p in self.m.named_parameters() if "lora_" in n]

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        """w (B, k, 512) -> (B, 768), mean over the k windows."""
        B, k, _ = w.shape
        x = w.reshape(B * k, 1, WIN).float().to(self.device, non_blocking=True)
        mask = torch.ones(B * k, WIN, device=self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            e = self.m(x_enc=x, input_mask=mask).embeddings
        return e.float().reshape(B, k, -1).mean(1)

    @torch.no_grad()
    def all_windows(self, sig: torch.Tensor, bs=64) -> torch.Tensor:
        """Per-window embeddings for every record: (N, 10, 768) on CPU."""
        out = []
        allw = torch.arange(N_WIN).expand(bs, -1)
        for i in range(0, len(sig), bs):
            s = sig[i:i + bs]
            w = window_batch(s, allw[:len(s)])
            B = len(s)
            x = w.reshape(B * N_WIN, 1, WIN).float().to(self.device)
            mask = torch.ones(B * N_WIN, WIN, device=self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                e = self.m(x_enc=x, input_mask=mask).embeddings
            out.append(e.float().reshape(B, N_WIN, -1).cpu())
        return torch.cat(out)


# ----------------------------------------------------------------------------- training
def run_arm(arm, seed, data, sig, winemb, a, device, ckpt_dir: Path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_lora = lora_param_count(a.lora_r)
    h = deep_width_for(n_lora)
    heads = nn.ModuleDict({
        "ts": DeepHead(D_TS, h) if arm == "deep_head" else Head(D_TS),
        "img": Head(data["train"]["img"].shape[1]),
        "txt": Head(data["train"]["txt"].shape[1])}).to(device)
    enc = Moment(device, lora_r=a.lora_r) if arm == "lora" else None
    extra = (sum(p.numel() for p in heads["ts"].block.parameters()) if arm == "deep_head"
             else sum(p.numel() for p in enc.lora_params()) if arm == "lora" else 0)
    if enc is not None:
        assert next(enc.m.parameters()).device.type == device
        assert all(p.requires_grad for p in enc.lora_params()) and extra == n_lora, extra
    print(f"== {arm} seed {seed}: extra trainable params {extra:,} "
          f"(LoRA budget {n_lora:,}; deep width {h})", flush=True)

    opt_h = torch.optim.AdamW(heads.parameters(), lr=a.lr, weight_decay=a.wd)
    opt_l = torch.optim.AdamW(enc.lora_params(), lr=a.lr_lora, weight_decay=a.wd) if enc else None
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda") if enc else None
    n = len(data["train"]["ts"])
    spe = n // a.bs
    total = spe * a.epochs
    warm = max(1, int(0.02 * total))
    lam = (lambda s: (s + 1) / warm if s < warm
           else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    sch = [torch.optim.lr_scheduler.LambdaLR(o, lam) for o in (opt_h, opt_l) if o]
    img_tr, txt_tr = data["train"]["img"].to(device), data["train"]["txt"].to(device)

    ck = ckpt_dir / f"ckpt_{arm}_seed{seed}_lead{a.lead}.pt"
    state = {"epoch": 0, "best": float("inf"), "bad": 0, "log": [], "best_state": None}
    if ck.exists():
        c = torch.load(ck, map_location=device, weights_only=False)
        heads.load_state_dict(c["heads"])
        opt_h.load_state_dict(c["opt_h"])
        if enc:
            enc.m.load_state_dict(c["enc"], strict=False)
            opt_l.load_state_dict(c["opt_l"])
            scaler.load_state_dict(c["scaler"])
        for s_, sd in zip(sch, c["sch"]):
            s_.load_state_dict(sd)
        state = c["state"]
        print(f"  resumed from epoch {state['epoch']}", flush=True)

    def ts_feats(split, train_windows=None, idx=None):
        """TS encoder features: cached per-window means (frozen arms) or live MOMENT (lora)."""
        if enc is None:
            src = winemb[split] if idx is None else winemb[split][idx]
            return (src.mean(1) if train_windows is None else
                    torch.gather(src, 1, train_windows.unsqueeze(-1).expand(-1, -1, D_TS)).mean(1)
                    ).to(device)
        s = sig[split] if idx is None else sig[split][idx]
        with torch.no_grad():
            if train_windows is None:
                return torch.cat([enc(window_batch(s[i:i + a.micro],
                                                   torch.arange(N_WIN).expand(len(s[i:i + a.micro]), -1)))
                                  for i in range(0, len(s), a.micro)])
            return torch.cat([enc(window_batch(s[i:i + a.micro], train_windows[i:i + a.micro]))
                              for i in range(0, len(s), a.micro)])

    def val_loss():
        heads.eval()
        with torch.no_grad():
            X = {"ts": ts_feats("val"), "img": data["val"]["img"], "txt": data["val"]["txt"]}
            z = embed(heads, X, device)
            nv = len(z["ts"])
            return float(np.mean([total_loss({m: z[m][i:i + a.bs] for m in MODS}, a.tau).item()
                                  for i in range(0, nv - a.bs + 1, a.bs)] or
                                 [total_loss(z, a.tau).item()]))

    t0 = time.time()
    for ep in range(state["epoch"], a.epochs):
        te = time.time()
        heads.train()
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(n, generator=g)
        wins = torch.randint(0, N_WIN, (n, a.k_win), generator=g)
        run = 0.0
        for i in range(spe):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            raw = ts_feats("train", wins[idx], idx).detach().requires_grad_(enc is not None)
            z = {"ts": heads["ts"](raw), "img": heads["img"](img_tr[idx.to(device)]),
                 "txt": heads["txt"](txt_tr[idx.to(device)])}
            loss = total_loss(z, a.tau)
            opt_h.zero_grad(set_to_none=True)
            loss.backward()
            if enc is not None:  # GradCache pass 2: backprop cached d(loss)/d(raw) through MOMENT
                gcache = raw.grad.detach()
                opt_l.zero_grad(set_to_none=True)
                s = sig["train"][idx]
                for j in range(0, len(idx), a.micro):
                    r = enc(window_batch(s[j:j + a.micro], wins[idx][j:j + a.micro]))
                    scaler.scale((r * gcache[j:j + a.micro]).sum()).backward()
                scaler.unscale_(opt_l)
                nn.utils.clip_grad_norm_(enc.lora_params(), 1.0)
                scaler.step(opt_l)
                scaler.update()
            nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            opt_h.step()
            for s_ in sch:
                s_.step()
            run += loss.item()
        vl = val_loss()
        state["log"].append({"epoch": ep + 1, "train_loss": run / spe, "val_loss": vl,
                             "sec": time.time() - te})
        print(f"  {arm} seed {seed} ep {ep + 1:02d} train {run / spe:.4f} val {vl:.4f} "
              f"({time.time() - te:.0f}s)", flush=True)
        if vl < state["best"] - 1e-4:
            state["best"], state["bad"] = vl, 0
            state["best_state"] = {"heads": {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
                                   "lora": ({k: v.detach().cpu().clone() for k, v in enc.m.state_dict().items()
                                             if "lora_" in k} if enc else None)}
        else:
            state["bad"] += 1
        state["epoch"] = ep + 1
        done = state["bad"] >= a.patience or ep + 1 == a.epochs
        torch.save({"heads": heads.state_dict(), "opt_h": opt_h.state_dict(),
                    "enc": ({k: v for k, v in enc.m.state_dict().items() if "lora_" in k} if enc else None),
                    "opt_l": opt_l.state_dict() if enc else None,
                    "scaler": scaler.state_dict() if enc else None,
                    "sch": [s_.state_dict() for s_ in sch], "state": state}, ck)
        if done:
            break

    heads.load_state_dict(state["best_state"]["heads"])
    if enc:
        enc.m.load_state_dict(state["best_state"]["lora"], strict=False)
    heads.eval()
    Xte = {"ts": ts_feats("test"), "img": data["test"]["img"], "txt": data["test"]["txt"]}
    zte_t = embed(heads, Xte, device)
    res = evaluate(zte_t)
    zte = {m: v.cpu().numpy() for m, v in zte_t.items()}
    with torch.no_grad():
        ttr = data["train"]["txt"]
        ztr_txt = torch.cat([heads["txt"](ttr[i:i + 2048].to(device))
                             for i in range(0, len(ttr), 2048)]).cpu().numpy()
    return {"seed": seed, "arm": arm, "extra_params": extra, "epochs_run": state["epoch"],
            "best_val_loss": state["best"], "log": state["log"], "test": res,
            "zte": zte, "ztr_txt": ztr_txt, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--lead", default="I")
    ap.add_argument("--arm", nargs="+", default=["head", "deep_head", "lora"],
                    choices=["head", "deep_head", "lora"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--k-win", type=int, default=2, help="windows per record per training step")
    ap.add_argument("--micro", type=int, default=32, help="records per GradCache micro-batch")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="records per split (smoke test)")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        print(f"[cuda ok] {torch.cuda.get_device_name(0)}", flush=True)
    elif a.allow_cpu:
        device = "cpu"
        print("WARNING: CPU (smoke test only)")
    else:
        sys.exit("CUDA not available. Runtime > Change runtime type > T4 GPU.")

    emb = Path(a.emb)
    split = pd.read_csv(emb / "split.csv", index_col=0)
    data = load_emb(emb, {"ts": "moment-base", "img": "dinov2-base", "txt": "qwen3-emb-0.6b"}, a.lead)
    tag = f"_lim{a.limit}" if a.limit else ""
    if a.limit:
        for s in data:
            data[s] = {m: v[:a.limit] for m, v in data[s].items()}
        split = split.groupby("split", group_keys=False).head(a.limit)
    sig = load_signals(emb, a.ptbxl, split, a.lead, tag)
    a.bs = min(a.bs, len(sig["train"]))

    winemb = {}
    if any(x != "lora" for x in a.arm):
        f = {s: emb / f"winemb_moment_{s}_lead{a.lead}{tag}.npy"
             for s in sig}
        if not all(p.exists() for p in f.values()):
            enc0 = Moment(device)
            print("  computing frozen per-window MOMENT embeddings ...", flush=True)
            for s in sig:
                np.save(f[s], enc0.all_windows(sig[s]).numpy().astype(np.float16))
            del enc0
        winemb = {s: torch.from_numpy(np.load(f[s]).astype(np.float32)) for s in sig}

    sc = superclasses(a.ptbxl)
    sets = {s: sc.loc[split.index[split.split == s]].tolist() for s in ("train", "test")}
    keep = np.array([len(x) > 0 for x in sets["test"]])
    y_test = multihot(sets["test"])[keep]
    ckpt_dir = emb / "adapt_ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    for arm in a.arm:
        runs = []
        for seed in a.seeds:
            r = run_arm(arm, seed, data, sig, winemb, a, device, ckpt_dir)
            zte, ztr_txt = r.pop("zte"), r.pop("ztr_txt")
            lab = {}
            for qm in ("ts", "img"):
                hit, chance = label_retrieval(zte[qm][keep], zte["txt"][keep],
                                              [x for x, k in zip(sets["test"], keep) if k])
                lab[f"{qm}2txt_label_p@10"], lab["label_p@10_chance"] = hit, chance
                lab[f"{qm}_zeroshot_auroc"] = zero_shot_auroc(zte[qm][keep], ztr_txt,
                                                              sets["train"], y_test)
            r["label"] = lab
            t = r["test"]
            print(f"  RESULT {arm} seed {seed}: " + "  ".join(
                f"{p} margin {t[p]['cos_margin']:.3f} proc {t[p]['procrustes']:.3f} "
                f"cka {t[p]['cka_rbf']:.3f} R@1 {t[p]['R@1']:.4f}" for p in t)
                + "  " + "  ".join(f"{k} {v:.3f}" for k, v in lab.items()), flush=True)
            runs.append(r)
        out = emb / f"results_adapt_{arm}_lead{a.lead}{'_lim' if a.limit else ''}.json"
        json.dump({"arm": arm, "lead": a.lead, "args": vars(a), "device": device, "runs": runs},
                  open(out, "w"), indent=2)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
