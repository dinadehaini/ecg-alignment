"""Stage 0 experiment: do signal-language methods track the exact claim posterior under partial views?

Methods (all receive the observed values, the observation mask and the time window):
  predictor    MLP(view, claim) -> P(claim true), BCE on sampled claim truth.
               Supervision: binary claim labels (more than positive pairs; a reference, not matched).
  ensemble     5 predictors with different seeds (mean; disagreement reported as a supplement).
  contrastive  CLIP-style InfoNCE between view encodings and one randomly drawn TRUE claim per
               example (positive pairs only, in-batch negatives). Decision rule: score a claim against
               its explicit negation, p = sigmoid(s(c) - s(not c)); calibrated version = Platt scaling
               on a labeled validation split.
  distill      student on partial views trained to match a full-view teacher (a predictor trained on
               full views) evaluated on the same latent sample's full noisy observation.
  oracle       exact posterior under the finite simulator.
"raw" = as trained; "cal" = temperature (predictor/student) or Platt (contrastive) fit on validation.
Abstention is evaluated through risk-coverage curves on every method's probabilities.

Held out at training time: 3 channel masks (all windows) and 4 (claim, view) compositions.

Usage: python stage0.py --out results [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import world as W

HELD_MASKS = [(1, 1, 0, 1), (0, 1, 0, 1), (1, 0, 1, 1)]
TRAIN_VIEWS = [i for i, (m, w) in enumerate(W.VIEWS) if m not in HELD_MASKS]
CN = [n for n, _, _ in W.CLAIMS]
VI = {v: i for i, v in enumerate(W.VIEWS)}
HELD_COMP = {(VI[((0, 1, 0, 0), "second")], CN.index("shift")), (VI[((0, 1, 0, 0), "second")], CN.index("not_shift")),
             (VI[((0, 0, 1, 0), "first")], CN.index("early")), (VI[((0, 0, 1, 0), "first")], CN.index("not_early"))}
BOW = np.array([[w in tok for w in W.VOCAB] for _, tok, _ in W.CLAIMS], np.float32)
D_IN = 2 * W.N_CH * W.T + 2


def feats(d, full=False):
    if full:
        y, m = d["y_full"], np.ones_like(d["mask"])
        win = np.tile(np.array([0, 1], np.float32), (len(y), 1))
    else:
        y, m, win = d["y"], d["mask"], d["win"]
    return torch.as_tensor(np.concatenate([y.reshape(len(y), -1), m.reshape(len(m), -1), win], 1))


class SigEnc(nn.Module):
    """1-D conv over (values, mask) channels; the time window is appended after the convolutions."""
    def __init__(self, out=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2 * W.N_CH, 32, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
                                  nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 * (W.T // 4) + 2, 256), nn.ReLU(), nn.Linear(256, out))

    def forward(self, x):
        n = len(x)
        seq = x[:, :2 * W.N_CH * W.T].reshape(n, 2 * W.N_CH, W.T)
        return self.fc(torch.cat([self.conv(seq).reshape(n, -1), x[:, -2:]], 1))


class ClaimEnc(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(W.VOCAB), 128), nn.ReLU(), nn.Linear(128, out))

    def forward(self, c):
        return self.net(torch.as_tensor(BOW[c]))


class Predictor(nn.Module):
    """Same encoders as the contrastive model; bilinear readout logit = f(view) . g(claim) + b."""
    def __init__(self):
        super().__init__()
        self.sig, self.cl = SigEnc(), ClaimEnc()
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, c):
        return (self.sig(x) * self.cl(c)).sum(1) / np.sqrt(128) + self.b


class Contrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig, self.cl = SigEnc(), ClaimEnc()

    def enc_x(self, x):
        return F.normalize(self.sig(x), dim=1)

    def enc_c(self, c):
        return F.normalize(self.cl(c), dim=1)


def fit(model, loss_fn, n, val_loss_fn, epochs=100, bs=256, lr=1e-3, patience=8, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, bad, state = 1e9, 0, None
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        for i in np.array_split(rng.permutation(n), max(1, n // bs)):
            opt.zero_grad()
            loss_fn(i).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(val_loss_fn())
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model


def temp_scale(logit_val, y_val):
    lv, yv = torch.as_tensor(logit_val), torch.as_tensor(y_val)
    best = min(np.exp(np.linspace(-3, 3, 121)), key=lambda T: float(F.binary_cross_entropy_with_logits(lv / T, yv)))
    return lambda lg: lg / best


def platt(s_val, y_val):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e4).fit(np.asarray(s_val)[:, None], y_val)
    return lambda s: lr.decision_function(np.asarray(s)[:, None])


def train_predictor(tr, va, seed=0, full=False):
    Xtr, Xva = feats(tr, full), feats(va, full)
    ytr, yva = torch.as_tensor(tr["truth"]), torch.as_tensor(va["truth"])
    m = Predictor()
    fit(m, lambda i: F.binary_cross_entropy_with_logits(m(Xtr[i], tr["c"][i]), ytr[i]), len(ytr),
        lambda: F.binary_cross_entropy_with_logits(m(Xva, va["c"]), yva), seed=seed)
    return m


def true_claim(h, rng, v=None):
    """One random true claim per example (respecting held-out compositions)."""
    out = np.empty(len(h), int)
    for k, hi in enumerate(h):
        ok = np.flatnonzero(W.TRUTH[hi])
        if v is not None:
            ok = np.array([c for c in ok if (v[k], c) not in HELD_COMP])
        out[k] = rng.choice(ok)
    return out


def train_contrastive(tr, va, seed=0, tau=0.1):
    rng = np.random.default_rng(seed)
    Xtr, Xva = feats(tr), feats(va)
    ctr, cva = true_claim(tr["h"], rng, tr["v"]), true_claim(va["h"], rng, va["v"])
    m = Contrastive()

    def nce(x, c):
        zx, zc = m.enc_x(x), m.enc_c(c)
        lg = zx @ zc.T / tau
        # identical captions in a batch are false negatives, as in real paired data
        y = torch.arange(len(lg))
        return 0.5 * (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y))
    vi = np.arange(min(len(cva), 2048))
    fit(m, lambda i: nce(Xtr[i], ctr[i]), len(ctr), lambda: nce(Xva[vi], cva[vi]), seed=seed)
    return m


def contrastive_score(m, d, tau=0.1):
    with torch.no_grad():
        zx = m.enc_x(feats(d))
        s_c = (zx * m.enc_c(d["c"])).sum(1) / tau
        s_n = (zx * m.enc_c(W.NEG[d["c"]])).sum(1) / tau
    return (s_c - s_n).numpy()


def train_distill(tr, va, teacher, seed=0):
    Xtr, Xva = feats(tr), feats(va)
    with torch.no_grad():
        ttr = torch.sigmoid(teacher(feats(tr, full=True), tr["c"]))
        tva = torch.sigmoid(teacher(feats(va, full=True), va["c"]))
    m = Predictor()
    fit(m, lambda i: F.binary_cross_entropy_with_logits(m(Xtr[i], tr["c"][i]), ttr[i]), len(ttr),
        lambda: F.binary_cross_entropy_with_logits(m(Xva, va["c"]), tva), seed=seed)
    return m


def logits(m, d):
    with torch.no_grad():
        return m(feats(d), d["c"]).numpy()


# ---------------- metrics ----------------
def aurc(p, y):
    conf = np.abs(p - 0.5)
    err = ((p > 0.5) != (y > 0.5)).astype(float)
    o = np.argsort(-conf)
    risk = np.cumsum(err[o]) / np.arange(1, len(o) + 1)
    return float(risk.mean())


def metrics(p, d, idx):
    p, y, o = p[idx], d["truth"][idx], d["oracle"][idx]
    eps = 1e-6
    ll = lambda q: float(-np.mean(y * np.log(q + eps) + (1 - y) * np.log(1 - q + eps)))
    return {"n": int(idx.sum()), "brier": float(np.mean((p - y) ** 2)),
            "excess_brier": float(np.mean((p - y) ** 2) - np.mean((o - y) ** 2)),
            "logloss": ll(p), "excess_logloss": ll(p) - ll(o),
            "mae_oracle": float(np.mean(np.abs(p - o))),
            "conf_wrong_side": float(np.mean(((p > 0.9) & (o < 0.5)) | ((p < 0.1) & (o > 0.5)))),
            "conf_when_oracle_ambig": float(np.mean((np.abs(p - 0.5) > 0.4) & (np.abs(o - 0.5) < 0.4))),
            "aurc": aurc(p, y), "aurc_oracle": aurc(o, y)}


def groups(d):
    vm = np.array([W.VIEWS[v][0] for v in d["v"]])
    seen = ~np.array([tuple(m) in HELD_MASKS for m in vm])
    comp = np.array([(v, c) in HELD_COMP for v, c in zip(d["v"], d["c"])])
    amb = (d["oracle"] > 0.1) & (d["oracle"] < 0.9)
    s = d["struct"]
    name = np.array(CN)[d["c"]]
    trap = (np.isin(name, ["shift", "not_shift"]) & (vm[:, 1] == 1) & (vm[:, 2] == 0) & s) | \
           (np.isin(name, ["lead", "not_lead"]) & (vm[:, 0] == 0) & s)
    return {"all": np.ones(len(s), bool),
            "identifiable, posterior-confident": s & ~amb,
            "identifiable, posterior-ambiguous (noise control)": s & amb,
            "structurally non-identifiable": ~s,
            "non-identifiable, posterior-confident (prior)": ~s & ~amb,
            "trap: evidence survives hidden channel": trap,
            "seen views": seen & ~comp,
            "unseen channel masks": ~seen,
            "held-out (claim, view) compositions": comp}


def run_condition(n, sigma, z_prior, seed, n_test, ens):
    rng = np.random.default_rng(seed)
    kw = dict(views=TRAIN_VIEWS, z_prior=z_prior, exclude=HELD_COMP)
    tr = W.sample(n, sigma, rng, **kw)
    va = W.sample(max(1000, n // 10), sigma, rng, **kw)
    te = W.sample(n_test, sigma, np.random.default_rng(10_000 + seed), z_prior=z_prior)
    probs = {"oracle": te["oracle"]}
    t0 = time.time()
    pred = train_predictor(tr, va, seed)
    lg_te, lg_va = logits(pred, te), logits(pred, va)
    probs["predictor raw"] = 1 / (1 + np.exp(-lg_te))
    probs["predictor cal"] = 1 / (1 + np.exp(-temp_scale(lg_va, va["truth"])(lg_te)))
    if ens:
        ms = [pred] + [train_predictor(tr, va, seed + 100 * k) for k in range(1, ens)]
        P = np.stack([1 / (1 + np.exp(-logits(m, te))) for m in ms])
        probs["ensemble"] = P.mean(0)
        dis = P.std(0)
    con = train_contrastive(tr, va, seed)
    s_te, s_va = contrastive_score(con, te), contrastive_score(con, va)
    probs["contrastive raw"] = 1 / (1 + np.exp(-s_te))
    probs["contrastive cal"] = 1 / (1 + np.exp(-platt(s_va, va["truth"])(s_te)))
    teacher = train_predictor(tr, va, seed + 7, full=True)
    st = train_distill(tr, va, teacher, seed)
    lg_te, lg_va = logits(st, te), logits(st, va)
    probs["distill raw"] = 1 / (1 + np.exp(-lg_te))
    probs["distill cal"] = 1 / (1 + np.exp(-temp_scale(lg_va, va["truth"])(lg_te)))
    G = groups(te)
    rows = []
    for meth, p in probs.items():
        for gname, idx in G.items():
            if idx.sum() == 0:
                continue
            rows.append(dict(n_train=n, sigma=sigma, z_prior=z_prior, seed=seed, method=meth, group=gname,
                             **metrics(p, te, idx)))
    extra = {}
    if ens:
        extra = {g: float(dis[idx].mean()) for g, idx in G.items() if idx.sum()}
    print(f"n={n} sigma={sigma} z_prior={z_prior} seed={seed} done in {time.time() - t0:.0f}s", flush=True)
    return rows, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    if a.quick:
        sizes, sweep, n_test, seeds, ens = [2000], [], 4000, 1, 2
    else:
        sizes, n_test, seeds, ens = [2000, 8000, 32000, 128000], 40000, a.seeds, 5
        sweep = [(32000, 0.5, 0.5), (32000, 2.0, 0.5), (32000, 1.0, 0.9)]
    conds = [(n, 1.0, 0.5) for n in sizes] + sweep
    rows, dis = [], {}
    for n, sigma, zp in conds:
        for seed in range(seeds):
            r, e = run_condition(n, sigma, zp, seed, n_test, ens if (sigma, zp) == (1.0, 0.5) and n <= 32000 else 0)
            rows += r
            if e:
                dis[f"n={n} seed={seed}"] = e
            pd.DataFrame(rows).to_csv(out / "stage0_rows.csv", index=False)
    json.dump(dis, open(out / "stage0_ensemble_disagreement.json", "w"), indent=1)
    df = pd.DataFrame(rows)
    key = ["sigma", "z_prior", "n_train", "group", "method"]
    summ = df.groupby(key)[["n", "brier", "excess_brier", "excess_logloss", "mae_oracle", "conf_wrong_side",
                            "conf_when_oracle_ambig", "aurc", "aurc_oracle"]].mean().reset_index()
    summ.to_csv(out / "stage0_summary.csv", index=False)
    main_n = max(sizes)
    sel = summ[(summ.sigma == 1.0) & (summ.z_prior == 0.5) & (summ.n_train == main_n)]
    print(sel.drop(columns=["sigma", "z_prior", "n_train"]).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
