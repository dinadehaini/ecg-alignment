"""Stage 0: a finite latent world with exactly computable claim posteriors and structural identifiability.

Latent hypothesis h = (a, l, e, z, u, s), 2*2*3*2*2*2 = 96 hypotheses, each with an analytic,
noiseless 4-channel trajectory of length T. A view v = (channel mask, time window). An observation is
the trajectory on the view's channels and window plus Gaussian noise (sigma).

Factors and where their evidence lives:
  a  oscillator A amplitude increasing / decreasing      ch1
  l  phase relation A leads B / B leads A                ch2 alone suffices (A's phase is fixed)
  e  event none / early (t=.25) / late (t=.75)            ch3 bump, only inside the window
  z  hidden marker absent / present                      ch4 only, coupled to nothing (decoupled)
  u  regime shift absent / present                       ch3 offset, AND ch2 frequency change in the
                                                         second half (coupled: survives hiding ch3)
  s  amplitude scale 0.8 / 1.2 (nuisance)                all channels

Claims are binary predicates of h, each with an explicit negation (12 claims). Two ground truths:
  structural identifiability of claim c from view v: c(h) is constant over every set of hypotheses
      whose noiseless observations under v are identical (with an explicit witness pair otherwise);
  posterior P(c | y, v): exact enumeration over the 96 hypotheses with the Gaussian likelihood.
Posterior categories: supported (>0.9), contradicted (<0.1), posterior-ambiguous (otherwise).
"""
from __future__ import annotations

import itertools

import numpy as np

T = 64
N_CH = 4
t = np.arange(T) / T

FACTORS = {"a": (0, 1), "l": (0, 1), "e": (0, 1, 2), "z": (0, 1), "u": (0, 1), "s": (0.8, 1.2)}
HYPS = [dict(zip(FACTORS, v)) for v in itertools.product(*FACTORS.values())]

# claim name -> (tokens, predicate on h)
BASE_CLAIMS = {
    "amp": (["oscillator", "a", "amplitude", "increasing"], lambda h: h["a"] == 1),
    "lead": (["oscillator", "a", "leads", "oscillator", "b"], lambda h: h["l"] == 1),
    "event": (["an", "event", "occurred"], lambda h: h["e"] != 0),
    "early": (["the", "event", "was", "early"], lambda h: h["e"] == 1),
    "marker": (["a", "hidden", "marker", "is", "present"], lambda h: h["z"] == 1),
    "shift": (["a", "regime", "shift", "occurred"], lambda h: h["u"] == 1),
}
CLAIMS = []  # list of (name, tokens, predicate, negation index)
for name, (tok, f) in BASE_CLAIMS.items():
    CLAIMS.append((name, tok, f))
    CLAIMS.append(("not_" + name, ["not"] + tok, (lambda g: (lambda h: not g(h)))(f)))
NEG = np.array([i + 1 if i % 2 == 0 else i - 1 for i in range(len(CLAIMS))])
VOCAB = sorted({w for _, tok, _ in CLAIMS for w in tok})
TRUTH = np.array([[bool(f(h)) for _, _, f in CLAIMS] for h in HYPS])  # (96, 12)

WINDOWS = {"full": (0, T), "first": (0, T // 2), "second": (T // 2, T)}
MASKS = [m for m in itertools.product((0, 1), repeat=N_CH) if any(m)]
VIEWS = [(m, w) for m in MASKS for w in WINDOWS]  # 45 views


def trajectory(h):
    s = h["s"]
    env = 1 + 0.6 * (t - 0.5) * (1 if h["a"] else -1)
    ch1 = s * env * np.sin(2 * np.pi * 4 * t)
    phi = np.pi / 2 if h["l"] else -np.pi / 2
    freq = 4 + 1.0 * h["u"] * (t >= 0.5)
    ch2 = s * np.sin(2 * np.pi * freq * t - phi)
    bump = {0: 0 * t, 1: np.exp(-((t - 0.25) / 0.04) ** 2), 2: np.exp(-((t - 0.75) / 0.04) ** 2)}[h["e"]]
    ch3 = s * (0.8 * h["u"] + 1.5 * bump)
    ch4 = s * 0.8 * h["z"] * np.sin(2 * np.pi * 2 * t)
    return np.stack([ch1, ch2, ch3, ch4])


TRAJ = np.stack([trajectory(h) for h in HYPS])  # (96, 4, T)


def view_mask(v):
    m, w = v
    M = np.zeros((N_CH, T), bool)
    a, b = WINDOWS[w]
    M[np.array(m, bool), a:b] = True
    return M


def structural(v, ci, tol=1e-9):
    """(identifiable, witness pair or None) for claim index ci under view v (noiseless)."""
    M = view_mask(v)
    obs = TRAJ[:, M]  # (96, n_obs)
    for i in range(len(HYPS)):
        same = np.flatnonzero(np.abs(obs - obs[i]).max(1) < tol)
        diff = same[TRUTH[same, ci] != TRUTH[i, ci]]
        if len(diff):
            return False, (i, int(diff[0]))
    return True, None


STRUCT = np.array([[structural(v, c)[0] for c in range(len(CLAIMS))] for v in VIEWS])  # (45, 12)


def prior(z_prior=0.5):
    p = np.array([z_prior if h["z"] == 1 else 1 - z_prior for h in HYPS])
    return p / p.sum()


def sample(n, sigma, rng, views=None, claims=None, z_prior=0.5, exclude=()):
    """Draw n examples. Returns dict of arrays incl. the exact oracle posterior.

    views/claims: allowed indices; exclude: set of (view, claim) pairs not to draw."""
    views = np.arange(len(VIEWS)) if views is None else np.asarray(views)
    claims = np.arange(len(CLAIMS)) if claims is None else np.asarray(claims)
    pr = prior(z_prior)
    h = rng.choice(len(HYPS), size=n, p=pr)
    v = rng.choice(views, size=n)
    c = rng.choice(claims, size=n)
    if exclude:
        bad = np.array([(vi, ci) in exclude for vi, ci in zip(v, c)])
        while bad.any():
            v[bad] = rng.choice(views, size=bad.sum())
            c[bad] = rng.choice(claims, size=bad.sum())
            bad = np.array([(vi, ci) in exclude for vi, ci in zip(v, c)])
    M = np.stack([view_mask(VIEWS[i]) for i in range(len(VIEWS))])[v]  # (n, 4, T)
    y_full = TRAJ[h] + sigma * rng.standard_normal((n, N_CH, T))
    y = y_full * M
    # exact posterior over hypotheses given the observed entries
    ll = np.concatenate([  # chunked to bound memory
        -0.5 * (((y[i:i + 2000, None] - TRAJ[None] * M[i:i + 2000, None]) ** 2) * M[i:i + 2000, None]).sum((2, 3))
        / sigma ** 2 for i in range(0, n, 2000)])
    lp = ll + np.log(pr)[None]
    lp -= lp.max(1, keepdims=True)
    post_h = np.exp(lp)
    post_h /= post_h.sum(1, keepdims=True)
    p_claim = (post_h * TRUTH[:, c].T).sum(1)
    win = np.array([[WINDOWS[VIEWS[i][1]][0] / T, WINDOWS[VIEWS[i][1]][1] / T] for i in v])
    return {"y": y.astype(np.float32), "y_full": y_full.astype(np.float32),
            "mask": M.astype(np.float32), "win": win.astype(np.float32),
            "h": h, "v": v, "c": c, "truth": TRUTH[h, c].astype(np.float32),
            "oracle": p_claim.astype(np.float32), "struct": STRUCT[v, c],
            "post_h": post_h.astype(np.float32)}


if __name__ == "__main__":
    print(f"{len(HYPS)} hypotheses, {len(VIEWS)} views, {len(CLAIMS)} claims")
    print("structurally non-identifiable (view, claim) share:", 1 - STRUCT.mean())
    for ci, (name, _, _) in enumerate(CLAIMS[::2]):
        ci *= 2
        print(f"{name:7s} identifiable in {STRUCT[:, ci].sum():2d}/45 views; witness example:",
              next(((VIEWS[vi], structural(VIEWS[vi], ci)[1]) for vi in range(45) if not STRUCT[vi, ci]), None))
