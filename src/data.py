"""PTB-XL loading, patient-disjoint split, and ECG plot rendering.

Paper setup (App. C.4): 10 s ECG at 500 Hz -> univariate series of length 5000; reports used
verbatim (German); 21,000 triplets with non-empty reports; split sizes matched to CaTS
(16k train / 1k val / 4k test, the same as their MIMIC split).

Not stated in the paper, so chosen here (see PAPER_NOTES.md, "Gaps"):
  * which lead -> default lead I (channel 0), configurable;
  * how the 21,000 are picked and split -> seeded sample, split disjoint by patient_id;
  * plot style -> plain line plot with axes and units, rendered at 384x384.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SPLIT_SIZES = {"train": 16000, "val": 1000, "test": 4000}
FS = 500
SEQ_LEN = 5000
IMG_SIZE = 384


def load_metadata(ptbxl_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(ptbxl_dir) / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["report"] = df["report"].fillna("").astype(str).str.strip()
    return df


def make_splits(df: pd.DataFrame, n_total: int = 21000, seed: int = 42,
                sizes: dict = SPLIT_SIZES) -> pd.DataFrame:
    """Pick n_total records with non-empty reports and split them disjointly by patient.

    Patients are shuffled with `seed` and assigned greedily to test, then val, then train,
    without letting a patient straddle two splits. Records of patients that no longer fit
    anywhere are left out, which is how we get from ~21.8k records down to exactly 21,000.
    """
    assert sum(sizes.values()) == n_total
    d = df[df["report"].str.len() > 0]
    rng = np.random.default_rng(seed)
    patients = d["patient_id"].unique()
    rng.shuffle(patients)
    groups = d.groupby("patient_id").groups
    order = ["test", "val", "train"]
    filled = {s: [] for s in order}
    for pid in patients:
        ids = list(groups[pid])
        for s in order:
            if len(filled[s]) + len(ids) <= sizes[s]:
                filled[s].extend(ids)
                break
    out = pd.concat([d.loc[sorted(filled[s])].assign(split=s) for s in order])
    return out


def load_signal(ptbxl_dir: str | Path, filename_hr: str, lead: int | str = 0) -> np.ndarray:
    """Return one lead of a 500 Hz record as float32 (mV), shape (5000,)."""
    import wfdb

    li = LEADS.index(lead) if isinstance(lead, str) else int(lead)
    sig, _ = wfdb.rdsamp(str(Path(ptbxl_dir) / filename_hr), channels=[li])
    x = np.nan_to_num(sig[:, 0].astype(np.float32))
    assert x.shape == (SEQ_LEN,), x.shape
    return x


def load_signal_12(ptbxl_dir: str | Path, filename_hr: str) -> np.ndarray:
    """All 12 leads, shape (12, 5000). Used in phase 2 only."""
    import wfdb

    sig, _ = wfdb.rdsamp(str(Path(ptbxl_dir) / filename_hr))
    return np.nan_to_num(sig.T.astype(np.float32))


class _Renderer:
    """One reusable matplotlib figure; only the line data and y-limits change per plot."""

    def __init__(self, size: int = IMG_SIZE):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.size = size
        self.fig, self.ax = plt.subplots(figsize=(4, 4), dpi=size / 4)
        t = np.arange(SEQ_LEN) / FS
        (self.line,) = self.ax.plot(t, np.zeros(SEQ_LEN), lw=0.6, color="black")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude (mV)")
        self.ax.set_xlim(0, t[-1])
        self.fig.tight_layout()

    def __call__(self, x: np.ndarray):
        from PIL import Image

        self.line.set_ydata(x)
        lo, hi = float(x.min()), float(x.max())
        pad = 0.05 * (hi - lo) or 0.1
        self.ax.set_ylim(lo - pad, hi + pad)
        self.fig.canvas.draw()
        img = Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[..., :3])
        return img if img.size == (self.size, self.size) else img.resize((self.size, self.size))


_RENDERER = None


def render_plot(x: np.ndarray, size: int = IMG_SIZE):
    """Render the full waveform as a line plot with axes and units -> PIL RGB image (size²)."""
    global _RENDERER
    if _RENDERER is None or _RENDERER.size != size:
        _RENDERER = _Renderer(size)
    return _RENDERER(x)


def load_and_render(args):
    """Worker for multiprocessing: (ptbxl_dir, filename_hr, lead) -> PIL image."""
    return render_plot(load_signal(*args))
