"""Verify the downloaded MIMIC-IV-ECG waveforms for the frozen subset before embedding extraction.

For every ECG in results/mimic/mimic_subset_labels.csv: both <study>.hea and <study>.dat present; header
declares 12 signals, 500 Hz, 5000 samples, format 16 (=> 120,000-byte .dat); the .dat has that size; and
wfdb.rdrecord decodes it to a finite (5000, 12) array. Records failing any check are written to
results/mimic/mimic_unavailable.csv with the reason; they are dropped from the analysis explicitly, never
replaced by other records.
Usage: python src/mimic_verify.py --wfdb data_mimic/wfdb --out results/mimic
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def check(wdir, study):
    hea, dat = wdir / f"{study}.hea", wdir / f"{study}.dat"
    if not hea.exists():
        return "missing .hea"
    if not dat.exists():
        return "missing .dat"
    first = hea.read_text().splitlines()[0].split()
    if first[1:4] != ["12", "500", "5000"]:
        return f"unexpected header {first[1:4]}"
    fmts = {l.split()[1] for l in hea.read_text().splitlines()[1:13]}
    if fmts != {"16"}:
        return f"unexpected format {fmts}"
    if os.path.getsize(dat) != 120000:
        return f".dat size {os.path.getsize(dat)}"
    import wfdb
    try:
        rec = wfdb.rdrecord(str(wdir / str(study)))
    except Exception as e:  # noqa: BLE001
        return f"wfdb error: {type(e).__name__}"
    sig = rec.p_signal
    if sig is None or sig.shape != (5000, 12):
        return f"decoded shape {None if sig is None else sig.shape}"
    if not np.isfinite(sig).all():
        return "non-finite samples"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfdb", default="data_mimic/wfdb")
    ap.add_argument("--out", default="results/mimic")
    a = ap.parse_args()
    d = pd.read_csv(Path(a.out) / "mimic_subset_labels.csv")
    wdir = Path(a.wfdb)
    status = [check(wdir, s) for s in d.study_id]
    d = d.assign(status=status)
    bad = d[d.status != "ok"][["subject_id", "study_id", "split", "status"]]
    bad.to_csv(Path(a.out) / "mimic_unavailable.csv", index=False)
    print(d.status.value_counts().to_string(), flush=True)
    print("ok per split:", d[d.status == "ok"].split.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
