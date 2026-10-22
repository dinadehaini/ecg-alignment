"""Inspect the MIMIC-IV-ECG machine-report vocabulary (no model results) to draft a concept inventory.

Each study's report = its non-empty report_0..report_17 lines, in order. Prints: number of studies and
subjects, share of studies with any report text, unique full reports and the duplicate share, and the most
frequent individual statements (lowercased, whitespace-collapsed) with their study counts.

Usage: python src/mimic_vocab.py --mimic /content/mimic --top 200
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def load_reports(mimic):
    mm = pd.read_csv(Path(mimic) / "machine_measurements.csv", low_memory=False)
    cols = [f"report_{i}" for i in range(18) if f"report_{i}" in mm]
    lines = mm[cols].apply(lambda r: [norm(x) for x in r if isinstance(x, str) and x.strip()], axis=1)
    return mm[["subject_id", "study_id"]].assign(lines=lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--top", type=int, default=200)
    a = ap.parse_args()
    df = load_reports(a.mimic)
    has = df.lines.map(len) > 0
    full = df.lines.map(" | ".join)
    print(f"studies {len(df):,}  subjects {df.subject_id.nunique():,}  with report text {has.mean():.3f}")
    vc = full[has].value_counts()
    print(f"unique full reports {len(vc):,} ({len(vc) / has.sum():.3f} of studies with text); "
          f"studies whose report appears elsewhere {(full[has].map(vc) > 1).mean():.3f}")
    print(f"lines per report: mean {df.lines[has].map(len).mean():.2f}, max {df.lines.map(len).max()}")
    c = Counter(l for ls in df.lines for l in set(ls))
    print(f"\ndistinct statements {len(c):,}; top {a.top} by number of studies:")
    for s, n in c.most_common(a.top):
        print(f"{n:8d}  {s}")
    print("\nmost common full reports:")
    for s, n in vc.head(15).items():
        print(f"{n:8d}  {s}")


if __name__ == "__main__":
    main()
