"""List every distinct MIMIC-IV-ECG machine statement that touches a concept, with study counts, for review.

Broad keyword nets (deliberately over-inclusive) per concept, plus statements carrying negation, history or
comparison wording. Output: <out>/mimic_statements_<concept>.txt, one "count<TAB>statement" line each, and
<out>/mimic_statements_all.csv (every distinct statement with its count).
Usage: python src/mimic_statements.py --mimic /content/mimic --out results/mimic
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from mimic_vocab import load_reports

NETS = {
    "normal": r"\bnormal ecg\b|\bwithin normal limits\b",
    "afib": r"fibrillation|flutter",
    "infarct": r"infarct|\bmi\b|myocardial",
    "ivca": r"bundle|bbb|fascicular|conduction|hemiblock|ivcd|intraventricular",
    "lvh": r"hypertroph|\blvh\b|voltage",
    "stt": r"st-t|\bst\b|t wave|t-wave|repolari|junctional|st elevation|st depression|ischemi",
    "brady": r"bradycardia",
    "tachy": r"tachycardia",
    "neg_hist": r"\bno\b|\bnot\b|no longer|previous|prior|compared|unchanged|resolved|now |replaced|absent|without",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--out", default="results/mimic")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_reports(a.mimic)
    c = Counter(l for ls in df.lines for l in set(ls))
    pd.DataFrame(c.most_common(), columns=["statement", "studies"]).to_csv(out / "mimic_statements_all.csv", index=False)
    for name, pat in NETS.items():
        rx = re.compile(pat)
        rows = sorted(((n, s) for s, n in c.items() if rx.search(s)), reverse=True)
        (out / f"mimic_statements_{name}.txt").write_text("\n".join(f"{n}\t{s}" for n, s in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} distinct statements, {sum(n for n, _ in rows):,} occurrences", flush=True)


if __name__ == "__main__":
    main()
