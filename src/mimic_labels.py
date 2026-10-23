"""MIMIC-IV-ECG report-derived concept labels and patient-disjoint subset (frozen before any model evaluation).

Labels are machine-report-derived targets: agreement with reported findings, not verified diagnoses.

Statement mapping: every report line is classified per concept as POS (unequivocal current assertion), UNC
(the finding itself is hedged: possible / probable / consider / cannot rule out / borderline / suggests),
NEG (explicit negation, "no longer present", historical or comparison statement about the finding), or
nothing. Rules are explicit regexes per concept (below), reviewed against the full statement list with
counts (results/mimic/mimic_statements_*.txt). Hedging that applies only to a proposed CAUSE (e.g. "st-t
changes may be due to myocardial ischemia") does not make the finding uncertain.

Report-level combination per concept:
  1. any POS and any NEG                -> excluded (conflict)
  2. else any POS                       -> positive (even if another line hedges a subtype or cause)
  3. else any UNC                       -> excluded (uncertain)
  4. else                               -> negative (missing-mention assumption)
Normal ECG additionally: excluded if the report also contains "abnormal ecg" (contradiction), and "normal ecg
except for rate" is UNC.

Subset: paced ECGs (any line mentioning pacing / paced rhythm / pacemaker) are excluded. Patients are shuffled
with a fixed seed; ALL remaining ECGs of each selected patient are kept; patients are added until ~21,000
ECGs, then assigned (patient-disjoint) to test (~4,000 ECGs), validation (~1,000) and train (rest).

Usage: python src/mimic_labels.py --mimic <dir with machine_measurements.csv> --out results/mimic
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_vocab import load_reports

SEED = 0
N_TOTAL, N_TEST, N_VAL = 21000, 4000, 1000

HEDGE = r"(possible|probable|probably|consider|cannot rule out|can not rule out|borderline|suggests|suggestive|questionable|\?)"
NEGHIST = r"(\bno longer\b|\bno\b|\bnot\b|\bresolved\b|\bprevious\b|\bprior\b|\bcompared\b|\breplaced\b|\babsent\b)"

CONCEPTS = {
    # concept: (positive pattern for the finding, extra uncertain patterns, extra negative patterns)
    "normal_ecg": (r"^normal ecg\.?$|^normal ecg\b(?! except)", r"normal ecg except for rate", None),
    "atrial_fibrillation": (r"atrial fibrillation", None, None),
    "reported_infarct_pattern": (r"infarct", r"cannot rule out .*infarct|may be due to lvh but cannot rule out", None),
    "intraventricular_conduction_abnormality": (
        r"right bundle branch block|left bundle branch block|\brbbb\b|\blbbb\b|iv conduction defect|"
        r"intraventricular conduction|left anterior fascicular block|left posterior fascicular block|"
        r"\blafb\b|\blpfb\b|bifascicular block|incomplete (right|left) bundle branch block|incomplete rbbb|incomplete lbbb",
        None, None),
    "left_ventricular_hypertrophy": (r"left ventricular hypertrophy|\blvh\b", r"qrs changes .* may be due to lvh", None),
    "st_t_changes": (
        r"st-t changes|st-t abnormalit|t wave changes|t-wave changes|t wave abnormalit|st junctional depression|"
        r"st depression|st elevation|repolarization abnormalit|t wave inversion",
        None, None),
    "sinus_bradycardia": (r"sinus bradycardia", None, None),
    "sinus_tachycardia": (r"sinus tachycardia", None, None),
}
PACED = re.compile(r"pac(ed|ing)\b|pacemaker|paced rhythm|ventricular pacing|atrial pacing|dual chamber pacing")


def classify_line(line, concept):
    pos, unc, neg = CONCEPTS[concept]
    if unc and re.search(unc, line):
        return "UNC"
    if not re.search(pos, line):
        return None
    if neg and re.search(neg, line):
        return "NEG"
    if re.search(NEGHIST, line):
        return "NEG"
    # hedge words count only if they qualify the finding, i.e. appear before the finding's match and not
    # only inside a cause clause ("may be due to ...", "are probably due to ...")
    m = re.search(pos, line)
    head = line[:m.start()]
    if re.search(HEDGE, head) and not re.search(r"due to", head):
        return "UNC"
    return "POS"


def report_label(lines, concept):
    cls = [classify_line(l, concept) for l in lines]
    if concept == "normal_ecg" and any(l.startswith("abnormal ecg") for l in lines) and "POS" in cls:
        return "excluded: contradiction", cls
    if "POS" in cls and "NEG" in cls:
        return "excluded: conflict", cls
    if "POS" in cls:
        return "positive", cls
    if "UNC" in cls:
        return "excluded: uncertain", cls
    return "negative", cls


def select_subset(df):
    df = df[~df.lines.map(lambda ls: any(PACED.search(l) for l in ls))]
    rng = np.random.default_rng(SEED)
    subj = df.subject_id.unique()
    rng.shuffle(subj)
    per = df.groupby("subject_id").size()
    chosen, total = [], 0
    for s in subj:
        if total >= N_TOTAL:
            break
        chosen.append(s)
        total += int(per[s])
    split = {}
    acc = {"test": 0, "val": 0}
    for s in chosen:
        if acc["test"] < N_TEST:
            split[s] = "test"; acc["test"] += int(per[s])
        elif acc["val"] < N_VAL:
            split[s] = "val"; acc["val"] += int(per[s])
        else:
            split[s] = "train"
    sub = df[df.subject_id.isin(chosen)].copy()
    sub["split"] = sub.subject_id.map(split)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--out", default="results/mimic")
    ap.add_argument("--audit", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_reports(a.mimic)
    sub = select_subset(df)
    sub["report"] = sub.lines.map(lambda ls: "\n".join(ls))
    rows = []
    for c in CONCEPTS:
        labs = sub.lines.map(lambda ls: report_label(ls, c)[0])
        sub[c] = labs
        for sp in ("train", "val", "test"):
            vc = labs[sub.split == sp].value_counts()
            rows.append({"concept": c, "split": sp, **vc.to_dict()})
    counts = pd.DataFrame(rows).fillna(0)
    sub.drop(columns=["lines"]).to_csv(out / "mimic_subset_labels.csv", index=False)
    rng = np.random.default_rng(SEED)
    audit = []
    for c in CONCEPTS:
        for lab in sorted(sub[c].unique()):
            idx = np.flatnonzero(sub[c].values == lab)
            for i in rng.choice(idx, size=min(a.audit, len(idx)), replace=False):
                audit.append({"concept": c, "label": lab, "report": sub.report.iloc[i].replace("\n", " | ")})
    pd.DataFrame(audit).to_csv(out / "mimic_label_audit.csv", index=False)
    te = sub[sub.split == "test"]
    vc = te.report.value_counts()
    print(f"subset: {len(sub)} ECGs, {sub.subject_id.nunique()} patients; per split "
          f"{sub.split.value_counts().to_dict()}", flush=True)
    print(f"test: {len(te)} ECGs, {len(vc)} unique reports ({len(vc) / len(te):.3f}); "
          f"ECGs whose report appears elsewhere in test {(te.report.map(vc) > 1).mean():.3f}", flush=True)
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
