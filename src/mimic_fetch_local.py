"""Fetch the frozen subset's WFDB files to local disk from two sources at once.

Colab's Drive mount serves small files about one at a time (~1.5 files/s regardless of parallel readers), and
PhysioNet throttles per address (~2 files/s), so each record is taken from one of the two, split 1:2, and the
other source is the fallback. Files are written to a temporary name and renamed only when complete (.dat must be
120,000 bytes), so later steps never see partial files. Records already present locally are skipped.
Usage: python src/mimic_fetch_local.py --subset results/mimic/mimic_subset_labels.csv \
           --drive data_mimic/wfdb --out /content/mimic_local/wfdb
"""
from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

URL = "https://physionet.org/files/mimic-iv-ecg/1.0/files/p{s4}/p{s}/s{t}/{t}{ext}"


def complete(p, ext):
    return p.exists() and (os.path.getsize(p) == 120000 if ext == ".dat" else os.path.getsize(p) > 0)


def fetch_drive(src, dst):
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def fetch_web(url, dst, attempts=5):
    """PhysioNet rejects some requests under load: retry with exponential backoff."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, dst)
            return
        except Exception:  # noqa: BLE001
            if k == attempts - 1:
                raise
            time.sleep(2 ** k + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, drive = Path(a.out), Path(a.drive)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.subset)
    jobs = []
    for i, (s, t) in enumerate(zip(d.subject_id.astype(str), d.study_id.astype(str))):
        for ext in (".hea", ".dat"):
            dst = out / f"{t}{ext}"
            if not complete(dst, ext):
                jobs.append((i % 3 == 0, drive / f"{t}{ext}", URL.format(s4=s[:4], s=s, t=t, ext=ext), dst, ext))
    print(f"{len(jobs)} files to fetch", flush=True)
    done, failed, lock, t0 = [0], [], threading.Lock(), time.time()

    def work(job):
        use_drive, src, url, dst, ext = job
        order = [("drive", src), ("web", url)] if use_drive else [("web", url), ("drive", src)]
        for kind, where in order:
            try:
                (fetch_drive if kind == "drive" else fetch_web)(where, dst)
                if complete(dst, ext):
                    break
            except Exception:  # noqa: BLE001
                continue
        with lock:
            done[0] += 1
            if not complete(dst, ext):
                failed.append(dst.name)
            if done[0] % 1000 == 0:
                print(f"{done[0]}/{len(jobs)} files, {time.time() - t0:.0f}s, failed {len(failed)}", flush=True)

    if not drive.exists():  # no Drive copy: every record comes from PhysioNet, with fewer parallel requests
        jobs = [(False,) + j[1:] for j in jobs]
    drive_jobs = [j for j in jobs if j[0]]
    web_jobs = [j for j in jobs if not j[0]]
    with ThreadPoolExecutor(8) as pd_, ThreadPoolExecutor(12) as pw:
        fs = [pd_.submit(work, j) for j in drive_jobs] + [pw.submit(work, j) for j in web_jobs]
        for f in fs:
            f.result()
    print(f"done: {done[0]} files in {time.time() - t0:.0f}s; failed {len(failed)}: {failed[:10]}", flush=True)


if __name__ == "__main__":
    main()
