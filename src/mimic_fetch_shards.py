"""Download the frozen subset's WFDB files from PhysioNet in resumable shards saved to Drive.

Records are taken in subset order and grouped into shards of 1,000 ECGs. For each shard not yet on Drive:
download its .hea/.dat files to local disk (retries with backoff; .dat must be 120,000 bytes), pack them into
one uncompressed tar, copy it to Drive under a temporary name and rename it when complete. A runtime death
loses at most the shard in progress; rerunning skips finished shards. Records that still fail after retries
are listed in the shard's manifest (never replaced).
Usage: python src/mimic_fetch_shards.py --ids mimic_ids.csv --out /content/drive/MyDrive/mimic_shards
"""
from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

URL = "https://physionet.org/files/mimic-iv-ecg/1.0/files/p{s4}/p{s}/s{t}/{t}{ext}"
SHARD = 1000


def fetch(url, dst, attempts=6):
    tmp = dst.with_suffix(dst.suffix + ".part")
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, dst)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 ** k + 1)
    return False


def ok(p, ext):
    return p.exists() and (os.path.getsize(p) == 120000 if ext == ".dat" else os.path.getsize(p) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="csv with subject_id, study_id in frozen subset order")
    ap.add_argument("--out", required=True, help="Drive folder for shard tars")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    ids = pd.read_csv(a.ids)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_shards = (len(ids) + SHARD - 1) // SHARD
    for i in range(n_shards):
        tar_path = out / f"shard_{i:02d}.tar"
        if tar_path.exists():
            print(f"shard {i:02d} already on Drive", flush=True)
            continue
        t0 = time.time()
        part = ids.iloc[i * SHARD:(i + 1) * SHARD]
        tmpdir = Path(f"/content/shard_tmp_{i:02d}")
        tmpdir.mkdir(exist_ok=True)
        jobs = [(URL.format(s4=str(s)[:4], s=s, t=t, ext=e), tmpdir / f"{t}{e}", e)
                for s, t in zip(part.subject_id, part.study_id) for e in (".hea", ".dat")]
        jobs = [j for j in jobs if not ok(j[1], j[2])]
        with ThreadPoolExecutor(a.threads) as ex:
            list(ex.map(lambda j: fetch(j[0], j[1]), jobs))
        failed = [f"{t}{e}" for t in part.study_id for e in (".hea", ".dat") if not ok(tmpdir / f"{t}{e}", e)]
        (tmpdir / "manifest.txt").write_text(f"records {len(part)}\nfailed {len(failed)}\n" + "\n".join(failed))
        local_tar = Path(f"/content/shard_{i:02d}.tar")
        with tarfile.open(local_tar, "w") as tf:
            for f in sorted(tmpdir.iterdir()):
                if not f.name.endswith(".part"):
                    tf.add(f, arcname=f.name)
        tmp_drive = out / f"shard_{i:02d}.tar.tmp"
        shutil.copyfile(local_tar, tmp_drive)
        os.replace(tmp_drive, tar_path)
        shutil.rmtree(tmpdir)
        local_tar.unlink()
        print(f"shard {i:02d}/{n_shards - 1} saved ({len(part)} ECGs, {len(failed)} files failed) in {time.time() - t0:.0f}s",
              flush=True)
    print("all shards on Drive", flush=True)


if __name__ == "__main__":
    main()
