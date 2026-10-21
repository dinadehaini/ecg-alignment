"""Download PTB-XL v1.0.3 (public, CC BY 4.0) from PhysioNet's official AWS open-data mirror.

physionet.org itself can be very slow; the same files are mirrored by PhysioNet at
s3://physionet-open/ptb-xl/1.0.3/ (no account needed). We fetch only the metadata and the
500 Hz records (records500/, ~2.6 GB, 21,799 records), checking each file against
PhysioNet's SHA256SUMS.txt.

Usage:
    python src/download_ptbxl.py --out /content/ptbxl              # everything (Colab)
    python src/download_ptbxl.py --out data/ptbxl --only 00001 00002  # a few records
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
META = ["ptbxl_database.csv", "scp_statements.csv", "SHA256SUMS.txt", "LICENSE.txt"]


def fetch(rel: str, out: Path, sums: dict, retries: int = 4) -> str:
    dst = out / rel
    if dst.exists() and (rel not in sums or sha(dst) == sums[rel]):
        return "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(BASE + rel, timeout=60) as r:
                data = r.read()
            if rel in sums and hashlib.sha256(data).hexdigest() != sums[rel]:
                raise IOError(f"checksum mismatch for {rel}")
            dst.write_bytes(data)
            return "ok"
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"failed {rel}: {e}") from e
    return "?"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="ecg_ids (zero-padded 5 digits) to fetch")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in META:
        fetch(m, out, {})
    sums = {}
    for line in (out / "SHA256SUMS.txt").read_text().splitlines():
        h, rel = line.split(maxsplit=1)
        sums[rel.strip()] = h
    files = sorted(r for r in sums if r.startswith("records500/") and r.endswith((".dat", ".hea")))
    if args.only:
        keep = {f"{int(i):05d}_hr" for i in args.only}
        files = [f for f in files if Path(f).stem in keep]
    print(f"fetching {len(files)} files with {args.threads} threads")
    done = 0
    with ThreadPoolExecutor(args.threads) as ex:
        for _ in ex.map(lambda r: fetch(r, out, sums), files):
            done += 1
            if done % 4000 == 0:
                print(f"  {done}/{len(files)}", flush=True)
    n_rec = len(list((out / "records500").rglob("*_hr.dat")))
    print(f"done: {n_rec} records500 .dat files in {out}")


if __name__ == "__main__":
    main()
