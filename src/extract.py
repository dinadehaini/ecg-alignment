"""Step 1 of phase 1: run the frozen encoders once and cache their embeddings.

The paper keeps all encoders frozen and does no data augmentation, so caching encoder
outputs and training the heads on the cache gives the same optimisation problem as running
the encoders inside the training loop, at a small fraction of the GPU time.

Usage (Colab):
    python src/extract.py --ptbxl /content/ptbxl --out /content/drive/MyDrive/rose-yu-ecg/emb \
        --config A
By default the script refuses to run without CUDA (pass --allow-cpu for the local smoke test).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SEQ_LEN, load_and_render, load_metadata, load_signal, make_splits  # noqa: E402

# Paper configs on PTB-XL (Fig. "PTB-XL vs MIMIC"). A and D are the two small ones.
CONFIGS = {
    "A": {"ts": "moment-base", "img": "dinov2-base", "txt": "qwen3-emb-0.6b"},
    "D": {"ts": "chronos-t5-base", "img": "siglip2-base-384", "txt": "qwen3-emb-0.6b"},
}
HF_IDS = {
    "moment-base": "AutonLab/MOMENT-1-base",
    "chronos-t5-base": "amazon/chronos-t5-base",
    "dinov2-base": "facebook/dinov2-base",
    "siglip2-base-384": "google/siglip2-base-patch16-384",
    "qwen3-emb-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}

# MOMENT and Chronos-T5 take at most 512 input steps. The paper says series are used at full
# length with no truncation but does not say how; we cover the full 5000 steps with 10
# overlapping 512-step windows and mean-pool the window embeddings (see PAPER_NOTES.md).
WIN, N_WIN = 512, 10
WIN_STARTS = np.round(np.linspace(0, SEQ_LEN - WIN, N_WIN)).astype(int)


def windows(x: np.ndarray) -> np.ndarray:
    """(B, 5000) -> (B, N_WIN, 512)"""
    return np.stack([x[:, s:s + WIN] for s in WIN_STARTS], axis=1)


# ----------------------------------------------------------------------------- encoders
class MomentEnc:
    def __init__(self, device):
        from momentfm import MOMENTPipeline
        self.m = MOMENTPipeline.from_pretrained(HF_IDS["moment-base"],
                                                model_kwargs={"task_name": "embedding"})
        self.m.init()
        self.m.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        B = x.shape[0]
        w = torch.from_numpy(windows(x)).reshape(B * N_WIN, 1, WIN).to(self.device)
        mask = torch.ones(B * N_WIN, WIN, device=self.device)
        e = self.m(x_enc=w, input_mask=mask).embeddings  # (B*N_WIN, d), mean over patches
        return e.reshape(B, N_WIN, -1).mean(1)


class ChronosEnc:
    def __init__(self, device):
        from chronos import ChronosPipeline
        self.p = ChronosPipeline.from_pretrained(HF_IDS["chronos-t5-base"], device_map=device,
                                                 torch_dtype=torch.float32)
        self.device = device

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        B = x.shape[0]
        w = torch.from_numpy(windows(x)).reshape(B * N_WIN, WIN)
        e, _ = self.p.embed(w)  # (B*N_WIN, WIN+1, d); last position is EOS
        e = e[:, :-1].mean(1).to(self.device)
        return e.reshape(B, N_WIN, -1).mean(1)


class Dinov2Enc:
    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(HF_IDS["dinov2-base"])
        self.m = AutoModel.from_pretrained(HF_IDS["dinov2-base"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, imgs) -> torch.Tensor:
        # Paper resizes images to 384x384; keep that (no 224 centre crop).
        px = self.proc(images=imgs, return_tensors="pt", do_center_crop=False,
                       size={"height": 384, "width": 384})["pixel_values"].to(self.device)
        return self.m(pixel_values=px).pooler_output  # layer-normed [CLS]


class Siglip2Enc:
    def __init__(self, device):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(HF_IDS["siglip2-base-384"])
        self.m = AutoModel.from_pretrained(HF_IDS["siglip2-base-384"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, imgs) -> torch.Tensor:
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device)
        return self.m.get_image_features(pixel_values=px)


class Qwen3EmbEnc:
    def __init__(self, device):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(HF_IDS["qwen3-emb-0.6b"], padding_side="left")
        self.m = AutoModel.from_pretrained(HF_IDS["qwen3-emb-0.6b"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, texts) -> torch.Tensor:
        # Reports are documents, so no query instruction; last-token pooling as recommended.
        b = self.tok(list(texts), padding=True, truncation=True, max_length=512,
                     return_tensors="pt").to(self.device)
        return self.m(**b).last_hidden_state[:, -1]


ENCODERS = {"moment-base": MomentEnc, "chronos-t5-base": ChronosEnc,
            "dinov2-base": Dinov2Enc, "siglip2-base-384": Siglip2Enc,
            "qwen3-emb-0.6b": Qwen3EmbEnc}
MODALITY_OF = {"moment-base": "ts", "chronos-t5-base": "ts", "dinov2-base": "img",
               "siglip2-base-384": "img", "qwen3-emb-0.6b": "txt"}


# ----------------------------------------------------------------------------- main
def check_device(enc, device: str, out: torch.Tensor):
    """Hard check that the encoder weights and outputs really are on the GPU."""
    mod = getattr(enc, "m", None) or getattr(getattr(enc, "p", None), "model", None)
    if mod is not None:
        p_dev = next(mod.parameters()).device.type
        assert p_dev == device, f"weights on {p_dev}, expected {device}"
    assert out.device.type == device, f"output on {out.device.type}, expected {device}"
    if device == "cuda":
        print(f"    [cuda ok] {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated", flush=True)


def batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", required=True, help="folder with ptbxl_database.csv, records500/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="A", choices=list(CONFIGS))
    ap.add_argument("--encoders", nargs="*", help="subset of encoders (default: all in config)")
    ap.add_argument("--lead", default="I")
    ap.add_argument("--limit", type=int, default=0, help="records per split (smoke test)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processes for loading + plotting ECGs")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif args.allow_cpu:
        device = "cpu"
        print("WARNING: running on CPU (smoke test only)")
    else:
        sys.exit("CUDA not available. In Colab: Runtime > Change runtime type > GPU. "
                 "(--allow-cpu exists only for the local smoke test.)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = make_splits(load_metadata(args.ptbxl))
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit)
    df[["patient_id", "filename_hr", "report", "split"]].to_csv(out / "split.csv")
    print({s: int((df.split == s).sum()) for s in ("train", "val", "test")})

    names = args.encoders or list(CONFIGS[args.config].values())
    for name in names:
        mod = MODALITY_OF[name]
        t0 = time.time()
        print(f"== {name} ({mod}) on {device}", flush=True)
        enc = ENCODERS[name](device)
        pool = mp.Pool(args.workers) if mod == "img" and args.workers > 1 else None
        for split in ("train", "val", "test"):
            path = out / f"{split}_{mod}_{name}_lead{args.lead}.npy" if mod != "txt" \
                else out / f"{split}_{mod}_{name}.npy"
            if path.exists():
                print(f"  {path.name} exists, skipping")
                continue
            sub = df[df.split == split]
            embs = []
            for k, sl in enumerate(batches(len(sub), args.bs)):
                rows = sub.iloc[sl]
                if mod == "txt":
                    inp = rows.report.tolist()
                elif mod == "ts":
                    inp = np.stack([load_signal(args.ptbxl, f, args.lead) for f in rows.filename_hr])
                else:
                    jobs = [(args.ptbxl, f, args.lead) for f in rows.filename_hr]
                    inp = pool.map(load_and_render, jobs) if pool else list(map(load_and_render, jobs))
                e = enc(inp)
                if k == 0:
                    check_device(enc, device, e)
                embs.append(e.float().cpu().numpy())
                if k % 50 == 0:
                    print(f"  {split} {sl.stop}/{len(sub)}  {time.time() - t0:.0f}s", flush=True)
            arr = np.concatenate(embs)
            assert np.isfinite(arr).all(), f"non-finite values in {path.name}"
            np.save(path, arr)
            print(f"  saved {path.name} {arr.shape}")
        if pool:
            pool.close()
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"== {name} done in {time.time() - t0:.0f}s")
    json.dump({"config": args.config, "encoders": {n: HF_IDS[n] for n in names},
               "lead": args.lead, "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "window": [WIN, N_WIN], "torch": torch.__version__},
              open(out / f"extract_meta_{args.config}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
