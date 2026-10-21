# Compute

Every script checks for CUDA. `extract.py`, `train_heads.py` and `adapt.py` **exit** if CUDA is
missing (pass `--allow-cpu` only for smoke tests). On the first batch they assert that model
weights and outputs are on `cuda`, and they log the GPU name, e.g.
`[cuda ok] Tesla T4, 2.41 GiB allocated`.

## What each step costs (one NVIDIA T4)

| Step | Time |
|---|---|
| Download PTB-XL records500 (~2.6 GB) from PhysioNet's AWS mirror | 3–10 min |
| Frozen encoders, config A (MOMENT-B, DINOv2-B, Qwen3-Emb-0.6B) | ~50 min (MOMENT 15, DINOv2 26, Qwen 8) |
| Projection heads on cached embeddings (`train_heads.py`, `train_heads_v2.py`) | ~1 min per run |
| LoRA arm of `adapt.py` (r = 8, GradCache, fp16 encoder) | ~205 s/epoch, ≤ 50 epochs |

The encoders are frozen in the baseline, so each runs **once** and its embeddings are cached
(~250 MB). With no augmentation, training heads on the cache is the same optimization problem as
running the encoders inside the training loop.

Each encoder writes its own `.npy` files and reruns skip finished files. `adapt.py` checkpoints
every epoch, so an interrupted run resumes where it stopped.

## Where to run
- **Google Colab** (free T4 or Colab Pro): `notebooks/phase1_colab.ipynb`.
- **Any CUDA machine:** `pip install torch -r requirements-colab.txt`, then
  `pip install momentfm==0.1.4 --no-deps`, then run the commands in `README.md`.

**Known environment issue (Oct 2026):** newer Colab images ship `torchao` 0.10, which peft 0.17's
version check rejects when creating LoRA layers. We don't use torchao, so run
`pip uninstall -y torchao` first.
