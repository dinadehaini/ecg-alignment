# Measuring time-series–text alignment on clinical ECGs

A re-implementation and extension of the frozen-encoder, projection-head baseline from
**Yashwante & Yu, "Time Series, Vision, and Language: Exploring the Limits of Alignment in
Contrastive Representation Spaces" (ICML 2026, [arXiv 2602.19367](https://arxiv.org/abs/2602.19367))**,
applied to the PTB-XL 12-lead ECG dataset. The official repository,
[Rose-STL-Lab/tvl-alignment](https://github.com/Rose-STL-Lab/tvl-alignment), doesn't include
the PTB-XL pipeline, so this one was built from the paper's appendix. Where the paper, the
official code and this implementation differ is documented in `results/official_code_diff.md`.

## What's here
1. **Reproduction.** A trimodal setup (ECG signal, rendered plot, report text) with frozen
   MOMENT-base, DINOv2-base and Qwen3-Embedding-0.6B encoders. Only the projection heads train,
   with symmetric InfoNCE. Scored with the paper's metrics: cosine margin, Procrustes, RBF-CKA,
   mutual kNN@5 and R@k. Leads I and II, 3 seeds.
2. **Label-level evaluation.** Zero-shot diagnostic AUROC and same-diagnosis retrieval on PTB-XL's
   five diagnostic superclasses, against a supervised linear-probe reference.
3. **Fairer metrics** (`src/metrics_v2.py`): duplicate-aware retrieval, a scale-symmetric
   Procrustes, and scores on a deduplicated subset. 51% of PTB-XL test reports are exact duplicates.
4. **Objective variants** (`src/train_heads_v2.py`): standard InfoNCE, duplicate-masked,
   soft targets from text similarity, and a label oracle.
5. **Matched-budget adaptation pilot** (`src/adapt.py`): LoRA inside the time-series encoder vs. a
   deeper head, at the same number of extra trainable parameters (~295k). LoRA training uses
   GradCache, which is verified to match full-batch gradients to 1e-6.

Results and readings: **`results/phase1_results.md`**. Paper notes and the gaps we had to fill:
`PAPER_NOTES.md`. Compute and CUDA checks: `COMPUTE.md`.

## Running it
```bash
python src/download_ptbxl.py --out data/ptbxl                       # PTB-XL v1.0.3, SHA256-checked
python src/extract.py --ptbxl data/ptbxl --out results/emb --config A   # frozen encoders -> cache
python src/train_heads.py --emb results/emb --config A --seeds 0 1 2    # paper baseline
python src/compare.py --emb results/emb --out results/phase1_results_auto.md
python src/label_eval.py --emb results/emb --ptbxl data/ptbxl --config A
python src/train_heads_v2.py --emb results/emb --ptbxl data/ptbxl --loss infonce dupmask soft label
python src/adapt.py --emb results/emb --ptbxl data/ptbxl --arm head deep_head lora --seeds 0 1 2
```
On Colab, `notebooks/phase1_colab.ipynb` runs the same steps. `tests/smoke_test.py` runs the
whole pipeline on CPU on a few records. It checks the code paths only, not the numbers.

PTB-XL is not redistributed here. It's public under CC BY 4.0 (Wagner et al., 2020); download it
from PhysioNet.

## Layout
```
src/download_ptbxl.py   PTB-XL v1.0.3 from PhysioNet's AWS mirror, SHA256-checked
src/data.py             metadata, patient-disjoint 16k/1k/4k split, lead selection, plot rendering
src/extract.py          frozen encoders -> cached embeddings (configs A and D)
src/train_heads.py      paper's projection heads + trimodal symmetric InfoNCE
src/metrics.py          paper metrics (App. E): cosine margin, Procrustes, RBF-CKA, mutual kNN, R@k
src/compare.py          ours vs. the paper's PTB-XL figure (digitized)
src/label_eval.py       zero-shot diagnostic AUROC, same-diagnosis retrieval, linear-probe reference
src/metrics_v2.py       duplicate-aware retrieval, symmetric Procrustes, deduplicated scoring
src/train_heads_v2.py   objective variants (infonce / dupmask / soft / label-oracle)
src/adapt.py            matched-budget LoRA vs. deeper head vs. head-only (GradCache, resumable)
```

## License
Code under the MIT License (see `LICENSE`). Cite the original paper if you use this.
