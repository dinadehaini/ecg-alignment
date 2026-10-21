# Phase 1 vs. the official code (Rose-STL-Lab/tvl-alignment), checked 2026-10-19

I read the released repo's source (commit HEAD as of 2026-10-19) on GitHub: `src/inf/geometric_metrics.py`, `src/inf/utils.py`, `src/inf/eval.py`, `src/main/{config,dataset,models,losses,train}.py`. I didn't run it. **The PTB-XL-specific script (`eval_ptbxl.py`, called by `src/data/ptbxl/eval/eval.sh`) and the PTB-XL training config are not in the repo.**

## Metrics: our implementation vs. theirs
| Metric | Theirs | Ours | Same? |
|---|---|---|---|
| Procrustes | center both; Kabsch rotation with reflection fix; disparity = ‖X̃R−Ỹ‖² / ‖Ỹ‖² (normalized by the target only, no scaling); on L2-normalized embeddings, full set | same formula, same normalization; on a subset of ≤2000 | **formula identical** |
| RBF CKA | median heuristic over pooled distances from 200 random points; random subset ≤1500 | median over all pooled pairs; subset ≤2000 | near-identical; bandwidth estimate differs slightly |
| Mutual kNN | k=5, cosine, self excluded; subset ≤1500 | k=5, cosine, self excluded | same |
| Retrieval R@k / MRR | exact index match, **no handling of duplicate captions** | exact index match | same |
| Cosine margin | matched − mean mismatched per row | same | same |

**So both phase-1 observations hold under their own metric code:**
1. **Duplicate reports.** Their retrieval also counts only the exact index as correct, so the cap of about 56% on TS↔TXT retrieval from duplicate reports applies to their numbers too. This is a property of PTB-XL plus the metric. It doesn't depend on our code.
2. **Procrustes above 1, and higher after training than before.** Their formula is the same one we use, normalized only by the target's spread. On unit-norm embeddings, the disparity rises whenever the target modality's embeddings are more concentrated than the source's. Duplicate captions would make text embeddings more concentrated ⚠ (a hypothesis, not tested). So this is "how the metric is defined," not a bug in our code. Whether it misleads is untested. The cheap test is to compute a scale-symmetric version (e.g., scipy's `procrustes`) on the cached embeddings.

## Training and preprocessing: where the paper, the code and our run differ
| Setting | Paper text | Released code | Our phase 1 |
|---|---|---|---|
| TS length fed to encoder | 5,000, no truncation (App. G.2) | `ts_length` **asserted = 256**, pad or truncate to the first 256 samples | 5,000 as 10 overlapping 512-step windows, mean-pooled |
| Batch size | effective 256 (App. G.1) | config 32 or 128; README example 32 × grad_acc 4. **Gradient accumulation doesn't add in-batch negatives**, so a contrastive batch of 32 | 256 |
| Head dropout | not stated ⚠ | 0.05 | 0.1 |
| Image size | 384² (per our notes) ⚠ | `image_size` 224 | 384² |
| Stopping | 50 epochs | 50 epochs, early stopping with patience 5 on val loss, grad clip 1.0, wd 1e-5 | 50 epochs, cosine schedule |
| Lead, split | not specified | not in repo | lead I (lead II as a check), patient-disjoint 16k/1k/4k |

**What this means:**
- We followed the *paper*. The code disagrees with the paper on TS length and batch size, and the PTB-XL path isn't released. So we **can't tell which settings produced the paper's PTB-XL bars**.
- The most likely sources of our "slightly stronger text alignment" are batch size (more negatives) and how the 5,000-step series reaches MOMENT. Both are plausible, and neither is confirmed.
- This is a legitimate, specific question to ask the author.
