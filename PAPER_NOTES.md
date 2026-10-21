# Paper notes: Yashwante & Yu, ICML 2026 (arXiv 2602.19367 v1, 22 Feb 2026)

Read from the arXiv LaTeX source on 2026-10-19.

## Is the code released?
**Yes. My first check was wrong.** The official repo is public at
https://github.com/Rose-STL-Lab/tvl-alignment (verified 2026-10-20). It isn't linked from the arXiv
page, which is why my first search missed it. Our phase-1 code is a re-implementation from the
paper's appendix, not a rerun of theirs. Their PTB-XL script and config are **not** in the repo.
See `results/official_code_diff.md` for where the paper, their code and our run differ: their code
uses TS length 256 and contrastive batch 32, against the paper's 5000 and 256.

## What the paper actually does (it matters for "reproduce")
- **Trimodal, not TS–text only.** Each sample is a triplet: the raw series (TS), a rendered line
  plot of it (IMG), and text (TXT). Three frozen encoders. One projection head per modality.
  The loss is symmetric InfoNCE summed with equal weights over TS–IMG, TS–TXT and IMG–TXT.
  So the baseline needs a vision encoder too. We include it to match their numbers.
- **Head** (App. G.4, their "improved" default): Linear(d_in→m) → LayerNorm → GELU → Dropout →
  Linear(m→1024), m = max(3·1024, 768) = 3072.
- **Training** (App. G.1): AdamW, lr 1e-4, wd 1e-5, cosine decay with 2% warmup, fixed τ = 0.2,
  grad-clip 1.0, batch 256, 50 epochs, early stopping patience 5, identical across all configs.
- **Pooling** (App. D.1): TS = mean over temporal hidden states; images = [CLS] or native
  embedding; Qwen3-Embedding = last token. Images resized to 384×384; text ≤ 512 tokens.
- **Metrics** (App. E), on the held-out test set, L2-normalized embeddings:
  - cosine matched / mismatched / **margin** (full test set);
  - **Procrustes** disparity, normalized by ‖Ỹ‖², Kabsch with reflection fix;
  - **CKA** with an RBF kernel, bandwidth from the median heuristic over both modalities;
  - **mutual kNN**, k = 5, cosine, self excluded;
  - **retrieval** R@1/5/10, both directions, ranked over the full test set.
  - Procrustes, CKA and mutual kNN use a random subset of ≤ 2000 pairs (seed 42).
- **PTB-XL** (App. C.4): 10 s at 500 Hz → length-5000 univariate series; German reports used
  verbatim; 21,000 triplets with non-empty reports; split sizes matched to CaTS. Their MIMIC
  split is 16k train / 4k test / 1k val, disjoint by subject. Only configs with Qwen text
  encoders were run on PTB-XL (Qwen is multilingual).

## The paper's PTB-XL numbers (all of them)
The paper prints **no PTB-XL table**. Its only PTB-XL result is one bar chart ("PTB-XL vs MIMIC")
with **cosine margin and Procrustes, for TS–TXT and IMG–TXT only**, over 4 configs. I read the
bar heights from the vector PDF (`figs/ptbxlvsmimic_1.pdf`; bar rectangles vs. tick positions,
about ±0.003):

| Config | Encoders (IMG + TXT + TS) | TS–TXT margin | TS–TXT Procrustes | IMG–TXT margin | IMG–TXT Procrustes |
|---|---|---|---|---|---|
| **A** | DINOv2-B + Qwen3-Emb-0.6B + MOMENT-B | 0.322 | 1.221 | 0.350 | 1.222 |
| B | DINOv2-G + Qwen3-Emb-4B + MOMENT-L | 0.336 | 1.176 | 0.372 | 1.184 |
| C | SigLIP-L + Qwen3-Emb-4B + Chronos-L | 0.332 | 1.235 | 0.397 | 1.121 |
| **D** | SigLIP2-B + Qwen3-Emb-0.6B + Chronos-B | 0.337 | 1.220 | 0.382 | 1.141 |

There are **no PTB-XL values for CKA, mutual kNN, retrieval, or any TS–IMG metric** in the paper.
We compute all of them anyway. For those, the closest published reference is MIMIC: Table 2
gives TS↔IMG R@1/5/10 = 31.31/61.84/73.62 % for config A and 20.33/45.79/58.69 % for D.

**Phase 1 target: config A** (smallest; matches row 10 of their Table 4). Config D is optional.
Note: the brief said "MOMENT-small or Chronos-small". The paper's smallest PTB-XL configs use
**MOMENT-base (125M)** and **Chronos-T5-base (201M)**, so we use those to be comparable.

## Gaps: choices the paper leaves open (we pick a default and say so)
1. **Which ECG lead.** Not stated. Default: **lead I** (channel 0, what `signal[:, 0]` gives).
   `--lead II` is one flag away if the numbers are far off.
2. **Length 5000 vs. 512-step encoders.** MOMENT and Chronos-T5 take ≤ 512 steps, and the paper
   says "no truncation" without saying how. We use 10 overlapping 512-step windows that cover all
   5000 steps, and mean-pool the window embeddings.
3. **Which 21,000 records and how they're split.** Not stated for PTB-XL. We take all records
   with a non-empty report, shuffle patients (seed 42), and fill test (4000), val (1000) and
   train (16000), keeping each patient in one split.
4. **Plot style.** "Rendered with axes and units". We draw a black line with Time (s) and
   Amplitude (mV) axes, at 384×384.
5. **Dropout rate, early-stopping criterion, CKA bandwidth form.** Not given. We use dropout 0.1,
   val-loss early stopping, and γ = 1/(2·median²).
6. **Seeds.** The paper reports single runs. We run 3 seeds and report mean ± sd.
7. **Caching.** We run the frozen encoders once and train the heads on cached embeddings. With
   frozen encoders and no augmentation, this is the same optimization problem.

## Prior-art check (redone 2026-10-19): LoRA vs. projection heads for contrastive TS–text alignment
- **Still no matched-budget, head-to-head study** of LoRA vs. head-only vs. partial unfreezing for
  *contrastive* TS–text (or TS–image) alignment that measures alignment geometry (CKA, mutual kNN,
  Procrustes). Searched arXiv/web plus the Semantic Scholar citations of 2602.19367: only 1 citing
  paper, and it's unrelated (SpatialSkill, 2609.34124).
- **Closest new item: GALA** (Zhang et al., UNC, arXiv 2608.13741, v3 13 Oct 2026). For
  text-to-time-series *generation*, Stage 1 aligns a text encoder and a TS foundation model with
  InfoNCE plus a generative auxiliary loss. Its ablation compares FT-Full, FT-Head (frozen backbones)
  and LoRA (r = 64): adapting the backbone helps, and FT-Head can't use the auxiliary loss.
  Differences from idea A: a generation objective, ETTh1 rather than ECG, budgets not matched, no
  geometry metrics, a single training seed. **Cite it** as related evidence that backbone
  adaptation matters. It does not cover idea A.
- Still as in the brief: CAMEL (2602.15677) and ECG-LLM (2607.16323) use projector-then-LoRA
  staging for *generative* ECG-LLMs. Different objective.
- The paper itself names "end-to-end fine-tuning or parameter-efficient adaptation" as future work
  (App. G.4 and Limitations), so idea A extends a limitation the authors state themselves.

## What was verified locally (2026-10-19, CPU only, no real numbers)
- `tests/smoke_test.py` passes: metric sanity checks; split is exactly 16,000/1,000/4,000 with
  **0 patients shared** across splits; all five real encoders load and embed real PTB-XL records
  (MOMENT-B, DINOv2-B, Qwen3-Emb-0.6B; Chronos-T5-B, SigLIP2-B); training and compare run.
- On synthetic triplets with a shared 32-d latent (16k train), trained heads reach R@1 ≈ 1.0 vs.
  ≈ 0.0 for untrained heads. So the training loop learns. It says nothing about PTB-XL numbers.
- Interpretation aid: for *random* isotropic unit vectors at N = 2000, d = 1024, our Procrustes
  is 0.85, CKA 0.46 and mutual kNN 0.002. The paper's PTB-XL Procrustes values (≈ 1.2) are above
  that isotropic-random level. That's possible because real embeddings are anisotropic, and the
  disparity isn't normalized by ‖X̃‖. So read Procrustes against our untrained-head floor, not
  against 0. Worth asking the authors about only if our numbers disagree.
