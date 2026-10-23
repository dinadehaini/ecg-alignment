# Frozen predictions for the MIMIC-IV-ECG replication

Written 2026-10-22, before any MIMIC-IV-ECG model was trained or any MIMIC labels were extracted. The PTB-XL
evidence behind them is exploratory (PTB-XL was examined extensively); MIMIC is the confirmation test.

## Protocol (frozen)
- Same frozen encoders (MOMENT-base, DINOv2-base, Qwen3-Embedding-0.6B), lead I, head architecture,
  training hyperparameters, and the 9 unsupervised configurations (InfoNCE; soft targets, mix 0.5 / 1.0 x
  soft-tau 0.02 / 0.05 / 0.1 / 0.2) at training temperature 0.2. Label-target arms reported separately.
- Patient-disjoint subset chosen without looking at model performance, comparable in size to PTB-XL.
- Text: the machine-generated report lines (report_0..report_17). Labels: extracted from those same reports
  with a concept inventory and extraction rules (negation, uncertainty, missing mentions) frozen before any
  model is evaluated. Claims are about recovery of report-derived concepts, not independent clinical validity.
- Readouts: report prototypes (single-concept training reports) and hand-written class descriptions
  (fixed template policy, written before scoring), each uncorrected and with text-side centering (mean of
  MIMIC training text embeddings). Selectors as in PTB-XL: validation alignment metrics, duplicate-aware
  retrieval, 25 / 50 / 100 labeled validation patients, full labeled validation.
- Primary endpoint: corrected (text-centered) selection regret for hand-written descriptions.

## Predictions
1. The soft-target advantage over InfoNCE is larger for hand-written descriptions than for report-derived
   prototypes.
2. That difference persists after text-side centering and with matched prototype sizes.
3. Within held-out-patient report prototypes, previously unseen wording does not show a larger gap than
   wording seen in training (as on PTB-XL). Identical strings give identical query vectors, so any
   train-vs-held-out difference reflects string selection, frequency and combination, not patient source.
4. Exploratory only (not a primary prediction; amended 2026-10-22 before any MIMIC work, because frozen-text
   proximity did not explain the PTB-XL gap): whether the gap grows with a query's distance from the
   training reports in the original frozen text-encoder space. (Distance can also track clinical meaning.)
5. Duplicate-aware retrieval selects within .01 of the full labeled-validation selector.

What would count against the PTB-XL account: no soft-target advantage for hand-written descriptions after
centering (prediction 1 fails), or the advantage being the same for report prototypes and hand-written text
(prediction 1 holds only uncorrected).

## Frozen label mapping, prompts and subset (2026-10-23, before any waveform download or MIMIC model)
- Label mapping: `src/mimic_labels.py` (statement-level POS / UNC / NEG rules reviewed against every
  concept-related statement with counts; report-level combination rule: conflict -> excluded, any unequivocal
  positive -> positive, otherwise uncertain -> excluded, otherwise report-negative). Labels are machine-report-
  derived targets: the primary result is replication of query-dependent transfer against machine-report labels;
  clinical diagnostic validity is outside its scope.
- Concepts: normal ECG; atrial fibrillation; reported infarct pattern (old / age-undetermined patterns
  included; no acute-MI wording); intraventricular conduction abnormality (bundle and fascicular blocks,
  intraventricular conduction delay; AV blocks excluded); left ventricular hypertrophy; ST-T changes;
  sinus bradycardia; sinus tachycardia.
- Prompts: `src/prompts_mimic.json`; primary readout = clinical-description ensemble; report-style family is
  secondary (written after viewing the statement vocabulary).
- Subset: all explicitly paced ECGs excluded; patients shuffled with seed 0, all remaining ECGs of each selected
  patient kept, patients added until 21,000 ECGs; patient-disjoint split: 21,011 ECGs, 4,358 patients; train
  16,005 / validation 1,001 / test 4,005 ECGs. Test set: 2,284 unique reports (0.570); 51.6% of test ECGs share
  their exact report with another test ECG.
- Eligible test counts (negative / positive / excluded-uncertain; no conflicts or contradictions occurred):
  normal ECG 3214 / 589 / 202; atrial fibrillation 3510 / 482 / 13; reported infarct pattern 3075 / 397 / 533;
  intraventricular conduction abnormality 3232 / 717 / 56; LVH 3646 / 245 / 114; ST-T changes 2396 / 1592 / 17;
  sinus bradycardia 3489 / 516 / 0; sinus tachycardia 3602 / 392 / 11.
