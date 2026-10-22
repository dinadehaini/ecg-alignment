# Stage 0 results (finite latent world; sigma 1.0 balanced prior unless noted; 2 seeds; 40k test examples per run)

Metric: exact excess Brier E[(p - q)^2] against the oracle posterior q (lower is better).

## Learning curves (calibrated methods)

| Group | Predictor 2k / 8k / 32k / 128k | Contrastive | Distillation |
|---|---|---|---|
| all | .098 / .043 / .033 / .023 | .100 / .058 / .029 / .023 | .093 / .037 / .027 / .026 |
| identifiable, posterior-confident | .234 / .083 / .066 / .041 | .227 / .094 / .030 / .020 | .219 / .060 / .046 / .043 |
| identifiable, posterior-ambiguous (noise control) | .067 / .060 / .052 / .048 | .067 / .067 / .055 / .051 | .068 / .059 / .051 / .048 |
| structurally non-identifiable | .012 / .012 / .006 / .006 | .022 / .033 / .022 / .019 | .014 / .017 / .008 / .010 |
| trap, seen | .249 / .038 / .005 / .002 | .244 / .066 / .007 / .004 | .247 / .010 / .005 / .003 |
| trap, held-out | .245 / .051 / .010 / .005 | .242 / .139 / .023 / .017 | .244 / .020 / .010 / .007 |

At 128k, contrastive minus predictor (paired 95% CI): non-identifiable +.0134 [+.0131, +.0137];
identifiable-confident -.0207 [-.0217, -.0198]; noise control +.0035 [+.0024, +.0047]; trap seen
+.0020 [+.0012, +.0029]; trap held-out +.0124 [+.0105, +.0142]. Noise sweep (32k): the non-identifiable
gap is stable across sigma (.020 / .015 / .014 at sigma .5 / 1 / 2); the noise-control gap is ~0.

## Mechanism: omitted claim prior (analytic check)

Under the InfoNCE optimum the score estimates log P(c|x)/P_caption(c) + const(x). Each latent state
here has exactly 6 true claims and the caption is one of them drawn uniformly, so P_caption(c) is
proportional to the claim's prevalence P(c) (except for the 4 held-out (claim, view) pairs, where the
choice set has 5). The predicted correction is therefore
logit P(c|x) = s(x,c) - s(x,not c) + log P(c)/P(not c), with s in training-logit units and P(c) known
from the simulator: no fitted parameters and no labels.

Exact excess Brier, seed 0 / seed 1 (per-claim fits are cross-fitted on test halves: a diagnostic of
correctability, not a held-out result):

| Condition, group | Raw | Global Platt | Analytic prior | Per-claim intercept | Per-claim slope+intercept | Predictor |
|---|---|---|---|---|---|---|
| 128k, structurally non-identifiable | .0227 / .0162 | .0213 / .0171 | .0087 / .0052 | .0087 / .0054 | .0090 / .0064 | .0075 / .0041 |
| 128k, all | .0251 / .0213 | .0247 / .0216 | .0172 / .0150 | .0172 / .0150 | .0173 / .0152 | .0155 / .0306 |
| 128k, unseen channel masks | .0349 / .0276 | .0344 / .0279 | .0233 / .0205 | .0231 / .0204 | .0225 / .0198 | .0202 / .0498 |
| 128k, trap held-out | .0254 / .0083 | .0274 / .0074 | .0254 / .0083 | .0252 / .0082 | .0205 / .0055 | .0066 / .0032 |
| prior 0.9 (32k), non-identifiable posterior-confident | .1283 / .1333 | .1251 / .1330 | .0028 / .0014 | .0027 / .0014 | .0022 / .0009 | .0066 / .0065 |
| prior 0.9 (32k), structurally non-identifiable | .0415 / .0410 | .0441 / .0429 | .0064 / .0068 | .0058 / .0062 | .0081 / .0086 | .0082 / .0126 |

The analytic prior correction alone reproduces the fitted per-claim intercept almost exactly, and
per-claim slopes add nothing on non-identifiable claims: the gap is explained by the omitted claim
prior. It does not touch the held-out-composition trap gap, which is compositional generalization.
Seed-to-seed variation is large for some cells (e.g. the predictor at 128k: .0155 vs .0306 overall),
so the tight bootstrap CIs above describe test sampling only, not training variability.

**Conclusion:** in this finite synthetic setting, the apparent ambiguity-specific contrastive error
is consistent with omitted prior information and is removed by the analytic prior correction. The
experiment does not justify a new observability-aware architecture. Still to do for a final result:
calibration fitted on a separate calibration set and frozen before evaluation (needs saved
validation scores), especially for unseen views and prior shift.
