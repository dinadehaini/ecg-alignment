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

## Mechanism: base-rate loss, not identifiability

The non-identifiable gap sits entirely on claims whose base rate is not 0.5 ("event" 2/3, "early"
1/3 and their negations: +.028 to +.036); claims with base rate 0.5 show none (-.008 to +.006).
Contrastive (InfoNCE) scores estimate a likelihood ratio and drop each claim's marginal, and the
"claim vs negation" rule cannot recover it; when the view carries no evidence the correct answer is
exactly that base rate. Per-claim Platt calibration (cross-fitted on test halves) cuts the
non-identifiable excess from .0198 to .0077 (predictor .0058), and overall contrastive + per-claim
calibration (.0162) beats the predictor (.0230).

**Conclusion:** stage 0 does not show an identifiability-specific failure beyond a known property
of contrastive scores (missing per-claim base rates). The remaining gaps are small (.0077 vs .0058)
or compositional (held-out trap). Calibrated probabilistic baselines track the oracle.
