# Model and evaluation conventions

## Prediction

A model receives historical power and reference forecasts, a future reference curve, and issue-time descriptors. A compact backbone and a gated residual adapter produce a corrected 24-hour trajectory. The adapter adjustment is bounded by its configured correction cap. This cap does not bound the total backbone correction. Final outputs are clipped to `[0, 1.05]` per unit and masked to zero at night.

The available backbones are DLinear, a dilated temporal convolutional network, and a compact patch Transformer. The patch implementations jointly embed historical channels and condition on future descriptors; they are reference-conditioned implementations rather than the original channel-independent PatchTST architecture.

All 21 S1–S4 descriptors enter the candidate backbone. The state mask selects which groups enter the residual adapter:

| Group | Information |
| --- | --- |
| S1 | Solar elevation proxy, clear-sky GHI, daylight |
| S2 | Reference trajectory, increments, peak, peak timing, energy |
| S3 | Historical power levels and ramp statistics available at issuance |
| S4 | Calendar and location |

Here, “state group” means a group of model features. Geographic states are used separately for sampling and metric aggregation.

## Training and repair

The loss combines daytime absolute error, optional ramp error, and an optional penalty on excessive curvature of the correction. AdamW updates parameters using state-then-site balanced sampling. Validation daytime nMAE controls early stopping. Test records are excluded from training and search evaluation.

Repair activates S1 when neither S1 nor S2 is selected, canonicalizes inactive patch settings, enforces patch compatibility and the adapter cap, and reduces model width/depth when the parameter constraint requires it. The configuration limits are one million parameters, 10 MB of FP32 parameter storage, and an adapter cap of 0.20 p.u.

## Objectives

All three objectives are minimized:

1. **Daytime nMAE:** mean absolute error over daylight observations within each geographic group, multiplied by 100, followed by an equal-weight mean across groups. Power is already capacity-normalized.
2. **Daily-error CVaR90:** calculate each site-day's daytime MAE, average the largest `ceil(0.1 * number_of_site_days)` losses in each geographic group, then average equally across groups.
3. **log10 MACs:** the base-10 logarithm of the multiply–accumulate count for one 24-hour forecast.

MAC estimates use half the forward-operation FLOP count on the non-fused model graph, including attention operations. MACs describe arithmetic work; runtime also depends on hardware, implementation, and batch size.

## Search and selection

NSGA-II uses constrained dominance, non-dominated sorting, crowding distance, binary tournaments, uniform crossover, and per-gene mutation. The search loop repairs each proposed configuration before training. A cache avoids retraining repeated configurations within that run, while each request still counts toward its budget.

Screening first retains the non-dominated configurations. It anchors minimum mean error, minimum tail error, minimum normalized ideal distance, and minimum computation within a 2% mean-error tolerance. Remaining slots use greedy maximin coverage in normalized objective space.

After longer training, final role assignment uses the complete supplied validation population for normalization. Proxy screening and final role assignment are separate operations. The package exposes both without using test performance to pick configurations.
