# Adaptive rho window and robustness notes

## What changed
- Ghost collection now uses a ball window with `rho_min=0.0` and explicitly drops the identity image instead of relying on a positive inner radius.
- A new adaptive chooser starts from the paper rho_max, probes a handful of base points, and expands `rho_max` by `0.25` until the probe median images per point and retention fraction clear minimum thresholds or a cap is reached.
- Protective guards return `NaN` chi² (instead of tiny misleading values) when too few points/images remain or the constraint matrix is under-populated.
- Spectrum metadata now records the paper rho_max, whether expansion occurred, probe statistics, and the expansion step count.

## Default tuning knobs
- `delta`: 0.25 increase per expansion step.
- `rho_max_cap`: 6.0 hard ceiling.
- `probe_points`: 12 base points probed (or fewer if fewer points exist).
- `min_images`: 10 images required per point (unchanged).
- `min_retention`: 0.5 fraction of probed points that must meet `min_images`.
- `min_base_points`: 8 retained points required to proceed.

## How to tune
- To allow more aggressive expansion for high k, raise `rho_max_cap` or increase `delta`.
- To make acceptance easier, lower `min_retention` (e.g., 0.3) or `min_images` (e.g., 8), but keep `min_base_points` high enough to avoid under-determined systems.
- To conserve work, reduce `probe_points`; to be more thorough, increase it (at extra cost).

## Logging and visibility
- Each k logs the paper and final rho window plus probe medians and retention.
- Any expansion is tagged with `[NON-PAPER] expanded rho_max ...` including step counts and caps.
- Fail-fast guards log the reason before setting chi² to NaN.
