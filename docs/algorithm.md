## Pipeline overview

This repository follows the Cornish & Spergel “method of ghosts” conventions:

- The scan variable is **k**.
- `nu == k` everywhere (no `sqrt(k^2+1)`), and `q^2 = k^2 + 1`.
- We work in pseudospherical coordinates `(rho, theta, phi)`; rho is the hyperbolic radial distance.

### Rho cutoff policy

`compute_rho_cutoffs(k, L, l_min, threshold)` scans for the first rho where
`|X_k^ell(rho) * sinh(rho)| <= threshold` for `ell = l_min` (rho_min) and `ell = L` (rho_max).
If no crossing is found up to a safety cap, it logs a warning, marks `fallback_used=True`, and falls
back to an envelope heuristic. The function always guarantees `0 <= rho_min < rho_max` or raises a
clear error.

### Ghost enumeration

`enumerate_ghost_images` loads SnapPy pairing matrices when available, augments them with inverses,
and performs a BFS over group words until images in the rho window are collected. Matrices are
deduplicated by hashing rounded entries. If SnapPy is unavailable, a deterministic synthetic
enumerator produces images in the requested rho band so tests and diagnostics still function.

### Points inside the Dirichlet domain

`sample_points_in_dirichlet_domain` applies the Dirichlet-domain inequality
`d(x, p0) <= d(x, gamma(p0))` for a finite set of group elements (words up to a small depth).
If generators are unavailable it falls back to rejection sampling inside the Poincaré ball. The
function returns both Poincaré coordinates and their pseudospherical transforms for diagnostics.

### Matrix assembly and invariants

`generate_matrix_system` expects `points_images` as `List[List[(rho, theta, phi)]]`. It builds
`A(k)` with `N = (L+1)^2` columns (one per `(l, m)`) and `M = sum_i n_i(n_i-1)/2` rows (one per
unordered pair of images for each base point). Assertions check both formulas.

### SVD diagnostics

`solve_system_via_svd_numeric` returns the smallest singular values (and their squared values
`chi^2`). These diagnostics are logged per-k to track multiplicities.

### Running locally vs HPC

- Quick local run: `python main.py --small-test --dry-run` to validate configuration, then
  `python main.py --small-test` to execute the fast end-to-end pipeline.
- For larger scans adjust `--k-min/--k-max/--num-k`, `--n-points`, and `--output-dir`.
Logging records per-k values of `(L, c, rho_min, rho_max)`, base-point retention, matrix shape,
and the smallest singular values.
