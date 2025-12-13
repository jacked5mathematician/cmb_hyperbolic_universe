import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from utils import (
    compute_rho_cutoffs,
    enumerate_ghost_images,
    generate_matrix_system,
    sample_points_in_dirichlet_domain,
    solve_system_via_svd_numeric,
)


LOGGER = logging.getLogger("pipeline")


def _default_L(k: float) -> int:
    return max(1, int(round(k)))


def _default_l_min(L: int) -> int:
    return max(0, min(2, L))


def run_pipeline(manifold_name: str, k_values: np.ndarray, n_points: int, seed: int,
                 small_test: bool, dry_run: bool, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Sampling %s base points for manifold %s", n_points, manifold_name)
    base_points, base_points_pseudo = sample_points_in_dirichlet_domain(
        manifold_name, n_points=n_points, seed=seed
    )
    LOGGER.info("Sampled %s base points (mean rho=%.3f)", len(base_points_pseudo), base_points_pseudo[:, 0].mean())

    summary = []
    for k in k_values:
        L = _default_L(k)
        l_min = _default_l_min(L)
        c_val = L  # TODO: replace with paper-specific c(k) heuristic once available
        rho_min, rho_max = compute_rho_cutoffs(k, L, l_min)
        LOGGER.info(
            "k=%.3f -> L=%s c=%s l_min=%s rho_min=%.3f rho_max=%.3f",
            k, L, c_val, l_min, rho_min, rho_max,
        )

        if dry_run:
            summary.append(
                {"k": float(k), "L": L, "c": c_val, "l_min": l_min, "rho_min": rho_min, "rho_max": rho_max}
            )
            continue

        points_images = enumerate_ghost_images(
            manifold_name,
            base_points,
            rho_min=rho_min,
            rho_max=rho_max,
            min_images=10,
            max_word_length=2 if small_test else 6,
        )
        kept = len(points_images)
        LOGGER.info("Retained %s/%s base points after ghost enumeration", kept, len(base_points))

        if not points_images:
            LOGGER.warning("No points retained for k=%.3f; skipping.", k)
            continue

        M, N, A = generate_matrix_system(points_images, L, k)
        LOGGER.info("Matrix A shape: %s x %s (M=%s N=%s)", A.shape[0], A.shape[1], M, N)

        chi_sq, _ = solve_system_via_svd_numeric(A)
        LOGGER.info("Smallest singular values^2 for k=%.3f: %s", k, chi_sq)

        summary.append(
            {
                "k": float(k),
                "L": L,
                "c": c_val,
                "l_min": l_min,
                "rho_min": rho_min,
                "rho_max": rho_max,
                "M": M,
                "N": N,
                "chi2": chi_sq,
                "kept_points": kept,
            }
        )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Wrote summary for %s k values to %s", len(summary), output_dir / "summary.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-faithful ghosts pipeline.")
    parser.add_argument("--manifold", default="m003(-2,3)")
    parser.add_argument("--k-min", type=float, default=1.0)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--num-k", type=int, default=3)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-test", action="store_true", help="Use small limits for quick runs.")
    parser.add_argument("--dry-run", action="store_true", help="Compute configuration only.")
    parser.add_argument("--output-dir", default="output_values")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)

    if args.small_test:
        args.num_k = min(args.num_k, 2)
        args.n_points = min(args.n_points, 6)

    k_values = np.linspace(args.k_min, args.k_max, args.num_k)
    LOGGER.info("Running pipeline for k in [%s, %s] (%s samples)", args.k_min, args.k_max, args.num_k)

    run_pipeline(
        manifold_name=args.manifold,
        k_values=k_values,
        n_points=args.n_points,
        seed=args.seed,
        small_test=args.small_test,
        dry_run=args.dry_run,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
