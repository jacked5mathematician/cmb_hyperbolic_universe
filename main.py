import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math

from utils import (
    compute_rho_cutoffs,
    enumerate_ghost_images,
    generate_matrix_system,
    generate_matrix_system_scalar,
    sample_points_in_dirichlet_domain,
    solve_system_via_svd_numeric,
)
from utils.eigenvalues import extract_eigenvalues_from_spectrum
from utils.ghosts import get_group_elements
from utils.points import DEFAULT_FALLBACK_RADIUS
from utils.transformations import (
    apply_so31_action,
    klein_to_poincare,
    poincare_distance,
    project_to_klein,
)

LOGGER = logging.getLogger("pipeline")
PAPER_L_MIN = 5
MAX_RANKS = 5


def _paper_L(k: float) -> int:
    return int(np.floor(k)) + 10


def _paper_c(k: float) -> int:
    return int(np.floor(100.0 / k)) + 10


def _target_M(L: int, c_val: int) -> int:
    N = (L + 1) ** 2
    return int(np.ceil(c_val * N))


def _cap_images_to_max_pairs(images: List[Tuple[float, float, float]], max_pairs: int):
    """Truncate images so that pair count <= max_pairs.

    Args:
        images: list of (rho, theta, phi)
        max_pairs: maximum allowed pairwise combinations

    Returns:
        truncated list of images
    """
    if len(images) < 2:
        return images
    # Solve n(n-1)/2 <= max_pairs -> n <= (1 + sqrt(1 + 8*max_pairs)) / 2
    n_cap = int((1 + math.isqrt(1 + 8 * max_pairs)) // 2)
    return images[: min(len(images), n_cap)]


def _build_dirichlet_checker(group_elements: List[np.ndarray], tolerance: float = 1e-6):
    base_point = np.zeros(3, dtype=float)
    gamma_p0 = []
    for mat in group_elements:
        transformed = apply_so31_action(mat, base_point)
        klein = project_to_klein(transformed)
        poincare = klein_to_poincare([klein])[0]
        if np.linalg.norm(poincare) >= 1.0:
            continue
        gamma_p0.append(poincare)

    def check(point: np.ndarray) -> bool:
        d0 = poincare_distance(point, base_point)
        for other in gamma_p0:
            if d0 > poincare_distance(point, other) + tolerance:
                return False
        return True

    return check


def _save_spectrum(output_dir: Path, k_values: np.ndarray, chi2_ranks: List[List[float]], meta: Dict[str, np.ndarray]) -> Path:
    payload = {
        "k_values": k_values,
        "chi2_rank_1": np.array(chi2_ranks[0], dtype=float),
    }
    for i in range(1, MAX_RANKS):
        payload[f"chi2_rank_{i+1}"] = np.array(chi2_ranks[i], dtype=float)
    payload.update({k: np.array(v) for k, v in meta.items()})
    spectrum_path = output_dir / "spectrum.npz"
    np.savez(spectrum_path, **payload)
    return spectrum_path


def _plot_spectrum(output_dir: Path, k_values: np.ndarray, chi2_ranks: List[List[float]]) -> Path:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, chi2_ranks[0], label="Rank 1")
    for idx in range(1, MAX_RANKS):
        plt.plot(k_values, chi2_ranks[idx], label=f"Rank {idx+1}", alpha=0.6)
    plt.xlabel("k")
    plt.ylabel(r"$\chi^2$")
    plt.title(r"$\chi^2$ spectrum")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = output_dir / "chi2_spectrum.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def run_pipeline(
    manifold_name: str,
    k_values: np.ndarray,
    n_points: int,
    seed: int,
    small_test: bool,
    dry_run: bool,
    output_dir: Path,
    self_check: bool = False,
    word_depth: int = 3,
    eigen_threshold: float | None = None,
    benchmark: bool = False,
    use_scalar_q: bool = False,
    chi2_mode: str = "paper",
) -> Dict:
    import time
    output_dir.mkdir(parents=True, exist_ok=True)
    max_word_length = word_depth
    
    # Timing dictionary for benchmark mode
    timings = {} if benchmark else None

    LOGGER.info("Sampling %s base points for manifold %s", n_points, manifold_name)
    t0 = time.perf_counter() if benchmark else None
    base_points, base_points_pseudo, sampling_meta = sample_points_in_dirichlet_domain(
        manifold_name, n_points=n_points, seed=seed, fallback_radius=DEFAULT_FALLBACK_RADIUS, word_depth=word_depth, return_metadata=True
    )
    if benchmark:
        timings["point_sampling"] = [time.perf_counter() - t0]
    LOGGER.info(
        "Sampled %s base points (mean rho=%.3f) [fallback=%s]",
        len(base_points_pseudo),
        base_points_pseudo[:, 0].mean(),
        sampling_meta.get("fallback_used"),
    )

    dirichlet_checker = None
    if self_check:
        group_elements, _ = get_group_elements(manifold_name, word_depth)
        if group_elements:
            dirichlet_checker = _build_dirichlet_checker(group_elements)

    chi2_ranks: List[List[float]] = [[] for _ in range(MAX_RANKS)]
    L_arr: List[int] = []
    rho_min_arr: List[float] = []
    rho_max_arr: List[float] = []
    M_arr: List[int] = []
    N_arr: List[int] = []
    kept_points_arr: List[int] = []
    fallback_arr: List[bool] = []
    M_target_arr: List[int] = []
    self_check_issues: List[str] = []
    
    # Enhanced diagnostics
    sigma_min_arr: List[float] = []
    sigma_max_arr: List[float] = []
    A_frobenius_arr: List[float] = []
    A_max_abs_arr: List[float] = []
    A_min_nonzero_arr: List[float] = []
    images_per_point_arr: List[float] = []

    for k in k_values:
        L = _paper_L(k)
        l_min = PAPER_L_MIN
        c_val = _paper_c(k)
        N = (L + 1) ** 2
        M_target = _target_M(L, c_val)
        
        t0 = time.perf_counter() if benchmark else None
        rho_min, rho_max, cutoff_fallback = compute_rho_cutoffs(k, L, l_min)
        if benchmark:
            timings.setdefault("cutoff_computation", []).append(time.perf_counter() - t0)
        
        LOGGER.info(
            "k=%.3f -> L=%s c=%s l_min=%s rho_min=%.3f rho_max=%.3f M_target=%s",
            k,
            L,
            c_val,
            l_min,
            rho_min,
            rho_max,
            M_target,
        )

        if dry_run:
            for idx in range(MAX_RANKS):
                chi2_ranks[idx].append(np.nan)
            L_arr.append(L)
            rho_min_arr.append(rho_min)
            rho_max_arr.append(rho_max)
            M_arr.append(0)
            N_arr.append(N)
            kept_points_arr.append(0)
            fallback_arr.append(cutoff_fallback or sampling_meta.get("fallback_used", False))
            M_target_arr.append(M_target)
            sigma_min_arr.append(np.nan)
            sigma_max_arr.append(np.nan)
            A_frobenius_arr.append(np.nan)
            A_max_abs_arr.append(np.nan)
            A_min_nonzero_arr.append(np.nan)
            images_per_point_arr.append(np.nan)
            continue

        group_elements, geom_fallback = get_group_elements(manifold_name, max_word_length)
        
        t0 = time.perf_counter() if benchmark else None
        points_images, ghost_meta = enumerate_ghost_images(
            manifold_name,
            base_points,
            rho_min=rho_min,
            rho_max=rho_max,
            min_images=10,
            max_word_length=max_word_length,
            group_elements=group_elements,
            return_metadata=True,
        )
        if benchmark:
            timings.setdefault("ghost_enumeration", []).append(time.perf_counter() - t0)
        
        kept_total = len(points_images)
        LOGGER.info("Retained %s/%s base points after ghost enumeration", kept_total, len(base_points))

        selected_points: List[List[Tuple[float, float, float]]] = []
        rows = 0
        required_min_points = min(10, max(8, len(points_images)))
        max_pairs_per_point = max(1500, M_target // max(required_min_points, 1))

        for imgs in points_images:
            capped_imgs = _cap_images_to_max_pairs(imgs, max_pairs_per_point)
            contribution = len(capped_imgs) * (len(capped_imgs) - 1) // 2
            if contribution == 0:
                continue

            if rows < M_target or len(selected_points) < required_min_points:
                selected_points.append(capped_imgs)
                rows += contribution

            if rows >= M_target and len(selected_points) >= required_min_points:
                break

        fallback_used = cutoff_fallback or sampling_meta.get("fallback_used", False) or geom_fallback or ghost_meta.get("fallback_used", False)

        if not selected_points:
            LOGGER.warning("No viable points retained for k=%.3f; marking NaN chi^2.", k)
            for idx in range(MAX_RANKS):
                chi2_ranks[idx].append(np.nan)
            L_arr.append(L)
            rho_min_arr.append(rho_min)
            rho_max_arr.append(rho_max)
            M_arr.append(0)
            N_arr.append(N)
            kept_points_arr.append(0)
            fallback_arr.append(fallback_used)
            M_target_arr.append(M_target)
            sigma_min_arr.append(np.nan)
            sigma_max_arr.append(np.nan)
            A_frobenius_arr.append(np.nan)
            A_max_abs_arr.append(np.nan)
            A_min_nonzero_arr.append(np.nan)
            images_per_point_arr.append(np.nan)
            continue

        t0 = time.perf_counter() if benchmark else None
        if use_scalar_q:
            M, N_check, A = generate_matrix_system_scalar(selected_points, L, k)
        else:
            M, N_check, A = generate_matrix_system(selected_points, L, k)
        if benchmark:
            timings.setdefault("matrix_build", []).append(time.perf_counter() - t0)
        
        if N_check != N:
            self_check_issues.append(f"N mismatch for k={k}: expected {N}, got {N_check}")

        t0 = time.perf_counter() if benchmark else None
        normalize_rows = (chi2_mode == "legacy")
        chi_sq, _, svd_diag = solve_system_via_svd_numeric(A, n_smallest=MAX_RANKS, normalize_rows=normalize_rows)
        if benchmark:
            timings.setdefault("svd", []).append(time.perf_counter() - t0)
        for idx in range(MAX_RANKS):
            chi2_ranks[idx].append(chi_sq[idx] if idx < len(chi_sq) else np.nan)
        
        # Collect diagnostics
        sigma_min_arr.append(svd_diag['sigma_min'])
        sigma_max_arr.append(svd_diag['sigma_max'])
        A_frobenius_arr.append(float(np.linalg.norm(A, ord='fro')))
        A_max_abs_arr.append(float(np.abs(A).max()) if A.size > 0 else np.nan)
        # Compute min of nonzero elements without creating intermediate copy
        nonzero_mask = (A != 0)
        A_min_nonzero_arr.append(float(np.abs(A[nonzero_mask]).min()) if nonzero_mask.any() else np.nan)
        images_per_point_arr.append(float(np.mean([len(imgs) for imgs in selected_points])))

        L_arr.append(L)
        rho_min_arr.append(rho_min)
        rho_max_arr.append(rho_max)
        M_arr.append(M)
        N_arr.append(N)
        kept_points_arr.append(len(selected_points))
        fallback_arr.append(fallback_used)
        M_target_arr.append(M_target)

        if self_check:
            if dirichlet_checker:
                for pt in base_points:
                    if not dirichlet_checker(pt):
                        self_check_issues.append(f"Base point outside Dirichlet approximation for k={k}")
                        break
            if not np.all(np.linalg.norm(base_points, axis=1) < 1.0):
                self_check_issues.append("Base point norm >=1 detected")
            if not all(rho_min <= img[0] <= rho_max for imgs in selected_points for img in imgs):
                self_check_issues.append(f"Ghost image outside rho window for k={k}")
            if not np.isfinite(chi2_ranks[0][-1]):
                self_check_issues.append(f"Non-finite chi^2 for k={k}")

    meta_arrays = {
        "L": np.array(L_arr, dtype=int),
        "rho_min": np.array(rho_min_arr, dtype=float),
        "rho_max": np.array(rho_max_arr, dtype=float),
        "M": np.array(M_arr, dtype=int),
        "N": np.array(N_arr, dtype=int),
        "kept_points": np.array(kept_points_arr, dtype=int),
        "fallback_used": np.array(fallback_arr, dtype=bool),
        "M_target": np.array(M_target_arr, dtype=int),
        "sigma_min": np.array(sigma_min_arr, dtype=float),
        "sigma_max": np.array(sigma_max_arr, dtype=float),
        "A_frobenius": np.array(A_frobenius_arr, dtype=float),
        "A_max_abs": np.array(A_max_abs_arr, dtype=float),
        "A_min_nonzero": np.array(A_min_nonzero_arr, dtype=float),
        "images_per_point": np.array(images_per_point_arr, dtype=float),
    }

    spectrum_path = _save_spectrum(output_dir, k_values, chi2_ranks, meta_arrays)
    plot_path = _plot_spectrum(output_dir, k_values, chi2_ranks)

    report = {"ok": len(self_check_issues) == 0, "issues": self_check_issues}
    if self_check:
        with open(output_dir / "self_check_report.json", "w") as f:
            json.dump(report, f, indent=2)
        LOGGER.info("Wrote self_check_report.json")
    
    # Write timing information if in benchmark mode
    if benchmark and timings:
        from utils.special_functions import get_cache_stats
        
        timing_summary = {}
        for phase, times in timings.items():
            if times:
                timing_summary[phase] = {
                    "mean": float(np.mean(times)),
                    "total": float(np.sum(times)),
                    "count": len(times),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                }
        
        # Add cache statistics
        cache_stats = get_cache_stats()
        timing_summary['cache_stats'] = cache_stats
        
        # Calculate hit rates
        if cache_stats['phi_calls'] > 0:
            timing_summary['cache_stats']['phi_hit_rate'] = cache_stats['phi_cache_hits'] / cache_stats['phi_calls']
        if cache_stats['y_lm_calls'] > 0:
            timing_summary['cache_stats']['y_lm_hit_rate'] = cache_stats['y_lm_cache_hits'] / cache_stats['y_lm_calls']
        
        timing_path = output_dir / "timings.json"
        with open(timing_path, "w") as f:
            json.dump(timing_summary, f, indent=2)
        LOGGER.info("Wrote benchmark timings to %s", timing_path)
        LOGGER.info("Cache stats: Phi calls=%d (hit rate=%.2f%%), Y_lm calls=%d (hit rate=%.2f%%), legenp calls=%d",
                    cache_stats['phi_calls'],
                    100.0 * cache_stats['phi_cache_hits'] / max(cache_stats['phi_calls'], 1),
                    cache_stats['y_lm_calls'],
                    100.0 * cache_stats['y_lm_cache_hits'] / max(cache_stats['y_lm_calls'], 1),
                    cache_stats['legenp_calls'])

    result = {
        "spectrum_path": spectrum_path,
        "plot_path": plot_path,
        "report": report,
    }
    if eigen_threshold is not None:
        eigen_path = extract_eigenvalues_from_spectrum(
            spectrum_path, output_dir, threshold=eigen_threshold, refine=True
        )
        result["eigenvalues_path"] = eigen_path
    return result


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
    parser.add_argument("--self-check", action="store_true", help="Run invariant checks and write self_check_report.json")
    parser.add_argument("--word-depth", type=int, default=3, help="Maximum group word depth for enumeration")
    parser.add_argument(
        "--eigen-threshold",
        type=float,
        default=float("inf"),
        help="Threshold for accepting local minima when extracting eigenvalues",
    )
    parser.add_argument(
        "--self-check-strict",
        action="store_true",
        help="Exit non-zero if self-check reports issues (default: do not fail).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark mode with detailed timing output to timings.json",
    )
    parser.add_argument(
        "--k-chunk-index",
        type=int,
        default=None,
        help="Index of k-value chunk to process (0-based, for HPC job arrays)",
    )
    parser.add_argument(
        "--k-num-chunks",
        type=int,
        default=None,
        help="Total number of chunks to split k-values into (for HPC job arrays)",
    )
    parser.add_argument(
        "--require-snappy",
        action="store_true",
        help="Exit with error if SnapPy cannot load manifold generators (prevents accidental synthetic runs)",
    )
    parser.add_argument(
        "--use-scalar-q",
        action="store_true",
        help="Use scalar Q_k_lm implementation instead of vectorized version for matrix assembly",
    )
    parser.add_argument(
        "--chi2-mode",
        type=str,
        choices=["paper", "legacy"],
        default="paper",
        help="Chi-squared computation mode: 'paper' (default, no row normalization) or 'legacy' (with row normalization)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile profiling and write profile.prof to output directory",
    )
    return parser.parse_args()


def _run_main_logic(args):
    """Main logic separated for profiling support."""
    output_dir = Path(args.output_dir)
    
    # Check SnapPy availability if required
    if args.require_snappy:
        try:
            import snappy
            # Try to load the manifold to verify it works
            snappy.Manifold(args.manifold)
            LOGGER.info("SnapPy check passed for manifold %s", args.manifold)
        except ImportError:
            LOGGER.error("--require-snappy flag set but SnapPy is not available")
            raise SystemExit(1)
        except Exception as e:
            LOGGER.error("--require-snappy flag set but failed to load manifold %s: %s", args.manifold, e)
            raise SystemExit(1)

    if args.small_test:
        args.num_k = min(args.num_k, 2)
        args.n_points = min(args.n_points, 6)

    k_values = np.linspace(args.k_min, args.k_max, args.num_k)
    
    # Handle k-value chunking for HPC
    if args.k_chunk_index is not None and args.k_num_chunks is not None:
        if args.k_chunk_index < 0 or args.k_chunk_index >= args.k_num_chunks:
            LOGGER.error("Invalid chunk index %s (must be 0 <= index < %s)", args.k_chunk_index, args.k_num_chunks)
            raise SystemExit(1)
        
        if args.k_num_chunks > len(k_values):
            LOGGER.error("Number of chunks (%s) cannot exceed number of k-values (%s)", 
                        args.k_num_chunks, len(k_values))
            raise SystemExit(1)
        
        # Split k_values into chunks
        chunk_size = len(k_values) // args.k_num_chunks
        remainder = len(k_values) % args.k_num_chunks
        
        # Distribute remainder evenly across first chunks
        if args.k_chunk_index < remainder:
            start_idx = args.k_chunk_index * (chunk_size + 1)
            end_idx = start_idx + chunk_size + 1
        else:
            start_idx = args.k_chunk_index * chunk_size + remainder
            end_idx = start_idx + chunk_size
        
        k_values = k_values[start_idx:end_idx]
        LOGGER.info("Processing chunk %s/%s: k-values from index %s to %s (%s values)",
                    args.k_chunk_index, args.k_num_chunks, start_idx, end_idx, len(k_values))
        
        # Update output directory to include chunk index
        output_dir = output_dir / f"chunk_{args.k_chunk_index}"
    
    LOGGER.info("Running pipeline for k in [%s, %s] (%s samples)", 
                k_values[0] if len(k_values) > 0 else args.k_min, 
                k_values[-1] if len(k_values) > 0 else args.k_max, 
                len(k_values))

    result = run_pipeline(
        manifold_name=args.manifold,
        k_values=k_values,
        n_points=args.n_points,
        seed=args.seed,
        small_test=args.small_test,
        dry_run=args.dry_run,
        output_dir=output_dir,
        self_check=args.self_check,
        word_depth=args.word_depth,
        eigen_threshold=args.eigen_threshold,
        benchmark=args.benchmark,
        use_scalar_q=args.use_scalar_q,
        chi2_mode=args.chi2_mode,
    )
    if args.self_check_strict and not result["report"]["ok"]:
        raise SystemExit(1)
    return result


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    if args.profile:
        import cProfile
        import pstats
        from pathlib import Path
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_path = output_dir / "profile.prof"
        
        LOGGER.info("Profiling enabled, output will be saved to %s", profile_path)
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = _run_main_logic(args)
        finally:
            profiler.disable()
            profiler.dump_stats(str(profile_path))
            LOGGER.info("Profile saved to %s", profile_path)
            
            # Also print top 20 functions by cumulative time
            stats = pstats.Stats(profiler)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            LOGGER.info("Top 20 functions by cumulative time:")
            stats.print_stats(20)
    else:
        result = _run_main_logic(args)
    
    return result


if __name__ == "__main__":
    main()
