#!/usr/bin/env python
"""Benchmark script for measuring pipeline performance.

This script runs the pipeline with different configurations and measures
performance improvements.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("benchmark")


def run_benchmark(config: dict, output_dir: Path) -> dict:
    """Run a single benchmark configuration.
    
    Args:
        config: Dictionary with benchmark parameters
        output_dir: Directory to write results
        
    Returns:
        Dictionary with timing results
    """
    cmd = [
        sys.executable,
        "main.py",
        "--manifold", config.get("manifold", "m188(-1,1)"),
        "--k-min", str(config.get("k_min", 3.5)),
        "--k-max", str(config.get("k_max", 4.2)),
        "--num-k", str(config.get("num_k", 5)),
        "--n-points", str(config.get("n_points", 15)),
        "--seed", str(config.get("seed", 42)),
        "--word-depth", str(config.get("word_depth", 3)),
        "--chi2-mode", config.get("chi2_mode", "paper"),
        "--output-dir", str(output_dir),
        "--benchmark",
    ]
    
    if config.get("use_scalar", False):
        cmd.append("--use-scalar-q")
    
    LOGGER.info("Running benchmark: %s", config.get("name", "unnamed"))
    LOGGER.info("Command: %s", " ".join(cmd))
    
    start_time = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.perf_counter() - start_time
        
        # Load timing data
        timing_path = output_dir / "timings.json"
        if timing_path.exists():
            with open(timing_path) as f:
                timing_data = json.load(f)
        else:
            timing_data = {}
        
        return {
            "name": config.get("name", "unnamed"),
            "elapsed_total": elapsed,
            "timing_data": timing_data,
            "success": True,
            "stdout": result.stdout,
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start_time
        LOGGER.error("Benchmark failed: %s", e)
        return {
            "name": config.get("name", "unnamed"),
            "elapsed_total": elapsed,
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


def main():
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument(
        "--output-dir",
        default="/tmp/benchmark_results",
        help="Base directory for benchmark results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (fewer k-values)"
    )
    parser.add_argument(
        "--compare-scalar",
        action="store_true",
        help="Also run scalar implementation for comparison"
    )
    args = parser.parse_args()
    
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Define benchmark configurations
    configs = []
    
    if args.quick:
        # Quick benchmark: fewer k-values
        configs.append({
            "name": "quick_vectorized",
            "manifold": "m188(-1,1)",
            "k_min": 3.5,
            "k_max": 4.0,
            "num_k": 3,
            "n_points": 10,
            "word_depth": 3,
            "use_scalar": False,
        })
        if args.compare_scalar:
            configs.append({
                "name": "quick_scalar",
                "manifold": "m188(-1,1)",
                "k_min": 3.5,
                "k_max": 4.0,
                "num_k": 3,
                "n_points": 10,
                "word_depth": 3,
                "use_scalar": True,
            })
    else:
        # Standard benchmark
        configs.append({
            "name": "standard_vectorized",
            "manifold": "m188(-1,1)",
            "k_min": 3.5,
            "k_max": 4.2,
            "num_k": 5,
            "n_points": 15,
            "word_depth": 3,
            "use_scalar": False,
        })
        if args.compare_scalar:
            configs.append({
                "name": "standard_scalar",
                "manifold": "m188(-1,1)",
                "k_min": 3.5,
                "k_max": 4.2,
                "num_k": 5,
                "n_points": 15,
                "word_depth": 3,
                "use_scalar": True,
            })
    
    # Run benchmarks
    results = []
    for config in configs:
        output_dir = output_base / config["name"]
        result = run_benchmark(config, output_dir)
        results.append(result)
        
        if result["success"]:
            LOGGER.info("✓ %s completed in %.2f seconds", result["name"], result["elapsed_total"])
            if "timing_data" in result:
                td = result["timing_data"]
                if "matrix_build" in td:
                    LOGGER.info("  Matrix build: %.2f s mean, %.2f s total", 
                               td["matrix_build"]["mean"], td["matrix_build"]["total"])
                if "cache_stats" in td:
                    cs = td["cache_stats"]
                    LOGGER.info("  legenp calls: %d, Phi hit rate: %.1f%%",
                               cs.get("legenp_calls", 0),
                               100.0 * cs.get("phi_hit_rate", 0.0))
        else:
            LOGGER.error("✗ %s failed", result["name"])
    
    # Write summary
    summary_path = output_base / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Wrote summary to %s", summary_path)
    
    # Print comparison if we have both scalar and vectorized
    if len(results) >= 2 and args.compare_scalar:
        vec = next((r for r in results if "vectorized" in r["name"]), None)
        scalar = next((r for r in results if "scalar" in r["name"]), None)
        
        if vec and scalar and vec["success"] and scalar["success"]:
            speedup = scalar["elapsed_total"] / vec["elapsed_total"]
            LOGGER.info("\n" + "="*60)
            LOGGER.info("SPEEDUP ANALYSIS")
            LOGGER.info("="*60)
            LOGGER.info("Vectorized: %.2f seconds", vec["elapsed_total"])
            LOGGER.info("Scalar:     %.2f seconds", scalar["elapsed_total"])
            LOGGER.info("Speedup:    %.2fx", speedup)
            
            vec_td = vec.get("timing_data", {})
            scalar_td = scalar.get("timing_data", {})
            
            if "matrix_build" in vec_td and "matrix_build" in scalar_td:
                vec_mb = vec_td["matrix_build"]["total"]
                scalar_mb = scalar_td["matrix_build"]["total"]
                mb_speedup = scalar_mb / vec_mb
                LOGGER.info("Matrix build speedup: %.2fx (%.2fs → %.2fs)", 
                           mb_speedup, scalar_mb, vec_mb)
            
            vec_cs = vec_td.get("cache_stats", {})
            scalar_cs = scalar_td.get("cache_stats", {})
            
            if "legenp_calls" in vec_cs and "legenp_calls" in scalar_cs:
                vec_calls = vec_cs["legenp_calls"]
                scalar_calls = scalar_cs["legenp_calls"]
                call_reduction = scalar_calls / vec_calls
                LOGGER.info("legenp call reduction: %.2fx (%d → %d)",
                           call_reduction, scalar_calls, vec_calls)
            LOGGER.info("="*60)
    
    # Return success if all benchmarks succeeded
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
