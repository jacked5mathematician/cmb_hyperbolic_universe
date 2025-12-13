#!/usr/bin/env python
"""
Plot chi-squared spectrum with different definitions.

This script loads a spectrum.npz file and recomputes chi-squared values
using all available definitions, plotting them on the same axes for comparison.

Usage:
    python scripts/plot_chi2_definitions.py --spectrum path/to/spectrum.npz --output comparison.png
"""
import argparse
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.chi2 import CHI2_DEFINITIONS

LOGGER = logging.getLogger(__name__)


def load_spectrum_data(spectrum_path: Path) -> dict:
    """Load spectrum data from npz file."""
    data = np.load(spectrum_path)
    return {key: data[key] for key in data.files}


def recompute_chi2_with_definitions(
    spectrum_data: dict,
    rank: int = 1
) -> dict:
    """
    Recompute chi-squared values using different definitions.
    
    This assumes the spectrum_data contains diagnostic information like
    sigma_min, sigma_max, A_frobenius, and M.
    
    Since we don't have the original A matrices, we'll use the relationship:
    chi2_raw = sigma^2 (which is what's stored as chi2_rank_X)
    
    Args:
        spectrum_data: Dictionary from spectrum.npz
        rank: Which rank to use (1-based)
    
    Returns:
        Dictionary mapping definition names to chi2 arrays
    """
    chi2_raw_key = f"chi2_rank_{rank}"
    if chi2_raw_key not in spectrum_data:
        raise ValueError(f"No {chi2_raw_key} in spectrum data")
    
    chi2_raw = spectrum_data[chi2_raw_key]
    
    # Extract metadata
    M = spectrum_data.get('M', None)
    sigma_min = spectrum_data.get('sigma_min', None)
    sigma_max = spectrum_data.get('sigma_max', None)
    A_frobenius = spectrum_data.get('A_frobenius', None)
    
    results = {
        'raw_residual': chi2_raw,
    }
    
    # Compute per_row if M is available
    if M is not None:
        # Handle array M (one value per k)
        M_vals = M if hasattr(M, '__len__') else np.full_like(chi2_raw, M)
        results['per_row'] = chi2_raw / M_vals
    
    # Compute ratio if sigma_min and sigma_max are available
    if sigma_min is not None and sigma_max is not None:
        # Note: This assumes chi2_raw = sigma^2 for the specific rank being analyzed.
        # For rank 1, this is sigma_min^2. For higher ranks, this is the k-th smallest sigma.
        # sigma_max is always the largest singular value across all ranks.
        sigma_for_rank = np.sqrt(chi2_raw)
        with np.errstate(divide='ignore', invalid='ignore'):
            results['ratio'] = (sigma_for_rank / sigma_max) ** 2
    
    # Compute frobenius if available
    if A_frobenius is not None:
        frobenius_sq = A_frobenius ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            results['frobenius'] = chi2_raw / frobenius_sq
    
    return results


def plot_chi2_comparison(
    k_values: np.ndarray,
    chi2_dict: dict,
    output_path: Path,
    rank: int = 1,
    log_scale: bool = True
):
    """
    Plot all chi2 definitions on the same axes.
    
    Args:
        k_values: Array of k values
        chi2_dict: Dictionary mapping definition names to chi2 arrays
        output_path: Where to save the plot
        rank: Which rank is being plotted
        log_scale: If True, use log scale for y-axis
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define colors and line styles for each definition
    styles = {
        'raw_residual': {'color': 'C0', 'linestyle': '-', 'label': r'Raw: $\sigma^2$'},
        'per_row': {'color': 'C1', 'linestyle': '--', 'label': r'Per-row: $\sigma^2/M$'},
        'ratio': {'color': 'C2', 'linestyle': '-.', 'label': r'Ratio: $(\sigma/\sigma_{max})^2$', 'linewidth': 2},
        'frobenius': {'color': 'C3', 'linestyle': ':', 'label': r'Frobenius: $\sigma^2/\|A\|_F^2$'},
    }
    
    # Top plot: All definitions
    for def_name, chi2_values in chi2_dict.items():
        if def_name not in styles:
            continue
        style = styles[def_name]
        ax1.plot(k_values, chi2_values, **style)
    
    ax1.set_xlabel('k', fontsize=12)
    ax1.set_ylabel(r'$\chi^2$', fontsize=12)
    ax1.set_title(f'Chi-squared Definitions Comparison (Rank {rank})', fontsize=14)
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylabel(r'$\chi^2$ (log scale)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Bottom plot: Focus on ratio and frobenius (normalized definitions)
    for def_name in ['ratio', 'frobenius']:
        if def_name in chi2_dict:
            style = styles[def_name]
            ax2.plot(k_values, chi2_dict[def_name], **style)
    
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel(r'$\chi^2$', fontsize=12)
    ax2.set_title(f'Normalized Chi-squared Definitions (Rank {rank})', fontsize=14)
    if log_scale:
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$\chi^2$ (log scale)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    LOGGER.info(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_all_ranks(
    k_values: np.ndarray,
    spectrum_data: dict,
    output_path: Path,
    max_ranks: int = 3,
    definition: str = 'ratio'
):
    """
    Plot multiple ranks for a single chi2 definition.
    
    Args:
        k_values: Array of k values
        spectrum_data: Full spectrum data dictionary
        output_path: Where to save the plot
        max_ranks: Number of ranks to plot
        definition: Which definition to use
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rank in range(1, max_ranks + 1):
        try:
            chi2_dict = recompute_chi2_with_definitions(spectrum_data, rank=rank)
            if definition in chi2_dict:
                ax.plot(k_values, chi2_dict[definition], 
                       label=f'Rank {rank}', linewidth=1.5, alpha=0.8)
        except (ValueError, KeyError) as e:
            LOGGER.warning(f"Could not compute rank {rank}: {e}")
            continue
    
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel(r'$\chi^2$', fontsize=12)
    ax.set_title(f'Chi-squared Spectrum ({definition} definition)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    LOGGER.info(f"Saved multi-rank plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot chi-squared with different definitions"
    )
    parser.add_argument(
        "--spectrum",
        type=Path,
        required=True,
        help="Path to spectrum.npz file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot path (default: spectrum_dir/chi2_comparison.png)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Which rank to plot for comparison (default: 1)",
    )
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Use linear scale instead of log scale",
    )
    parser.add_argument(
        "--all-ranks",
        action="store_true",
        help="Also plot all ranks with a single definition",
    )
    parser.add_argument(
        "--max-ranks",
        type=int,
        default=3,
        help="Maximum number of ranks to plot (default: 3)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    if not args.spectrum.exists():
        LOGGER.error(f"Spectrum file not found: {args.spectrum}")
        raise SystemExit(1)
    
    # Load spectrum data
    LOGGER.info(f"Loading spectrum from {args.spectrum}")
    spectrum_data = load_spectrum_data(args.spectrum)
    
    if "k_values" not in spectrum_data:
        LOGGER.error("No k_values in spectrum file")
        raise SystemExit(1)
    
    k_values = spectrum_data["k_values"]
    LOGGER.info(f"Loaded {len(k_values)} k-values from {k_values[0]:.2f} to {k_values[-1]:.2f}")
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        output_dir = args.spectrum.parent
        output_path = output_dir / f"chi2_comparison_rank{args.rank}.png"
    
    # Recompute chi2 with different definitions
    LOGGER.info(f"Recomputing chi-squared with all definitions for rank {args.rank}")
    chi2_dict = recompute_chi2_with_definitions(spectrum_data, rank=args.rank)
    
    LOGGER.info(f"Available definitions: {list(chi2_dict.keys())}")
    
    # Plot comparison
    plot_chi2_comparison(
        k_values, 
        chi2_dict, 
        output_path, 
        rank=args.rank,
        log_scale=not args.linear_scale
    )
    
    # Optionally plot all ranks with ratio definition
    if args.all_ranks:
        all_ranks_path = output_path.parent / f"chi2_all_ranks_ratio.png"
        plot_all_ranks(
            k_values,
            spectrum_data,
            all_ranks_path,
            max_ranks=args.max_ranks,
            definition='ratio'
        )
    
    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
