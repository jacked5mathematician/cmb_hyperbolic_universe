"""
Adaptive rho window strategies for the hyperbolic universe CMB pipeline.

This module provides two adaptive window expansion strategies as alternatives
to the paper-faithful fixed window approach:

1. adaptive_images: Expand ρ_max until median images per point reaches a target
2. adaptive_rank: Expand ρ_max until numerical rank of A matrix reaches a target fraction

The paper-faithful approach remains the default; these are experimental alternatives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Callable

import numpy as np

from .cutoffs import compute_rho_cutoffs
from .ghosts import enumerate_ghost_images, get_group_elements

LOGGER = logging.getLogger(__name__)


@dataclass
class AdaptiveRhoResult:
    """Result of adaptive rho window computation."""
    rho_min: float
    rho_max: float
    rho_max_original: float  # Paper-faithful value before expansion
    expansions: int  # Number of expansion steps taken
    stopped_by: str  # 'target_met', 'cap_reached', 'no_improvement'
    metric_history: List[float]  # Values of target metric at each step
    images_per_point: Optional[List[int]] = None  # Final images per point


def adaptive_rho_images(
    k: float,
    L: int,
    l_min: int,
    manifold_name: str,
    base_points: Sequence[Sequence[float]],
    *,
    target_median_images: int = 10,
    expand_step: float = 0.25,
    rho_max_cap: float = 6.0,
    min_images: int = 10,
    max_word_length: int = 6,
    max_images_per_point: int = 200,
    threshold: float = 0.25,
) -> AdaptiveRhoResult:
    """
    Expand ρ_max until median images per kept point reaches target.

    Starting from the paper-faithful ρ_max, repeatedly expand the window
    until:
    - median(images_per_point) >= target_median_images, OR
    - rho_max >= rho_max_cap, OR
    - expansion produces no improvement (plateau)

    Parameters
    ----------
    k : float
        Wavenumber.
    L : int
        Maximum angular momentum (typically floor(k) + 10).
    l_min : int
        Minimum angular momentum for cutoff computation.
    manifold_name : str
        Name of the SnapPy manifold.
    base_points : sequence
        Base points in the Dirichlet domain.
    target_median_images : int
        Target median number of images per point.
    expand_step : float
        Amount to expand ρ_max at each iteration.
    rho_max_cap : float
        Hard cap on ρ_max.
    min_images : int
        Minimum images required to keep a point.
    max_word_length : int
        Maximum word length for group element enumeration.
    max_images_per_point : int
        Maximum images to collect per point.
    threshold : float
        Paper threshold for cutoff computation.

    Returns
    -------
    AdaptiveRhoResult
        Result containing final window and diagnostic info.
    """
    # Get paper-faithful window as starting point
    rho_min, rho_max_paper, _ = compute_rho_cutoffs(k, L, l_min, threshold=threshold)
    rho_max_original = rho_max_paper
    
    # Pre-compute group elements once
    group_elements, _ = get_group_elements(manifold_name, max_word_length)
    if not group_elements:
        LOGGER.warning("No group elements available for %s; cannot do adaptive expansion", manifold_name)
        return AdaptiveRhoResult(
            rho_min=rho_min,
            rho_max=rho_max_paper,
            rho_max_original=rho_max_original,
            expansions=0,
            stopped_by='no_group_elements',
            metric_history=[],
        )
    
    rho_max = rho_max_paper
    metric_history = []
    expansions = 0
    prev_median = -1
    
    while True:
        # Enumerate images with current window
        points_images = enumerate_ghost_images(
            manifold_name,
            base_points,
            rho_min=rho_min,
            rho_max=rho_max,
            min_images=min_images,
            max_word_length=max_word_length,
            max_images=max_images_per_point,
            group_elements=group_elements,
        )
        
        if not points_images:
            images_counts = [0]
        else:
            images_counts = [len(imgs) for imgs in points_images]
        
        current_median = float(np.median(images_counts)) if images_counts else 0
        metric_history.append(current_median)
        
        LOGGER.debug(
            "k=%.2f adaptive_images: rho_max=%.3f, kept_points=%d, median_images=%.1f",
            k, rho_max, len(points_images), current_median
        )
        
        # Check stopping conditions
        if current_median >= target_median_images:
            return AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='target_met',
                metric_history=metric_history,
                images_per_point=images_counts,
            )
        
        if rho_max >= rho_max_cap:
            return AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='cap_reached',
                metric_history=metric_history,
                images_per_point=images_counts,
            )
        
        # Check for plateau (no improvement)
        if expansions > 0 and current_median <= prev_median:
            return AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='no_improvement',
                metric_history=metric_history,
                images_per_point=images_counts,
            )
        
        # Expand
        prev_median = current_median
        rho_max = min(rho_max + expand_step, rho_max_cap)
        expansions += 1


def adaptive_rho_rank(
    k: float,
    L: int,
    l_min: int,
    manifold_name: str,
    base_points: Sequence[Sequence[float]],
    matrix_builder: Callable,
    *,
    target_rank_frac: float = 0.7,
    target_nullity: int = 50,
    expand_step: float = 0.25,
    rho_max_cap: float = 6.0,
    min_images: int = 10,
    max_word_length: int = 6,
    max_images_per_point: int = 200,
    threshold: float = 0.25,
    svd_tol: float = 1e-10,
) -> AdaptiveRhoResult:
    """
    Expand ρ_max until numerical rank of A matrix reaches target.

    Starting from the paper-faithful ρ_max, repeatedly expand the window
    until:
    - rank(A) / N >= target_rank_frac, OR
    - nullity(A) = N - rank(A) >= target_nullity, OR
    - rho_max >= rho_max_cap, OR
    - expansion produces no improvement (plateau)

    Parameters
    ----------
    k : float
        Wavenumber.
    L : int
        Maximum angular momentum.
    l_min : int
        Minimum angular momentum for cutoff computation.
    manifold_name : str
        Name of the SnapPy manifold.
    base_points : sequence
        Base points in the Dirichlet domain.
    matrix_builder : callable
        Function to build the A matrix. Signature:
        matrix_builder(k, L, l_min, points_images) -> (A, M, N)
    target_rank_frac : float
        Target rank / N ratio.
    target_nullity : int
        Target nullity (N - rank).
    expand_step : float
        Amount to expand ρ_max at each iteration.
    rho_max_cap : float
        Hard cap on ρ_max.
    min_images : int
        Minimum images required to keep a point.
    max_word_length : int
        Maximum word length for group element enumeration.
    max_images_per_point : int
        Maximum images to collect per point.
    threshold : float
        Paper threshold for cutoff computation.
    svd_tol : float
        Tolerance for determining numerical rank.

    Returns
    -------
    AdaptiveRhoResult
        Result containing final window and diagnostic info.
    """
    # Get paper-faithful window as starting point
    rho_min, rho_max_paper, _ = compute_rho_cutoffs(k, L, l_min, threshold=threshold)
    rho_max_original = rho_max_paper
    
    # Pre-compute group elements once
    group_elements, _ = get_group_elements(manifold_name, max_word_length)
    if not group_elements:
        LOGGER.warning("No group elements available for %s; cannot do adaptive expansion", manifold_name)
        return AdaptiveRhoResult(
            rho_min=rho_min,
            rho_max=rho_max_paper,
            rho_max_original=rho_max_original,
            expansions=0,
            stopped_by='no_group_elements',
            metric_history=[],
        )
    
    rho_max = rho_max_paper
    metric_history = []  # Store rank fractions
    expansions = 0
    prev_rank_frac = -1
    
    while True:
        # Enumerate images with current window
        points_images = enumerate_ghost_images(
            manifold_name,
            base_points,
            rho_min=rho_min,
            rho_max=rho_max,
            min_images=min_images,
            max_word_length=max_word_length,
            max_images=max_images_per_point,
            group_elements=group_elements,
        )
        
        if not points_images or len(points_images) == 0:
            metric_history.append(0.0)
            if rho_max >= rho_max_cap:
                return AdaptiveRhoResult(
                    rho_min=rho_min,
                    rho_max=rho_max,
                    rho_max_original=rho_max_original,
                    expansions=expansions,
                    stopped_by='cap_reached',
                    metric_history=metric_history,
                )
            rho_max = min(rho_max + expand_step, rho_max_cap)
            expansions += 1
            continue
        
        # Build matrix and compute SVD
        try:
            A, M, N = matrix_builder(k, L, l_min, points_images)
            if M == 0 or N == 0:
                raise ValueError("Empty matrix")
            
            # Compute numerical rank
            s = np.linalg.svd(A, compute_uv=False)
            rank = np.sum(s > svd_tol * s[0])
            rank_frac = rank / N
            nullity = N - rank
            
        except Exception as e:
            LOGGER.debug("Matrix build failed for k=%.2f, rho_max=%.3f: %s", k, rho_max, e)
            rank_frac = 0.0
            nullity = 0
        
        metric_history.append(rank_frac)
        
        LOGGER.debug(
            "k=%.2f adaptive_rank: rho_max=%.3f, M=%d, N=%d, rank_frac=%.3f, nullity=%d",
            k, rho_max, M, N, rank_frac, nullity
        )
        
        # Check stopping conditions
        if rank_frac >= target_rank_frac or nullity >= target_nullity:
            return AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='target_met',
                metric_history=metric_history,
            )
        
        if rho_max >= rho_max_cap:
            return AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='cap_reached',
                metric_history=metric_history,
            )
        
        # Check for plateau (no improvement)
        if expansions > 0 and rank_frac <= prev_rank_frac + 0.01:
            return AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='no_improvement',
                metric_history=metric_history,
            )
        
        # Expand
        prev_rank_frac = rank_frac
        rho_max = min(rho_max + expand_step, rho_max_cap)
        expansions += 1


__all__ = ['AdaptiveRhoResult', 'adaptive_rho_images', 'adaptive_rho_rank']
