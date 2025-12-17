"""
Adaptive rho window strategies for the hyperbolic universe CMB pipeline.

This module provides two adaptive window expansion strategies as alternatives
to the paper-faithful fixed window approach:

1. adaptive_images: Expand ρ_max until quantile-based image criteria are met
   - Q10 images ≥ target (ensures even tail points have enough images)
   - Drop fraction ≤ max_drop_fraction (limits point loss)

2. adaptive_rank: Expand ρ_max until numerical rank reaches target, with σ_min vs τ health gate
   - Records full SVD diagnostics: σ_min, σ_max, τ, cond, rank, nullity, frac≤5τ

The paper-faithful approach remains the default; these are experimental alternatives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional, Callable, Dict, Any

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
    stopped_by: str  # 'target_met', 'cap_reached', 'no_improvement', 'health_failed'
    metric_history: List[float]  # Values of target metric at each step
    images_per_point: Optional[List[int]] = None  # Final images per point
    kept_points: int = 0  # Number of points retained
    drop_fraction: float = 0.0  # Fraction of points dropped
    # SVD health diagnostics (for adaptive_rank)
    svd_health: Optional[Dict[str, Any]] = None


@dataclass
class SVDHealth:
    """SVD-based health diagnostics for matrix A."""
    sigma_min: float
    sigma_max: float
    sigma_2: float  # Second singular value (useful for rank diagnostics)
    tau: float  # Machine epsilon * max(M, N) * sigma_max
    condition_number: float
    numerical_rank: int
    nullity: int
    N: int  # Total number of columns
    M: int  # Total number of rows
    frac_below_5tau: float  # Fraction of singular values ≤ 5τ
    healthy: bool  # True if σ_min > τ (well-conditioned)
    singular_values: Optional[np.ndarray] = None  # Full singular value spectrum (optional)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'sigma_2': self.sigma_2,
            'tau': self.tau,
            'condition_number': self.condition_number,
            'numerical_rank': self.numerical_rank,
            'nullity': self.nullity,
            'N': self.N,
            'M': self.M,
            'frac_below_5tau': self.frac_below_5tau,
            'healthy': self.healthy,
        }


def compute_svd_health(A: np.ndarray, svd_tol: float = 1e-10, store_singular_values: bool = False) -> SVDHealth:
    """
    Compute comprehensive SVD health diagnostics for matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to analyze (M x N).
    svd_tol : float
        Relative tolerance for numerical rank (default 1e-10).
    store_singular_values : bool
        If True, store full singular value array in result.
    
    Returns
    -------
    SVDHealth
        Comprehensive SVD diagnostics.
    """
    M, N = A.shape
    s = np.linalg.svd(A, compute_uv=False)
    
    sigma_max = s[0] if len(s) > 0 else 0.0
    sigma_2 = s[1] if len(s) > 1 else 0.0
    sigma_min = s[-1] if len(s) > 0 else 0.0
    
    # τ = machine epsilon * max(M, N) * σ_max (standard numerical threshold)
    eps = np.finfo(A.dtype).eps
    tau = eps * max(M, N) * sigma_max
    
    # Condition number
    cond = sigma_max / sigma_min if sigma_min > 0 else np.inf
    
    # Numerical rank using relative tolerance
    rank_threshold = svd_tol * sigma_max
    numerical_rank = int(np.sum(s > rank_threshold))
    nullity = N - numerical_rank
    
    # Fraction of singular values ≤ 5τ
    frac_below_5tau = float(np.sum(s <= 5 * tau)) / len(s) if len(s) > 0 else 1.0
    
    # Health check: σ_min > τ means well-conditioned relative to machine precision
    healthy = sigma_min > tau
    
    return SVDHealth(
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        sigma_2=float(sigma_2),
        tau=float(tau),
        condition_number=float(cond),
        numerical_rank=numerical_rank,
        nullity=nullity,
        N=N,
        M=M,
        frac_below_5tau=frac_below_5tau,
        healthy=healthy,
        singular_values=s if store_singular_values else None,
    )


def adaptive_rho_images(
    k: float,
    L: int,
    l_min: int,
    manifold_name: str,
    base_points: Sequence[Sequence[float]],
    *,
    target_q10_images: int = 10,
    max_drop_fraction: float = 0.3,
    expand_step: float = 0.25,
    rho_max_cap: float = 6.0,
    min_images: int = 10,
    max_word_length: int = 6,
    max_images_per_point: int = 200,
    threshold: float = 0.25,
) -> AdaptiveRhoResult:
    """
    Expand ρ_max until quantile-based image criteria are met.

    Criteria (all must be satisfied):
    - Q10 (10th percentile) of images per point ≥ target_q10_images
    - Drop fraction ≤ max_drop_fraction

    Always expands up to rho_max_cap if criteria not met.

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
    target_q10_images : int
        Target 10th percentile of images per point.
    max_drop_fraction : float
        Maximum fraction of points that can be dropped.
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
    n_base = len(base_points)
    
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
            kept_points=0,
            drop_fraction=1.0,
        )
    
    rho_max = rho_max_paper
    metric_history = []  # Store Q10 values
    expansions = 0
    best_result = None
    best_score = -np.inf
    
    while rho_max <= rho_max_cap:
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
        
        kept_points = len(points_images)
        drop_fraction = 1.0 - (kept_points / n_base) if n_base > 0 else 1.0
        
        if kept_points > 0:
            images_counts = [len(imgs) for imgs in points_images]
            q10 = float(np.percentile(images_counts, 10))
            q50 = float(np.percentile(images_counts, 50))
        else:
            images_counts = []
            q10 = 0.0
            q50 = 0.0
        
        metric_history.append(q10)
        
        # Score: prioritize Q10, then kept points
        score = q10 * 10 + kept_points
        
        LOGGER.debug(
            "k=%.2f adaptive_images: rho_max=%.3f, kept=%d/%d (drop=%.1f%%), Q10=%.1f, Q50=%.1f",
            k, rho_max, kept_points, n_base, drop_fraction * 100, q10, q50
        )
        
        # Track best result
        if score > best_score:
            best_score = score
            best_result = AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='best_found',
                metric_history=list(metric_history),
                images_per_point=images_counts,
                kept_points=kept_points,
                drop_fraction=drop_fraction,
            )
        
        # Check if criteria are met
        if q10 >= target_q10_images and drop_fraction <= max_drop_fraction:
            return AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='target_met',
                metric_history=metric_history,
                images_per_point=images_counts,
                kept_points=kept_points,
                drop_fraction=drop_fraction,
            )
        
        # Always expand up to cap
        if rho_max >= rho_max_cap:
            break
        
        rho_max = min(rho_max + expand_step, rho_max_cap)
        expansions += 1
    
    # Return best result if criteria never met
    if best_result is not None:
        best_result.stopped_by = 'cap_reached'
        best_result.metric_history = metric_history
        return best_result
    
    # Fallback
    return AdaptiveRhoResult(
        rho_min=rho_min,
        rho_max=rho_max_cap,
        rho_max_original=rho_max_original,
        expansions=expansions,
        stopped_by='cap_reached',
        metric_history=metric_history,
        kept_points=0,
        drop_fraction=1.0,
    )


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
    require_healthy: bool = True,
) -> AdaptiveRhoResult:
    """
    Expand ρ_max until numerical rank of A matrix reaches target.

    Includes σ_min vs τ health gate: expansion continues until:
    - rank(A) / N >= target_rank_frac, OR
    - nullity(A) = N - rank(A) >= target_nullity, OR
    - rho_max >= rho_max_cap
    
    If require_healthy=True, also requires σ_min > τ for target_met.

    Records comprehensive SVD diagnostics at each step.

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
    require_healthy : bool
        If True, require σ_min > τ for target_met status.

    Returns
    -------
    AdaptiveRhoResult
        Result containing final window and SVD health diagnostics.
    """
    # Get paper-faithful window as starting point
    rho_min, rho_max_paper, _ = compute_rho_cutoffs(k, L, l_min, threshold=threshold)
    rho_max_original = rho_max_paper
    n_base = len(base_points)
    
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
    best_result = None
    best_rank_frac = -1
    
    while rho_max <= rho_max_cap:
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
        
        kept_points = len(points_images)
        drop_fraction = 1.0 - (kept_points / n_base) if n_base > 0 else 1.0
        
        if not points_images or kept_points == 0:
            metric_history.append(0.0)
            if rho_max >= rho_max_cap:
                break
            rho_max = min(rho_max + expand_step, rho_max_cap)
            expansions += 1
            continue
        
        # Build matrix and compute SVD health
        svd_health = None
        rank_frac = 0.0
        nullity = 0
        M = 0
        N = 0
        
        try:
            A, M, N = matrix_builder(k, L, l_min, points_images)
            if M == 0 or N == 0:
                raise ValueError("Empty matrix")
            
            svd_health = compute_svd_health(A, svd_tol=svd_tol)
            rank_frac = svd_health.numerical_rank / N
            nullity = svd_health.nullity
            
        except Exception as e:
            LOGGER.debug("Matrix build failed for k=%.2f, rho_max=%.3f: %s", k, rho_max, e)
        
        metric_history.append(rank_frac)
        
        LOGGER.debug(
            "k=%.2f adaptive_rank: rho_max=%.3f, M=%d, N=%d, rank_frac=%.3f, nullity=%d, "
            "σ_min=%.2e, τ=%.2e, healthy=%s",
            k, rho_max, M, N,
            rank_frac, nullity,
            svd_health.sigma_min if svd_health else 0,
            svd_health.tau if svd_health else 0,
            svd_health.healthy if svd_health else False
        )
        
        # Track best result
        if rank_frac > best_rank_frac:
            best_rank_frac = rank_frac
            best_result = AdaptiveRhoResult(
                rho_min=rho_min,
                rho_max=rho_max,
                rho_max_original=rho_max_original,
                expansions=expansions,
                stopped_by='best_found',
                metric_history=list(metric_history),
                kept_points=kept_points,
                drop_fraction=drop_fraction,
                svd_health=svd_health.to_dict() if svd_health else None,
            )
        
        # Check stopping conditions
        meets_rank_target = rank_frac >= target_rank_frac or nullity >= target_nullity
        is_healthy = svd_health.healthy if svd_health else False
        
        if meets_rank_target:
            if not require_healthy or is_healthy:
                return AdaptiveRhoResult(
                    rho_min=rho_min,
                    rho_max=rho_max,
                    rho_max_original=rho_max_original,
                    expansions=expansions,
                    stopped_by='target_met',
                    metric_history=metric_history,
                    kept_points=kept_points,
                    drop_fraction=drop_fraction,
                    svd_health=svd_health.to_dict() if svd_health else None,
                )
            else:
                LOGGER.debug(
                    "k=%.2f: rank target met but σ_min=%.2e < τ=%.2e; continuing expansion",
                    k, svd_health.sigma_min, svd_health.tau
                )
        
        # Always expand up to cap
        if rho_max >= rho_max_cap:
            break
        
        rho_max = min(rho_max + expand_step, rho_max_cap)
        expansions += 1
    
    # Return best result if target never met
    if best_result is not None:
        best_result.stopped_by = 'cap_reached'
        best_result.metric_history = metric_history
        return best_result
    
    # Fallback
    return AdaptiveRhoResult(
        rho_min=rho_min,
        rho_max=rho_max_cap,
        rho_max_original=rho_max_original,
        expansions=expansions,
        stopped_by='cap_reached',
        metric_history=metric_history,
    )


__all__ = ['AdaptiveRhoResult', 'SVDHealth', 'compute_svd_health', 
           'adaptive_rho_images', 'adaptive_rho_rank']
