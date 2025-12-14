"""
Test tiling and image enumeration correctness.

Validates:
1. Stability: Same seed → same number of images (deterministic)
2. No duplicates: Images are unique (no clustering at machine epsilon)
3. Growth sanity: Image count grows with radius (hyperbolic volume)
4. Deep word tests: word_depth=2,3,4 produce valid results

Critical for ensuring ghost image enumeration is correct before χ² construction.
"""
import os
import sys
import numpy as np
import pytest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.ghosts import enumerate_ghost_images, get_group_elements
from utils.transformations import poincare_distance


def compute_pairwise_distances(images):
    """Compute all pairwise distances between images (as pseudo-spherical coords)."""
    distances = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            # Images are in (rho, theta, phi) format
            rho1, theta1, phi1 = images[i]
            rho2, theta2, phi2 = images[j]
            
            # Simple Euclidean distance in (rho, theta, phi) space
            # (Not hyperbolic distance, just for duplicate detection)
            d = np.sqrt((rho1 - rho2)**2 + (theta1 - theta2)**2 + (phi1 - phi2)**2)
            distances.append(d)
    
    return np.array(distances)


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("manifold_name", ["m003(-2,3)", "m188(-1,1)"])
def test_image_count_stability_across_runs(seed, manifold_name):
    """Test that same seed produces same number of images (deterministic)."""
    base_point = [0.1, 0.05, -0.07]
    rho_min, rho_max = 0.5, 3.0
    min_images = 10
    
    # Run enumeration twice with same parameters
    images1 = enumerate_ghost_images(
        manifold_name,
        base_points=[base_point],
        rho_min=rho_min,
        rho_max=rho_max,
        min_images=min_images,
        max_word_length=3,
    )
    
    images2 = enumerate_ghost_images(
        manifold_name,
        base_points=[base_point],
        rho_min=rho_min,
        rho_max=rho_max,
        min_images=min_images,
        max_word_length=3,
    )
    
    assert images1, "No images returned (first run)"
    assert images2, "No images returned (second run)"
    
    n1 = len(images1[0])
    n2 = len(images2[0])
    
    assert n1 == n2, (
        f"{manifold_name}, seed {seed}: Image count unstable!\n"
        f"Run 1: {n1} images\n"
        f"Run 2: {n2} images"
    )
    
    print(f"{manifold_name}, seed {seed}: Stable with {n1} images")


@pytest.mark.parametrize("manifold_name", ["m003(-2,3)", "m004(-2,3)", "m188(-1,1)"])
def test_no_duplicate_images(manifold_name):
    """Test that images are unique (no clustering at machine epsilon)."""
    base_point = [0.0, 0.0, 0.0]  # Use origin for simplicity
    rho_min, rho_max = 0.0, 4.0
    
    images = enumerate_ghost_images(
        manifold_name,
        base_points=[base_point],
        rho_min=rho_min,
        rho_max=rho_max,
        min_images=20,
        max_word_length=3,
    )
    
    if not images or not images[0]:
        pytest.skip(f"No images generated for {manifold_name}")
    
    images_list = images[0]
    n_images = len(images_list)
    
    # Compute pairwise distances in (rho, theta, phi) space
    distances = compute_pairwise_distances(images_list)
    
    if len(distances) == 0:
        pytest.skip("Only 1 image, cannot check duplicates")
    
    # Check for near-duplicates (distance < 1e-10)
    near_duplicates = np.sum(distances < 1e-10)
    
    assert near_duplicates == 0, (
        f"{manifold_name}: Found {near_duplicates} near-duplicate image pairs!\n"
        f"Total images: {n_images}\n"
        f"Min distance: {np.min(distances):.2e}\n"
        f"Distances < 1e-10: {near_duplicates}"
    )
    
    # Additional check: No exact duplicates (set membership)
    # Round to reasonable precision for set comparison
    rounded_images = [tuple(np.round(img, decimals=12)) for img in images_list]
    unique_count = len(set(rounded_images))
    
    assert unique_count == n_images, (
        f"{manifold_name}: Duplicate images detected!\n"
        f"Total images: {n_images}\n"
        f"Unique (rounded): {unique_count}\n"
        f"Duplicates: {n_images - unique_count}"
    )
    
    print(f"{manifold_name}: {n_images} images, all unique")
    print(f"  Min pairwise distance: {np.min(distances):.2e}")
    print(f"  Median pairwise distance: {np.median(distances):.2e}")


@pytest.mark.parametrize("manifold_name", ["m003(-2,3)", "m188(-1,1)"])
def test_image_growth_with_radius(manifold_name):
    """Test that image count grows with rho_max (hyperbolic volume growth)."""
    base_point = [0.0, 0.0, 0.0]
    
    # Test different rho_max values
    test_radii = [1.5, 2.5, 3.5, 4.5]
    image_counts = []
    
    for rho_max in test_radii:
        images = enumerate_ghost_images(
            manifold_name,
            base_points=[base_point],
            rho_min=0.0,
            rho_max=rho_max,
            min_images=5,
            max_word_length=4,
        )
        
        if not images or not images[0]:
            continue
        
        count = len(images[0])
        image_counts.append(count)
    
    if len(image_counts) < 3:
        pytest.skip(f"Insufficient data points for {manifold_name}")
    
    # Check that counts generally increase
    # (Allow for some non-monotonicity due to discrete group structure)
    increasing_pairs = sum(1 for i in range(len(image_counts)-1) 
                          if image_counts[i+1] >= image_counts[i])
    total_pairs = len(image_counts) - 1
    
    assert increasing_pairs >= total_pairs * 0.6, (
        f"{manifold_name}: Image count not growing with radius!\n"
        f"Radii: {test_radii[:len(image_counts)]}\n"
        f"Counts: {image_counts}\n"
        f"Increasing pairs: {increasing_pairs}/{total_pairs}"
    )
    
    print(f"{manifold_name}: Image growth with radius:")
    for r, c in zip(test_radii[:len(image_counts)], image_counts):
        print(f"  rho_max={r:.1f}: {c} images")


@pytest.mark.parametrize("word_depth", [1, 2, 3, 4])
@pytest.mark.parametrize("manifold_name", ["m003(-2,3)", "m188(-1,1)"])
def test_deep_word_enumeration(word_depth, manifold_name):
    """Test that deeper word depths produce valid group elements."""
    group_elements, fallback = get_group_elements(manifold_name, max_depth=word_depth)
    
    if fallback:
        pytest.skip(f"SnapPy not available for {manifold_name}")
    
    assert len(group_elements) > 0, f"No group elements at depth {word_depth}"
    
    # Check that we have identity
    identity = np.eye(4)
    has_identity = any(np.allclose(elem, identity, atol=1e-10) for elem in group_elements)
    assert has_identity, f"Identity not found at depth {word_depth}"
    
    # Check that all elements are valid 4x4 matrices
    for elem in group_elements:
        assert elem.shape == (4, 4), f"Invalid matrix shape: {elem.shape}"
        assert np.all(np.isfinite(elem)), "Matrix contains inf/nan"
    
    # Check growth: deeper depths should have more elements (generally)
    if word_depth > 1:
        group_shallow, _ = get_group_elements(manifold_name, max_depth=word_depth-1)
        assert len(group_elements) >= len(group_shallow), (
            f"Element count decreased at depth {word_depth}?\n"
            f"Depth {word_depth-1}: {len(group_shallow)} elements\n"
            f"Depth {word_depth}: {len(group_elements)} elements"
        )
    
    print(f"{manifold_name}, depth {word_depth}: {len(group_elements)} group elements")


@pytest.mark.parametrize("word_depth", [2, 3, 4])
def test_images_with_deep_words(word_depth):
    """Test that image enumeration works with deeper word depths."""
    manifold_name = "m003(-2,3)"
    base_point = [0.1, 0.0, 0.0]
    
    images = enumerate_ghost_images(
        manifold_name,
        base_points=[base_point],
        rho_min=0.0,
        rho_max=3.0,
        min_images=10,
        max_word_length=word_depth,
    )
    
    assert images, f"No images returned at word depth {word_depth}"
    assert len(images[0]) >= 10, f"Fewer than min_images at depth {word_depth}"
    
    # All images should be in valid range
    for rho, theta, phi in images[0]:
        assert rho >= 0.0, f"Invalid rho={rho} < 0"
        assert 0.0 <= theta <= np.pi, f"Invalid theta={theta}"
        assert -np.pi <= phi <= np.pi, f"Invalid phi={phi}"
    
    print(f"Word depth {word_depth}: {len(images[0])} images generated")


def test_word_depth_comparison():
    """Compare image counts across different word depths for same parameters."""
    manifold_name = "m003(-2,3)"
    base_point = [0.0, 0.0, 0.0]
    rho_min, rho_max = 0.0, 4.0
    
    results = {}
    for depth in [1, 2, 3, 4]:
        images = enumerate_ghost_images(
            manifold_name,
            base_points=[base_point],
            rho_min=rho_min,
            rho_max=rho_max,
            min_images=10,
            max_word_length=depth,
        )
        
        if images and images[0]:
            results[depth] = len(images[0])
    
    assert len(results) >= 3, "Insufficient depths tested"
    
    print(f"\n{manifold_name}: Image count vs word depth (rho_max={rho_max}):")
    for depth in sorted(results.keys()):
        print(f"  Depth {depth}: {results[depth]} images")
    
    # Deeper words should generally give more images
    # (or at least not significantly fewer)
    for i in range(len(results) - 1):
        d1 = list(results.keys())[i]
        d2 = list(results.keys())[i + 1]
        
        # Allow some decrease due to finite rho_max cutoff
        assert results[d2] >= results[d1] * 0.8, (
            f"Significant decrease in images from depth {d1} to {d2}:\n"
            f"Depth {d1}: {results[d1]} images\n"
            f"Depth {d2}: {results[d2]} images"
        )


def test_group_element_uniqueness():
    """Test that BFS enumeration doesn't produce duplicate group elements."""
    manifold_name = "m188(-1,1)"
    
    for depth in [2, 3, 4]:
        group_elements, fallback = get_group_elements(manifold_name, max_depth=depth)
        
        if fallback:
            pytest.skip(f"SnapPy not available for {manifold_name}")
        
        # Round matrices for comparison
        rounded_matrices = []
        for mat in group_elements:
            rounded = tuple(np.round(mat.flatten(), decimals=8))
            rounded_matrices.append(rounded)
        
        unique_count = len(set(rounded_matrices))
        total_count = len(group_elements)
        
        assert unique_count == total_count, (
            f"{manifold_name}, depth {depth}: Duplicate group elements!\n"
            f"Total: {total_count}\n"
            f"Unique: {unique_count}\n"
            f"Duplicates: {total_count - unique_count}"
        )
        
        print(f"{manifold_name}, depth {depth}: {total_count} unique group elements")


@pytest.mark.parametrize("manifold_name", ["m003(-2,3)", "m188(-1,1)"])
def test_hyperbolic_volume_scaling(manifold_name):
    """Test that image counts scale roughly with hyperbolic volume."""
    # Hyperbolic volume of ball: V(r) = π(sinh(2r) - 2r) / 2
    # Should grow exponentially: V(r) ≈ (π/4) exp(2r) for large r
    
    base_point = [0.0, 0.0, 0.0]
    test_data = []
    
    for rho_max in [2.0, 3.0, 4.0]:
        images = enumerate_ghost_images(
            manifold_name,
            base_points=[base_point],
            rho_min=0.0,
            rho_max=rho_max,
            min_images=5,
            max_word_length=4,
        )
        
        if images and images[0]:
            count = len(images[0])
            # Theoretical volume (in fundamental domain)
            volume = np.pi * (np.sinh(2 * rho_max) - 2 * rho_max) / 2
            test_data.append((rho_max, count, volume))
    
    if len(test_data) < 2:
        pytest.skip("Insufficient data for scaling test")
    
    print(f"\n{manifold_name}: Image count vs hyperbolic volume:")
    for rho, count, vol in test_data:
        print(f"  rho={rho:.1f}: {count} images, V≈{vol:.1f}")
    
    # Check that ratio of counts is roughly proportional to ratio of volumes
    # (within a factor of 3, accounting for discrete group structure)
    if len(test_data) >= 2:
        rho1, count1, vol1 = test_data[0]
        rho2, count2, vol2 = test_data[-1]
        
        count_ratio = count2 / count1
        volume_ratio = vol2 / vol1
        
        # Ratio should be within a factor of 3
        ratio_of_ratios = count_ratio / volume_ratio
        
        assert 0.3 < ratio_of_ratios < 3.0, (
            f"Image scaling doesn't match volume scaling:\n"
            f"Count ratio ({count1}→{count2}): {count_ratio:.2f}\n"
            f"Volume ratio: {volume_ratio:.2f}\n"
            f"Ratio of ratios: {ratio_of_ratios:.2f} (expected ≈1)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
