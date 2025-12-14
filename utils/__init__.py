# This file exports the core pipeline functions from the utils package
# For the current paper-faithful pipeline (main.py)

# Core pipeline functions
from .cutoffs import compute_rho_cutoffs
from .ghosts import enumerate_ghost_images, get_group_elements
from .points import sample_points_in_dirichlet_domain, DEFAULT_FALLBACK_RADIUS
from .sys_generation import generate_matrix_system, generate_matrix_system_scalar
from .svd import solve_system_via_svd_numeric
from .eigenvalues import extract_eigenvalues_from_spectrum
from .diagnostics import diagnose_base_points, write_point_diagnostics

# Transformations used by pipeline
from .transformations import (
    apply_so31_action,
    klein_to_poincare,
    poincare_distance,
    project_to_klein,
)

# Special functions (internal use, but exported for completeness)
from .special_functions import (
    Q_k_lm, 
    Q_k_lm_vectorized,
    clear_special_function_caches,
    get_cache_stats,
    get_call_counters,
    reset_call_counters,
)

# Legacy/optional imports (for backward compatibility with old scripts in legacy/)
try:
    from .domain_building import (
        convert_to_4x4_matrices,
        build_dirichlet_domain,
        generate_random_points_in_domain,
        filter_points_in_domain,
        select_points,
    )
except Exception:  # pragma: no cover - optional dependency
    pass

try:
    from .parameter_control import (
        compute_target_M,
        filter_points_for_overconstraint,
        select_points_for_c,
        determine_tiling_radius,
    )
except Exception:  # pragma: no cover - optional dependency
    pass
