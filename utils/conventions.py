"""
Shared conventions for the Cornish & Spergel “method of ghosts” pipeline.

* The scan variable is k.
* nu is identified with k everywhere (no sqrt(k**2 + 1)).
* q^2 = k^2 + 1.
"""

from __future__ import annotations


def k_to_nu(k: float) -> float:
    """Return nu for a given k. By convention nu == k."""
    return float(k)


def q_squared(k: float) -> float:
    """Return q^2 for a given k using q^2 = k^2 + 1."""
    k = float(k)
    return k * k + 1.0


__all__ = ["k_to_nu", "q_squared"]
