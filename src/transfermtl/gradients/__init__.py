"""gradients/ — regional gradient affinity (plan §2.7).

Public surface:
- compute_gradient_vector / compute_regional_gradient
- cosine_affinity, dot_product_affinity
- compute_trajectory_affinity
- write_region_affinity
"""

from transfermtl.gradients.affinity import cosine_affinity, dot_product_affinity
from transfermtl.gradients.extract import (
    compute_gradient_vector,
    compute_regional_gradient,
    encoder_param_order_hash,
)
from transfermtl.gradients.io import write_region_affinity
from transfermtl.gradients.trajectory import compute_trajectory_affinity

__all__ = [
    "compute_gradient_vector",
    "compute_regional_gradient",
    "compute_trajectory_affinity",
    "cosine_affinity",
    "dot_product_affinity",
    "encoder_param_order_hash",
    "write_region_affinity",
]
