"""Region partitioning schemes (plan §2.6).

Owned by A5. Public API:
- ``compute_scaffold_partition`` — primary scheme, HAC on Tanimoto over scaffold FPs.
- ``compute_latent_partition``  — secondary scheme, k-means on encoder embeddings.
- ``compute_knn_partition``     — tertiary scheme, scaffold-centroid kNN in encoder space.
- ``generate_random_partitions`` — negative-control random partitions.

The ``*_from_*`` helpers expose a pure-compute form for tests, so modules can
be exercised on the synthetic fixture without touching disk or requiring a
trained encoder.
"""

from transfermtl.partition.io import write_partition
from transfermtl.partition.knn import compute_knn_partition, knn_partition_from_embeddings
from transfermtl.partition.latent import (
    compute_latent_partition,
    latent_partition_from_embeddings,
)
from transfermtl.partition.random_null import (
    generate_random_partitions,
    random_partition_from_sizes,
)
from transfermtl.partition.scaffold import (
    compute_scaffold_partition,
    scaffold_partition_from_fps,
)

__all__ = [
    "compute_knn_partition",
    "compute_latent_partition",
    "compute_scaffold_partition",
    "generate_random_partitions",
    "knn_partition_from_embeddings",
    "latent_partition_from_embeddings",
    "random_partition_from_sizes",
    "scaffold_partition_from_fps",
    "write_partition",
]
