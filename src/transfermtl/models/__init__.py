"""GCN encoder, MLP head, multi-head composition, and encoder registry."""

# Importing registry triggers @register("encoder", "gcn") side effects.
from transfermtl.models import registry as _registry  # noqa: F401
from transfermtl.models.gcn import GCNEncoder
from transfermtl.models.heads import TaskHead
from transfermtl.models.multihead import MultiHeadModel

__all__ = ["GCNEncoder", "MultiHeadModel", "TaskHead"]
