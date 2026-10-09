"""Pre-rename module name, kept as an alias: the layer now lives in `noodl.layers.allocation`."""

from noodl.layers.allocation import AllocatedFlowLayer, CapacitatedTransferLayer

__all__ = ["AllocatedFlowLayer", "CapacitatedTransferLayer"]
