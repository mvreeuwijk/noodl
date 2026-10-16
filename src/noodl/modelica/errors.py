"""The error both Modelica readers raise."""

from __future__ import annotations

__all__ = ["ModelicaImportError"]


class ModelicaImportError(ValueError):
    """Raised for a malformed `noodl-modelica/1` (building) or `noodl-modelica-electrical/1`
    (circuit) document, or for model content the reader does not support.

    The message names every offending field, instance, connector or class (unsupported content
    is never silently approximated or dropped). Both readers gather every refusal they find
    and raise exactly one of these, rather than stopping at the first one.
    """
