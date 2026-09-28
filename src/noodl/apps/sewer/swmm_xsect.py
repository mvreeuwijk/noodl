"""Alias, the pre-rename name of `noodl.apps.sewer.xsect_tables` (the tabulated circular
cross-section). Kept so existing imports keep working; every name resolves to that module."""

from noodl.apps.sewer import xsect_tables as _module
from noodl.apps.sewer.xsect_tables import *  # noqa: F403
from noodl.apps.sewer.xsect_tables import __all__  # noqa: F401


def __getattr__(name: str):
    return getattr(_module, name)
