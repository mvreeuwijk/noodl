"""Alias, the pre-rename name of `noodl.elements.powerlaw_regularized` (the regularised power
law). Kept so existing imports keep working; every name resolves to that module, including
the pre-rename names ``MBLPowerLaw`` and ``mbl_*``."""

from noodl.elements import powerlaw_regularized as _module
from noodl.elements.powerlaw_regularized import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_module, name)
