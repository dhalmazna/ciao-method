"""
Data package initialization.
"""

from ciao.data.data_module import CIAODataModule
from ciao.data.datasets import ColorectalCancer, ProstateCancer, ProstateCancerPredict

__all__ = [
    "CIAODataModule",
    "ColorectalCancer",
    "ProstateCancer",
    "ProstateCancerPredict",
]
