"""
CIAO: Contextual Importance Assessment via Obfuscation
Enterprise Architecture Implementation
"""

__version__ = "0.1.0"
__author__ = "Your Lab"

from ciao.components.factory import make_ciao_explainer
from ciao.data.data_module import CIAODataModule
from ciao.data.datasets import ColorectalCancer, ProstateCancer

__all__ = [
    "make_ciao_explainer",
    "CIAODataModule",
    "ColorectalCancer",
    "ProstateCancer",
]
