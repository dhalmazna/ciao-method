"""
CIAO components package initialization.
"""

from ciao.components.factory import (
    make_ciao_explainer,
    make_classifier,
    make_obfuscator,
    make_segmenter,
)
from ciao.components.obfuscation import (
    BaseObfuscation,
    BlurObfuscation,
    NoiseObfuscation,
    PixelInterlacing,
    ZeroOut,
)
from ciao.components.segmentation import (
    BaseSegmentation,
    HexagonalSegmentation,
    SuperpixelSegmentation,
)

__all__ = [
    # Factory functions
    "make_ciao_explainer",
    "make_classifier",
    "make_obfuscator",
    "make_segmenter",
    # Segmentation components
    "BaseSegmentation",
    "SuperpixelSegmentation",
    "HexagonalSegmentation",
    # Obfuscation components
    "BaseObfuscation",
    "PixelInterlacing",
    "ZeroOut",
    "NoiseObfuscation",
    "BlurObfuscation",
]
