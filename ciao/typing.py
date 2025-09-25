"""
CIAO typing definitions
"""

from typing import Any, Dict, List, Protocol, Tuple, Union

import numpy as np
import torch

# Type aliases for CIAO
ImageTensor = torch.Tensor
ImageArray = np.ndarray
Image = Union[ImageTensor, ImageArray]
SegmentMask = np.ndarray
ExplanationResult = Dict[str, Any]
FeatureGroups = List[List[int]]

# Medical imaging dataset types
Sample = tuple[torch.Tensor, torch.Tensor]  # (image, label)
PredictSample = tuple[torch.Tensor, Dict[str, Any]]  # (image, metadata)


class Segmenter(Protocol):
    """Protocol for segmentation methods."""

    def segment(self, image: Image) -> SegmentMask:
        """Segment image into regions."""
        ...


class Obfuscator(Protocol):
    """Protocol for obfuscation methods."""

    def obfuscate(self, image: Image, mask: SegmentMask, segments: List[int]) -> Image:
        """Obfuscate specified segments in image."""
        ...


class Classifier(Protocol):
    """Protocol for classifiers."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Classify input tensor."""
        ...

    def eval(self) -> None:
        """Set to evaluation mode."""
        ...
