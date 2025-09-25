"""
Factory for creating CIAO components following counterfactuals pattern.
"""

from pathlib import Path
from typing import Literal, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

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
from ciao.typing import Classifier


def make_segmenter(
    method: Literal["superpixel", "hexagonal"], **kwargs
) -> BaseSegmentation:
    """
    Create segmentation component.

    Args:
        method: Segmentation method type
        **kwargs: Additional parameters for the segmenter

    Returns:
        Segmentation component
    """
    if method == "superpixel":
        return SuperpixelSegmentation(**kwargs)
    elif method == "hexagonal":
        return HexagonalSegmentation(**kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def make_obfuscator(
    method: Literal["interlacing", "zero", "noise", "blur"], **kwargs
) -> BaseObfuscation:
    """
    Create obfuscation component.

    Args:
        method: Obfuscation method type
        **kwargs: Additional parameters for the obfuscator

    Returns:
        Obfuscation component
    """
    if method == "interlacing":
        return PixelInterlacing(**kwargs)
    elif method == "zero":
        return ZeroOut(**kwargs)
    elif method == "noise":
        return NoiseObfuscation(**kwargs)
    elif method == "blur":
        return BlurObfuscation(**kwargs)
    else:
        raise ValueError(f"Unknown obfuscation method: {method}")


def make_classifier(
    variant: Literal["prostate", "colorectal"], model_path: Optional[Path] = None
) -> Classifier:
    """
    Create classifier for CIAO explanations.
    Uses standalone classifier implementation.

    Args:
        variant: Dataset variant (prostate or colorectal)
        model_path: Optional path to model weights

    Returns:
        Classifier model
    """
    from ciao.components.classifiers import load_pretrained_classifier

    # Default model path if not provided
    if model_path is None:
        model_path = Path(
            f"/mnt/data/rationai/data/Counterfactuals/models/{variant}_classifier.pt"
        )

    classifier = load_pretrained_classifier(variant, model_path)
    return classifier


def make_ciao_explainer(
    cfg: Union[DictConfig, ListConfig],
    variant: Literal["prostate", "colorectal"] = "prostate",
):
    """
    Create complete CIAO explainer from configuration.

    Args:
        cfg: Hydra configuration object
        variant: Dataset variant

    Returns:
        CIAO explainer instance
    """
    # This will be implemented after we create the main CIAO class
    from ciao.components.explainer.ciao_explainer import CIAOExplainer

    # Create classifier
    classifier = make_classifier(variant)

    # Create segmenter from config
    segmenter = instantiate(cfg.ciao.segmentation)

    # Create obfuscator from config
    obfuscator = instantiate(cfg.ciao.obfuscation)

    # Create explainer
    explainer = CIAOExplainer(
        classifier=classifier,
        segmenter=segmenter,
        obfuscator=obfuscator,
        **cfg.ciao.get("explainer_params", {}),
    )

    return explainer
