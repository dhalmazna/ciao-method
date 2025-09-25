"""
Main CIAO explainer class following Lightning framework.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from ciao.components.explainer.feature_selector import CIAOFeatureSelector
from ciao.components.obfuscation.base import BaseObfuscation
from ciao.components.segmentation.base import BaseSegmentation
from ciao.typing import Classifier, ExplanationResult, Image


class CIAOExplainer:
    """
    Main CIAO explainer class.
    Provides unified interface for generating CIAO explanations.
    """

    def __init__(
        self,
        classifier: Classifier,
        segmenter: BaseSegmentation,
        obfuscator: BaseObfuscation,
        eta: float = 0.01,
        radius_range: List[int] = [1, 2],
        adaptive_eta: bool = True,
        target_class: Optional[int] = None,
        use_fallback_groups: bool = True,
        max_group_size_ratio: float = 0.05,
    ):
        """
        Initialize CIAO explainer.

        Args:
            classifier: Classifier model to explain
            segmenter: Segmentation method
            obfuscator: Obfuscation method
            eta: Importance gain threshold
            radius_range: List of radii for local groups
            adaptive_eta: Whether to adapt eta threshold
            target_class: Class to explain (None for top prediction)
            use_fallback_groups: Whether to use fallback grouping when no groups form
            max_group_size_ratio: Maximum group size as ratio of total segments
        """
        self.classifier = classifier
        self.segmenter = segmenter
        self.obfuscator = obfuscator

        # Initialize feature selector
        self.feature_selector = CIAOFeatureSelector(
            model=classifier,
            target_class=target_class,
            eta=eta,
            radius_range=radius_range,
            adaptive_eta=adaptive_eta,
            use_fallback_groups=use_fallback_groups,
            max_group_size_ratio=max_group_size_ratio,
        )

    def explain(
        self,
        image: Image,
        target_class: Optional[int] = None,
        save_visualization: bool = True,
        output_path: Optional[Path] = None,
    ) -> ExplanationResult:
        """
        Generate CIAO explanation for an image.

        Args:
            image: Input image
            target_class: Class to explain (overrides default)
            save_visualization: Whether to save visualization
            output_path: Path to save results

        Returns:
            Dictionary containing explanation results
        """
        # Convert to numpy array for segmentation
        if isinstance(image, torch.Tensor):
            # Move to CPU and convert to numpy
            image_np = image.detach().cpu().numpy()
            if len(image_np.shape) == 4:  # Remove batch dimension if present
                image_np = image_np.squeeze(0)
            if image_np.shape[0] == 3:  # CHW to HWC
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image

        # Segment image
        segments, adjacency_graph = self.segmenter.segment(image_np)

        # Update target class if provided
        if target_class is not None:
            self.feature_selector.target_class = target_class

        # Get model prediction for original image
        original_prediction = self.feature_selector._get_model_prediction(image)
        predicted_class = original_prediction.argmax().item()
        confidence = original_prediction.max().item()

        # Select features using CIAO algorithm
        feature_groups = self.feature_selector.select_features(
            image_np, segments, adjacency_graph, self.obfuscator
        )

        # Create explanation result
        result = {
            "original_image": image_np,
            "segments": segments,
            "adjacency_graph": adjacency_graph,
            "feature_groups": feature_groups,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "target_class": (
                target_class
                if target_class is not None
                else self.feature_selector.target_class
            ),
            "eta_used": self.feature_selector.eta,
            "n_segments": len(np.unique(segments)),
            "n_feature_groups": len(feature_groups),
        }

        # Generate visualization if requested
        if save_visualization and output_path is not None:
            self._create_visualization(result, output_path)

        return result

    def explain_batch(
        self,
        images: List[Image],
        target_classes: Optional[List[int]] = None,
        save_visualizations: bool = True,
        output_dir: Optional[Path] = None,
    ) -> List[ExplanationResult]:
        """
        Generate CIAO explanations for a batch of images.

        Args:
            images: List of input images
            target_classes: List of target classes (optional)
            save_visualizations: Whether to save visualizations
            output_dir: Directory to save results

        Returns:
            List of explanation results
        """
        results = []

        for i, image in enumerate(images):
            target_class = target_classes[i] if target_classes is not None else None
            output_path = (
                output_dir / f"explanation_{i:03d}.png"
                if output_dir is not None
                else None
            )

            result = self.explain(
                image=image,
                target_class=target_class,
                save_visualization=save_visualizations,
                output_path=output_path,
            )
            results.append(result)

        return results

    def _create_visualization(
        self, result: ExplanationResult, output_path: Path
    ) -> None:
        """Create and save CIAO visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(result["original_image"])
        axes[0].set_title(
            f"Original\nPred: {result['predicted_class']} "
            f"(conf: {result['confidence']:.3f})"
        )
        axes[0].axis("off")

        # Segmentation
        axes[1].imshow(result["segments"], cmap="tab20")
        axes[1].set_title(f"Segmentation\n{result['n_segments']} segments")
        axes[1].axis("off")

        # Feature groups heatmap
        heatmap = self._create_heatmap(result)
        axes[2].imshow(result["original_image"])
        axes[2].imshow(heatmap, alpha=0.6, cmap="Reds")
        axes[2].set_title(
            f"CIAO Explanation\n{result['n_feature_groups']} groups, "
            f"η={result['eta_used']:.4f}"
        )
        axes[2].axis("off")

        plt.tight_layout()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _create_heatmap(self, result: ExplanationResult) -> np.ndarray:
        """Create importance heatmap from feature groups."""
        segments = result["segments"]
        feature_groups = result["feature_groups"]

        # Create importance map
        importance_map = np.zeros_like(segments, dtype=float)

        # Assign importance values to segments
        for i, group in enumerate(feature_groups):
            # Higher importance for earlier groups (found first)
            importance = 1.0 - (i / len(feature_groups)) * 0.8

            for segment_id in group:
                mask = segments == segment_id
                importance_map[mask] = importance

        return importance_map
