"""
Standalone classifier implementations for CIAO.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """
    Simple classifier that can load pretrained weights.

    For CIAO explanations, we just need a callable that takes images
    and returns predictions. This is a minimal implementation that can
    load weights from various model formats.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        # For now, use a simple ResNet-based classifier
        from torchvision.models import resnet18

        self.backbone = resnet18(pretrained=True)

        # Replace the final layer for binary classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make the classifier callable for CIAO protocol."""
        return self.forward(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities for CIAO explanations."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def load_pretrained_classifier(
    variant: str, model_path: Optional[Path] = None, device: str = "cpu"
) -> SimpleClassifier:
    """
    Load a pretrained classifier for the given variant.

    Args:
        variant: Dataset variant (prostate, colorectal)
        model_path: Optional path to model weights
        device: Device to load model on

    Returns:
        Loaded classifier model
    """
    classifier = SimpleClassifier(num_classes=1)

    if model_path and model_path.exists():
        try:
            # Try to load weights
            state_dict = torch.load(model_path, map_location=device)

            # Handle different state dict formats
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            classifier.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {model_path}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Using randomly initialized classifier")
    else:
        print("Using randomly initialized classifier (for demonstration)")

    classifier.eval()
    return classifier
